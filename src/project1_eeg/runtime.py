from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluation import compute_retrieval_metrics
from .image_banks import TensorBank
from .retrieval import RetrievalOutputs
from .utils import ensure_dir


def make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def compute_retrieval_outputs(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    model.eval()
    outputs_by_head: dict[str, list[torch.Tensor]] = {
        "semantic": [],
        "perceptual": [],
        "rerank": [],
        "legacy": [],
        "contour": [],
        "object": [],
        "context": [],
        "fused": [],
        "route_weights": [],
    }
    image_ids: list[str] = []
    image_paths: list[str] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        outputs: RetrievalOutputs = model.encode_all(eeg)
        if outputs.semantic is not None:
            outputs_by_head["semantic"].append(outputs.semantic.float().cpu())
        if outputs.perceptual is not None:
            outputs_by_head["perceptual"].append(outputs.perceptual.float().cpu())
        rerank_output = getattr(outputs, "rerank", None)
        if rerank_output is not None:
            outputs_by_head["rerank"].append(rerank_output.float().cpu())
        if outputs.legacy is not None:
            outputs_by_head["legacy"].append(outputs.legacy.float().cpu())
        contour_output = getattr(outputs, "contour", None)
        if contour_output is not None:
            outputs_by_head["contour"].append(contour_output.float().cpu())
        object_output = getattr(outputs, "object", None)
        if object_output is not None:
            outputs_by_head["object"].append(object_output.float().cpu())
        context_output = getattr(outputs, "context", None)
        if context_output is not None:
            outputs_by_head["context"].append(context_output.float().cpu())
        fused_output = getattr(outputs, "fused", None)
        if fused_output is not None:
            outputs_by_head["fused"].append(fused_output.float().cpu())
        route_weights = getattr(outputs, "route_weights", None)
        if route_weights is not None:
            outputs_by_head["route_weights"].append(route_weights.float().cpu())
        image_ids.extend(list(batch["image_id"]))
        image_paths.extend(list(batch["image_path"]))

    stacked = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in outputs_by_head.items()
        if chunks
    }
    return stacked, image_ids, image_paths


@torch.no_grad()
def compute_embeddings(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    head: str | None = None,
) -> tuple[torch.Tensor, list[str], list[str]]:
    outputs, image_ids, image_paths = compute_retrieval_outputs(model, loader, device)
    resolved_head = head
    if resolved_head is None:
        if "fused" in outputs:
            resolved_head = "fused"
        elif "context" in outputs:
            resolved_head = "context"
        elif "object" in outputs:
            resolved_head = "object"
        elif "contour" in outputs:
            resolved_head = "contour"
        elif "semantic" in outputs:
            resolved_head = "semantic"
        elif "perceptual" in outputs:
            resolved_head = "perceptual"
        else:
            resolved_head = "legacy"
    if resolved_head not in outputs:
        raise KeyError(f"Retrieval head '{resolved_head}' was not produced by the model.")
    return outputs[resolved_head], image_ids, image_paths


def _resolve_candidate_image_ids(
    *banks: TensorBank | None,
    candidate_image_ids: Iterable[str] | None = None,
) -> list[str]:
    if candidate_image_ids is not None:
        return list(candidate_image_ids)

    resolved_banks = [bank for bank in banks if bank is not None]
    if not resolved_banks:
        raise ValueError("At least one candidate bank is required.")
    base_ids = list(resolved_banks[0].image_ids)
    for bank in resolved_banks[1:]:
        if bank.image_ids != base_ids:
            raise ValueError("All retrieval banks must share the same candidate ordering.")
    return base_ids


def _exclude_self_matches(
    logits: torch.Tensor,
    candidate_image_ids: list[str],
    *,
    query_image_ids: Iterable[str] | None,
    exclude_self: bool,
) -> torch.Tensor:
    if not exclude_self or query_image_ids is None:
        return logits

    masked = logits.clone()
    index = {image_id: idx for idx, image_id in enumerate(candidate_image_ids)}
    for row, image_id in enumerate(query_image_ids):
        candidate_idx = index.get(image_id)
        if candidate_idx is not None:
            masked[row, candidate_idx] = float("-inf")
    return masked


def compute_retrieval_logits(
    model,
    outputs: dict[str, torch.Tensor],
    *,
    semantic_bank: TensorBank | None = None,
    perceptual_bank: TensorBank | None = None,
    rerank_bank: TensorBank | None = None,
    contour_bank: TensorBank | None = None,
    object_bank: TensorBank | None = None,
    context_bank: TensorBank | None = None,
    fused_bank: TensorBank | None = None,
    candidate_image_ids: Iterable[str] | None = None,
    alpha: float = 0.5,
    rerank_alpha: float = 0.0,
    rerank_topk: int = 0,
) -> tuple[torch.Tensor, list[str], dict[str, torch.Tensor]]:
    candidate_ids = _resolve_candidate_image_ids(
        semantic_bank,
        perceptual_bank,
        rerank_bank,
        contour_bank,
        object_bank,
        context_bank,
        fused_bank,
        candidate_image_ids=candidate_image_ids,
    )
    model_device = next(model.parameters()).device
    component_logits: dict[str, torch.Tensor] = {}

    semantic_query = outputs.get("semantic")
    if semantic_query is None and semantic_bank is not None and outputs.get("legacy") is not None:
        semantic_query = outputs["legacy"]

    if semantic_query is not None and semantic_bank is not None:
        semantic_head = "semantic" if outputs.get("semantic") is not None else "legacy"
        semantic_values = semantic_bank.align(candidate_ids, device=model_device).float()
        component_logits["semantic"] = model.similarity(
            semantic_query.to(model_device),
            semantic_values,
            head=semantic_head,
        ).cpu()

    if outputs.get("perceptual") is not None and perceptual_bank is not None:
        perceptual_values = perceptual_bank.align(candidate_ids, device=model_device).float()
        component_logits["perceptual"] = model.similarity(
            outputs["perceptual"].to(model_device),
            perceptual_values,
            head="perceptual",
        ).cpu()

    coarse_logits: torch.Tensor | None = None
    route_logits: torch.Tensor | None = None

    if "semantic" in component_logits and "perceptual" in component_logits:
        coarse_logits = alpha * component_logits["semantic"] + (1.0 - alpha) * component_logits["perceptual"]
    elif "semantic" in component_logits:
        coarse_logits = component_logits["semantic"]
    elif "perceptual" in component_logits:
        coarse_logits = component_logits["perceptual"]

    if outputs.get("contour") is not None or outputs.get("object") is not None or outputs.get("context") is not None:
        if outputs.get("contour") is not None and contour_bank is not None:
            contour_values = contour_bank.align(candidate_ids, device=model_device).float()
            component_logits["contour"] = model.similarity(
                outputs["contour"].to(model_device),
                contour_values,
                head="contour",
            ).cpu()
        if outputs.get("object") is not None and object_bank is not None:
            object_values = object_bank.align(candidate_ids, device=model_device).float()
            component_logits["object"] = model.similarity(
                outputs["object"].to(model_device),
                object_values,
                head="object",
            ).cpu()
        if outputs.get("context") is not None and context_bank is not None:
            context_values = context_bank.align(candidate_ids, device=model_device).float()
            component_logits["context"] = model.similarity(
                outputs["context"].to(model_device),
                context_values,
                head="context",
            ).cpu()
        if outputs.get("fused") is not None and fused_bank is not None:
            fused_values = fused_bank.align(candidate_ids, device=model_device).float()
            component_logits["fused"] = model.similarity(
                outputs["fused"].to(model_device),
                fused_values,
                head="fused",
            ).cpu()

        branch_names = [name for name in ["contour", "object", "context", "fused"] if name in component_logits]
        if not branch_names:
            raise ValueError("No visible retrieval branches were available for the requested banks.")
        if outputs.get("route_weights") is not None:
            route_index = {"contour": 0, "object": 1, "context": 2, "fused": 3}
            weights = outputs["route_weights"][:, [route_index[name] for name in branch_names]].cpu()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            stacked = torch.stack([component_logits[name] for name in branch_names], dim=1)
            route_logits = (weights.unsqueeze(-1) * stacked).sum(dim=1)
            component_logits["route"] = route_logits
        elif "fused" in component_logits:
            route_logits = component_logits["fused"]
            component_logits["route"] = route_logits
        else:
            route_logits = torch.stack([component_logits[name] for name in branch_names], dim=0).mean(dim=0)
            component_logits["route"] = route_logits

    if coarse_logits is not None and route_logits is not None:
        fused = alpha * coarse_logits + (1.0 - alpha) * route_logits
    elif coarse_logits is not None:
        fused = coarse_logits
    elif route_logits is not None:
        fused = route_logits
    else:
        raise ValueError("No compatible retrieval heads were available for the requested banks.")

    if coarse_logits is not None:
        component_logits["coarse"] = coarse_logits
    component_logits["hybrid"] = fused

    if outputs.get("rerank") is not None and rerank_bank is not None and rerank_alpha > 0.0 and rerank_topk > 0:
        rerank_values = rerank_bank.align(candidate_ids, device=model_device).float()
        rerank_logits = model.similarity(
            outputs["rerank"].to(model_device),
            rerank_values,
            head="rerank",
        ).cpu()
        component_logits["rerank"] = rerank_logits
        topk = min(int(rerank_topk), fused.shape[1])
        if topk > 0:
            reranked = fused.clone()
            topk_indices = fused.topk(k=topk, dim=1).indices
            row_indices = torch.arange(fused.shape[0]).unsqueeze(1)
            reranked[row_indices, topk_indices] = reranked[row_indices, topk_indices] + (
                rerank_alpha * rerank_logits[row_indices, topk_indices]
            )
            fused = reranked

    return fused, candidate_ids, component_logits


def select_best_alpha(
    *,
    semantic_logits: torch.Tensor | None,
    perceptual_logits: torch.Tensor | None,
    ordered_image_ids: list[str],
    candidate_image_ids: list[str],
    alpha_grid: Iterable[float] | None = None,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    if semantic_logits is None and perceptual_logits is None:
        raise ValueError("At least one logit matrix is required.")
    if semantic_logits is not None and perceptual_logits is None:
        metrics = compute_retrieval_metrics(
            semantic_logits,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        return 1.0, metrics, [{"alpha": 1.0, **metrics}]
    if perceptual_logits is not None and semantic_logits is None:
        metrics = compute_retrieval_metrics(
            perceptual_logits,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        return 0.0, metrics, [{"alpha": 0.0, **metrics}]

    assert semantic_logits is not None
    assert perceptual_logits is not None
    search_grid = list(alpha_grid or [step / 10.0 for step in range(11)])

    best_alpha = search_grid[0]
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    alpha_history: list[dict[str, float]] = []

    for alpha in search_grid:
        fused = alpha * semantic_logits + (1.0 - alpha) * perceptual_logits
        metrics = compute_retrieval_metrics(
            fused,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        alpha_history.append({"alpha": float(alpha), **metrics})
        score = (
            float(metrics["top1_acc"]),
            float(metrics["top5_acc"]),
            -abs(float(alpha) - 0.5),
        )
        if best_score is None or score > best_score:
            best_alpha = float(alpha)
            best_metrics = metrics
            best_score = score

    assert best_metrics is not None
    return best_alpha, best_metrics, alpha_history


def prototype_lookup(
    query_embeddings: torch.Tensor,
    candidate_bank: TensorBank,
    *,
    query_image_ids: Iterable[str] | None = None,
    exclude_self: bool = False,
    topk: int = 1,
) -> tuple[torch.Tensor, list[list[str]]]:
    candidate_values = candidate_bank.values.float()
    logits = query_embeddings.float().cpu() @ candidate_values.T.cpu()
    logits = _exclude_self_matches(
        logits,
        list(candidate_bank.image_ids),
        query_image_ids=query_image_ids,
        exclude_self=exclude_self,
    )

    topk_idx = logits.topk(k=topk, dim=1).indices
    topk_ids = [
        [candidate_bank.image_ids[index] for index in row.tolist()]
        for row in topk_idx
    ]
    return topk_idx, topk_ids


def prototype_lookup_from_logits(
    logits: torch.Tensor,
    *,
    candidate_image_ids: list[str],
    query_image_ids: Iterable[str] | None = None,
    exclude_self: bool = False,
    topk: int = 1,
) -> tuple[torch.Tensor, list[list[str]], torch.Tensor]:
    filtered = _exclude_self_matches(
        logits.cpu(),
        candidate_image_ids,
        query_image_ids=query_image_ids,
        exclude_self=exclude_self,
    )
    topk_result = filtered.topk(k=topk, dim=1)
    topk_ids = [[candidate_image_ids[index] for index in row.tolist()] for row in topk_result.indices]
    return topk_result.indices, topk_ids, topk_result.values


def aggregate_prototype_latents(
    candidate_latent_bank: TensorBank,
    topk_ids: list[list[str]],
    topk_scores: torch.Tensor,
    *,
    device: torch.device,
    prototype_mode: str = "top1",
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = len(topk_ids[0])
    flat_ids = [image_id for row in topk_ids for image_id in row]
    latent_shape = candidate_latent_bank.values.shape[1:]
    latents = candidate_latent_bank.align(flat_ids, device=device).float().reshape(-1, topk, *latent_shape)

    if prototype_mode == "top1":
        weights = torch.zeros(topk_scores.shape, device=device, dtype=torch.float32)
        weights[:, 0] = 1.0
        return latents[:, 0], weights
    if prototype_mode != "score_weighted_topk":
        raise ValueError(f"Unknown prototype_mode: {prototype_mode}")

    weights = torch.softmax(topk_scores.to(device).float(), dim=-1)
    fused = (latents * weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)).sum(dim=1)
    return fused, weights


@torch.no_grad()
def select_prototype_latents(
    model,
    eeg: torch.Tensor,
    *,
    candidate_semantic_bank: TensorBank | None,
    candidate_perceptual_bank: TensorBank | None,
    candidate_latent_bank: TensorBank,
    alpha: float,
    query_image_ids: list[str] | None,
    exclude_self: bool,
    device: torch.device,
    prototype_topk: int = 1,
    prototype_mode: str = "top1",
) -> dict[str, object]:
    outputs = model.encode_all(eeg)
    output_dict = {
        name: tensor
        for name, tensor in {
            "semantic": outputs.semantic,
            "perceptual": outputs.perceptual,
            "legacy": outputs.legacy,
        }.items()
        if tensor is not None
    }
    fused_logits, candidate_ids, component_logits = compute_retrieval_logits(
        model,
        output_dict,
        semantic_bank=candidate_semantic_bank,
        perceptual_bank=candidate_perceptual_bank,
        alpha=alpha,
    )
    _, topk_ids, topk_scores = prototype_lookup_from_logits(
        fused_logits,
        candidate_image_ids=candidate_ids,
        query_image_ids=query_image_ids,
        exclude_self=exclude_self,
        topk=prototype_topk,
    )
    prototype_latents, prototype_weights = aggregate_prototype_latents(
        candidate_latent_bank,
        topk_ids,
        topk_scores,
        device=device,
        prototype_mode=prototype_mode,
    )
    return {
        "prototype_latents": prototype_latents,
        "topk_ids": topk_ids,
        "topk_scores": topk_scores,
        "topk_weights": prototype_weights,
        "candidate_image_ids": candidate_ids,
        "component_logits": component_logits,
        "fused_logits": fused_logits,
    }


def prepare_vae(model_name: str, device: torch.device):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_name).to(device)
    vae.eval()
    return vae


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    scaled = latents / vae.config.scaling_factor
    images = vae.decode(scaled).sample
    return (images / 2.0 + 0.5).clamp(0.0, 1.0)


def save_image_batch(images: torch.Tensor, image_ids: list[str], output_dir: str | Path) -> list[str]:
    from torchvision.utils import save_image

    output_dir = ensure_dir(output_dir)
    saved_paths: list[str] = []
    for image, image_id in zip(images, image_ids, strict=True):
        path = output_dir / f"{image_id}.png"
        save_image(image.cpu(), path)
        saved_paths.append(str(path))
    return saved_paths


def clip_image_loss_model(device: torch.device, model_name: str = "ViT-B/32"):
    import clip

    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval().requires_grad_(False)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def encode(images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - mean) / std
        return F.normalize(model.encode_image(images), dim=-1)

    return encode
