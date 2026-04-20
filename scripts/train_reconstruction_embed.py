#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split
from project1_eeg.evaluation import compute_retrieval_metrics, eval_images
from project1_eeg.image_banks import TensorBank, default_bank_path
from project1_eeg.kandinsky import (
    DEFAULT_KANDINSKY_DECODER_MODEL,
    DEFAULT_KANDINSKY_PRIOR_MODEL,
    generate_best_images,
    load_kandinsky_decoder_pipeline,
    load_kandinsky_prior_pipeline,
    negative_image_embed_from_bank,
)
from project1_eeg.reconstruction import EEGEmbeddingRegressor, GatedRetrievalResidualRegressor
from project1_eeg.retrieval import RetrievalOutputs, build_retrieval_model_from_config
from project1_eeg.runtime import make_dataloader, prototype_lookup_from_logits
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    format_seconds,
    load_image_batch,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-to-Kandinsky-embedding reconstruction model.")
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=["regression", "rag_residual"],
        default="regression",
    )
    parser.add_argument(
        "--embedding-bank",
        type=Path,
        default=default_bank_path(DEFAULT_OUTPUT_DIR, "kandinsky", "train"),
    )
    parser.add_argument("--retrieval-bank", type=Path, default=None)
    parser.add_argument("--text-bank", type=Path, default=None)
    parser.add_argument("--use-text-aux", action="store_true")
    parser.add_argument("--use-text-context", action="store_true")
    parser.add_argument("--pretrain-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "experiments" / "reconstruction_kandinsky_embed",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--head-hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embedding-eval-every", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["loss", "val_subset_top1_then_top5"],
        default="loss",
    )
    parser.add_argument("--image-eval-every", type=int, default=10)
    parser.add_argument("--image-eval-limit", type=int, default=16)
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--decoder-steps", type=int, default=50)
    parser.add_argument("--decoder-guidance-scale", type=float, default=4.0)
    parser.add_argument("--decoder-height", type=int, default=512)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--prior-model", type=str, default=DEFAULT_KANDINSKY_PRIOR_MODEL)
    parser.add_argument("--decoder-model", type=str, default=DEFAULT_KANDINSKY_DECODER_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--staged-regression-finetune", action="store_true")
    parser.add_argument("--retrieval-topk", type=int, default=4)
    parser.add_argument(
        "--exclude-self-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--final-mse-weight", type=float, default=0.5)
    parser.add_argument("--final-cosine-weight", type=float, default=1.0)
    parser.add_argument("--direct-mse-weight", type=float, default=0.25)
    parser.add_argument("--direct-cosine-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-contrastive-weight", type=float, default=0.2)
    parser.add_argument("--retrieval-distill-weight", type=float, default=0.2)
    parser.add_argument("--text-cosine-weight", type=float, default=0.1)
    parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--attention-heads", type=int, default=8)
    return parser.parse_args()


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def resolve_artifact_path(path: str | Path | None, *, base_paths: Iterable[Path] = ()) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for base_path in base_paths:
        resolved = base_path / candidate
        if resolved.exists():
            return resolved
    return candidate


def compute_embedding_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse_loss = F.mse_loss(predicted, target)
    cosine_loss = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()

    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    contrastive_loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    total_loss = mse_weight * mse_loss + cosine_weight * cosine_loss + contrastive_weight * contrastive_loss
    metrics = {
        "mse_loss": float(mse_loss.item()),
        "cosine_loss": float(cosine_loss.item()),
        "contrastive_loss": float(contrastive_loss.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics


def contrastive_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def extract_teacher_output(outputs: RetrievalOutputs, *, head: str) -> torch.Tensor:
    if head == "perceptual":
        if outputs.perceptual is None:
            raise KeyError("Teacher retrieval model does not expose a perceptual head.")
        return outputs.perceptual
    if head == "semantic":
        if outputs.semantic is None:
            raise KeyError("Teacher retrieval model does not expose a semantic head.")
        return outputs.semantic
    if outputs.legacy is None:
        raise KeyError("Teacher retrieval model does not expose a legacy head.")
    return outputs.legacy


def resolve_teacher_head(model, retrieval_bank: TensorBank) -> str:
    if retrieval_bank.bank_type == "dreamsim" and hasattr(model, "perceptual_head"):
        return "perceptual"
    if retrieval_bank.bank_type in {"clip", "clip_text"} and hasattr(model, "semantic_head"):
        return "semantic"
    if hasattr(model, "perceptual_head") and retrieval_bank.values.shape[1] == getattr(model, "perceptual_dim", None):
        return "perceptual"
    if hasattr(model, "semantic_head") and retrieval_bank.values.shape[1] == getattr(model, "semantic_dim", None):
        return "semantic"
    return "legacy"


def load_encoder_init_payload(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidate = payload.get("encoder_state")
    if candidate is None:
        candidate = payload.get("model_state", payload)
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if key.startswith("eeg_encoder."):
            cleaned[key.removeprefix("eeg_encoder.")] = value
        elif key.startswith("encoder."):
            cleaned[key.removeprefix("encoder.")] = value
        else:
            cleaned[key] = value
    return cleaned


def apply_encoder_init(module: torch.nn.Module, checkpoint_path: Path | None) -> dict[str, list[str]]:
    if checkpoint_path is None:
        return {"missing_keys": [], "unexpected_keys": []}
    state_dict = load_encoder_init_payload(checkpoint_path)
    result = module.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def set_encoder_trainable(model, trainable: bool) -> None:
    if not hasattr(model, "eeg_encoder"):
        return
    for parameter in model.eeg_encoder.parameters():
        parameter.requires_grad = trainable


def build_stage_optimizer(model, args: argparse.Namespace, *, stage: str) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if args.model_type == "regression":
        if not args.staged_regression_finetune:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
            return optimizer, scheduler

        if stage == "freeze":
            set_encoder_trainable(model, False)
            params = list(model.head_parameters())
            stage_epochs = max(1, min(args.freeze_encoder_epochs, args.epochs))
            optimizer = torch.optim.AdamW(params, lr=args.head_learning_rate, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
            return optimizer, scheduler

        set_encoder_trainable(model, True)
        params = [
            {"params": list(model.encoder_parameters()), "lr": args.encoder_learning_rate},
            {"params": list(model.head_parameters()), "lr": args.head_learning_rate},
        ]
        stage_epochs = max(1, args.epochs - args.freeze_encoder_epochs)
        optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
        return optimizer, scheduler

    if stage == "freeze":
        set_encoder_trainable(model, False)
        params = list(model.head_parameters())
        stage_epochs = max(1, min(args.freeze_encoder_epochs, args.epochs))
        optimizer = torch.optim.AdamW(params, lr=args.head_learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
        return optimizer, scheduler

    set_encoder_trainable(model, True)
    params = [
        {"params": list(model.encoder_parameters()), "lr": args.encoder_learning_rate},
        {"params": list(model.head_parameters()), "lr": args.head_learning_rate},
    ]
    stage_epochs = max(1, args.epochs - args.freeze_encoder_epochs)
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
    return optimizer, scheduler


def load_retrieval_assets(
    checkpoint_path: Path,
    *,
    device: torch.device,
    retrieval_bank_path: Path | None,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    retrieval_model = build_retrieval_model_from_config(config).to(device)
    retrieval_model.load_state_dict(payload["model_state"])
    retrieval_model.eval()
    for parameter in retrieval_model.parameters():
        parameter.requires_grad = False

    if getattr(retrieval_model, "legacy_mode", False):
        backbone_dim = int(retrieval_model.output_dim())
    else:
        backbone_dim = int(retrieval_model.embedding_dim)

    resolved_bank_path = retrieval_bank_path
    if resolved_bank_path is None:
        fallback = (
            config.get("perceptual_bank")
            or config.get("semantic_bank")
            or config.get("clip_bank")
        )
        resolved_bank_path = resolve_artifact_path(
            fallback,
            base_paths=[PROJECT_ROOT, checkpoint_path.parent, checkpoint_path.parent.parent],
        )

    retrieval_bank = None
    teacher_head = None
    if resolved_bank_path is not None and resolved_bank_path.exists():
        retrieval_bank = TensorBank.load(resolved_bank_path)
        teacher_head = resolve_teacher_head(retrieval_model, retrieval_bank)
    else:
        resolved_bank_path = None

    return {
        "payload": payload,
        "config": config,
        "teacher_model": retrieval_model,
        "eeg_encoder": copy.deepcopy(retrieval_model.encoder),
        "backbone_dim": backbone_dim,
        "retrieval_bank": retrieval_bank,
        "retrieval_bank_path": resolved_bank_path,
        "teacher_head": teacher_head,
    }


def build_context_batch(
    query_embeddings: torch.Tensor,
    *,
    retrieval_bank: TensorBank,
    embedding_bank: TensorBank,
    device: torch.device,
    topk: int,
    text_bank: TensorBank | None = None,
    query_image_ids: list[str] | None = None,
    exclude_self: bool = False,
) -> dict[str, object]:
    effective_topk = min(topk, len(retrieval_bank.image_ids) - (1 if exclude_self and query_image_ids else 0))
    if effective_topk <= 0:
        raise ValueError("retrieval_topk is too large for the candidate bank after self-exclusion.")

    logits = query_embeddings.float().cpu() @ retrieval_bank.values.float().T.cpu()
    _, topk_ids, topk_scores = prototype_lookup_from_logits(
        logits,
        candidate_image_ids=list(retrieval_bank.image_ids),
        query_image_ids=query_image_ids,
        exclude_self=exclude_self,
        topk=effective_topk,
    )
    flat_ids = [image_id for row in topk_ids for image_id in row]
    retrieved_embeddings = embedding_bank.align(flat_ids, device=device).float()
    retrieved_embeddings = retrieved_embeddings.reshape(query_embeddings.shape[0], effective_topk, -1)

    retrieved_text_embeddings = None
    if text_bank is not None:
        retrieved_text_embeddings = text_bank.align(flat_ids, device=device).float()
        retrieved_text_embeddings = retrieved_text_embeddings.reshape(query_embeddings.shape[0], effective_topk, -1)

    topk_probs = torch.softmax(topk_scores.float(), dim=-1)
    confidence = topk_probs[:, 0].to(device)
    return {
        "topk_ids": topk_ids,
        "topk_scores": topk_scores,
        "topk_probs": topk_probs,
        "confidence": confidence,
        "retrieved_embeddings": retrieved_embeddings,
        "retrieved_text_embeddings": retrieved_text_embeddings,
    }


def forward_batch(
    *,
    model,
    args: argparse.Namespace,
    batch,
    device: torch.device,
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object] | None]:
    eeg = batch["eeg"].to(device)
    if args.model_type == "regression":
        predicted = model(eeg)
        outputs = {"final_embedding": predicted}
        return predicted, outputs, None

    if retrieval_bank is None:
        raise ValueError("rag_residual requires a retrieval bank.")

    shared = model.encode_backbone(eeg)
    base_outputs = model.project_shared(shared)
    context = build_context_batch(
        base_outputs.retrieval_embedding.detach(),
        retrieval_bank=retrieval_bank,
        embedding_bank=embedding_bank,
        device=device,
        topk=args.retrieval_topk,
        text_bank=text_bank if args.use_text_context else None,
        query_image_ids=list(batch["image_id"]),
        exclude_self=args.exclude_self_retrieval,
    )
    final_outputs = model.project_shared(
        shared,
        retrieved_embeddings=context["retrieved_embeddings"],
        retrieved_text_embeddings=context["retrieved_text_embeddings"],
        retrieval_confidence=context["confidence"],
    )
    outputs = {
        "final_embedding": final_outputs.final_embedding,
        "direct_embedding": final_outputs.direct_embedding,
        "retrieval_embedding": final_outputs.retrieval_embedding,
        "residual_embedding": final_outputs.residual_embedding,
        "gate": final_outputs.gate,
    }
    if final_outputs.predicted_text_embedding is not None:
        outputs["predicted_text_embedding"] = final_outputs.predicted_text_embedding
    return final_outputs.final_embedding, outputs, context


def compute_rag_losses(
    *,
    batch,
    outputs: dict[str, torch.Tensor],
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank,
    text_bank: TensorBank | None,
    teacher_model,
    teacher_head: str | None,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_ids = list(batch["image_id"])
    eeg = batch["eeg"].to(device)

    target_embedding = embedding_bank.align(image_ids, device=device).float()
    final_embedding = outputs["final_embedding"]
    direct_embedding = outputs["direct_embedding"]

    final_mse = F.mse_loss(final_embedding, target_embedding)
    final_cosine = 1.0 - F.cosine_similarity(final_embedding, target_embedding, dim=-1).mean()
    direct_mse = F.mse_loss(direct_embedding, target_embedding)
    direct_cosine = 1.0 - F.cosine_similarity(direct_embedding, target_embedding, dim=-1).mean()

    retrieval_target = F.normalize(retrieval_bank.align(image_ids, device=device).float(), dim=-1)
    retrieval_contrastive = contrastive_alignment_loss(
        outputs["retrieval_embedding"],
        retrieval_target,
        temperature=args.temperature,
    )

    retrieval_distill = torch.tensor(0.0, device=device)
    if teacher_model is not None and teacher_head is not None:
        with torch.no_grad():
            teacher_outputs = teacher_model.encode_all(eeg)
            teacher_embedding = extract_teacher_output(teacher_outputs, head=teacher_head)
        retrieval_distill = 1.0 - F.cosine_similarity(
            outputs["retrieval_embedding"],
            F.normalize(teacher_embedding, dim=-1),
            dim=-1,
        ).mean()

    residual_l2 = outputs["residual_embedding"].pow(2).mean()
    gate_mean = outputs["gate"].mean()

    text_cosine = torch.tensor(0.0, device=device)
    if args.use_text_aux and text_bank is not None and "predicted_text_embedding" in outputs:
        text_target = F.normalize(text_bank.align(image_ids, device=device).float(), dim=-1)
        text_cosine = 1.0 - F.cosine_similarity(outputs["predicted_text_embedding"], text_target, dim=-1).mean()

    total_loss = (
        args.final_mse_weight * final_mse
        + args.final_cosine_weight * final_cosine
        + args.direct_mse_weight * direct_mse
        + args.direct_cosine_weight * direct_cosine
        + args.retrieval_contrastive_weight * retrieval_contrastive
        + args.retrieval_distill_weight * retrieval_distill
        + args.residual_l2_weight * residual_l2
        + args.text_cosine_weight * text_cosine
    )
    metrics = {
        "final_mse_loss": float(final_mse.item()),
        "final_cosine_loss": float(final_cosine.item()),
        "direct_mse_loss": float(direct_mse.item()),
        "direct_cosine_loss": float(direct_cosine.item()),
        "retrieval_contrastive_loss": float(retrieval_contrastive.item()),
        "retrieval_distill_loss": float(retrieval_distill.item()),
        "residual_l2_loss": float(residual_l2.item()),
        "text_cosine_loss": float(text_cosine.item()),
        "gate_mean": float(gate_mean.item()),
        "total_loss": float(total_loss.item()),
    }
    return total_loss, metrics


def train_one_epoch(
    model,
    loader,
    *,
    args: argparse.Namespace,
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    teacher_model,
    teacher_head: str | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        target = embedding_bank.align(batch["image_id"], device=device).float()

        if args.model_type == "regression":
            predicted = model(eeg)
            loss, batch_metrics = compute_embedding_loss(
                predicted,
                target,
                mse_weight=args.mse_weight,
                cosine_weight=args.cosine_weight,
                contrastive_weight=args.contrastive_weight,
                temperature=args.temperature,
            )
        else:
            _, outputs, _ = forward_batch(
                model=model,
                args=args,
                batch=batch,
                device=device,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
            )
            loss, batch_metrics = compute_rag_losses(
                batch=batch,
                outputs=outputs,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
                teacher_model=teacher_model,
                teacher_head=teacher_head,
                device=device,
                args=args,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append(batch_metrics)

    return average_metrics(metrics)


@torch.no_grad()
def evaluate_losses(
    model,
    loader,
    *,
    args: argparse.Namespace,
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    teacher_model,
    teacher_head: str | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        target = embedding_bank.align(batch["image_id"], device=device).float()

        if args.model_type == "regression":
            predicted = model(eeg)
            _, batch_metrics = compute_embedding_loss(
                predicted,
                target,
                mse_weight=args.mse_weight,
                cosine_weight=args.cosine_weight,
                contrastive_weight=args.contrastive_weight,
                temperature=args.temperature,
            )
        else:
            _, outputs, _ = forward_batch(
                model=model,
                args=args,
                batch=batch,
                device=device,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
            )
            _, batch_metrics = compute_rag_losses(
                batch=batch,
                outputs=outputs,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
                teacher_model=teacher_model,
                teacher_head=teacher_head,
                device=device,
                args=args,
            )
        metrics.append(batch_metrics)

    return average_metrics(metrics)


@torch.no_grad()
def evaluate_embedding_proxy(
    model,
    loader,
    *,
    args: argparse.Namespace,
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    predicted_embeddings: list[torch.Tensor] = []
    ordered_ids: list[str] = []
    gate_values: list[torch.Tensor] = []

    for batch in loader:
        predicted, outputs, _ = forward_batch(
            model=model,
            args=args,
            batch=batch,
            device=device,
            embedding_bank=embedding_bank,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
        )
        predicted_embeddings.append(predicted.float().cpu())
        if "gate" in outputs:
            gate_values.append(outputs["gate"].float().cpu())
        ordered_ids.extend(list(batch["image_id"]))

    if not predicted_embeddings:
        return {}

    preds = F.normalize(torch.cat(predicted_embeddings, dim=0), dim=-1)
    target = F.normalize(embedding_bank.align(ordered_ids).float(), dim=-1)
    avg_target_cosine = float((preds * target).sum(dim=-1).mean().item())

    full_bank = F.normalize(embedding_bank.values.float(), dim=-1)
    full_logits = preds @ full_bank.T
    full_metrics = compute_retrieval_metrics(
        full_logits,
        ordered_image_ids=ordered_ids,
        candidate_image_ids=embedding_bank.image_ids,
    )

    subset_bank = embedding_bank.subset(ordered_ids)
    subset_values = F.normalize(subset_bank.values.float(), dim=-1)
    subset_logits = preds @ subset_values.T
    subset_metrics = compute_retrieval_metrics(
        subset_logits,
        ordered_image_ids=ordered_ids,
        candidate_image_ids=ordered_ids,
    )

    metrics = {
        "avg_target_cosine": avg_target_cosine,
        "full_train_top1_acc": float(full_metrics["top1_acc"]),
        "full_train_top5_acc": float(full_metrics["top5_acc"]),
        "val_subset_top1_acc": float(subset_metrics["top1_acc"]),
        "val_subset_top5_acc": float(subset_metrics["top5_acc"]),
    }
    if gate_values:
        metrics["avg_gate"] = float(torch.cat(gate_values, dim=0).mean().item())
    return metrics


@torch.no_grad()
def evaluate_images_on_probe(
    *,
    model,
    records,
    args: argparse.Namespace,
    embedding_bank: TensorBank,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    prior_pipe,
    decoder_pipe,
    device: torch.device,
) -> dict[str, float]:
    if not records:
        return {}

    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.decode_batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_fake_images: list[torch.Tensor] = []
    all_real_images: list[torch.Tensor] = []
    negative_image_embed = negative_image_embed_from_bank(
        embedding_bank,
        batch_size=1,
        device=device,
        dtype=next(decoder_pipe.unet.parameters()).dtype,
    )

    for batch in loader:
        predicted_embeddings, _, _ = forward_batch(
            model=model,
            args=args,
            batch=batch,
            device=device,
            embedding_bank=embedding_bank,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
        )
        result = generate_best_images(
            predicted_embeddings=predicted_embeddings,
            prior_pipe=prior_pipe,
            decoder_pipe=decoder_pipe,
            negative_image_embeds=negative_image_embed,
            num_candidates=args.num_candidates,
            num_inference_steps=args.decoder_steps,
            guidance_scale=args.decoder_guidance_scale,
            height=args.decoder_height,
            width=args.decoder_width,
        )
        selected_images = result["selected_images"]
        all_fake_images.extend(list(selected_images))
        real_images = load_image_batch(list(batch["image_path"]), image_size=args.decoder_height)
        all_real_images.extend(list(real_images))

    return eval_images(
        real_images=torch.stack(all_real_images, dim=0),
        fake_images=torch.stack(all_fake_images, dim=0),
        device=device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if not args.embedding_bank.exists():
        raise FileNotFoundError(
            f"Embedding bank not found: {args.embedding_bank}. "
            "Run scripts/cache_image_bank.py --bank-type kandinsky --split train first."
        )

    embedding_bank = TensorBank.load(args.embedding_bank)
    retrieval_assets = load_retrieval_assets(
        args.retrieval_checkpoint,
        device=device,
        retrieval_bank_path=args.retrieval_bank,
    )
    retrieval_bank = retrieval_assets["retrieval_bank"]
    text_bank_path = resolve_artifact_path(
        args.text_bank,
        base_paths=[PROJECT_ROOT, args.retrieval_checkpoint.parent, args.retrieval_checkpoint.parent.parent],
    )
    text_bank = TensorBank.load(text_bank_path) if text_bank_path is not None and text_bank_path.exists() else None

    if args.model_type == "rag_residual" and retrieval_bank is None:
        raise ValueError(
            "rag_residual requires a retrieval bank. Pass --retrieval-bank or use a retrieval checkpoint "
            "whose config stores a perceptual/semantic bank path."
        )
    if (args.use_text_aux or args.use_text_context) and text_bank is None:
        raise ValueError("Text features were requested but no text bank could be resolved.")

    eeg_encoder = retrieval_assets["eeg_encoder"]
    init_report = apply_encoder_init(
        eeg_encoder,
        resolve_artifact_path(
            args.pretrain_checkpoint,
            base_paths=[PROJECT_ROOT, args.output_dir.parent, args.output_dir],
        ),
    )

    backbone_dim = int(retrieval_assets["backbone_dim"])
    if args.model_type == "regression":
        model = EEGEmbeddingRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=backbone_dim,
            target_dim=int(embedding_bank.values.shape[1]),
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        text_dim = None if text_bank is None else int(text_bank.values.shape[1])
        model = GatedRetrievalResidualRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=backbone_dim,
            target_dim=int(embedding_bank.values.shape[1]),
            retrieval_dim=int(retrieval_bank.values.shape[1]),
            text_dim=text_dim if (args.use_text_aux or args.use_text_context) else None,
            hidden_dim=args.head_hidden_dim,
            attention_heads=args.attention_heads,
            dropout=args.dropout,
        ).to(device)

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=train_ids,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=val_ids,
    )

    train_loader = make_dataloader(
        EEGImageDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        EEGImageDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task": "reconstruction_embed",
        "mode": "kandinsky_image_embedding",
        "model_type": args.model_type,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "mse_weight": args.mse_weight,
        "cosine_weight": args.cosine_weight,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "embedding_eval_every": args.embedding_eval_every,
        "selection_metric": args.selection_metric,
        "embedding_bank": str(args.embedding_bank),
        "retrieval_bank": None if retrieval_assets["retrieval_bank_path"] is None else str(retrieval_assets["retrieval_bank_path"]),
        "text_bank": None if text_bank_path is None else str(text_bank_path),
        "embedding_dim": int(embedding_bank.values.shape[1]),
        "retrieval_bank_dim": None if retrieval_bank is None else int(retrieval_bank.values.shape[1]),
        "text_dim": None if text_bank is None else int(text_bank.values.shape[1]),
        "backbone_dim": int(backbone_dim),
        "retrieval_checkpoint": str(args.retrieval_checkpoint),
        "retrieval_config": retrieval_assets["config"],
        "teacher_head": retrieval_assets["teacher_head"],
        "train_avg_trials": True,
        "val_avg_trials": True,
        "image_eval_every": args.image_eval_every,
        "image_eval_limit": args.image_eval_limit,
        "decode_batch_size": args.decode_batch_size,
        "num_candidates": args.num_candidates,
        "decoder_steps": args.decoder_steps,
        "decoder_guidance_scale": args.decoder_guidance_scale,
        "decoder_height": args.decoder_height,
        "decoder_width": args.decoder_width,
        "prior_model": args.prior_model,
        "decoder_model": args.decoder_model,
        "local_files_only": bool(args.local_files_only),
        "use_text_aux": bool(args.use_text_aux),
        "use_text_context": bool(args.use_text_context),
        "pretrain_checkpoint": None if args.pretrain_checkpoint is None else str(args.pretrain_checkpoint),
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "staged_regression_finetune": bool(args.staged_regression_finetune),
        "retrieval_topk": args.retrieval_topk,
        "exclude_self_retrieval": bool(args.exclude_self_retrieval),
        "final_mse_weight": args.final_mse_weight,
        "final_cosine_weight": args.final_cosine_weight,
        "direct_mse_weight": args.direct_mse_weight,
        "direct_cosine_weight": args.direct_cosine_weight,
        "retrieval_contrastive_weight": args.retrieval_contrastive_weight,
        "retrieval_distill_weight": args.retrieval_distill_weight,
        "text_cosine_weight": args.text_cosine_weight,
        "residual_l2_weight": args.residual_l2_weight,
        "attention_heads": args.attention_heads,
        "encoder_init_missing_keys": init_report["missing_keys"],
        "encoder_init_unexpected_keys": init_report["unexpected_keys"],
    }
    save_json(config, run_dir / "config.json")
    probe_ids = val_ids[: min(len(val_ids), args.image_eval_limit)]
    save_json({"train_ids": train_ids, "val_ids": val_ids, "probe_ids": probe_ids}, run_dir / "split.json")

    current_stage = None
    optimizer = None
    scheduler = None
    prior_pipe = None
    decoder_pipe = None
    history: list[dict[str, float]] = []
    best_score = (float("-inf"),)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        stage = "train"
        if args.model_type == "rag_residual" and epoch <= args.freeze_encoder_epochs:
            stage = "freeze"
        if args.model_type == "regression" and args.staged_regression_finetune and epoch <= args.freeze_encoder_epochs:
            stage = "freeze"
        if stage != current_stage:
            optimizer, scheduler = build_stage_optimizer(model, args, stage=stage)
            current_stage = stage

        train_metrics = train_one_epoch(
            model,
            train_loader,
            args=args,
            embedding_bank=embedding_bank,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
            teacher_model=retrieval_assets["teacher_model"],
            teacher_head=retrieval_assets["teacher_head"],
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate_losses(
            model,
            val_loader,
            args=args,
            embedding_bank=embedding_bank,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
            teacher_model=retrieval_assets["teacher_model"],
            teacher_head=retrieval_assets["teacher_head"],
            device=device,
        )
        scheduler.step()

        combined_metrics = {
            "epoch": float(epoch),
            "stage": 0.0 if stage == "freeze" else 1.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        for prefix, metric_dict in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metric_dict.items():
                combined_metrics[f"{prefix}_{key}"] = float(value)

        proxy_metrics: dict[str, float] = {}
        should_eval_proxy = args.embedding_eval_every > 0 and (
            epoch % args.embedding_eval_every == 0 or epoch == args.epochs
        )
        if should_eval_proxy:
            proxy_metrics = evaluate_embedding_proxy(
                model,
                val_loader,
                args=args,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
                device=device,
            )
            combined_metrics.update(proxy_metrics)

        image_metrics: dict[str, float] = {}
        should_eval_images = (
            args.image_eval_limit > 0
            and (epoch % args.image_eval_every == 0 or epoch == args.epochs)
            and probe_ids
        )
        if should_eval_images:
            if prior_pipe is None:
                prior_pipe = load_kandinsky_prior_pipeline(
                    model_name=args.prior_model,
                    device=device,
                    local_files_only=args.local_files_only,
                )
            if decoder_pipe is None:
                decoder_pipe = load_kandinsky_decoder_pipeline(
                    model_name=args.decoder_model,
                    device=device,
                    local_files_only=args.local_files_only,
                )
            probe_records = val_records[: min(len(val_records), args.image_eval_limit)]
            image_metrics = evaluate_images_on_probe(
                model=model,
                records=probe_records,
                args=args,
                embedding_bank=embedding_bank,
                retrieval_bank=retrieval_bank,
                text_bank=text_bank,
                prior_pipe=prior_pipe,
                decoder_pipe=decoder_pipe,
                device=device,
            )
            combined_metrics.update(image_metrics)

        history.append(combined_metrics)
        save_json({"history": history}, run_dir / "history.json")
        save_checkpoint(
            run_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            config=config,
            metrics=combined_metrics,
        )

        if args.selection_metric == "val_subset_top1_then_top5":
            if not proxy_metrics:
                raise ValueError(
                    "selection_metric=val_subset_top1_then_top5 requires embedding proxy evaluation. "
                    "Set --embedding-eval-every to a positive value."
                )
            score = (
                float(proxy_metrics["val_subset_top1_acc"]),
                float(proxy_metrics["val_subset_top5_acc"]),
                float(proxy_metrics["avg_target_cosine"]),
                -float(val_metrics["total_loss"]),
            )
        elif image_metrics:
            score = (float(image_metrics["eval_clip"]) + 0.25 * float(image_metrics["eval_ssim"]),)
        else:
            score = (-float(val_metrics["total_loss"]),)

        if score > best_score:
            best_score = score
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=combined_metrics,
            )

        elapsed = format_seconds(time.time() - start_time)
        if image_metrics:
            print(
                f"epoch={epoch:03d} "
                f"stage={stage} "
                f"train_total={train_metrics['total_loss']:.4f} "
                f"val_total={val_metrics['total_loss']:.4f} "
                f"eval_clip={image_metrics['eval_clip']:.4f} "
                f"eval_ssim={image_metrics['eval_ssim']:.4f} "
                f"elapsed={elapsed}"
            )
        elif proxy_metrics:
            print(
                f"epoch={epoch:03d} "
                f"stage={stage} "
                f"train_total={train_metrics['total_loss']:.4f} "
                f"val_total={val_metrics['total_loss']:.4f} "
                f"val_subset_top1={proxy_metrics['val_subset_top1_acc']:.4f} "
                f"val_subset_top5={proxy_metrics['val_subset_top5_acc']:.4f} "
                f"target_cos={proxy_metrics['avg_target_cosine']:.4f} "
                f"elapsed={elapsed}"
            )
        else:
            print(
                f"epoch={epoch:03d} "
                f"stage={stage} "
                f"train_total={train_metrics['total_loss']:.4f} "
                f"val_total={val_metrics['total_loss']:.4f} "
                f"elapsed={elapsed}"
            )


if __name__ == "__main__":
    main()
