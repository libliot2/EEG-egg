#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import eval_images
from project1_eeg.image_banks import TensorBank, default_bank_path
from project1_eeg.kandinsky import (
    DEFAULT_KANDINSKY_DECODER_MODEL,
    DEFAULT_KANDINSKY_PRIOR_MODEL,
    generate_best_images,
    load_kandinsky_decoder_pipeline,
    load_kandinsky_img2img_decoder_pipeline,
    load_kandinsky_prior_pipeline,
    negative_image_embed_from_bank,
)
from project1_eeg.reconstruction import EEGEmbeddingRegressor, EmbeddingAdapter, GatedRetrievalResidualRegressor
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, make_dataloader, prototype_lookup_from_logits, save_image_batch
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    load_image_batch,
    resolve_device,
    save_json,
)


CHANNEL_PRESETS: dict[str, list[str]] = {
    "visual17": [
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reconstructions from EEG via Kandinsky image embeddings.")
    parser.add_argument("--reconstruction-checkpoint", type=Path, default=None)
    parser.add_argument("--retrieval-checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", choices=["regression", "rag_residual"], default=None)
    parser.add_argument(
        "--embedding-bank",
        type=Path,
        default=None,
        help="Generation bank for the Kandinsky decoder. Defaults to the bank recorded in the reconstruction checkpoint config.",
    )
    parser.add_argument(
        "--conditioning-bank",
        type=Path,
        default=None,
        help="Optional source bank used to produce conditioning embeddings before any adapter is applied.",
    )
    parser.add_argument("--conditioning-adapter", type=Path, default=None)
    parser.add_argument("--retrieval-bank", type=Path, default=None)
    parser.add_argument("--text-bank", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "reconstruction_predictions_embed",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument(
        "--embedding-source",
        choices=["predicted", "ground_truth", "retrieval_top1"],
        default="predicted",
    )
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--prior-model", type=str, default=None)
    parser.add_argument("--decoder-model", type=str, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--decoder-steps", type=int, default=None)
    parser.add_argument("--decoder-guidance-scale", type=float, default=None)
    parser.add_argument("--decoder-height", type=int, default=None)
    parser.add_argument("--decoder-width", type=int, default=None)
    parser.add_argument("--init-image-dir", type=Path, default=None)
    parser.add_argument("--img2img-strength", type=float, default=0.5)
    parser.add_argument("--candidate-seed-offset", type=int, default=0)
    parser.add_argument(
        "--candidate-selection-mode",
        choices=["semantic", "semantic_lowlevel"],
        default="semantic",
        help="How to select the best image among decoder candidates.",
    )
    parser.add_argument(
        "--candidate-lowlevel-weight",
        type=float,
        default=0.0,
        help="Weight for the low-level init-image similarity term when using semantic_lowlevel selection.",
    )
    parser.add_argument(
        "--candidate-lowlevel-metric",
        choices=["pixel_cosine", "neg_mse"],
        default="pixel_cosine",
    )
    parser.add_argument(
        "--candidate-score-normalization",
        choices=["none", "per_query_minmax", "per_query_zscore"],
        default="per_query_minmax",
    )
    parser.add_argument("--retrieval-topk", type=int, default=None)
    parser.add_argument("--use-text-context", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    parser.add_argument("--channel-subset-name", type=str, default=None)
    return parser.parse_args()


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


def resolve_bank(path: Path | None, fallback: str | None, *, name: str, base_paths: Iterable[Path]) -> TensorBank | None:
    resolved = path or resolve_artifact_path(fallback, base_paths=base_paths)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(f"{name} bank not found: {resolved}")
    return TensorBank.load(resolved)


def resolve_alpha(args: argparse.Namespace, payload: dict, *, has_semantic: bool, has_perceptual: bool) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    metrics = payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    return 0.5


def resolve_embedding_bank_path(args: argparse.Namespace, config: dict | None, checkpoint_path: Path | None) -> Path:
    if args.embedding_bank is not None:
        return args.embedding_bank
    if config is not None and config.get("embedding_bank"):
        resolved = resolve_artifact_path(
            config["embedding_bank"],
            base_paths=[PROJECT_ROOT, checkpoint_path.parent if checkpoint_path is not None else PROJECT_ROOT],
        )
        if resolved is not None:
            return resolved
    default_path = default_bank_path(DEFAULT_OUTPUT_DIR, "kandinsky", "train")
    if default_path.exists():
        return default_path
    raise ValueError(
        "Unable to resolve an embedding bank. Pass --embedding-bank or provide a reconstruction checkpoint "
        "whose config contains `embedding_bank`."
    )


def resolve_conditioning_bank_path(
    args: argparse.Namespace,
    config: dict | None,
    checkpoint_path: Path | None,
    *,
    generation_bank_path: Path,
) -> Path:
    if args.conditioning_bank is not None:
        return args.conditioning_bank
    if config is not None and config.get("embedding_bank"):
        resolved = resolve_artifact_path(
            config["embedding_bank"],
            base_paths=[PROJECT_ROOT, checkpoint_path.parent if checkpoint_path is not None else PROJECT_ROOT],
        )
        if resolved is not None:
            return resolved
    return generation_bank_path


def resolve_generation_config(
    args: argparse.Namespace,
    config: dict | None,
    embedding_bank_path: Path,
) -> dict[str, object]:
    def coalesce(*values):
        for value in values:
            if value is not None:
                return value
        return None

    return {
        "embedding_bank": str(embedding_bank_path),
        "prior_model": coalesce(
            args.prior_model,
            config.get("prior_model") if config is not None else None,
            DEFAULT_KANDINSKY_PRIOR_MODEL,
        ),
        "decoder_model": coalesce(
            args.decoder_model,
            config.get("decoder_model") if config is not None else None,
            DEFAULT_KANDINSKY_DECODER_MODEL,
        ),
        "num_candidates": int(coalesce(args.num_candidates, config.get("num_candidates") if config else None, 4)),
        "decoder_steps": int(coalesce(args.decoder_steps, config.get("decoder_steps") if config else None, 50)),
        "decoder_guidance_scale": float(
            coalesce(args.decoder_guidance_scale, config.get("decoder_guidance_scale") if config else None, 4.0)
        ),
        "decoder_height": int(coalesce(args.decoder_height, config.get("decoder_height") if config else None, 512)),
        "decoder_width": int(coalesce(args.decoder_width, config.get("decoder_width") if config else None, 512)),
    }


def resolve_init_image_paths(query_ids: list[str], init_image_dir: Path) -> list[Path]:
    init_paths: list[Path] = []
    for image_id in query_ids:
        path = init_image_dir / f"{image_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"Init image not found for '{image_id}': {path}")
        init_paths.append(path)
    return init_paths


def load_embedding_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    retrieval_model = build_retrieval_model_from_config(config["retrieval_config"])
    eeg_encoder = copy.deepcopy(retrieval_model.encoder)
    model_type = config.get("model_type", "regression")

    if model_type == "rag_residual":
        model = GatedRetrievalResidualRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=int(config["backbone_dim"]),
            target_dim=int(config["embedding_dim"]),
            retrieval_dim=int(config.get("retrieval_bank_dim") or config["retrieval_config"].get("perceptual_dim") or config["retrieval_config"].get("semantic_dim") or config["embedding_dim"]),
            text_dim=config.get("text_dim"),
            hidden_dim=int(config.get("head_hidden_dim", 1024)),
            attention_heads=int(config.get("attention_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    else:
        model = EEGEmbeddingRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=int(config["backbone_dim"]),
            target_dim=int(config["embedding_dim"]),
            hidden_dim=int(config.get("head_hidden_dim", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, model_type


def load_conditioning_adapter(checkpoint_path: Path, device: torch.device) -> tuple[EmbeddingAdapter, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = EmbeddingAdapter(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        hidden_dim=int(config.get("hidden_dim", 2048)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def load_retrieval_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, payload


def resolve_selected_channels(
    args: argparse.Namespace,
    *,
    embedding_config: dict | None,
    retrieval_config: dict | None,
) -> list[str] | None:
    if args.selected_channels is not None:
        return list(args.selected_channels)
    if args.channel_preset is not None:
        return list(CHANNEL_PRESETS[args.channel_preset])
    if args.channel_subset_name is not None:
        source_config = retrieval_config or (embedding_config.get("retrieval_config") if embedding_config else None)
        selected_channels = None if source_config is None else source_config.get("selected_channels")
        if not selected_channels:
            raise ValueError(
                f"--channel-subset-name={args.channel_subset_name} was provided, but no selected_channels "
                "were stored in the checkpoint config."
            )
        return list(selected_channels)
    if retrieval_config is not None and retrieval_config.get("selected_channels") is not None:
        return list(retrieval_config["selected_channels"])
    if embedding_config is not None:
        if embedding_config.get("selected_channels") is not None:
            return list(embedding_config["selected_channels"])
        nested_retrieval_config = embedding_config.get("retrieval_config")
        if isinstance(nested_retrieval_config, dict) and nested_retrieval_config.get("selected_channels") is not None:
            return list(nested_retrieval_config["selected_channels"])
    return None


def build_context_batch(
    query_embeddings: torch.Tensor,
    *,
    retrieval_bank: TensorBank,
    embedding_bank: TensorBank,
    device: torch.device,
    topk: int,
    text_bank: TensorBank | None = None,
) -> dict[str, object]:
    effective_topk = min(topk, len(retrieval_bank.image_ids))
    logits = query_embeddings.float().cpu() @ retrieval_bank.values.float().T.cpu()
    _, topk_ids, topk_scores = prototype_lookup_from_logits(
        logits,
        candidate_image_ids=list(retrieval_bank.image_ids),
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


def resolve_conditioning_embeddings(
    *,
    args: argparse.Namespace,
    batch: dict[str, object],
    device: torch.device,
    conditioning_bank: TensorBank,
    embedding_model,
    embedding_config: dict | None,
    model_type: str | None,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    retrieval_model: torch.nn.Module | None,
    semantic_bank: TensorBank | None,
    perceptual_bank: TensorBank | None,
    alpha: float | None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    query_ids = list(batch["image_id"])
    eeg = batch["eeg"].to(device)

    if args.embedding_source == "ground_truth":
        target = conditioning_bank.align(query_ids, device=device).float()
        context = [{"conditioning_source": "ground_truth", "conditioning_image_id": image_id} for image_id in query_ids]
        return target, context

    if args.embedding_source == "retrieval_top1":
        if retrieval_model is None or alpha is None:
            raise ValueError("Retrieval conditioning requires a retrieval checkpoint and a resolved alpha.")

        with torch.no_grad():
            output_bundle = retrieval_model.encode_all(eeg)
        output_dict = {
            name: tensor
            for name, tensor in {
                "semantic": output_bundle.semantic,
                "perceptual": output_bundle.perceptual,
                "legacy": output_bundle.legacy,
            }.items()
            if tensor is not None
        }
        logits, candidate_ids, _ = compute_retrieval_logits(
            retrieval_model,
            output_dict,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            alpha=alpha,
        )
        _, top1_ids, top1_scores = prototype_lookup_from_logits(
            logits,
            candidate_image_ids=candidate_ids,
            topk=1,
        )
        conditioning_image_ids = [row[0] for row in top1_ids]
        embeddings = conditioning_bank.align(conditioning_image_ids, device=device).float()
        context = [
            {
                "conditioning_source": "retrieval_top1",
                "conditioning_image_id": image_id,
                "conditioning_score": float(score),
            }
            for image_id, score in zip(conditioning_image_ids, top1_scores[:, 0].tolist(), strict=True)
        ]
        return embeddings, context

    if embedding_model is None:
        raise ValueError("--reconstruction-checkpoint is required when --embedding-source=predicted.")

    effective_model_type = args.model_type or model_type or (embedding_config.get("model_type") if embedding_config else "regression")
    if effective_model_type == "rag_residual":
        if retrieval_bank is None:
            raise ValueError("rag_residual prediction requires a retrieval bank.")
        with torch.no_grad():
            shared = embedding_model.encode_backbone(eeg)
            base_outputs = embedding_model.project_shared(shared)
            effective_topk = args.retrieval_topk or int(embedding_config.get("retrieval_topk", 4))
            use_text_context = args.use_text_context or bool(embedding_config.get("use_text_context", False))
            context_batch = build_context_batch(
                base_outputs.retrieval_embedding,
                retrieval_bank=retrieval_bank,
                embedding_bank=conditioning_bank,
                device=device,
                topk=effective_topk,
                text_bank=text_bank if use_text_context else None,
            )
            final_outputs = embedding_model.project_shared(
                shared,
                retrieved_embeddings=context_batch["retrieved_embeddings"],
                retrieved_text_embeddings=context_batch["retrieved_text_embeddings"],
                retrieval_confidence=context_batch["confidence"],
            )
        conditioning_context = []
        for topk_ids, topk_scores, topk_probs, gate, conf in zip(
            context_batch["topk_ids"],
            context_batch["topk_scores"].tolist(),
            context_batch["topk_probs"].tolist(),
            final_outputs.gate.squeeze(-1).tolist(),
            context_batch["confidence"].tolist(),
            strict=True,
        ):
            conditioning_context.append(
                {
                    "conditioning_source": "predicted",
                    "conditioning_image_id": None,
                    "retrieval_topk_ids": list(topk_ids),
                    "retrieval_topk_scores": [float(score) for score in topk_scores],
                    "retrieval_topk_probs": [float(prob) for prob in topk_probs],
                    "retrieval_confidence": float(conf),
                    "residual_gate": float(gate),
                }
            )
        return final_outputs.final_embedding, conditioning_context

    with torch.no_grad():
        predicted = embedding_model(eeg)
    context = [{"conditioning_source": "predicted", "conditioning_image_id": None} for _ in query_ids]
    return predicted, context


def adapt_conditioning_embeddings(
    conditioning_embeddings: torch.Tensor,
    conditioning_context: list[dict[str, object]],
    *,
    generation_bank: TensorBank,
    conditioning_adapter,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    source_dim = int(conditioning_embeddings.shape[1])
    target_dim = int(generation_bank.values.shape[1])
    if source_dim == target_dim and conditioning_adapter is None:
        return conditioning_embeddings, conditioning_context
    if conditioning_adapter is None:
        raise ValueError(
            f"Conditioning embedding dim {source_dim} does not match generation bank dim {target_dim}. "
            "Pass --conditioning-adapter."
        )
    with torch.no_grad():
        adapted = conditioning_adapter(conditioning_embeddings).float()
    updated_context = []
    for item in conditioning_context:
        new_item = dict(item)
        new_item["conditioning_adapter_applied"] = True
        new_item["conditioning_source_dim"] = source_dim
        new_item["conditioning_target_dim"] = target_dim
        updated_context.append(new_item)
    return adapted, updated_context


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.init_image_dir is not None and not args.init_image_dir.exists():
        raise FileNotFoundError(f"Init image directory not found: {args.init_image_dir}")
    if not (0.0 < args.img2img_strength <= 1.0):
        raise ValueError("--img2img-strength must be in (0, 1].")

    embedding_model = None
    embedding_config = None
    checkpoint_model_type = None
    conditioning_adapter = None
    if args.reconstruction_checkpoint is not None:
        embedding_model, embedding_config, checkpoint_model_type = load_embedding_model(args.reconstruction_checkpoint, device)
    if args.conditioning_adapter is not None:
        conditioning_adapter, _ = load_conditioning_adapter(args.conditioning_adapter, device)

    if args.embedding_source == "predicted" and embedding_model is None:
        raise ValueError("--reconstruction-checkpoint is required when --embedding-source=predicted.")

    generation_bank_path = resolve_embedding_bank_path(args, embedding_config, args.reconstruction_checkpoint)
    conditioning_bank_path = resolve_conditioning_bank_path(
        args,
        embedding_config,
        args.reconstruction_checkpoint,
        generation_bank_path=generation_bank_path,
    )
    generation_bank = TensorBank.load(generation_bank_path)
    conditioning_bank = TensorBank.load(conditioning_bank_path)
    generation_config = resolve_generation_config(args, embedding_config, generation_bank_path)

    effective_model_type = args.model_type or checkpoint_model_type or (embedding_config.get("model_type") if embedding_config else "regression")
    base_paths = [PROJECT_ROOT]
    if args.reconstruction_checkpoint is not None:
        base_paths.extend([args.reconstruction_checkpoint.parent, args.reconstruction_checkpoint.parent.parent])

    retrieval_bank = resolve_bank(
        args.retrieval_bank,
        embedding_config.get("retrieval_bank") if embedding_config is not None else None,
        name="Retrieval",
        base_paths=base_paths,
    )
    text_bank = resolve_bank(
        args.text_bank,
        embedding_config.get("text_bank") if embedding_config is not None else None,
        name="Text",
        base_paths=base_paths,
    )

    retrieval_model = None
    retrieval_config = None
    retrieval_checkpoint = args.retrieval_checkpoint
    if retrieval_checkpoint is None and embedding_config is not None and embedding_config.get("retrieval_checkpoint"):
        retrieval_checkpoint = resolve_artifact_path(
            embedding_config["retrieval_checkpoint"],
            base_paths=base_paths,
        )

    semantic_bank = None
    perceptual_bank = None
    alpha = None
    if args.embedding_source == "retrieval_top1":
        if retrieval_checkpoint is None:
            raise ValueError(
                "--retrieval-checkpoint is required when --embedding-source=retrieval_top1 "
                "and no retrieval checkpoint is stored in the reconstruction config."
            )
        retrieval_model, retrieval_config, retrieval_payload = load_retrieval_model(retrieval_checkpoint, device)
        semantic_bank = resolve_bank(
            args.semantic_bank,
            retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank"),
            name="Semantic",
            base_paths=[PROJECT_ROOT, retrieval_checkpoint.parent, retrieval_checkpoint.parent.parent],
        )
        perceptual_bank = resolve_bank(
            args.perceptual_bank,
            retrieval_config.get("perceptual_bank"),
            name="Perceptual",
            base_paths=[PROJECT_ROOT, retrieval_checkpoint.parent, retrieval_checkpoint.parent.parent],
        )
        if semantic_bank is None and perceptual_bank is None:
            raise ValueError("At least one retrieval bank must be available for retrieval_top1 conditioning.")
        alpha = resolve_alpha(
            args,
            retrieval_payload,
            has_semantic=semantic_bank is not None,
            has_perceptual=perceptual_bank is not None,
        )

    selected_channels = resolve_selected_channels(
        args,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
    )

    prior_pipe = load_kandinsky_prior_pipeline(
        model_name=str(generation_config["prior_model"]),
        device=device,
        local_files_only=args.local_files_only,
    )
    if args.init_image_dir is None:
        decoder_pipe = load_kandinsky_decoder_pipeline(
            model_name=str(generation_config["decoder_model"]),
            device=device,
            local_files_only=args.local_files_only,
        )
    else:
        decoder_pipe = load_kandinsky_img2img_decoder_pipeline(
            model_name=str(generation_config["decoder_model"]),
            device=device,
            local_files_only=args.local_files_only,
        )

    negative_image_embed = negative_image_embed_from_bank(
        generation_bank,
        batch_size=1,
        device=device,
        dtype=next(decoder_pipe.unet.parameters()).dtype,
    )

    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        image_ids=selected_image_ids,
        selected_channels=selected_channels,
    )
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    all_fake_images: list[torch.Tensor] = []
    all_real_images: list[torch.Tensor] = []

    for batch in loader:
        init_paths = None
        init_images = None
        if args.init_image_dir is not None:
            init_paths = resolve_init_image_paths(list(batch["image_id"]), args.init_image_dir)
            init_images = load_image_batch(init_paths, image_size=int(generation_config["decoder_height"]))
        conditioning_embeddings, conditioning_context = resolve_conditioning_embeddings(
            args=args,
            batch=batch,
            device=device,
            conditioning_bank=conditioning_bank,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
            model_type=effective_model_type,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
            retrieval_model=retrieval_model,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            alpha=alpha,
        )
        conditioning_embeddings, conditioning_context = adapt_conditioning_embeddings(
            conditioning_embeddings,
            conditioning_context,
            generation_bank=generation_bank,
            conditioning_adapter=conditioning_adapter,
        )
        generation = generate_best_images(
            predicted_embeddings=conditioning_embeddings,
            prior_pipe=prior_pipe,
            decoder_pipe=decoder_pipe,
            negative_image_embeds=negative_image_embed,
            num_candidates=int(generation_config["num_candidates"]),
            num_inference_steps=int(generation_config["decoder_steps"]),
            guidance_scale=float(generation_config["decoder_guidance_scale"]),
            height=int(generation_config["decoder_height"]),
            width=int(generation_config["decoder_width"]),
            init_images=init_images,
            img2img_strength=args.img2img_strength,
            candidate_seed_offset=args.candidate_seed_offset,
            selection_mode=args.candidate_selection_mode,
            candidate_lowlevel_weight=args.candidate_lowlevel_weight,
            candidate_lowlevel_metric=args.candidate_lowlevel_metric,
            candidate_score_normalization=args.candidate_score_normalization,
        )
        selected_images = generation["selected_images"]
        saved_paths = save_image_batch(selected_images, list(batch["image_id"]), output_images_dir)
        all_fake_images.extend(list(selected_images))

        if args.evaluate:
            real_images = load_image_batch(list(batch["image_path"]), image_size=int(generation_config["decoder_height"]))
            all_real_images.extend(list(real_images))

        candidate_scores = generation["candidate_scores"].tolist()
        candidate_semantic_scores = generation["candidate_semantic_scores"].tolist()
        raw_candidate_lowlevel_scores = generation["candidate_lowlevel_scores"]
        candidate_lowlevel_scores = (
            None if raw_candidate_lowlevel_scores is None else raw_candidate_lowlevel_scores.tolist()
        )
        selected_indices = generation["selected_candidate_indices"].tolist()
        selected_scores = generation["selected_scores"].tolist()
        candidate_seeds = generation["candidate_seeds"]
        selected_seeds = generation["selected_candidate_seeds"]

        for row_index, (query_id, output_path, selected_index, selected_seed, selected_score, scores, semantic_scores, context) in enumerate(zip(
            batch["image_id"],
            saved_paths,
            selected_indices,
            selected_seeds,
            selected_scores,
            candidate_scores,
            candidate_semantic_scores,
            conditioning_context,
            strict=True,
        )):
            init_image_path = None
            if init_paths is not None:
                init_image_path = str((args.init_image_dir / f"{query_id}.png").resolve())
            lowlevel_scores = None
            if candidate_lowlevel_scores is not None:
                lowlevel_scores = candidate_lowlevel_scores[row_index]
            item = {
                "query_image_id": query_id,
                "output_path": output_path,
                "mode": "embedding_decoder",
                "model_type": effective_model_type,
                "embedding_source": args.embedding_source,
                "init_image_mode": "external_dir" if args.init_image_dir is not None else "none",
                "init_image_path": init_image_path,
                "img2img_strength": float(args.img2img_strength) if args.init_image_dir is not None else None,
                "candidate_seed_offset": int(args.candidate_seed_offset),
                "generation_mode": generation["generation_mode"],
                "selection_mode": generation["selection_mode"],
                "candidate_lowlevel_weight": float(args.candidate_lowlevel_weight),
                "candidate_lowlevel_metric": args.candidate_lowlevel_metric,
                "candidate_score_normalization": args.candidate_score_normalization,
                "candidate_seeds": candidate_seeds,
                "selected_candidate_index": int(selected_index),
                "selected_candidate_seed": int(selected_seed),
                "selected_score": float(selected_score),
                "candidate_scores": [float(score) for score in scores],
                "candidate_semantic_scores": [float(score) for score in semantic_scores],
                "candidate_lowlevel_scores": None
                if lowlevel_scores is None
                else [float(score) for score in lowlevel_scores],
            }
            item.update(context)
            metadata.append(item)

    save_json(
        {
            "split": args.split,
            "split_file": None if args.split_file is None else str(args.split_file.resolve()),
            "image_id_source": args.image_id_source,
            "embedding_source": args.embedding_source,
            "selected_channels": selected_channels,
            "conditioning_bank": str(conditioning_bank_path),
            "generation_bank": str(generation_bank_path),
            "retrieval_bank": None if retrieval_bank is None else retrieval_bank.bank_type,
            "text_bank": None if text_bank is None else text_bank.bank_type,
            "mode": "embedding_decoder",
            "model_type": effective_model_type,
            "conditioning_adapter": None if args.conditioning_adapter is None else str(args.conditioning_adapter.resolve()),
            "decoder_model": str(generation_config["decoder_model"]),
            "prior_model": str(generation_config["prior_model"]),
            "num_candidates": float(generation_config["num_candidates"]),
            "decoder_steps": float(generation_config["decoder_steps"]),
            "decoder_guidance_scale": float(generation_config["decoder_guidance_scale"]),
            "decoder_height": float(generation_config["decoder_height"]),
            "decoder_width": float(generation_config["decoder_width"]),
            "init_image_mode": "external_dir" if args.init_image_dir is not None else "none",
            "init_image_dir": None if args.init_image_dir is None else str(args.init_image_dir.resolve()),
            "img2img_strength": float(args.img2img_strength) if args.init_image_dir is not None else None,
            "candidate_seed_offset": int(args.candidate_seed_offset),
            "candidate_selection_mode": args.candidate_selection_mode,
            "candidate_lowlevel_weight": float(args.candidate_lowlevel_weight),
            "candidate_lowlevel_metric": args.candidate_lowlevel_metric,
            "candidate_score_normalization": args.candidate_score_normalization,
            "reconstruction_checkpoint": None
            if args.reconstruction_checkpoint is None
            else str(args.reconstruction_checkpoint.resolve()),
            "retrieval_checkpoint": None if retrieval_checkpoint is None else str(retrieval_checkpoint.resolve()),
            "alpha": None if alpha is None else float(alpha),
            "local_files_only": bool(args.local_files_only),
            "predictions": metadata,
        },
        args.output_dir / "reconstruction_metadata.json",
    )

    if args.evaluate:
        metrics = eval_images(
            real_images=torch.stack(all_real_images, dim=0),
            fake_images=torch.stack(all_fake_images, dim=0),
            device=device,
        )
        metrics["num_candidates"] = float(generation_config["num_candidates"])
        metrics["decoder_steps"] = float(generation_config["decoder_steps"])
        metrics["decoder_guidance_scale"] = float(generation_config["decoder_guidance_scale"])
        metrics["candidate_lowlevel_weight"] = float(args.candidate_lowlevel_weight)
        save_json(metrics, args.output_dir / "reconstruction_metrics.json")
        print(metrics)
    else:
        print(f"saved reconstruction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
