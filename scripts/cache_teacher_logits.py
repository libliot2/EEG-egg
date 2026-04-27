#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.image_banks import TeacherLogitsBank, TensorBank
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache teacher retrieval logits for distillation.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--rerank-bank", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rerank-alpha", type=float, default=None)
    parser.add_argument("--rerank-topk", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--normalize", choices=["none", "zscore", "l2"], default="zscore")
    return parser.parse_args()


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_alpha(explicit_alpha: float | None, payload: dict, has_semantic: bool, has_perceptual: bool) -> float:
    if explicit_alpha is not None:
        return float(explicit_alpha)
    metrics = payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    return 0.5


def resolve_rerank_alpha(explicit_alpha: float | None, config: dict) -> float:
    if explicit_alpha is not None:
        return float(explicit_alpha)
    return float(config.get("rerank_alpha", 0.0))


def resolve_rerank_topk(explicit_topk: int | None, config: dict) -> int:
    if explicit_topk is not None:
        return int(explicit_topk)
    return int(config.get("rerank_topk", 0))


def normalize_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return logits
    if mode == "zscore":
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (logits - mean) / std
    if mode == "l2":
        return logits / logits.norm(dim=1, keepdim=True).clamp_min(1e-6)
    raise ValueError(f"Unknown normalize mode: {mode}")


def resolve_weights(num_items: int, weights: list[float] | None) -> list[float]:
    if not weights:
        return [1.0 / num_items] * num_items
    if len(weights) != num_items:
        raise ValueError(f"Expected {num_items} weights, got {len(weights)}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [weight / total for weight in weights]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    weights = resolve_weights(len(args.checkpoints), args.weights)
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)

    blended_logits = None
    resolved_query_ids = None
    resolved_candidate_ids = None
    source_metadata: list[dict[str, object]] = []

    for checkpoint, weight in zip(args.checkpoints, weights, strict=True):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = build_retrieval_model_from_config(config).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()

        semantic_bank = resolve_bank(
            args.semantic_bank,
            config.get("semantic_bank") or config.get("clip_bank"),
            name="Semantic",
        )
        perceptual_bank = resolve_bank(args.perceptual_bank, config.get("perceptual_bank"), name="Perceptual")
        rerank_bank = resolve_bank(args.rerank_bank, config.get("rerank_bank"), name="Rerank")
        if semantic_bank is None and perceptual_bank is None:
            raise ValueError(f"{checkpoint} does not resolve any semantic/perceptual bank.")

        records = load_eeg_records(
            data_dir=args.data_dir,
            split=args.split,
            avg_trials=True,
            selected_channels=config.get("selected_channels"),
            image_ids=selected_image_ids,
        )
        loader = make_dataloader(
            EEGImageDataset(records),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        outputs, query_image_ids, _ = compute_retrieval_outputs(model, loader, device)
        candidate_image_ids = query_image_ids if args.split == "train" else None
        alpha = resolve_alpha(args.alpha, payload, semantic_bank is not None, perceptual_bank is not None)
        rerank_alpha = resolve_rerank_alpha(args.rerank_alpha, config)
        rerank_topk = resolve_rerank_topk(args.rerank_topk, config)
        logits, candidate_ids, _ = compute_retrieval_logits(
            model,
            outputs,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            rerank_bank=rerank_bank,
            candidate_image_ids=candidate_image_ids,
            alpha=alpha,
            rerank_alpha=rerank_alpha,
            rerank_topk=rerank_topk,
        )
        logits = normalize_logits(logits.float().cpu(), args.normalize)

        if resolved_query_ids is None:
            resolved_query_ids = list(query_image_ids)
            resolved_candidate_ids = list(candidate_ids)
            blended_logits = weight * logits
        else:
            if query_image_ids != resolved_query_ids:
                raise ValueError("Teacher checkpoints produced different query ordering.")
            if candidate_ids != resolved_candidate_ids:
                raise ValueError("Teacher checkpoints produced different candidate ordering.")
            assert blended_logits is not None
            blended_logits = blended_logits + weight * logits

        source_metadata.append(
            {
                "checkpoint": str(checkpoint),
                "weight": float(weight),
                "alpha": float(alpha),
                "rerank_alpha": float(rerank_alpha),
                "rerank_topk": int(rerank_topk),
            }
        )

    assert resolved_query_ids is not None
    assert resolved_candidate_ids is not None
    assert blended_logits is not None

    ensure_dir(args.output.parent)
    TeacherLogitsBank(
        query_image_ids=resolved_query_ids,
        candidate_image_ids=resolved_candidate_ids,
        logits=blended_logits.cpu(),
        metadata={
            "split": args.split,
            "image_id_source": args.image_id_source,
            "normalize": args.normalize,
            "sources": source_metadata,
        },
    ).save(str(args.output))
    print(f"saved teacher logits to {args.output}")


if __name__ == "__main__":
    main()
