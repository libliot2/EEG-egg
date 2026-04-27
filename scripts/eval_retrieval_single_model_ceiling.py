#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import compute_retrieval_metrics
from project1_eeg.image_banks import TensorBank
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-model retrieval ceiling via checkpoints or soup.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--rerank-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_single_model_ceiling")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rerank-alpha", type=float, default=None)
    parser.add_argument("--rerank-topk", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--no-soup", action="store_true")
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


def resolve_weights(num_items: int, weights: list[float] | None) -> list[float]:
    if not weights:
        return [1.0 / num_items] * num_items
    if len(weights) != num_items:
        raise ValueError(f"Expected {num_items} weights, got {len(weights)}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [weight / total for weight in weights]


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    averaged = copy.deepcopy(state_dicts[0])
    for key, value in averaged.items():
        if not torch.is_tensor(value):
            continue
        if value.dtype.is_floating_point:
            weighted = sum(weight * state[key].float() for weight, state in zip(weights, state_dicts, strict=True))
            averaged[key] = weighted.to(dtype=value.dtype)
        else:
            averaged[key] = state_dicts[0][key]
    return averaged


def evaluate_payload(
    payload: dict,
    *,
    args: argparse.Namespace,
    device: torch.device,
    records,
    loader,
) -> dict[str, float]:
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
        raise ValueError("At least one semantic/perceptual bank is required for evaluation.")

    outputs, query_image_ids, _ = compute_retrieval_outputs(model, loader, device)
    candidate_image_ids = query_image_ids if args.split == "train" else None
    alpha = resolve_alpha(args.alpha, payload, semantic_bank is not None, perceptual_bank is not None)
    rerank_alpha = resolve_rerank_alpha(args.rerank_alpha, config)
    rerank_topk = resolve_rerank_topk(args.rerank_topk, config)
    logits, resolved_candidate_ids, _ = compute_retrieval_logits(
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
    metrics = compute_retrieval_metrics(
        logits,
        ordered_image_ids=[record.image_id for record in records],
        candidate_image_ids=resolved_candidate_ids,
    )
    metrics["alpha"] = float(alpha)
    metrics["rerank_alpha"] = float(rerank_alpha)
    metrics["rerank_topk"] = int(rerank_topk)
    return metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        image_ids=selected_image_ids,
    )
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    payloads = [torch.load(checkpoint, map_location="cpu", weights_only=False) for checkpoint in args.checkpoints]
    results: dict[str, dict[str, float]] = {}

    for checkpoint, payload in zip(args.checkpoints, payloads, strict=True):
        results[str(checkpoint)] = evaluate_payload(
            payload,
            args=args,
            device=device,
            records=records,
            loader=loader,
        )

    if len(payloads) > 1 and not args.no_soup:
        weights = resolve_weights(len(payloads), args.weights)
        soup_payload = copy.deepcopy(payloads[0])
        soup_payload["model_state"] = average_state_dicts([payload["model_state"] for payload in payloads], weights)
        soup_name = "soup:" + ",".join(str(checkpoint) for checkpoint in args.checkpoints)
        soup_metrics = evaluate_payload(
            soup_payload,
            args=args,
            device=device,
            records=records,
            loader=loader,
        )
        soup_metrics["soup_weights"] = weights
        results[soup_name] = soup_metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "results": results,
        },
        args.output_dir / "retrieval_single_model_ceiling.json",
    )
    for name, metrics in results.items():
        print(name)
        print(metrics)


if __name__ == "__main__":
    main()
