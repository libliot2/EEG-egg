#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.utils import DEFAULT_OUTPUT_DIR, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse retrieval logit files and evaluate the ensemble.")
    parser.add_argument("--logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_ensemble")
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--normalize", choices=["none", "zscore", "l2", "softmax"], default="zscore")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def normalize_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    values = logits.float()
    if mode == "none":
        return values
    if mode == "zscore":
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (values - mean) / std
    if mode == "l2":
        norm = values.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return values / norm
    if mode == "softmax":
        return torch.softmax(values, dim=1)
    raise ValueError(f"Unknown normalize mode: {mode}")


def check_compatibility(payloads: list[dict], paths: list[Path]) -> tuple[list[str], list[str], str]:
    reference = payloads[0]
    query_ids = list(reference["ordered_query_image_ids"])
    candidate_ids = list(reference["candidate_image_ids"])
    split = str(reference.get("split", "unknown"))

    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query ordering mismatch in {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate ordering mismatch in {path}")
        if str(payload.get("split", "unknown")) != split:
            raise ValueError(f"Split mismatch in {path}")

    return query_ids, candidate_ids, split


def fuse_logits(logits_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    fused = torch.zeros_like(logits_list[0], dtype=torch.float32)
    for weight, logits in zip(weights, logits_list, strict=True):
        fused += float(weight) * logits.float()
    return fused


def resolve_weights(args: argparse.Namespace, num_files: int) -> list[float] | None:
    if args.weights is None:
        return None
    if len(args.weights) != num_files:
        raise ValueError(f"--weights expected {num_files} values, got {len(args.weights)}.")
    total = sum(float(weight) for weight in args.weights)
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value.")
    return [float(weight) / total for weight in args.weights]


def search_two_way_weights(
    logits_list: list[torch.Tensor],
    *,
    ordered_image_ids: list[str],
    candidate_image_ids: list[str],
    step: float,
) -> tuple[list[float], dict[str, float], list[dict[str, float]]]:
    if len(logits_list) != 2:
        raise ValueError("--grid-search currently supports exactly two logit files.")
    if step <= 0.0 or step > 1.0:
        raise ValueError("--grid-step must be in (0, 1].")

    num_steps = int(round(1.0 / step))
    candidates = [round(index * step, 10) for index in range(num_steps + 1)]
    if candidates[-1] != 1.0:
        candidates.append(1.0)

    best_weights = [0.5, 0.5]
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    history: list[dict[str, float]] = []

    for alpha in candidates:
        weights = [float(alpha), float(1.0 - alpha)]
        fused = fuse_logits(logits_list, weights)
        metrics = compute_retrieval_metrics(
            fused,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        record = {"weight_0": weights[0], "weight_1": weights[1], **metrics}
        history.append(record)
        score = (
            float(metrics["top1_acc"]),
            float(metrics["top5_acc"]),
            -abs(weights[0] - 0.5),
        )
        if best_score is None or score > best_score:
            best_weights = weights
            best_metrics = metrics
            best_score = score

    assert best_metrics is not None
    return best_weights, best_metrics, history


def main() -> None:
    args = parse_args()
    if len(args.logit_files) < 2:
        raise ValueError("--logit-files requires at least two inputs.")

    payloads = [load_payload(path) for path in args.logit_files]
    ordered_image_ids, candidate_image_ids, split = check_compatibility(payloads, args.logit_files)
    normalized_logits = [normalize_logits(torch.as_tensor(payload["logits"]), args.normalize) for payload in payloads]

    explicit_weights = resolve_weights(args, len(args.logit_files))
    grid_history: list[dict[str, float]] = []
    if args.grid_search:
        weights, metrics, grid_history = search_two_way_weights(
            normalized_logits,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
            step=args.grid_step,
        )
    else:
        weights = explicit_weights or [1.0 / len(normalized_logits)] * len(normalized_logits)
        fused = fuse_logits(normalized_logits, weights)
        metrics = compute_retrieval_metrics(
            fused,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )

    fused_logits = fuse_logits(normalized_logits, weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "split": split,
            "ordered_query_image_ids": ordered_image_ids,
            "candidate_image_ids": candidate_image_ids,
            "weights": weights,
            "normalize": args.normalize,
            "source_logit_files": [str(path) for path in args.logit_files],
            "logits": fused_logits,
        },
        args.output_dir / "retrieval_logits.pt",
    )

    rankings = rank_candidate_ids(fused_logits, candidate_image_ids)
    save_json(
        {
            "split": split,
            "weights": weights,
            "normalize": args.normalize,
            "source_logit_files": [str(path) for path in args.logit_files],
            "predictions": [
                {
                    "query_image_id": image_id,
                    "ranked_candidate_ids": ranked_ids,
                }
                for image_id, ranked_ids in zip(ordered_image_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )

    save_json(
        {
            "split": split,
            "weights": weights,
            "normalize": args.normalize,
            "source_logit_files": [str(path) for path in args.logit_files],
            "grid_history": grid_history,
            **metrics,
        },
        args.output_dir / "retrieval_metrics.json",
    )
    print({"split": split, "weights": weights, "normalize": args.normalize, **metrics})


if __name__ == "__main__":
    main()
