#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.utils import DEFAULT_OUTPUT_DIR, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select retrieval ensemble/calibration on validation 200-way episodes, "
            "then apply the fixed setting to test logits."
        )
    )
    parser.add_argument("--val-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--test-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_episode_calibration")
    parser.add_argument("--normalize", choices=["none", "zscore", "l2", "softmax"], default="zscore")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--max-active", type=int, default=3)
    parser.add_argument("--episode-size", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--csls-k", type=int, nargs="*", default=[0, 5, 10, 20])
    parser.add_argument("--column-center", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--mnn-k", type=int, nargs="*", default=[0, 5, 10])
    parser.add_argument("--mnn-bonus", type=float, nargs="*", default=[0.0, 0.05, 0.1])
    parser.add_argument("--top-history", type=int, default=50)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def check_group(payloads: list[dict[str, Any]], paths: list[Path]) -> tuple[list[str], list[str]]:
    query_ids = list(payloads[0]["ordered_query_image_ids"])
    candidate_ids = list(payloads[0]["candidate_image_ids"])
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query ordering mismatch in {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate ordering mismatch in {path}")
    return query_ids, candidate_ids


def check_matching_sources(val_paths: list[Path], test_paths: list[Path]) -> None:
    if len(val_paths) != len(test_paths):
        raise ValueError("--val-logit-files and --test-logit-files must have the same length.")


def target_indices(query_ids: list[str], candidate_ids: list[str]) -> torch.Tensor:
    index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
    missing = [image_id for image_id in query_ids if image_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} query ids are missing from candidates; first={missing[0]}")
    return torch.tensor([index[image_id] for image_id in query_ids], dtype=torch.long)


def normalize_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    values = logits.float()
    if mode == "none":
        return values
    if mode == "zscore":
        return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)
    if mode == "l2":
        return values / values.norm(dim=1, keepdim=True).clamp_min(1e-6)
    if mode == "softmax":
        return torch.softmax(values, dim=1)
    raise ValueError(f"Unknown normalize mode: {mode}")


def fuse_logits(logits_list: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    fused = torch.zeros_like(logits_list[0], dtype=torch.float32)
    for weight, logits in zip(weights, logits_list, strict=True):
        fused += float(weight) * logits.float()
    return fused


def weight_grid(num_files: int, *, step: float, max_active: int) -> list[list[float]]:
    if num_files < 1:
        raise ValueError("At least one file is required.")
    if step <= 0.0 or step > 1.0:
        raise ValueError("--weight-step must be in (0, 1].")
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-6:
        raise ValueError("--weight-step must evenly divide 1.0.")
    max_active = min(max(1, int(max_active)), num_files)
    weights: list[list[float]] = []
    for active in range(1, max_active + 1):
        for indices in itertools.combinations(range(num_files), active):
            for parts in integer_compositions(units, active):
                values = [0.0] * num_files
                for idx, part in zip(indices, parts, strict=True):
                    values[idx] = float(part) / float(units)
                weights.append(values)
    # Preserve order while dropping duplicates from e.g. active=1.
    seen: set[tuple[float, ...]] = set()
    unique: list[list[float]] = []
    for values in weights:
        key = tuple(round(value, 10) for value in values)
        if key not in seen:
            seen.add(key)
            unique.append(values)
    return unique


def integer_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for tail in integer_compositions(total - first, parts - 1):
            result.append((first, *tail))
    return result


def apply_calibration(
    logits: torch.Tensor,
    *,
    csls_k: int,
    column_center: float,
    mnn_k: int,
    mnn_bonus: float,
) -> torch.Tensor:
    values = logits.float()
    adjusted = values
    if csls_k > 0:
        k_row = min(int(csls_k), values.shape[1])
        k_col = min(int(csls_k), values.shape[0])
        row_local = values.topk(k=k_row, dim=1).values.mean(dim=1, keepdim=True)
        col_local = values.topk(k=k_col, dim=0).values.mean(dim=0, keepdim=True)
        adjusted = 2.0 * adjusted - row_local - col_local
    if column_center > 0.0:
        adjusted = adjusted - float(column_center) * adjusted.mean(dim=0, keepdim=True)
    if mnn_k > 0 and mnn_bonus > 0.0:
        k_row = min(int(mnn_k), values.shape[1])
        k_col = min(int(mnn_k), values.shape[0])
        row_top = values.topk(k=k_row, dim=1).indices
        col_top = values.topk(k=k_col, dim=0).indices
        row_mask = torch.zeros_like(values, dtype=torch.bool)
        row_mask.scatter_(1, row_top, True)
        col_mask = torch.zeros_like(values, dtype=torch.bool)
        for col_idx in range(values.shape[1]):
            col_mask[col_top[:, col_idx], col_idx] = True
        adjusted = adjusted + float(mnn_bonus) * (row_mask & col_mask).float()
    return adjusted


def build_episode_indices(
    *,
    num_queries: int,
    targets: torch.Tensor,
    candidate_count: int,
    episode_size: int,
    num_episodes: int,
    seed: int,
) -> torch.Tensor:
    episode_size = min(int(episode_size), candidate_count)
    if episode_size < 2:
        raise ValueError("--episode-size must be at least 2.")
    rng = random.Random(seed)
    all_indices = list(range(candidate_count))
    episodes: list[torch.Tensor] = []
    for _ in range(int(num_episodes)):
        per_query: list[list[int]] = []
        for row in range(num_queries):
            target = int(targets[row].item())
            pool = [idx for idx in all_indices if idx != target]
            sampled = rng.sample(pool, episode_size - 1)
            candidates = [target, *sampled]
            rng.shuffle(candidates)
            per_query.append(candidates)
        episodes.append(torch.tensor(per_query, dtype=torch.long))
    return torch.stack(episodes, dim=0)


def episode_metrics(logits: torch.Tensor, targets: torch.Tensor, episodes: torch.Tensor) -> dict[str, float]:
    expanded = logits.unsqueeze(0).expand(episodes.shape[0], -1, -1)
    scores = expanded.gather(2, episodes)
    target_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    ranks = (scores > target_scores.view(1, -1, 1)).sum(dim=2) + 1
    return {
        "episode_top1_acc": float((ranks == 1).float().mean().item()),
        "episode_top5_acc": float((ranks <= 5).float().mean().item()),
    }


def select_setting(
    logits_list: list[torch.Tensor],
    *,
    targets: torch.Tensor,
    episodes: torch.Tensor,
    weights_grid: list[list[float]],
    csls_values: list[int],
    column_center_values: list[float],
    mnn_k_values: list[int],
    mnn_bonus_values: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for weights in weights_grid:
        fused = fuse_logits(logits_list, weights)
        for csls_k in csls_values:
            for column_center in column_center_values:
                for mnn_k in mnn_k_values:
                    for mnn_bonus in mnn_bonus_values:
                        if mnn_k <= 0 and mnn_bonus > 0.0:
                            continue
                        calibrated = apply_calibration(
                            fused,
                            csls_k=int(csls_k),
                            column_center=float(column_center),
                            mnn_k=int(mnn_k),
                            mnn_bonus=float(mnn_bonus),
                        )
                        metrics = episode_metrics(calibrated, targets, episodes)
                        record = {
                            "weights": weights,
                            "csls_k": int(csls_k),
                            "column_center": float(column_center),
                            "mnn_k": int(mnn_k),
                            "mnn_bonus": float(mnn_bonus),
                            **metrics,
                        }
                        history.append(record)
                        score = (
                            float(metrics["episode_top1_acc"]),
                            float(metrics["episode_top5_acc"]),
                            -sum(1 for value in weights if value > 0.0),
                            -abs(max(weights) - 1.0),
                        )
                        if best is None or score > best["_score"]:
                            best = {**record, "_score": score}
    assert best is not None
    best.pop("_score", None)
    history.sort(key=lambda item: (item["episode_top1_acc"], item["episode_top5_acc"]), reverse=True)
    return best, history


def subset_payload(payload: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    result = dict(payload)
    result["logits"] = logits
    return result


def main() -> None:
    args = parse_args()
    check_matching_sources(args.val_logit_files, args.test_logit_files)
    if args.names is not None and len(args.names) not in {0, len(args.val_logit_files)}:
        raise ValueError("--names must be omitted or match the number of logit files.")

    val_payloads = [load_payload(path) for path in args.val_logit_files]
    test_payloads = [load_payload(path) for path in args.test_logit_files]
    val_query_ids, val_candidate_ids = check_group(val_payloads, args.val_logit_files)
    test_query_ids, test_candidate_ids = check_group(test_payloads, args.test_logit_files)
    val_targets = target_indices(val_query_ids, val_candidate_ids)

    val_logits = [normalize_logits(torch.as_tensor(payload["logits"]), args.normalize) for payload in val_payloads]
    test_logits = [normalize_logits(torch.as_tensor(payload["logits"]), args.normalize) for payload in test_payloads]
    weights_grid = weight_grid(len(val_logits), step=float(args.weight_step), max_active=int(args.max_active))
    episodes = build_episode_indices(
        num_queries=len(val_query_ids),
        targets=val_targets,
        candidate_count=len(val_candidate_ids),
        episode_size=int(args.episode_size),
        num_episodes=int(args.num_episodes),
        seed=int(args.seed),
    )
    best, history = select_setting(
        val_logits,
        targets=val_targets,
        episodes=episodes,
        weights_grid=weights_grid,
        csls_values=[int(value) for value in args.csls_k],
        column_center_values=[float(value) for value in args.column_center],
        mnn_k_values=[int(value) for value in args.mnn_k],
        mnn_bonus_values=[float(value) for value in args.mnn_bonus],
    )

    fused_test = fuse_logits(test_logits, [float(value) for value in best["weights"]])
    calibrated_test = apply_calibration(
        fused_test,
        csls_k=int(best["csls_k"]),
        column_center=float(best["column_center"]),
        mnn_k=int(best["mnn_k"]),
        mnn_bonus=float(best["mnn_bonus"]),
    )
    test_metrics = compute_retrieval_metrics(
        calibrated_test,
        ordered_image_ids=test_query_ids,
        candidate_image_ids=test_candidate_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": test_payloads[0].get("split", "test"),
            "ordered_query_image_ids": test_query_ids,
            "candidate_image_ids": test_candidate_ids,
            "source_test_logit_files": [str(path) for path in args.test_logit_files],
            "source_val_logit_files": [str(path) for path in args.val_logit_files],
            "names": args.names,
            "normalize": args.normalize,
            "selected_setting": best,
            "logits": calibrated_test,
        },
        args.output_dir / "retrieval_logits.pt",
    )
    rankings = rank_candidate_ids(calibrated_test, test_candidate_ids)
    save_json(
        {
            "split": test_payloads[0].get("split", "test"),
            "selected_setting": best,
            "normalize": args.normalize,
            "source_test_logit_files": [str(path) for path in args.test_logit_files],
            "predictions": [
                {"query_image_id": image_id, "ranked_candidate_ids": ranked_ids}
                for image_id, ranked_ids in zip(test_query_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )
    save_json(
        {
            "selected_setting": best,
            "test_metrics": {key: float(value) for key, value in test_metrics.items()},
            "normalize": args.normalize,
            "episode_size": int(args.episode_size),
            "num_episodes": int(args.num_episodes),
            "seed": int(args.seed),
            "num_weight_candidates": len(weights_grid),
            "top_history": history[: int(args.top_history)],
            "source_val_logit_files": [str(path) for path in args.val_logit_files],
            "source_test_logit_files": [str(path) for path in args.test_logit_files],
        },
        args.output_dir / "retrieval_metrics.json",
    )
    print({"selected_setting": best, **{key: float(value) for key, value in test_metrics.items()}})


if __name__ == "__main__":
    main()
