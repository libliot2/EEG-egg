#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.evaluation import compute_retrieval_metrics
from project1_eeg.utils import DEFAULT_OUTPUT_DIR, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose retrieval logits, oracle top-k coverage, and model overlap.")
    parser.add_argument("--logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_diagnostics")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 3, 4, 5, 10])
    parser.add_argument("--max-misses", type=int, default=50)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def check_compatible(payloads: list[dict[str, Any]], paths: list[Path]) -> tuple[list[str], list[str]]:
    query_ids = list(payloads[0]["ordered_query_image_ids"])
    candidate_ids = list(payloads[0]["candidate_image_ids"])
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query id ordering mismatch: {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate id ordering mismatch: {path}")
    return query_ids, candidate_ids


def normalize_name(path: Path, index: int, names: list[str] | None) -> str:
    if names:
        if len(names) != index + 1 and len(names) < index + 1:
            raise ValueError("--names must match --logit-files length.")
        return names[index]
    parent = path.parent.name
    return parent if parent else path.stem


def target_indices(query_ids: list[str], candidate_ids: list[str]) -> torch.Tensor:
    index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
    missing = [image_id for image_id in query_ids if image_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} query image ids are not present in candidates; first={missing[0]}")
    return torch.tensor([index[image_id] for image_id in query_ids], dtype=torch.long)


def topk_hit_mask(logits: torch.Tensor, targets: torch.Tensor, topk: int) -> torch.Tensor:
    k = min(int(topk), logits.shape[1])
    indices = logits.topk(k=k, dim=1).indices
    return (indices == targets.unsqueeze(1)).any(dim=1)


def summarize_model(
    *,
    name: str,
    path: Path,
    logits: torch.Tensor,
    query_ids: list[str],
    candidate_ids: list[str],
    targets: torch.Tensor,
    topks: list[int],
    max_misses: int,
) -> dict[str, Any]:
    metrics = compute_retrieval_metrics(logits, ordered_image_ids=query_ids, candidate_image_ids=candidate_ids)
    ranking = logits.argsort(dim=1, descending=True)
    target_ranks = (ranking == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    margins = top2[:, 0] - top2[:, 1] if top2.shape[1] == 2 else torch.zeros(logits.shape[0])
    top1 = ranking[:, 0]
    hit_top1 = top1 == targets
    misses = []
    miss_rows = (~hit_top1).nonzero(as_tuple=False).flatten().tolist()
    for row in miss_rows[:max_misses]:
        misses.append(
            {
                "query_image_id": query_ids[row],
                "target_rank": int(target_ranks[row].item()),
                "top1_candidate_id": candidate_ids[int(top1[row].item())],
                "top1_margin": float(margins[row].item()),
            }
        )
    return {
        "name": name,
        "path": str(path),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "oracle_coverage": {
            f"top{k}": float(topk_hit_mask(logits, targets, k).float().mean().item()) for k in topks
        },
        "mean_target_rank": float(target_ranks.float().mean().item()),
        "median_target_rank": float(target_ranks.float().median().item()),
        "mean_top1_margin": float(margins.mean().item()),
        "mean_correct_top1_margin": float(margins[hit_top1].mean().item()) if hit_top1.any() else 0.0,
        "mean_wrong_top1_margin": float(margins[~hit_top1].mean().item()) if (~hit_top1).any() else 0.0,
        "misses": misses,
    }


def summarize_union(
    *,
    names: list[str],
    logits_list: list[torch.Tensor],
    query_ids: list[str],
    candidate_ids: list[str],
    targets: torch.Tensor,
    topks: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in topks:
        union_hits = torch.zeros(len(query_ids), dtype=torch.bool)
        for logits in logits_list:
            union_hits |= topk_hit_mask(logits, targets, k)
        summary[f"union_top{k}"] = float(union_hits.float().mean().item())

    pairwise = []
    top1_hits = [(logits.argmax(dim=1) == targets) for logits in logits_list]
    for i in range(len(logits_list)):
        for j in range(i + 1, len(logits_list)):
            both = top1_hits[i] & top1_hits[j]
            either = top1_hits[i] | top1_hits[j]
            pairwise.append(
                {
                    "left": names[i],
                    "right": names[j],
                    "both_top1_correct": float(both.float().mean().item()),
                    "either_top1_correct": float(either.float().mean().item()),
                    "jaccard_correct_sets": float(both.sum().item() / max(1, either.sum().item())),
                }
            )
    return {"oracle_coverage": summary, "pairwise_top1_overlap": pairwise}


def main() -> None:
    args = parse_args()
    if args.names is not None and len(args.names) not in {0, len(args.logit_files)}:
        raise ValueError("--names must be omitted or match --logit-files length.")
    payloads = [load_payload(path) for path in args.logit_files]
    query_ids, candidate_ids = check_compatible(payloads, args.logit_files)
    targets = target_indices(query_ids, candidate_ids)
    names = [normalize_name(path, idx, args.names) for idx, path in enumerate(args.logit_files)]
    logits_list = [torch.as_tensor(payload["logits"]).float() for payload in payloads]
    topks = sorted({int(k) for k in args.topk if int(k) > 0})

    per_model = [
        summarize_model(
            name=name,
            path=path,
            logits=logits,
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            targets=targets,
            topks=topks,
            max_misses=args.max_misses,
        )
        for name, path, logits in zip(names, args.logit_files, logits_list, strict=True)
    ]
    payload = {
        "num_queries": len(query_ids),
        "num_candidates": len(candidate_ids),
        "models": per_model,
        "union": summarize_union(
            names=names,
            logits_list=logits_list,
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            targets=targets,
            topks=topks,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "retrieval_diagnostics.json")
    print(payload)


if __name__ == "__main__":
    main()
