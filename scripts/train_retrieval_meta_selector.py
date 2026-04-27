#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.utils import DEFAULT_OUTPUT_DIR, save_json, set_seed


class CandidateSelector(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ShortlistDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, features: torch.Tensor, target_positions: torch.Tensor) -> None:
        self.features = features
        self.target_positions = target_positions

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "target_positions": self.target_positions[index]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a validation-only retrieval meta-selector. The selector is trained on a split of validation "
            "queries, selected on held-out validation queries, then applied once to test logits."
        )
    )
    parser.add_argument("--val-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--test-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_meta_selector")
    parser.add_argument("--base-index", type=int, default=0, help="Logit source used as the fallback/base ranking.")
    parser.add_argument("--topk-per-model", type=int, default=10)
    parser.add_argument("--shortlist-max-size", type=int, default=40)
    parser.add_argument("--meta-train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rerank-weight-grid", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--margin-gate-grid", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.4])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def check_compatibility(payloads: list[dict[str, Any]], paths: list[Path]) -> tuple[list[str], list[str], str]:
    query_ids = list(payloads[0]["ordered_query_image_ids"])
    candidate_ids = list(payloads[0]["candidate_image_ids"])
    split = str(payloads[0].get("split", "unknown"))
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query id ordering mismatch in {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate id ordering mismatch in {path}")
    return query_ids, candidate_ids, split


def normalize_rows(logits: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)


def target_indices(query_ids: list[str], candidate_ids: list[str]) -> torch.Tensor:
    index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
    missing = [image_id for image_id in query_ids if image_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} query ids are missing from candidates; first={missing[0]}")
    return torch.tensor([index[image_id] for image_id in query_ids], dtype=torch.long)


def ranks_from_logits(logits: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def build_shortlist(
    logits_list: list[torch.Tensor],
    *,
    topk_per_model: int,
    shortlist_max_size: int,
) -> torch.Tensor:
    num_queries, num_candidates = logits_list[0].shape
    rows: list[torch.Tensor] = []
    for row_idx in range(num_queries):
        selected: list[int] = []
        seen: set[int] = set()
        for logits in logits_list:
            topk = min(topk_per_model, num_candidates)
            for candidate_idx in logits[row_idx].topk(k=topk).indices.tolist():
                if candidate_idx not in seen:
                    selected.append(candidate_idx)
                    seen.add(candidate_idx)
                if len(selected) >= shortlist_max_size:
                    break
            if len(selected) >= shortlist_max_size:
                break
        if not selected:
            selected = [int(logits_list[0][row_idx].argmax().item())]
        selected = selected[:shortlist_max_size]
        while len(selected) < shortlist_max_size:
            selected.append(selected[-1])
        rows.append(torch.tensor(selected, dtype=torch.long))
    return torch.stack(rows, dim=0)


def target_positions(shortlist: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.full((shortlist.shape[0],), -1, dtype=torch.long)
    valid = torch.zeros(shortlist.shape[0], dtype=torch.bool)
    for row_idx in range(shortlist.shape[0]):
        matches = (shortlist[row_idx] == targets[row_idx]).nonzero(as_tuple=False)
        if len(matches):
            positions[row_idx] = int(matches[0, 0].item())
            valid[row_idx] = True
    return positions, valid


def build_features(
    logits_list: list[torch.Tensor],
    shortlist: torch.Tensor,
    *,
    topk_per_model: int,
    base_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = [normalize_rows(logits) for logits in logits_list]
    ranks = [ranks_from_logits(logits) for logits in normalized]
    base_logits = normalized[base_index]
    base_shortlist_scores = base_logits.gather(1, shortlist)

    per_model_features: list[torch.Tensor] = []
    for logits, ranks_for_model in zip(normalized, ranks, strict=True):
        scores = logits.gather(1, shortlist)
        rank_values = ranks_for_model.gather(1, shortlist).float()
        reciprocal_rank = 1.0 / (rank_values + 1.0)
        is_top1 = (rank_values == 0).float()
        in_model_topk = (rank_values < float(topk_per_model)).float()
        top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1).expand_as(scores)
        per_model_features.extend([scores, reciprocal_rank, is_top1, in_model_topk, margin])

    gathered_scores = torch.stack([logits.gather(1, shortlist) for logits in normalized], dim=-1)
    gathered_ranks = torch.stack([rank.gather(1, shortlist).float() for rank in ranks], dim=-1)
    aggregate_features = [
        gathered_scores.mean(dim=-1),
        gathered_scores.max(dim=-1).values,
        gathered_scores.std(dim=-1, unbiased=False),
        (gathered_ranks < float(topk_per_model)).float().mean(dim=-1),
        base_shortlist_scores,
        1.0 / (gathered_ranks.min(dim=-1).values + 1.0),
    ]
    features = torch.stack(per_model_features + aggregate_features, dim=-1)
    return features.float().detach(), base_shortlist_scores.float().detach(), base_logits.float().detach()


@torch.no_grad()
def score_features(model: CandidateSelector, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    scores: list[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_size):
        scores.append(model(features[start : start + batch_size]).cpu())
    return torch.cat(scores, dim=0)


def apply_selector(
    *,
    base_logits: torch.Tensor,
    shortlist: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    rerank_weight: float,
    margin_gate: float,
) -> torch.Tensor:
    row_weight = torch.full((base_logits.shape[0], 1), float(rerank_weight), dtype=base_logits.dtype)
    if margin_gate > 0.0:
        top2 = base_logits.topk(k=min(2, base_logits.shape[1]), dim=1).values
        if top2.shape[1] == 2:
            margins = top2[:, 0] - top2[:, 1]
            row_weight = row_weight.masked_fill((margins >= margin_gate).unsqueeze(1), 0.0)

    order_scores = base_shortlist_scores + row_weight * selector_scores.cpu()
    sorted_base_scores = base_shortlist_scores.sort(dim=1, descending=True).values
    rerank_order = order_scores.argsort(dim=1, descending=True)
    reassigned_scores = torch.zeros_like(sorted_base_scores).scatter(1, rerank_order, sorted_base_scores)
    output = base_logits.clone()
    output.scatter_(1, shortlist, reassigned_scores)
    return output


def select_on_meta_val(
    *,
    base_logits: torch.Tensor,
    shortlist: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    query_ids: list[str],
    candidate_ids: list[str],
    rerank_weight_grid: list[float],
    margin_gate_grid: list[float],
) -> tuple[float, float, dict[str, float], list[dict[str, float]]]:
    best_weight = float(rerank_weight_grid[0])
    best_gate = float(margin_gate_grid[0])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float] | None = None
    history: list[dict[str, float]] = []
    for weight in rerank_weight_grid:
        for gate in margin_gate_grid:
            logits = apply_selector(
                base_logits=base_logits,
                shortlist=shortlist,
                base_shortlist_scores=base_shortlist_scores,
                selector_scores=selector_scores,
                rerank_weight=float(weight),
                margin_gate=float(gate),
            )
            metrics = compute_retrieval_metrics(logits, ordered_image_ids=query_ids, candidate_image_ids=candidate_ids)
            record = {"rerank_weight": float(weight), "margin_gate": float(gate), **metrics}
            history.append(record)
            score = (
                float(metrics["top1_acc"]),
                float(metrics["top5_acc"]),
                -abs(float(weight) - 0.5),
                -float(gate),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_weight = float(weight)
                best_gate = float(gate)
                best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_gate, best_metrics, history


def subset_list(values: list[str], indices: torch.Tensor) -> list[str]:
    return [values[int(index)] for index in indices.tolist()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if len(args.val_logit_files) != len(args.test_logit_files):
        raise ValueError("--val-logit-files and --test-logit-files must have the same length.")
    if args.names is not None and len(args.names) not in (0, len(args.val_logit_files)):
        raise ValueError("--names must be omitted or have the same length as --val-logit-files.")
    if not (0 <= args.base_index < len(args.val_logit_files)):
        raise ValueError("--base-index is out of range.")
    if not (0.0 < args.meta_train_fraction < 1.0):
        raise ValueError("--meta-train-fraction must be in (0, 1).")

    val_payloads = [load_payload(path) for path in args.val_logit_files]
    test_payloads = [load_payload(path) for path in args.test_logit_files]
    val_query_ids, val_candidate_ids, val_split = check_compatibility(val_payloads, args.val_logit_files)
    test_query_ids, test_candidate_ids, test_split = check_compatibility(test_payloads, args.test_logit_files)

    val_logits_list = [torch.as_tensor(payload["logits"]).float() for payload in val_payloads]
    test_logits_list = [torch.as_tensor(payload["logits"]).float() for payload in test_payloads]
    val_targets = target_indices(val_query_ids, val_candidate_ids)
    test_targets = target_indices(test_query_ids, test_candidate_ids)

    val_shortlist = build_shortlist(
        val_logits_list,
        topk_per_model=args.topk_per_model,
        shortlist_max_size=args.shortlist_max_size,
    )
    test_shortlist = build_shortlist(
        test_logits_list,
        topk_per_model=args.topk_per_model,
        shortlist_max_size=args.shortlist_max_size,
    )
    val_positions, val_valid = target_positions(val_shortlist, val_targets)
    test_positions, test_valid = target_positions(test_shortlist, test_targets)

    val_features, val_base_shortlist_scores, val_base_logits = build_features(
        val_logits_list,
        val_shortlist,
        topk_per_model=args.topk_per_model,
        base_index=args.base_index,
    )
    test_features, test_base_shortlist_scores, test_base_logits = build_features(
        test_logits_list,
        test_shortlist,
        topk_per_model=args.topk_per_model,
        base_index=args.base_index,
    )

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(val_query_ids), generator=generator)
    split_at = int(round(len(permutation) * args.meta_train_fraction))
    meta_train_indices = permutation[:split_at]
    meta_val_indices = permutation[split_at:]

    train_valid_indices = meta_train_indices[val_valid[meta_train_indices]]
    if len(train_valid_indices) == 0:
        raise RuntimeError("No meta-train query has the true target in the union shortlist.")

    train_dataset = ShortlistDataset(val_features[train_valid_indices], val_positions[train_valid_indices])
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = CandidateSelector(
        feature_dim=int(val_features.shape[-1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    meta_val_query_ids = subset_list(val_query_ids, meta_val_indices)
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float, float] | None = None
    best_weight = 0.0
    best_gate = 0.0
    best_meta_val_metrics: dict[str, float] | None = None
    best_weight_history: list[dict[str, float]] = []
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            scores = model(batch["features"])
            loss = F.cross_entropy(scores, batch["target_positions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        meta_val_scores = score_features(model, val_features[meta_val_indices], args.batch_size)
        selected_weight, selected_gate, meta_val_metrics, weight_history = select_on_meta_val(
            base_logits=val_base_logits[meta_val_indices],
            shortlist=val_shortlist[meta_val_indices],
            base_shortlist_scores=val_base_shortlist_scores[meta_val_indices],
            selector_scores=meta_val_scores,
            query_ids=meta_val_query_ids,
            candidate_ids=val_candidate_ids,
            rerank_weight_grid=args.rerank_weight_grid,
            margin_gate_grid=args.margin_gate_grid,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "selected_weight": float(selected_weight),
            "selected_margin_gate": float(selected_gate),
            "meta_val_top1": float(meta_val_metrics["top1_acc"]),
            "meta_val_top5": float(meta_val_metrics["top5_acc"]),
        }
        history.append(record)
        score = (
            float(meta_val_metrics["top1_acc"]),
            float(meta_val_metrics["top5_acc"]),
            -float(epoch),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_weight = float(selected_weight)
            best_gate = float(selected_gate)
            best_meta_val_metrics = meta_val_metrics
            best_weight_history = weight_history

    assert best_state is not None and best_meta_val_metrics is not None
    model.load_state_dict(best_state)
    test_selector_scores = score_features(model, test_features, args.batch_size)
    test_logits = apply_selector(
        base_logits=test_base_logits,
        shortlist=test_shortlist,
        base_shortlist_scores=test_base_shortlist_scores,
        selector_scores=test_selector_scores,
        rerank_weight=best_weight,
        margin_gate=best_gate,
    )
    base_val_metrics = compute_retrieval_metrics(
        val_base_logits[meta_val_indices],
        ordered_image_ids=meta_val_query_ids,
        candidate_image_ids=val_candidate_ids,
    )
    base_test_metrics = compute_retrieval_metrics(
        test_base_logits,
        ordered_image_ids=test_query_ids,
        candidate_image_ids=test_candidate_ids,
    )
    test_metrics = compute_retrieval_metrics(
        test_logits,
        ordered_image_ids=test_query_ids,
        candidate_image_ids=test_candidate_ids,
    )

    names = args.names if args.names else [path.stem for path in args.val_logit_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": test_split,
            "ordered_query_image_ids": test_query_ids,
            "candidate_image_ids": test_candidate_ids,
            "logits": test_logits,
            "base_logits": test_base_logits,
            "source_names": names,
            "val_logit_files": [str(path) for path in args.val_logit_files],
            "test_logit_files": [str(path) for path in args.test_logit_files],
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
            "topk_per_model": int(args.topk_per_model),
            "shortlist_max_size": int(args.shortlist_max_size),
        },
        args.output_dir / "retrieval_logits.pt",
    )
    torch.save(
        {
            "model_state": best_state,
            "config": vars(args),
            "feature_dim": int(val_features.shape[-1]),
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
            "source_names": names,
        },
        args.output_dir / "best.pt",
    )
    rankings = rank_candidate_ids(test_logits, test_candidate_ids)
    save_json(
        {
            "split": test_split,
            "source_names": names,
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
            "predictions": [
                {"query_image_id": query_id, "ranked_candidate_ids": ranked_ids}
                for query_id, ranked_ids in zip(test_query_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )
    metrics_payload = {
        "protocol": "validation_only_meta_selector_no_test_distribution",
        "val_split": val_split,
        "test_split": test_split,
        "source_names": names,
        "val_logit_files": [str(path) for path in args.val_logit_files],
        "test_logit_files": [str(path) for path in args.test_logit_files],
        "base_index": int(args.base_index),
        "base_name": names[args.base_index],
        "topk_per_model": int(args.topk_per_model),
        "shortlist_max_size": int(args.shortlist_max_size),
        "meta_train_size": int(len(meta_train_indices)),
        "meta_train_usable_size": int(len(train_valid_indices)),
        "meta_val_size": int(len(meta_val_indices)),
        "meta_train_shortlist_target_coverage": float(val_valid[meta_train_indices].float().mean().item()),
        "meta_val_shortlist_target_coverage": float(val_valid[meta_val_indices].float().mean().item()),
        "test_shortlist_target_coverage_diagnostic": float(test_valid.float().mean().item()),
        "selected_rerank_weight": float(best_weight),
        "selected_margin_gate": float(best_gate),
        "meta_val_base_metrics": {key: float(value) for key, value in base_val_metrics.items()},
        "meta_val_selector_metrics": {key: float(value) for key, value in best_meta_val_metrics.items()},
        "test_base_metrics": {key: float(value) for key, value in base_test_metrics.items()},
        "test_selector_metrics": {key: float(value) for key, value in test_metrics.items()},
        "best_weight_history": best_weight_history,
        "history": history,
    }
    save_json(metrics_payload, args.output_dir / "retrieval_metrics.json")
    print(
        {
            "protocol": metrics_payload["protocol"],
            "base_name": metrics_payload["base_name"],
            "meta_val_base": base_val_metrics,
            "meta_val_selector": best_meta_val_metrics,
            "test_base": base_test_metrics,
            "test_selector": test_metrics,
            "selected_rerank_weight": best_weight,
            "selected_margin_gate": best_gate,
            "meta_val_shortlist_target_coverage": metrics_payload["meta_val_shortlist_target_coverage"],
            "test_shortlist_target_coverage_diagnostic": metrics_payload["test_shortlist_target_coverage_diagnostic"],
        }
    )


if __name__ == "__main__":
    main()
