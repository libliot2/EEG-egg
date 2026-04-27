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


class LogitReranker(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
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


class ShortlistFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        features: torch.Tensor,
        shortlist_indices: torch.Tensor,
        target_positions: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> None:
        self.features = features
        self.shortlist_indices = shortlist_indices
        self.target_positions = target_positions
        self.base_scores = base_scores

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "shortlist_indices": self.shortlist_indices[index],
            "target_positions": self.target_positions[index],
            "base_scores": self.base_scores[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a constrained union-shortlist reranker from retrieval logits.")
    parser.add_argument("--train-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_logit_reranker")
    parser.add_argument("--topk-per-model", type=int, default=5)
    parser.add_argument("--shortlist-max-size", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin-gate-threshold", type=float, default=0.0)
    parser.add_argument("--weight-grid", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
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
            raise ValueError(f"Query id ordering mismatch: {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate id ordering mismatch: {path}")
    return query_ids, candidate_ids, split


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (values - mean) / std


def target_indices(query_ids: list[str], candidate_ids: list[str]) -> torch.Tensor:
    index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
    missing = [image_id for image_id in query_ids if image_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} query ids are missing from candidates; first={missing[0]}")
    return torch.tensor([index[image_id] for image_id in query_ids], dtype=torch.long)


def build_union_shortlist(
    logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    *,
    topk_per_model: int,
    shortlist_max_size: int,
    ensure_positive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries = logits_list[0].shape[0]
    rows: list[torch.Tensor] = []
    target_positions = torch.zeros(num_queries, dtype=torch.long)
    for row_idx in range(num_queries):
        selected: list[int] = []
        seen: set[int] = set()
        for logits in logits_list:
            for candidate_idx in logits[row_idx].topk(k=min(topk_per_model, logits.shape[1])).indices.tolist():
                if candidate_idx not in seen:
                    seen.add(candidate_idx)
                    selected.append(candidate_idx)
        if ensure_positive:
            target_idx = int(targets[row_idx].item())
            if target_idx not in seen:
                selected.append(target_idx)
        if len(selected) > shortlist_max_size:
            target_idx = int(targets[row_idx].item())
            selected = selected[:shortlist_max_size]
            if ensure_positive and target_idx not in selected:
                selected[-1] = target_idx
        while len(selected) < shortlist_max_size:
            selected.append(selected[-1])
        shortlist = torch.tensor(selected, dtype=torch.long)
        matches = (shortlist == targets[row_idx]).nonzero(as_tuple=False)
        target_positions[row_idx] = int(matches[0, 0].item()) if len(matches) else 0
        rows.append(shortlist)
    return torch.stack(rows, dim=0), target_positions


def ranks_from_logits(logits: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def build_features(
    logits_list: list[torch.Tensor],
    shortlist_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = [normalize_logits(logits) for logits in logits_list]
    ranks = [ranks_from_logits(logits) for logits in normalized]
    base_scores = torch.stack(normalized, dim=0).mean(dim=0)
    shortlist_base_scores = base_scores.gather(1, shortlist_indices)
    model_features: list[torch.Tensor] = []

    for logits, rank in zip(normalized, ranks, strict=True):
        gathered_scores = logits.gather(1, shortlist_indices)
        gathered_ranks = rank.gather(1, shortlist_indices).float()
        reciprocal_rank = 1.0 / (gathered_ranks + 1.0)
        is_top1 = (gathered_ranks == 0).float()
        top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1).expand_as(gathered_scores)
        model_features.extend([gathered_scores, reciprocal_rank, is_top1, margin])

    stacked_scores = torch.stack([logits.gather(1, shortlist_indices) for logits in normalized], dim=-1)
    occurrence = torch.stack(
        [(rank.gather(1, shortlist_indices) < 5).float() for rank in ranks],
        dim=-1,
    ).mean(dim=-1)
    aggregate_features = [
        stacked_scores.mean(dim=-1),
        stacked_scores.max(dim=-1).values,
        stacked_scores.std(dim=-1, unbiased=False),
        occurrence,
        shortlist_base_scores,
    ]
    features = torch.stack(model_features + aggregate_features, dim=-1)
    return features.float().detach(), shortlist_base_scores.float().detach(), base_scores.float().detach()


def apply_rerank(
    *,
    base_logits: torch.Tensor,
    shortlist_indices: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    rerank_scores: torch.Tensor,
    weight: float,
    margin_gate_threshold: float,
) -> torch.Tensor:
    row_weight = torch.full((base_logits.shape[0], 1), float(weight), dtype=base_logits.dtype)
    if margin_gate_threshold > 0.0:
        top2 = base_logits.topk(k=min(2, base_logits.shape[1]), dim=1).values
        if top2.shape[1] == 2:
            margins = top2[:, 0] - top2[:, 1]
            row_weight = row_weight.masked_fill((margins >= margin_gate_threshold).unsqueeze(1), 0.0)
    order_scores = base_shortlist_scores + row_weight * rerank_scores.cpu()
    reranked_logits = base_logits.clone()
    sorted_base_scores = base_shortlist_scores.sort(dim=1, descending=True).values
    rerank_order = order_scores.argsort(dim=1, descending=True)
    reassigned_scores = torch.zeros_like(sorted_base_scores).scatter(1, rerank_order, sorted_base_scores)
    reranked_logits.scatter_(1, shortlist_indices, reassigned_scores)
    return reranked_logits


@torch.no_grad()
def score_features(model: LogitReranker, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    scores = []
    for start in range(0, features.shape[0], batch_size):
        scores.append(model(features[start : start + batch_size]).cpu())
    return torch.cat(scores, dim=0)


def select_weight(
    *,
    base_logits: torch.Tensor,
    shortlist_indices: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    rerank_scores: torch.Tensor,
    query_ids: list[str],
    candidate_ids: list[str],
    weight_grid: list[float],
    margin_gate_threshold: float,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    best_weight = float(weight_grid[0])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    history = []
    for weight in weight_grid:
        logits = apply_rerank(
            base_logits=base_logits,
            shortlist_indices=shortlist_indices,
            base_shortlist_scores=base_shortlist_scores,
            rerank_scores=rerank_scores,
            weight=float(weight),
            margin_gate_threshold=margin_gate_threshold,
        )
        metrics = compute_retrieval_metrics(logits, ordered_image_ids=query_ids, candidate_image_ids=candidate_ids)
        record = {"weight": float(weight), **metrics}
        history.append(record)
        score = (float(metrics["top1_acc"]), float(metrics["top5_acc"]), -abs(float(weight) - 1.0))
        if best_score is None or score > best_score:
            best_score = score
            best_weight = float(weight)
            best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_metrics, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_payloads = [load_payload(path) for path in args.train_logit_files]
    eval_payloads = [load_payload(path) for path in args.eval_logit_files]
    train_query_ids, train_candidate_ids, train_split = check_compatibility(train_payloads, args.train_logit_files)
    eval_query_ids, eval_candidate_ids, eval_split = check_compatibility(eval_payloads, args.eval_logit_files)
    if len(train_payloads) != len(eval_payloads):
        raise ValueError("--train-logit-files and --eval-logit-files must have the same length.")

    train_logits_list = [torch.as_tensor(payload["logits"]).float() for payload in train_payloads]
    eval_logits_list = [torch.as_tensor(payload["logits"]).float() for payload in eval_payloads]
    train_targets = target_indices(train_query_ids, train_candidate_ids)
    eval_targets = target_indices(eval_query_ids, eval_candidate_ids)

    train_shortlist, train_positions = build_union_shortlist(
        train_logits_list,
        train_targets,
        topk_per_model=args.topk_per_model,
        shortlist_max_size=args.shortlist_max_size,
        ensure_positive=True,
    )
    eval_shortlist, _ = build_union_shortlist(
        eval_logits_list,
        eval_targets,
        topk_per_model=args.topk_per_model,
        shortlist_max_size=args.shortlist_max_size,
        ensure_positive=False,
    )
    train_features, train_base_shortlist_scores, train_base_logits = build_features(train_logits_list, train_shortlist)
    eval_features, eval_base_shortlist_scores, eval_base_logits = build_features(eval_logits_list, eval_shortlist)

    model = LogitReranker(
        feature_dim=int(train_features.shape[-1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        ShortlistFeatureDataset(train_features, train_shortlist, train_positions, train_base_shortlist_scores),
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_state = None
    best_score: tuple[float, float] | None = None
    best_weight = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            scores = model(batch["features"])
            loss = F.cross_entropy(scores, batch["target_positions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_scores = score_features(model, train_features, args.batch_size)
        selected_weight, train_metrics, weight_history = select_weight(
            base_logits=train_base_logits,
            shortlist_indices=train_shortlist,
            base_shortlist_scores=train_base_shortlist_scores,
            rerank_scores=train_scores,
            query_ids=train_query_ids,
            candidate_ids=train_candidate_ids,
            weight_grid=args.weight_grid,
            margin_gate_threshold=args.margin_gate_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "selected_weight": float(selected_weight),
            "train_top1": float(train_metrics["top1_acc"]),
            "train_top5": float(train_metrics["top5_acc"]),
            "weight_history": weight_history,
        }
        history.append(epoch_record)
        score = (float(train_metrics["top1_acc"]), float(train_metrics["top5_acc"]))
        if best_score is None or score > best_score:
            best_score = score
            best_weight = float(selected_weight)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    eval_scores = score_features(model, eval_features, args.batch_size)
    eval_logits = apply_rerank(
        base_logits=eval_base_logits,
        shortlist_indices=eval_shortlist,
        base_shortlist_scores=eval_base_shortlist_scores,
        rerank_scores=eval_scores,
        weight=best_weight,
        margin_gate_threshold=args.margin_gate_threshold,
    )
    base_metrics = compute_retrieval_metrics(
        eval_base_logits,
        ordered_image_ids=eval_query_ids,
        candidate_image_ids=eval_candidate_ids,
    )
    rerank_metrics = compute_retrieval_metrics(
        eval_logits,
        ordered_image_ids=eval_query_ids,
        candidate_image_ids=eval_candidate_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": vars(args),
            "feature_dim": int(train_features.shape[-1]),
            "selected_weight": float(best_weight),
        },
        args.output_dir / "best.pt",
    )
    torch.save(
        {
            "split": eval_split,
            "ordered_query_image_ids": eval_query_ids,
            "candidate_image_ids": eval_candidate_ids,
            "logits": eval_logits,
            "base_logits": eval_base_logits,
            "source_logit_files": [str(path) for path in args.eval_logit_files],
            "selected_weight": float(best_weight),
        },
        args.output_dir / "retrieval_logits.pt",
    )
    save_json(
        {
            "train_split": train_split,
            "eval_split": eval_split,
            "train_logit_files": [str(path) for path in args.train_logit_files],
            "eval_logit_files": [str(path) for path in args.eval_logit_files],
            "selected_weight": float(best_weight),
            "base_metrics": {key: float(value) for key, value in base_metrics.items()},
            "rerank_metrics": {key: float(value) for key, value in rerank_metrics.items()},
            "history": history,
        },
        args.output_dir / "retrieval_metrics.json",
    )
    rankings = rank_candidate_ids(eval_logits, eval_candidate_ids)
    save_json(
        {
            "split": eval_split,
            "source_logit_files": [str(path) for path in args.eval_logit_files],
            "selected_weight": float(best_weight),
            "predictions": [
                {"query_image_id": image_id, "ranked_candidate_ids": ranked_ids}
                for image_id, ranked_ids in zip(eval_query_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )
    print({"base": base_metrics, "rerank": rerank_metrics, "selected_weight": best_weight})


if __name__ == "__main__":
    main()
