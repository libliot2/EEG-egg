#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
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


class EpisodeSelector(nn.Module):
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


class EpisodeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        logits_list: list[torch.Tensor],
        *,
        num_episodes: int,
        episode_size: int,
        topk_per_model: int,
        shortlist_max_size: int,
        base_index: int,
        seed: int,
    ) -> None:
        self.features: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        rng = random.Random(seed)
        num_queries = int(logits_list[0].shape[0])
        all_indices = list(range(num_queries))

        for _ in range(num_episodes):
            query_idx = rng.randrange(num_queries)
            negatives = [idx for idx in rng.sample(all_indices, k=min(len(all_indices), episode_size + 16)) if idx != query_idx]
            candidates = [query_idx] + negatives[: max(0, episode_size - 1)]
            rng.shuffle(candidates)
            candidate_indices = torch.tensor(candidates, dtype=torch.long)
            target_position = int((candidate_indices == query_idx).nonzero(as_tuple=False)[0, 0].item())

            episode_logits = [logits[query_idx, candidate_indices].unsqueeze(0) for logits in logits_list]
            shortlist = build_shortlist(
                episode_logits,
                topk_per_model=topk_per_model,
                shortlist_max_size=shortlist_max_size,
            )[0]
            matches = (shortlist == target_position).nonzero(as_tuple=False)
            if len(matches) == 0:
                continue
            features, _, _ = build_features(
                episode_logits,
                shortlist.unsqueeze(0),
                topk_per_model=topk_per_model,
                base_index=base_index,
            )
            self.features.append(features[0])
            self.targets.append(matches[0, 0].long())

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "target": self.targets[index]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a retrieval reranker on sampled train episodes, select on validation, apply once to test."
    )
    parser.add_argument("--train-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--val-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--test-logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_episode_selector")
    parser.add_argument("--base-index", type=int, default=0)
    parser.add_argument("--episode-size", type=int, default=200)
    parser.add_argument("--num-train-episodes", type=int, default=50000)
    parser.add_argument("--topk-per-model", type=int, default=10)
    parser.add_argument("--shortlist-max-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
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


def check_compatible(payloads: list[dict[str, Any]], paths: list[Path]) -> tuple[list[str], list[str], str]:
    query_ids = list(payloads[0]["ordered_query_image_ids"])
    candidate_ids = list(payloads[0]["candidate_image_ids"])
    split = str(payloads[0].get("split", "unknown"))
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query id mismatch in {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate id mismatch in {path}")
    return query_ids, candidate_ids, split


def normalize_rows(logits: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)


def ranks_from_logits(logits: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def target_indices(query_ids: list[str], candidate_ids: list[str]) -> torch.Tensor:
    index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
    return torch.tensor([index[image_id] for image_id in query_ids], dtype=torch.long)


def build_shortlist(
    logits_list: list[torch.Tensor],
    *,
    topk_per_model: int,
    shortlist_max_size: int,
) -> torch.Tensor:
    rows = []
    for row_idx in range(logits_list[0].shape[0]):
        selected: list[int] = []
        seen: set[int] = set()
        for logits in logits_list:
            for idx in logits[row_idx].topk(k=min(topk_per_model, logits.shape[1])).indices.tolist():
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
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
    features: list[torch.Tensor] = []
    for logits, rank in zip(normalized, ranks, strict=True):
        scores = logits.gather(1, shortlist)
        rank_values = rank.gather(1, shortlist).float()
        features.extend(
            [
                scores,
                1.0 / (rank_values + 1.0),
                (rank_values == 0).float(),
                (rank_values < float(topk_per_model)).float(),
            ]
        )
    gathered_scores = torch.stack([logits.gather(1, shortlist) for logits in normalized], dim=-1)
    gathered_ranks = torch.stack([rank.gather(1, shortlist).float() for rank in ranks], dim=-1)
    features.extend(
        [
            gathered_scores.mean(dim=-1),
            gathered_scores.max(dim=-1).values,
            gathered_scores.std(dim=-1, unbiased=False),
            (gathered_ranks < float(topk_per_model)).float().mean(dim=-1),
            base_shortlist_scores,
        ]
    )
    return torch.stack(features, dim=-1).float(), base_shortlist_scores.float(), base_logits.float()


@torch.no_grad()
def score_features(model: EpisodeSelector, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        chunks.append(model(features[start : start + batch_size]).cpu())
    return torch.cat(chunks, dim=0)


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
            row_weight = row_weight.masked_fill(((top2[:, 0] - top2[:, 1]) >= margin_gate).unsqueeze(1), 0.0)
    order_scores = base_shortlist_scores + row_weight * selector_scores.cpu()
    sorted_base_scores = base_shortlist_scores.sort(dim=1, descending=True).values
    rerank_order = order_scores.argsort(dim=1, descending=True)
    reassigned = torch.zeros_like(sorted_base_scores).scatter(1, rerank_order, sorted_base_scores)
    output = base_logits.clone()
    output.scatter_(1, shortlist, reassigned)
    return output


def select_params(
    *,
    base_logits: torch.Tensor,
    shortlist: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    selector_scores: torch.Tensor,
    query_ids: list[str],
    candidate_ids: list[str],
    weight_grid: list[float],
    gate_grid: list[float],
) -> tuple[float, float, dict[str, float], list[dict[str, float]]]:
    best_weight = 0.0
    best_gate = 0.0
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    history = []
    for weight in weight_grid:
        for gate in gate_grid:
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
            score = (float(metrics["top1_acc"]), float(metrics["top5_acc"]), -abs(float(weight) - 0.5))
            if best_score is None or score > best_score:
                best_score = score
                best_weight = float(weight)
                best_gate = float(gate)
                best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_gate, best_metrics, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_payloads = [load_payload(path) for path in args.train_logit_files]
    val_payloads = [load_payload(path) for path in args.val_logit_files]
    test_payloads = [load_payload(path) for path in args.test_logit_files]
    train_query_ids, train_candidate_ids, train_split = check_compatible(train_payloads, args.train_logit_files)
    val_query_ids, val_candidate_ids, val_split = check_compatible(val_payloads, args.val_logit_files)
    test_query_ids, test_candidate_ids, test_split = check_compatible(test_payloads, args.test_logit_files)
    if len(train_payloads) != len(val_payloads) or len(train_payloads) != len(test_payloads):
        raise ValueError("train/val/test must have the same number of logit sources.")

    train_logits = [torch.as_tensor(payload["logits"]).float() for payload in train_payloads]
    val_logits = [torch.as_tensor(payload["logits"]).float() for payload in val_payloads]
    test_logits = [torch.as_tensor(payload["logits"]).float() for payload in test_payloads]

    train_dataset = EpisodeDataset(
        train_logits,
        num_episodes=args.num_train_episodes,
        episode_size=args.episode_size,
        topk_per_model=args.topk_per_model,
        shortlist_max_size=args.shortlist_max_size,
        base_index=args.base_index,
        seed=args.seed,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No train episodes contain the target in the shortlist.")

    val_shortlist = build_shortlist(val_logits, topk_per_model=args.topk_per_model, shortlist_max_size=args.shortlist_max_size)
    test_shortlist = build_shortlist(test_logits, topk_per_model=args.topk_per_model, shortlist_max_size=args.shortlist_max_size)
    val_features, val_base_shortlist_scores, val_base_logits = build_features(
        val_logits,
        val_shortlist,
        topk_per_model=args.topk_per_model,
        base_index=args.base_index,
    )
    test_features, test_base_shortlist_scores, test_base_logits = build_features(
        test_logits,
        test_shortlist,
        topk_per_model=args.topk_per_model,
        base_index=args.base_index,
    )

    model = EpisodeSelector(feature_dim=int(val_features.shape[-1]), hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float, float] | None = None
    best_weight = 0.0
    best_gate = 0.0
    best_val_metrics: dict[str, float] | None = None
    best_param_history: list[dict[str, float]] = []
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            scores = model(batch["features"])
            loss = F.cross_entropy(scores, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_scores = score_features(model, val_features, args.batch_size)
        selected_weight, selected_gate, val_metrics, param_history = select_params(
            base_logits=val_base_logits,
            shortlist=val_shortlist,
            base_shortlist_scores=val_base_shortlist_scores,
            selector_scores=val_scores,
            query_ids=val_query_ids,
            candidate_ids=val_candidate_ids,
            weight_grid=args.rerank_weight_grid,
            gate_grid=args.margin_gate_grid,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "selected_rerank_weight": float(selected_weight),
            "selected_margin_gate": float(selected_gate),
            "val_top1": float(val_metrics["top1_acc"]),
            "val_top5": float(val_metrics["top5_acc"]),
        }
        history.append(record)
        score = (float(val_metrics["top1_acc"]), float(val_metrics["top5_acc"]), -float(epoch))
        if best_score is None or score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_weight = float(selected_weight)
            best_gate = float(selected_gate)
            best_val_metrics = val_metrics
            best_param_history = param_history

    assert best_state is not None and best_val_metrics is not None
    model.load_state_dict(best_state)
    test_scores = score_features(model, test_features, args.batch_size)
    test_output_logits = apply_selector(
        base_logits=test_base_logits,
        shortlist=test_shortlist,
        base_shortlist_scores=test_base_shortlist_scores,
        selector_scores=test_scores,
        rerank_weight=best_weight,
        margin_gate=best_gate,
    )
    base_test_metrics = compute_retrieval_metrics(test_base_logits, ordered_image_ids=test_query_ids, candidate_image_ids=test_candidate_ids)
    test_metrics = compute_retrieval_metrics(test_output_logits, ordered_image_ids=test_query_ids, candidate_image_ids=test_candidate_ids)

    names = args.names if args.names else [path.stem for path in args.train_logit_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": test_split,
            "ordered_query_image_ids": test_query_ids,
            "candidate_image_ids": test_candidate_ids,
            "logits": test_output_logits,
            "base_logits": test_base_logits,
            "source_names": names,
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
        },
        args.output_dir / "retrieval_logits.pt",
    )
    torch.save(
        {
            "model_state": best_state,
            "config": vars(args),
            "source_names": names,
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
        },
        args.output_dir / "best.pt",
    )
    rankings = rank_candidate_ids(test_output_logits, test_candidate_ids)
    save_json(
        {
            "split": test_split,
            "predictions": [
                {"query_image_id": query_id, "ranked_candidate_ids": ranked_ids}
                for query_id, ranked_ids in zip(test_query_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )
    save_json(
        {
            "protocol": "train_episode_selector_val_selected_no_test_distribution",
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "source_names": names,
            "train_logit_files": [str(path) for path in args.train_logit_files],
            "val_logit_files": [str(path) for path in args.val_logit_files],
            "test_logit_files": [str(path) for path in args.test_logit_files],
            "num_requested_train_episodes": int(args.num_train_episodes),
            "num_usable_train_episodes": int(len(train_dataset)),
            "selected_rerank_weight": float(best_weight),
            "selected_margin_gate": float(best_gate),
            "best_val_metrics": {key: float(value) for key, value in best_val_metrics.items()},
            "test_base_metrics": {key: float(value) for key, value in base_test_metrics.items()},
            "test_selector_metrics": {key: float(value) for key, value in test_metrics.items()},
            "best_param_history": best_param_history,
            "history": history,
        },
        args.output_dir / "retrieval_metrics.json",
    )
    print(
        {
            "protocol": "train_episode_selector_val_selected_no_test_distribution",
            "usable_train_episodes": len(train_dataset),
            "best_val": best_val_metrics,
            "test_base": base_test_metrics,
            "test_selector": test_metrics,
            "selected_rerank_weight": best_weight,
            "selected_margin_gate": best_gate,
        }
    )


if __name__ == "__main__":
    main()
