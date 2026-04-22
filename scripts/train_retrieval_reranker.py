#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split
from project1_eeg.image_banks import TensorBank
from project1_eeg.reranker import (
    TopKRetrievalReranker,
    build_shortlist_indices,
    reranker_loss,
    select_rerank_weight,
)
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    format_seconds,
    load_json,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


class ShortlistDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        query_embeddings: torch.Tensor,
        base_shortlist_scores: torch.Tensor,
        shortlist_indices: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> None:
        self.query_embeddings = query_embeddings
        self.base_shortlist_scores = base_shortlist_scores
        self.shortlist_indices = shortlist_indices
        self.target_positions = target_positions

    def __len__(self) -> int:
        return int(self.query_embeddings.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "query_embedding": self.query_embeddings[index],
            "base_shortlist_score": self.base_shortlist_scores[index],
            "shortlist_indices": self.shortlist_indices[index],
            "target_position": self.target_positions[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a second-stage reranker on top of a frozen retrieval model.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "experiments" / "retrieval_reranker")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["constant", "warmup_cosine"], default="constant")
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--min-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--scorer-type",
        choices=["cosine", "mlp_pairwise", "contextual_transformer", "listwise_transformer"],
        default="cosine",
    )
    parser.add_argument("--adapter-hidden-dim", type=int, default=512)
    parser.add_argument("--score-hidden-dim", type=int, default=None)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-beta", type=float, default=0.1)
    parser.add_argument("--share-adapters", action="store_true")
    parser.add_argument("--use-top1-head", action="store_true")
    parser.add_argument("--top1-head-hidden-dim", type=int, default=None)
    parser.add_argument("--top1-score-coef", type=float, default=0.25)
    parser.add_argument("--shortlist-topk", type=int, default=5)
    parser.add_argument("--ce-loss-coef", type=float, default=1.0)
    parser.add_argument("--margin-loss-coef", type=float, default=0.1)
    parser.add_argument("--pairwise-loss-coef", type=float, default=0.0)
    parser.add_argument("--focal-loss-coef", type=float, default=0.0)
    parser.add_argument("--hard-negative-logistic-loss-coef", type=float, default=0.0)
    parser.add_argument("--top1-ce-loss-coef", type=float, default=0.0)
    parser.add_argument("--top1-focal-loss-coef", type=float, default=0.0)
    parser.add_argument("--top1-hard-negative-loss-coef", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pairwise-rank-weight-power", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    return parser.parse_args()


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if scheduler_name == "constant":
        return None
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    min_scale = float(min_learning_rate_scale)
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def init_ema_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return clone_state_dict(module)


@torch.no_grad()
def update_ema_state(
    module: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    *,
    decay: float,
) -> None:
    param_names = {name for name, _ in module.named_parameters()}
    current_state = module.state_dict()
    for key, value in current_state.items():
        target = ema_state[key]
        if key in param_names and torch.is_floating_point(value):
            target.mul_(float(decay)).add_(value.detach(), alpha=1.0 - float(decay))
        else:
            target.copy_(value.detach())


def resolve_eval_state(
    module: torch.nn.Module,
    ema_state: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if ema_state is None:
        return clone_state_dict(module)
    current_state = module.state_dict()
    resolved: dict[str, torch.Tensor] = {}
    for key, value in current_state.items():
        if key in ema_state:
            resolved[key] = ema_state[key].detach().clone()
        else:
            resolved[key] = value.detach().clone()
    return resolved


def load_split(
    *,
    base_checkpoint: Path,
    data_dir: Path,
    fallback_seed: int,
    fallback_val_ratio: float,
) -> tuple[list[str], list[str]]:
    split_path = base_checkpoint.parent / "split.json"
    if split_path.exists():
        payload = load_json(split_path)
        return [str(item) for item in payload["train_ids"]], [str(item) for item in payload["val_ids"]]
    return make_train_val_split(data_dir=data_dir, val_ratio=fallback_val_ratio, seed=fallback_seed)


@torch.no_grad()
def encode_queries(
    base_model,
    records,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, list[str]]:
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    outputs, image_ids, _ = compute_retrieval_outputs(base_model, loader, device)
    if "perceptual" not in outputs:
        raise ValueError("Base checkpoint does not expose a perceptual retrieval head.")
    return outputs["perceptual"].float().cpu(), image_ids


@torch.no_grad()
def compute_base_logits(
    base_model,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    candidate_embeddings = candidate_embeddings.to(device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings), batch_size):
        query_chunk = query_embeddings[start : start + batch_size].to(device)
        logits = base_model.similarity(query_chunk, candidate_embeddings, head="perceptual").cpu()
        chunks.append(logits)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def compute_shortlist_rerank_scores(
    reranker: TopKRetrievalReranker,
    query_embeddings: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    shortlist_indices: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    candidate_embeddings = candidate_embeddings.float()
    scores: list[torch.Tensor] = []
    for start in range(0, len(query_embeddings), batch_size):
        query_chunk = query_embeddings[start : start + batch_size].to(device)
        base_score_chunk = base_shortlist_scores[start : start + batch_size].to(device)
        shortlist_chunk = shortlist_indices[start : start + batch_size]
        candidate_chunk = candidate_embeddings[shortlist_chunk].to(device)
        scores.append(
            reranker.score_shortlist(
                query_chunk,
                candidate_chunk,
                base_shortlist_scores=base_score_chunk,
            ).cpu()
        )
    return torch.cat(scores, dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    base_config = dict(base_payload["config"])
    base_model = build_retrieval_model_from_config(base_config).to(device)
    base_model.load_state_dict(base_payload["model_state"], strict=True)
    base_model.eval().requires_grad_(False)

    perceptual_bank_path = args.perceptual_bank
    if perceptual_bank_path is None:
        fallback = base_config.get("perceptual_bank")
        if fallback is None:
            raise ValueError("--perceptual-bank is required when the base checkpoint config does not define one.")
        perceptual_bank_path = Path(str(fallback))
    perceptual_bank = TensorBank.load(perceptual_bank_path)

    train_ids, val_ids = load_split(
        base_checkpoint=args.base_checkpoint,
        data_dir=args.data_dir,
        fallback_seed=int(base_config.get("seed", args.seed)),
        fallback_val_ratio=float(base_config.get("val_ratio", 0.1)),
    )
    selected_channels = base_config.get("selected_channels")
    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        selected_channels=selected_channels,
        image_ids=train_ids,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        selected_channels=selected_channels,
        image_ids=val_ids,
    )

    train_query_embeddings, resolved_train_ids = encode_queries(
        base_model,
        train_records,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_query_embeddings, resolved_val_ids = encode_queries(
        base_model,
        val_records,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_bank = perceptual_bank.subset(train_ids)
    val_bank = perceptual_bank.subset(val_ids)

    train_base_logits = compute_base_logits(
        base_model,
        train_query_embeddings,
        train_bank.values.float(),
        device=device,
    )
    val_base_logits = compute_base_logits(
        base_model,
        val_query_embeddings,
        val_bank.values.float(),
        device=device,
    )
    train_shortlist_indices, train_target_positions = build_shortlist_indices(
        train_base_logits,
        candidate_image_ids=train_ids,
        query_image_ids=resolved_train_ids,
        shortlist_topk=args.shortlist_topk,
        ensure_positive=True,
    )
    val_shortlist_indices, _ = build_shortlist_indices(
        val_base_logits,
        candidate_image_ids=val_ids,
        shortlist_topk=args.shortlist_topk,
        ensure_positive=False,
    )
    assert train_target_positions is not None

    train_base_shortlist_scores = train_base_logits.gather(1, train_shortlist_indices)
    val_base_shortlist_scores = val_base_logits.gather(1, val_shortlist_indices)
    dataset = ShortlistDataset(
        train_query_embeddings,
        train_base_shortlist_scores,
        train_shortlist_indices,
        train_target_positions,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    reranker = TopKRetrievalReranker(
        int(train_query_embeddings.shape[1]),
        scorer_type=args.scorer_type,
        hidden_dim=args.adapter_hidden_dim,
        score_hidden_dim=args.score_hidden_dim,
        dropout=args.adapter_dropout,
        beta=args.adapter_beta,
        share_adapters=args.share_adapters,
        use_top1_head=bool(args.use_top1_head),
        top1_head_hidden_dim=args.top1_head_hidden_dim,
        top1_score_coef=args.top1_score_coef,
    ).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = int(args.epochs) * max(1, len(train_loader))
    warmup_steps = int(float(args.warmup_epochs) * max(1, len(train_loader)))
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_learning_rate_scale=args.min_learning_rate_scale,
    )
    ema_state = None
    if float(args.ema_decay) > 0.0:
        ema_state = init_ema_state(reranker)

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "retrieval_reranker",
        "base_checkpoint": str(args.base_checkpoint),
        "perceptual_bank": str(perceptual_bank_path),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": float(args.warmup_epochs),
        "min_learning_rate_scale": float(args.min_learning_rate_scale),
        "ema_decay": float(args.ema_decay),
        "scorer_type": args.scorer_type,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "score_hidden_dim": args.score_hidden_dim,
        "adapter_dropout": args.adapter_dropout,
        "adapter_beta": args.adapter_beta,
        "share_adapters": bool(args.share_adapters),
        "use_top1_head": bool(args.use_top1_head),
        "top1_head_hidden_dim": args.top1_head_hidden_dim,
        "top1_score_coef": float(args.top1_score_coef),
        "shortlist_topk": int(args.shortlist_topk),
        "ce_loss_coef": float(args.ce_loss_coef),
        "margin_loss_coef": float(args.margin_loss_coef),
        "pairwise_loss_coef": float(args.pairwise_loss_coef),
        "focal_loss_coef": float(args.focal_loss_coef),
        "hard_negative_logistic_loss_coef": float(args.hard_negative_logistic_loss_coef),
        "top1_ce_loss_coef": float(args.top1_ce_loss_coef),
        "top1_focal_loss_coef": float(args.top1_focal_loss_coef),
        "top1_hard_negative_loss_coef": float(args.top1_hard_negative_loss_coef),
        "focal_gamma": float(args.focal_gamma),
        "pairwise_rank_weight_power": float(args.pairwise_rank_weight_power),
        "margin": float(args.margin),
        "weight_grid": [float(item) for item in args.weight_grid],
        "selected_channels": selected_channels,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    base_val_metrics = select_rerank_weight(
        base_logits=val_base_logits,
        rerank_scores=torch.zeros_like(val_shortlist_indices, dtype=torch.float32),
        shortlist_indices=val_shortlist_indices,
        ordered_image_ids=resolved_val_ids,
        candidate_image_ids=val_ids,
        weight_grid=[0.0],
    )[1]
    save_json(
        {
            "base_val_top1": float(base_val_metrics["top1_acc"]),
            "base_val_top5": float(base_val_metrics["top5_acc"]),
        },
        run_dir / "base_val_metrics.json",
    )

    best_score = (float("-inf"), float("-inf"))
    history: list[dict[str, float]] = []
    weight_history: list[dict[str, object]] = []
    start_time = time.time()

    train_candidate_values = train_bank.values.float()
    val_candidate_values = val_bank.values.float()

    for epoch in range(1, args.epochs + 1):
        reranker.train()
        batch_metrics: list[dict[str, float]] = []
        for batch in train_loader:
            query_embeddings = batch["query_embedding"].to(device)
            base_shortlist_scores = batch["base_shortlist_score"].to(device)
            shortlist_indices = batch["shortlist_indices"].long()
            target_positions = batch["target_position"].long().to(device)
            shortlist_candidates = train_candidate_values[shortlist_indices].to(device)
            score_outputs = reranker.score_shortlist_details(
                query_embeddings,
                shortlist_candidates,
                base_shortlist_scores=base_shortlist_scores,
            )
            shortlist_scores = score_outputs["combined_scores"]
            auxiliary_top1_scores = score_outputs["top1_logits"]
            loss, metrics = reranker_loss(
                shortlist_scores,
                target_positions,
                auxiliary_top1_scores=auxiliary_top1_scores,
                ce_loss_coef=args.ce_loss_coef,
                margin_loss_coef=args.margin_loss_coef,
                pairwise_loss_coef=args.pairwise_loss_coef,
                focal_loss_coef=args.focal_loss_coef,
                hard_negative_logistic_loss_coef=args.hard_negative_logistic_loss_coef,
                top1_ce_loss_coef=args.top1_ce_loss_coef,
                top1_focal_loss_coef=args.top1_focal_loss_coef,
                top1_hard_negative_loss_coef=args.top1_hard_negative_loss_coef,
                focal_gamma=args.focal_gamma,
                pairwise_rank_weight_power=args.pairwise_rank_weight_power,
                margin=args.margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema_state is not None:
                update_ema_state(reranker, ema_state, decay=float(args.ema_decay))
            batch_metrics.append(metrics)

        train_metrics = average_metrics(batch_metrics)
        train_state = clone_state_dict(reranker)
        eval_state = resolve_eval_state(reranker, ema_state)
        reranker.load_state_dict(eval_state, strict=True)
        reranker.eval()
        val_rerank_scores = compute_shortlist_rerank_scores(
            reranker,
            val_query_embeddings,
            val_base_shortlist_scores,
            val_shortlist_indices,
            val_candidate_values,
            device=device,
            batch_size=args.batch_size,
        )
        selected_weight, val_metrics, search_history = select_rerank_weight(
            base_logits=val_base_logits,
            rerank_scores=val_rerank_scores,
            shortlist_indices=val_shortlist_indices,
            ordered_image_ids=resolved_val_ids,
            candidate_image_ids=val_ids,
            weight_grid=args.weight_grid,
        )
        reranker.load_state_dict(train_state, strict=True)

        epoch_metrics = {
            "epoch": float(epoch),
            "train_total_loss": float(train_metrics.get("total_loss", 0.0)),
            "train_ce_loss": float(train_metrics.get("ce_loss", 0.0)),
            "train_margin_loss": float(train_metrics.get("margin_loss", 0.0)),
            "train_pairwise_loss": float(train_metrics.get("pairwise_loss", 0.0)),
            "train_focal_loss": float(train_metrics.get("focal_loss", 0.0)),
            "train_hard_negative_logistic_loss": float(train_metrics.get("hard_negative_logistic_loss", 0.0)),
            "train_top1_ce_loss": float(train_metrics.get("top1_ce_loss", 0.0)),
            "train_top1_focal_loss": float(train_metrics.get("top1_focal_loss", 0.0)),
            "train_top1_hard_negative_loss": float(train_metrics.get("top1_hard_negative_loss", 0.0)),
            "val_top1": float(val_metrics["top1_acc"]),
            "val_top5": float(val_metrics["top5_acc"]),
            "val_selected_rerank_weight": float(selected_weight),
            "eval_uses_ema": float(ema_state is not None),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)
        weight_history.append({"epoch": epoch, "weight_results": search_history})
        save_json({"history": history}, run_dir / "history.json")
        save_json({"weight_history": weight_history}, run_dir / "weight_history.json")

        save_checkpoint(
            run_dir / "last.pt",
            model_state=train_state,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None if scheduler is None else scheduler.state_dict(),
            config=config,
            metrics=epoch_metrics,
            extra_state={
                "base_checkpoint": str(args.base_checkpoint),
                "ema_state": ema_state,
            },
        )

        score = (epoch_metrics["val_top1"], epoch_metrics["val_top5"])
        if score > best_score:
            best_score = score
            save_checkpoint(
                run_dir / "best.pt",
                model_state=eval_state,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=None if scheduler is None else scheduler.state_dict(),
                config=config,
                metrics=epoch_metrics,
                extra_state={
                    "base_checkpoint": str(args.base_checkpoint),
                    "ema_state": ema_state,
                },
            )

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"epoch={epoch:03d} "
            f"train_total={epoch_metrics['train_total_loss']:.4f} "
            f"train_ce={epoch_metrics['train_ce_loss']:.4f} "
            f"train_margin={epoch_metrics['train_margin_loss']:.4f} "
            f"train_pairwise={epoch_metrics['train_pairwise_loss']:.4f} "
            f"train_focal={epoch_metrics['train_focal_loss']:.4f} "
            f"train_hn={epoch_metrics['train_hard_negative_logistic_loss']:.4f} "
            f"train_top1_ce={epoch_metrics['train_top1_ce_loss']:.4f} "
            f"val_top1={epoch_metrics['val_top1']:.4f} "
            f"val_top5={epoch_metrics['val_top5']:.4f} "
            f"weight={epoch_metrics['val_selected_rerank_weight']:.2f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
