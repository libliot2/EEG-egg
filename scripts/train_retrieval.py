#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split, ordered_image_ids
from project1_eeg.image_banks import TensorBank
from project1_eeg.retrieval import RetrievalModel, weighted_retrieval_loss
from project1_eeg.runtime import (
    compute_retrieval_logits,
    compute_retrieval_outputs,
    make_dataloader,
    select_best_alpha,
)
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    format_seconds,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG retrieval model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--channel-dropout", type=float, default=0.1)
    parser.add_argument("--time-mask-ratio", type=float, default=0.1)
    parser.add_argument("--encoder-type", choices=["legacy_cnn", "atm_small", "atm_base", "atm_large"], default="atm_small")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-loss-weight", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-no-avg-trials", action="store_true")
    parser.add_argument("--train-trial-sampling", action="store_true")
    parser.add_argument("--train-trial-k-min", type=int, default=1)
    parser.add_argument("--train-trial-k-max", type=int, default=4)
    parser.add_argument("--selection-metric", choices=["top1", "top5", "blend_top1_top5"], default="blend_top1_top5")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--keep-last-k", type=int, default=5)
    return parser.parse_args()


def load_bank(path: Path | None, *, name: str) -> TensorBank | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{name} bank not found: {path}.")
    return TensorBank.load(path)


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def selection_score(metrics: dict[str, float], *, selection_metric: str) -> tuple[float, ...]:
    top1 = float(metrics["top1_acc"])
    top5 = float(metrics["top5_acc"])
    if selection_metric == "top1":
        return (top1, top5)
    if selection_metric == "top5":
        return (top5, top1)
    if selection_metric == "blend_top1_top5":
        return (0.5 * top1 + 0.5 * top5, top1, top5)
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def train_one_epoch(
    model: RetrievalModel,
    loader,
    *,
    semantic_bank: TensorBank | None,
    perceptual_bank: TensorBank | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    semantic_loss_weight: float,
    perceptual_loss_weight: float,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        semantic_targets = None
        perceptual_targets = None
        if semantic_bank is not None:
            semantic_targets = semantic_bank.align(batch["image_id"], device=device).float()
        if perceptual_bank is not None:
            perceptual_targets = perceptual_bank.align(batch["image_id"], device=device).float()

        outputs = model.encode_all(eeg)
        loss, batch_metrics = weighted_retrieval_loss(
            model,
            outputs,
            semantic_targets=semantic_targets,
            perceptual_targets=perceptual_targets,
            semantic_weight=semantic_loss_weight,
            perceptual_weight=perceptual_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append(batch_metrics)

    return average_metrics(metrics)


@torch.no_grad()
def evaluate(
    model: RetrievalModel,
    loader,
    *,
    semantic_bank: TensorBank | None,
    perceptual_bank: TensorBank | None,
    device: torch.device,
) -> dict[str, float]:
    outputs, image_ids, _ = compute_retrieval_outputs(model, loader, device)
    _, candidate_ids, component_logits = compute_retrieval_logits(
        model,
        outputs,
        semantic_bank=semantic_bank,
        perceptual_bank=perceptual_bank,
        candidate_image_ids=image_ids,
        alpha=0.5,
    )
    selected_alpha, metrics, alpha_history = select_best_alpha(
        semantic_logits=component_logits.get("semantic"),
        perceptual_logits=component_logits.get("perceptual"),
        ordered_image_ids=image_ids,
        candidate_image_ids=candidate_ids,
    )
    return {
        "top1_acc": float(metrics["top1_acc"]),
        "top5_acc": float(metrics["top5_acc"]),
        "selected_alpha": float(selected_alpha),
        "alpha_history": alpha_history,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.train_no_avg_trials and args.train_trial_sampling:
        raise ValueError("--train-no-avg-trials and --train-trial-sampling cannot both be enabled.")

    semantic_bank = load_bank(args.semantic_bank, name="Semantic") if args.semantic_bank is not None else None
    perceptual_bank = load_bank(args.perceptual_bank, name="Perceptual") if args.perceptual_bank is not None else None
    if semantic_bank is None and perceptual_bank is None:
        raise ValueError("At least one of --semantic-bank or --perceptual-bank must be provided.")

    if args.encoder_type == "legacy_cnn":
        if perceptual_bank is not None:
            raise ValueError("encoder_type=legacy_cnn only supports the semantic bank.")
        if semantic_bank is None:
            raise ValueError("encoder_type=legacy_cnn requires a semantic bank.")
        if semantic_bank.values.shape[1] != args.embedding_dim:
            raise ValueError(
                "legacy_cnn requires semantic bank dimension to match --embedding-dim. "
                f"Got bank_dim={semantic_bank.values.shape[1]} and embedding_dim={args.embedding_dim}."
            )

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=not args.train_no_avg_trials and not args.train_trial_sampling,
        preserve_trials=args.train_trial_sampling,
        image_ids=train_ids,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=val_ids,
    )

    train_loader = make_dataloader(
        EEGImageDataset(
            train_records,
            trial_sampling="random_avg" if args.train_trial_sampling else "none",
            trial_k_min=args.train_trial_k_min,
            trial_k_max=args.train_trial_k_max,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        EEGImageDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    semantic_dim = None if semantic_bank is None else int(semantic_bank.values.shape[1])
    perceptual_dim = None if perceptual_bank is None else int(perceptual_bank.values.shape[1])
    model = RetrievalModel(
        in_channels=train_records[0].eeg.shape[0],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        channel_dropout=args.channel_dropout,
        time_mask_ratio=args.time_mask_ratio,
        encoder_type=args.encoder_type,
        semantic_dim=semantic_dim,
        perceptual_dim=perceptual_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "retrieval",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "channel_dropout": args.channel_dropout,
        "time_mask_ratio": args.time_mask_ratio,
        "encoder_type": args.encoder_type,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "semantic_dim": semantic_dim,
        "perceptual_dim": perceptual_dim,
        "semantic_loss_weight": args.semantic_loss_weight,
        "perceptual_loss_weight": args.perceptual_loss_weight,
        "train_avg_trials": not args.train_no_avg_trials and not args.train_trial_sampling,
        "train_trial_sampling": bool(args.train_trial_sampling),
        "train_trial_k_min": int(args.train_trial_k_min),
        "train_trial_k_max": int(args.train_trial_k_max),
        "val_avg_trials": True,
        "in_channels": int(train_records[0].eeg.shape[0]),
        "semantic_bank": None if args.semantic_bank is None else str(args.semantic_bank),
        "perceptual_bank": None if args.perceptual_bank is None else str(args.perceptual_bank),
        "ordered_val_image_ids": ordered_image_ids(val_records),
        "alpha_grid": [step / 10.0 for step in range(11)],
        "alpha_selection_rule": "maximize_val_top1_then_val_top5_then_closest_to_0.5",
        "selection_metric": args.selection_metric,
        "save_every_epoch": bool(args.save_every_epoch),
        "keep_last_k": int(args.keep_last_k),
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    best_selection = (float("-inf"), float("-inf"), float("-inf"))
    best_top1 = (float("-inf"), float("-inf"))
    best_top5 = (float("-inf"), float("-inf"))
    history: list[dict[str, float]] = []
    alpha_search_history: list[dict[str, object]] = []
    epoch_checkpoints: list[Path] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            optimizer=optimizer,
            device=device,
            semantic_loss_weight=args.semantic_loss_weight,
            perceptual_loss_weight=args.perceptual_loss_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            device=device,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": float(epoch),
            "train_total_loss": float(train_metrics.get("total_loss", 0.0)),
            "val_top1": float(val_metrics["top1_acc"]),
            "val_top5": float(val_metrics["top5_acc"]),
            "val_blend_top1_top5": float(0.5 * val_metrics["top1_acc"] + 0.5 * val_metrics["top5_acc"]),
            "val_selected_alpha": float(val_metrics["selected_alpha"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if "semantic_loss" in train_metrics:
            epoch_metrics["train_semantic_loss"] = float(train_metrics["semantic_loss"])
        if "perceptual_loss" in train_metrics:
            epoch_metrics["train_perceptual_loss"] = float(train_metrics["perceptual_loss"])
        if "legacy_loss" in train_metrics:
            epoch_metrics["train_legacy_loss"] = float(train_metrics["legacy_loss"])

        history.append(epoch_metrics)
        alpha_search_history.append({"epoch": epoch, "alpha_results": val_metrics["alpha_history"]})
        save_json({"history": history}, run_dir / "history.json")
        save_json({"alpha_search_history": alpha_search_history}, run_dir / "alpha_search.json")

        save_checkpoint(
            run_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            config=config,
            metrics=epoch_metrics,
        )

        if args.save_every_epoch or args.keep_last_k > 0:
            epoch_path = run_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_path,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=epoch_metrics,
            )
            epoch_checkpoints.append(epoch_path)
            while args.keep_last_k > 0 and len(epoch_checkpoints) > args.keep_last_k:
                stale = epoch_checkpoints.pop(0)
                if stale.exists():
                    stale.unlink()

        selection = selection_score(val_metrics, selection_metric=args.selection_metric)
        score_top1 = selection_score(val_metrics, selection_metric="top1")
        score_top5 = selection_score(val_metrics, selection_metric="top5")
        if selection > best_selection:
            best_selection = selection
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=epoch_metrics,
            )
        if score_top1 > best_top1:
            best_top1 = score_top1
            save_checkpoint(
                run_dir / "best_top1.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=epoch_metrics,
            )
        if score_top5 > best_top5:
            best_top5 = score_top5
            save_checkpoint(
                run_dir / "best_top5.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=epoch_metrics,
            )

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"epoch={epoch:03d} "
            f"train_total={epoch_metrics['train_total_loss']:.4f} "
            f"val_top1={epoch_metrics['val_top1']:.4f} "
            f"val_top5={epoch_metrics['val_top5']:.4f} "
            f"blend={epoch_metrics['val_blend_top1_top5']:.4f} "
            f"alpha={epoch_metrics['val_selected_alpha']:.2f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
