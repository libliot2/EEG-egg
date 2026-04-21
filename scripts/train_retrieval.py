#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
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


CHANNEL_PRESETS: dict[str, list[str]] = {
    "visual17": [
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG retrieval model.")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
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
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--channel-dropout", type=float, default=0.1)
    parser.add_argument("--time-mask-ratio", type=float, default=0.1)
    parser.add_argument("--encoder-type", choices=["legacy_cnn", "atm_small", "atm_base", "atm_large"], default="atm_small")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-eeg-perturbation", action="store_true")
    parser.add_argument("--use-semantic-target-adapter", action="store_true")
    parser.add_argument("--use-perceptual-target-adapter", action="store_true")
    parser.add_argument("--target-adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--target-adapter-dropout", type=float, default=0.1)
    parser.add_argument("--target-adapter-beta", type=float, default=0.1)
    parser.add_argument("--target-adapter-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-loss-weight", type=float, default=0.7)
    parser.add_argument("--retrieval-loss-type", choices=["clip", "neuroclip"], default="clip")
    parser.add_argument("--soft-target-beta", type=float, default=10.0)
    parser.add_argument("--clip-loss-coef", type=float, default=1.0)
    parser.add_argument("--soft-loss-coef", type=float, default=0.3)
    parser.add_argument("--relation-loss-coef", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-no-avg-trials", action="store_true")
    parser.add_argument("--train-trial-sampling", action="store_true")
    parser.add_argument("--train-trial-k-min", type=int, default=1)
    parser.add_argument("--train-trial-k-max", type=int, default=4)
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    parser.add_argument("--selection-metric", choices=["top1", "top5", "blend_top1_top5"], default="blend_top1_top5")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--keep-last-k", type=int, default=5)
    parser.add_argument("--freeze-eeg-encoder", action="store_true")
    parser.add_argument("--freeze-retrieval-heads", action="store_true")
    parser.add_argument("--freeze-target-adapters", action="store_true")
    return parser.parse_args()


def resolve_selected_channels(args: argparse.Namespace) -> list[str] | None:
    if args.selected_channels and args.channel_preset:
        raise ValueError("--selected-channels and --channel-preset cannot both be set.")
    if args.selected_channels:
        return [str(channel) for channel in args.selected_channels]
    if args.channel_preset:
        return list(CHANNEL_PRESETS[args.channel_preset])
    return None


def load_bank(path: Path | None, *, name: str) -> TensorBank | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{name} bank not found: {path}.")
    return TensorBank.load(path)


def apply_freeze_options(model: RetrievalModel, args: argparse.Namespace) -> None:
    if args.freeze_eeg_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    if args.freeze_retrieval_heads:
        for name in ("semantic_head", "perceptual_head", "logit_scale", "semantic_logit_scale", "perceptual_logit_scale"):
            module_or_param = getattr(model, name, None)
            if module_or_param is None:
                continue
            if isinstance(module_or_param, torch.nn.Parameter):
                module_or_param.requires_grad = False
            else:
                for parameter in module_or_param.parameters():
                    parameter.requires_grad = False
    if args.freeze_target_adapters:
        for name in ("semantic_target_adapter", "perceptual_target_adapter"):
            module = getattr(model, name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False


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
    retrieval_loss_type: str,
    soft_target_beta: float,
    clip_loss_coef: float,
    soft_loss_coef: float,
    relation_loss_coef: float,
    grad_accum_steps: int,
    max_grad_norm: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    target_adapter_loss_weight: float,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(loader, start=1):
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
            retrieval_loss_type=retrieval_loss_type,
            soft_target_beta=soft_target_beta,
            clip_loss_coef=clip_loss_coef,
            soft_loss_coef=soft_loss_coef,
            relation_loss_coef=relation_loss_coef,
            target_adapter_loss_weight=target_adapter_loss_weight,
        )
        loss = loss / max(1, grad_accum_steps)

        loss.backward()
        should_step = step_idx % max(1, grad_accum_steps) == 0 or step_idx == len(loader)
        if should_step:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append(batch_metrics)

    return average_metrics(metrics)


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    selected_channels = resolve_selected_channels(args)

    if args.train_no_avg_trials and args.train_trial_sampling:
        raise ValueError("--train-no-avg-trials and --train-trial-sampling cannot both be enabled.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1).")

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

    sample_eeg = train_records[0].eeg
    in_channels = int(sample_eeg.shape[0] if sample_eeg.ndim == 2 else sample_eeg.shape[1])
    semantic_dim = None if semantic_bank is None else int(semantic_bank.values.shape[1])
    perceptual_dim = None if perceptual_bank is None else int(perceptual_bank.values.shape[1])
    model = RetrievalModel(
        in_channels=in_channels,
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
        use_eeg_perturbation=args.use_eeg_perturbation,
        use_semantic_target_adapter=args.use_semantic_target_adapter,
        use_perceptual_target_adapter=args.use_perceptual_target_adapter,
        target_adapter_hidden_dim=args.target_adapter_hidden_dim,
        target_adapter_dropout=args.target_adapter_dropout,
        target_adapter_beta=args.target_adapter_beta,
    ).to(device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)

    apply_freeze_options(model, args)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("All model parameters are frozen; nothing to train.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_optimizer_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(round(total_optimizer_steps * args.warmup_ratio))
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "retrieval",
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_accum_steps": int(args.grad_accum_steps),
        "max_grad_norm": args.max_grad_norm,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "channel_dropout": args.channel_dropout,
        "time_mask_ratio": args.time_mask_ratio,
        "encoder_type": args.encoder_type,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "use_eeg_perturbation": bool(args.use_eeg_perturbation),
        "use_semantic_target_adapter": bool(args.use_semantic_target_adapter),
        "use_perceptual_target_adapter": bool(args.use_perceptual_target_adapter),
        "target_adapter_hidden_dim": int(args.target_adapter_hidden_dim),
        "target_adapter_dropout": float(args.target_adapter_dropout),
        "target_adapter_beta": float(args.target_adapter_beta),
        "target_adapter_loss_weight": float(args.target_adapter_loss_weight),
        "semantic_dim": semantic_dim,
        "perceptual_dim": perceptual_dim,
        "semantic_loss_weight": args.semantic_loss_weight,
        "perceptual_loss_weight": args.perceptual_loss_weight,
        "retrieval_loss_type": args.retrieval_loss_type,
        "soft_target_beta": args.soft_target_beta,
        "clip_loss_coef": args.clip_loss_coef,
        "soft_loss_coef": args.soft_loss_coef,
        "relation_loss_coef": args.relation_loss_coef,
        "train_avg_trials": not args.train_no_avg_trials and not args.train_trial_sampling,
        "train_trial_sampling": bool(args.train_trial_sampling),
        "train_trial_k_min": int(args.train_trial_k_min),
        "train_trial_k_max": int(args.train_trial_k_max),
        "val_avg_trials": True,
        "selected_channels": selected_channels,
        "channel_preset": args.channel_preset,
        "in_channels": in_channels,
        "semantic_bank": None if args.semantic_bank is None else str(args.semantic_bank),
        "perceptual_bank": None if args.perceptual_bank is None else str(args.perceptual_bank),
        "ordered_val_image_ids": ordered_image_ids(val_records),
        "alpha_grid": [step / 10.0 for step in range(11)],
        "alpha_selection_rule": "maximize_val_top1_then_val_top5_then_closest_to_0.5",
        "selection_metric": args.selection_metric,
        "save_every_epoch": bool(args.save_every_epoch),
        "keep_last_k": int(args.keep_last_k),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": warmup_steps,
        "freeze_eeg_encoder": bool(args.freeze_eeg_encoder),
        "freeze_retrieval_heads": bool(args.freeze_retrieval_heads),
        "freeze_target_adapters": bool(args.freeze_target_adapters),
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
            retrieval_loss_type=args.retrieval_loss_type,
            soft_target_beta=args.soft_target_beta,
            clip_loss_coef=args.clip_loss_coef,
            soft_loss_coef=args.soft_loss_coef,
            relation_loss_coef=args.relation_loss_coef,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            scheduler=scheduler,
            target_adapter_loss_weight=args.target_adapter_loss_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            device=device,
        )

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
        if "semantic_clip_loss" in train_metrics:
            epoch_metrics["train_semantic_clip_loss"] = float(train_metrics["semantic_clip_loss"])
        if "semantic_soft_loss" in train_metrics:
            epoch_metrics["train_semantic_soft_loss"] = float(train_metrics["semantic_soft_loss"])
        if "semantic_relation_loss" in train_metrics:
            epoch_metrics["train_semantic_relation_loss"] = float(train_metrics["semantic_relation_loss"])
        if "perceptual_clip_loss" in train_metrics:
            epoch_metrics["train_perceptual_clip_loss"] = float(train_metrics["perceptual_clip_loss"])
        if "perceptual_soft_loss" in train_metrics:
            epoch_metrics["train_perceptual_soft_loss"] = float(train_metrics["perceptual_soft_loss"])
        if "perceptual_relation_loss" in train_metrics:
            epoch_metrics["train_perceptual_relation_loss"] = float(train_metrics["perceptual_relation_loss"])
        if "semantic_target_adapter_reg" in train_metrics:
            epoch_metrics["train_semantic_target_adapter_reg"] = float(train_metrics["semantic_target_adapter_reg"])
        if "perceptual_target_adapter_reg" in train_metrics:
            epoch_metrics["train_perceptual_target_adapter_reg"] = float(train_metrics["perceptual_target_adapter_reg"])

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
