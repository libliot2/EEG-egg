#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split
from project1_eeg.retrieval import ATMSmallEncoder
from project1_eeg.runtime import make_dataloader
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
    parser = argparse.ArgumentParser(description="Pretrain an EEG encoder with masked temporal reconstruction.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "experiments" / "eeg_mask_pretrain",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--mask-patch-size", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


class MaskedEEGAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        time_steps: int,
        hidden_dim: int,
        embedding_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
        channel_dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.encoder = ATMSmallEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            channel_dropout=channel_dropout,
            time_mask_ratio=0.0,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 4, in_channels * time_steps),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(eeg)
        reconstruction = self.decoder(embedding)
        return reconstruction.view(eeg.shape[0], self.in_channels, self.time_steps)


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def apply_temporal_mask(
    eeg: torch.Tensor,
    *,
    mask_ratio: float,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1).")

    batch_size, channels, time_steps = eeg.shape
    num_patches = max(1, time_steps // patch_size)
    effective_time_steps = num_patches * patch_size
    num_masked = max(1, int(round(num_patches * mask_ratio)))

    patch_mask = torch.zeros((batch_size, num_patches), device=eeg.device, dtype=torch.bool)
    for row in range(batch_size):
        indices = torch.randperm(num_patches, device=eeg.device)[:num_masked]
        patch_mask[row, indices] = True

    time_mask = patch_mask.repeat_interleave(patch_size, dim=1)
    if effective_time_steps < time_steps:
        tail = torch.zeros((batch_size, time_steps - effective_time_steps), device=eeg.device, dtype=torch.bool)
        time_mask = torch.cat([time_mask, tail], dim=1)
    mask = time_mask.unsqueeze(1).expand(-1, channels, -1)
    masked_eeg = eeg.clone()
    masked_eeg[mask] = 0.0
    return masked_eeg, mask


def reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (predicted - target).pow(2)
    masked_error = error[mask]
    return masked_error.mean()


def train_one_epoch(
    model: MaskedEEGAutoencoder,
    loader,
    *,
    optimizer: torch.optim.Optimizer,
    mask_ratio: float,
    patch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        masked_eeg, mask = apply_temporal_mask(eeg, mask_ratio=mask_ratio, patch_size=patch_size)
        predicted = model(masked_eeg)
        loss = reconstruction_loss(predicted, eeg, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append({"total_loss": float(loss.item())})
    return average_metrics(metrics)


@torch.no_grad()
def evaluate(
    model: MaskedEEGAutoencoder,
    loader,
    *,
    mask_ratio: float,
    patch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        masked_eeg, mask = apply_temporal_mask(eeg, mask_ratio=mask_ratio, patch_size=patch_size)
        predicted = model(masked_eeg)
        loss = reconstruction_loss(predicted, eeg, mask)
        metrics.append({"total_loss": float(loss.item())})
    return average_metrics(metrics)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=False,
        image_ids=train_ids,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=False,
        image_ids=val_ids,
    )
    if not train_records:
        raise ValueError("No EEG records were loaded for pretraining.")

    train_loader = make_dataloader(
        EEGImageDataset(train_records),
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

    in_channels = train_records[0].eeg.shape[0]
    time_steps = train_records[0].eeg.shape[1]
    model = MaskedEEGAutoencoder(
        in_channels=in_channels,
        time_steps=time_steps,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        channel_dropout=args.channel_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "eeg_mask_pretrain",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "channel_dropout": args.channel_dropout,
        "mask_ratio": args.mask_ratio,
        "mask_patch_size": args.mask_patch_size,
        "in_channels": in_channels,
        "time_steps": time_steps,
        "train_avg_trials": False,
        "val_avg_trials": False,
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            mask_ratio=args.mask_ratio,
            patch_size=args.mask_patch_size,
            device=device,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            mask_ratio=args.mask_ratio,
            patch_size=args.mask_patch_size,
            device=device,
        )
        scheduler.step()

        combined_metrics = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total_loss": float(train_metrics["total_loss"]),
            "val_total_loss": float(val_metrics["total_loss"]),
        }
        history.append(combined_metrics)
        save_json({"history": history}, run_dir / "history.json")
        save_checkpoint(
            run_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            config=config,
            metrics=combined_metrics,
        )
        torch.save(
            {
                "encoder_state": model.encoder.state_dict(),
                "config": config,
                "metrics": combined_metrics,
            },
            run_dir / "encoder_last.pt",
        )

        if combined_metrics["val_total_loss"] < best_loss:
            best_loss = combined_metrics["val_total_loss"]
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=combined_metrics,
            )
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "config": config,
                    "metrics": combined_metrics,
                },
                run_dir / "encoder_best.pt",
            )

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"val_total={val_metrics['total_loss']:.4f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
