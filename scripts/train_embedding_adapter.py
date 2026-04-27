#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import make_train_val_split
from project1_eeg.image_banks import TensorBank
from project1_eeg.reconstruction import EmbeddingAdapter
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    format_seconds,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


class PairedEmbeddingDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        *,
        image_ids: list[str],
        source_bank: TensorBank,
        target_bank: TensorBank,
    ) -> None:
        self.image_ids = list(image_ids)
        self.source_bank = source_bank
        self.target_bank = target_bank

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_id = self.image_ids[index]
        return {
            "image_id": image_id,
            "source": self.source_bank.align([image_id]).squeeze(0).float(),
            "target": self.target_bank.align([image_id]).squeeze(0).float(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight embedding adapter between two image banks.")
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--target-bank", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "experiments" / "embedding_adapter",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def compute_adapter_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse_loss = F.mse_loss(predicted, target)
    cosine_loss = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()

    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    contrastive_loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    total_loss = mse_weight * mse_loss + cosine_weight * cosine_loss + contrastive_weight * contrastive_loss
    metrics = {
        "mse_loss": float(mse_loss.item()),
        "cosine_loss": float(cosine_loss.item()),
        "contrastive_loss": float(contrastive_loss.item()),
        "total_loss": float(total_loss.item()),
        "avg_target_cosine": float(F.cosine_similarity(predicted, target, dim=-1).mean().item()),
    }
    return total_loss, metrics


def run_epoch(
    model: EmbeddingAdapter,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    metrics: list[dict[str, float]] = []

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(is_train):
            predicted = model(source)
            loss, batch_metrics = compute_adapter_loss(
                predicted,
                target,
                mse_weight=args.mse_weight,
                cosine_weight=args.cosine_weight,
                contrastive_weight=args.contrastive_weight,
                temperature=args.temperature,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metrics.append(batch_metrics)

    return average_metrics(metrics)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    source_bank = TensorBank.load(args.source_bank)
    target_bank = TensorBank.load(args.target_bank)
    shared_image_ids = sorted(set(source_bank.image_ids) & set(target_bank.image_ids))
    if not shared_image_ids:
        raise ValueError("The source bank and target bank do not share any image ids.")

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_ids = [image_id for image_id in train_ids if image_id in source_bank._index and image_id in target_bank._index]
    val_ids = [image_id for image_id in val_ids if image_id in source_bank._index and image_id in target_bank._index]
    if not train_ids or not val_ids:
        raise ValueError("After intersecting with the banks, train/val split became empty.")

    train_loader = DataLoader(
        PairedEmbeddingDataset(image_ids=train_ids, source_bank=source_bank, target_bank=target_bank),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PairedEmbeddingDataset(image_ids=val_ids, source_bank=source_bank, target_bank=target_bank),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = EmbeddingAdapter(
        input_dim=int(source_bank.values.shape[1]),
        output_dim=int(target_bank.values.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task": "embedding_adapter",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "mse_weight": args.mse_weight,
        "cosine_weight": args.cosine_weight,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "source_bank": str(args.source_bank),
        "target_bank": str(args.target_bank),
        "source_bank_type": source_bank.bank_type,
        "target_bank_type": target_bank.bank_type,
        "input_dim": int(source_bank.values.shape[1]),
        "output_dim": int(target_bank.values.shape[1]),
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    history: list[dict[str, float]] = []
    best_score = (float("-inf"), float("-inf"))
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device, args=args)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, args=args)
        scheduler.step()

        combined_metrics = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        for prefix, metric_dict in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metric_dict.items():
                combined_metrics[f"{prefix}_{key}"] = float(value)

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

        score = (
            float(val_metrics["avg_target_cosine"]),
            -float(val_metrics["total_loss"]),
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=combined_metrics,
            )

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"val_total={val_metrics['total_loss']:.4f} "
            f"val_cos={val_metrics['avg_target_cosine']:.4f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
