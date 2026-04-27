#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids, make_train_val_split
from project1_eeg.evaluation import eval_images
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import make_dataloader
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    format_seconds,
    load_image_batch,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


class LowLevelImagePrior(nn.Module):
    def __init__(
        self,
        *,
        eeg_embedder: nn.Module,
        embedding_dim: int,
        base_channels: int = 256,
        image_size: int = 256,
        finetune_encoder: bool = False,
    ) -> None:
        super().__init__()
        if image_size != 256:
            raise ValueError("LowLevelImagePrior currently supports image_size=256.")
        self.eeg_embedder = eeg_embedder
        self.finetune_encoder = finetune_encoder
        self.fc = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, base_channels * 16 * 16),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            self._block(base_channels, base_channels // 2),
            self._block(base_channels // 2, base_channels // 4),
            self._block(base_channels // 4, base_channels // 8),
            self._block(base_channels // 8, base_channels // 16),
            nn.Conv2d(base_channels // 16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels),
            nn.GELU(),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.finetune_encoder and self.training):
            embedding = self.eeg_embedder(eeg)
        feature = self.fc(embedding).view(eeg.shape[0], -1, 16, 16)
        return self.decoder(feature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EEG-conditioned RGB low-level init image prior.")
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "experiments" / "lowlevel_image_prior")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--ssim-weight", type=float, default=0.5)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.1)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-split-file", type=Path, default=None)
    parser.add_argument("--eval-image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--eval-output-name", type=str, default="eval_val64")
    parser.add_argument("--selected-channels", nargs="+", default=None)
    return parser.parse_args()


def build_embedder(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    output_dim = model.output_dim(model.primary_head())
    embedder = model.build_primary_embedder().to(device)
    embedder.eval()
    return embedder, config, int(output_dim)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )


def load_targets(paths: list[str], *, image_size: int, device: torch.device) -> torch.Tensor:
    transform = image_transform(image_size)
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors, dim=0).to(device)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, kernel_size=7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=7, stride=1, padding=3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - score.mean()


def clip_loss(pred: torch.Tensor, target: torch.Tensor, clip_encode) -> torch.Tensor:
    pred_clip = clip_encode(pred)
    with torch.no_grad():
        target_clip = clip_encode(target)
    return 1.0 - (pred_clip * target_clip).sum(dim=-1).mean()


def average(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in metrics[0]}


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def run_epoch(
    *,
    model: LowLevelImagePrior,
    loader,
    optimizer: torch.optim.Optimizer | None,
    lpips_model,
    clip_encode,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    rows: list[dict[str, float]] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        target = load_targets(list(batch["image_path"]), image_size=args.image_size, device=device)
        with torch.set_grad_enabled(is_train):
            pred = model(eeg)
            l1 = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
            ssim = ssim_loss(pred, target)
            perceptual = lpips_model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()
            semantic = clip_loss(pred, target, clip_encode) if args.clip_weight > 0.0 else pred.new_tensor(0.0)
            total = (
                args.l1_weight * l1
                + args.mse_weight * mse
                + args.ssim_weight * ssim
                + args.lpips_weight * perceptual
                + args.clip_weight * semantic
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        rows.append(
            {
                "total_loss": float(total.item()),
                "l1": float(l1.item()),
                "mse": float(mse.item()),
                "ssim_loss": float(ssim.item()),
                "lpips": float(perceptual.item()),
                "clip_loss": float(semantic.item()),
            }
        )
    return average(rows)


@torch.no_grad()
def generate_eval_images(
    *,
    model: LowLevelImagePrior,
    records,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    all_fake = []
    all_real = []
    for batch in loader:
        pred = model(batch["eeg"].to(device)).cpu()
        for image_id, tensor in zip(batch["image_id"], pred, strict=True):
            image = transforms.ToPILImage()(tensor.clamp(0.0, 1.0))
            image.resize((512, 512), Image.BICUBIC).save(image_dir / f"{image_id}.png")
        all_fake.extend(list(pred))
        real = load_image_batch(list(batch["image_path"]), image_size=args.image_size)
        all_real.extend(list(real))
    fake_images = torch.stack(all_fake, dim=0)
    real_images = torch.stack(all_real, dim=0)
    metrics = eval_images(real_images=real_images, fake_images=fake_images, device=device)
    save_json(metrics, output_dir / "reconstruction_metrics.json")
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    embedder, retrieval_config, embedding_dim = build_embedder(args.retrieval_checkpoint, device)
    selected_channels = args.selected_channels or retrieval_config.get("selected_channels")
    if not args.finetune_encoder:
        for parameter in embedder.parameters():
            parameter.requires_grad = False

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=train_ids,
        selected_channels=selected_channels,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=val_ids,
        selected_channels=selected_channels,
    )
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

    model = LowLevelImagePrior(
        eeg_embedder=embedder,
        embedding_dim=embedding_dim,
        base_channels=args.base_channels,
        image_size=args.image_size,
        finetune_encoder=args.finetune_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    lpips_model.requires_grad_(False)
    from project1_eeg.runtime import clip_image_loss_model

    clip_encode = clip_image_loss_model(device, model_name="ViT-B/32")

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = jsonable(vars(args).copy())
    config.update(
        {
            "retrieval_config": jsonable(retrieval_config),
            "embedding_dim": embedding_dim,
            "selected_channels": selected_channels,
            "train_count": len(train_records),
            "val_count": len(val_records),
        }
    )
    save_json(config, run_dir / "config.json")
    history: list[dict[str, float]] = []
    best_score = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            lpips_model=lpips_model,
            clip_encode=clip_encode,
            args=args,
            device=device,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            lpips_model=lpips_model,
            clip_encode=clip_encode,
            args=args,
            device=device,
        )
        scheduler.step()
        row = {"epoch": float(epoch), "lr": float(optimizer.param_groups[0]["lr"])}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        save_json({"history": history}, run_dir / "history.json")
        save_checkpoint(
            run_dir / "last.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            config=config,
            metrics=row,
        )
        if val_metrics["total_loss"] < best_score:
            best_score = val_metrics["total_loss"]
            save_checkpoint(
                run_dir / "best.pt",
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=config,
                metrics=row,
            )
        print(
            f"epoch={epoch:03d} train={train_metrics['total_loss']:.4f} "
            f"val={val_metrics['total_loss']:.4f} elapsed={format_seconds(time.time() - start_time)}",
            flush=True,
        )

    best_payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    model.eval()
    eval_ids = load_split_image_ids(args.eval_split_file, image_id_source=args.eval_image_id_source)
    eval_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=eval_ids,
        selected_channels=selected_channels,
    )
    metrics = generate_eval_images(
        model=model,
        records=eval_records,
        args=args,
        output_dir=run_dir / args.eval_output_name,
        device=device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
