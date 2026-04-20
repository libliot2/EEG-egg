#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split
from project1_eeg.evaluation import eval_images
from project1_eeg.image_banks import TensorBank, default_bank_path
from project1_eeg.reconstruction import PrototypeResidualModel
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import (
    clip_image_loss_model,
    decode_latents,
    make_dataloader,
    prepare_vae,
    select_prototype_latents,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the residual latent reconstruction model.")
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--latent-bank", type=Path, default=default_bank_path(DEFAULT_OUTPUT_DIR, "vae", "train"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "reconstruction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--clip-loss-model", type=str, default="ViT-B/32")
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--prototype-topk", type=int, default=4)
    parser.add_argument("--prototype-mode", choices=["top1", "score_weighted_topk"], default="score_weighted_topk")
    parser.add_argument("--train-no-avg-trials", action="store_true")
    parser.add_argument("--official-eval-limit", type=int, default=0)
    parser.add_argument("--official-eval-every", type=int, default=5)
    return parser.parse_args()


def build_retrieval_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, payload


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_alpha(args: argparse.Namespace, retrieval_payload: dict, has_semantic: bool, has_perceptual: bool) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    metrics = retrieval_payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    return 0.5


def compute_losses(
    *,
    pred_latents: torch.Tensor,
    target_latents: torch.Tensor,
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    lpips_model,
    clip_encode,
    latent_weight: float,
    image_weight: float,
    lpips_weight: float,
    clip_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_l1 = F.l1_loss(pred_latents, target_latents)
    image_l1 = F.l1_loss(pred_images, target_images)
    perceptual = lpips_model(pred_images * 2.0 - 1.0, target_images * 2.0 - 1.0).mean()
    pred_clip = clip_encode(pred_images)
    with torch.no_grad():
        target_clip = clip_encode(target_images)
    clip_loss = 1.0 - (pred_clip * target_clip).sum(dim=-1).mean()

    total = (
        latent_weight * latent_l1
        + image_weight * image_l1
        + lpips_weight * perceptual
        + clip_weight * clip_loss
    )
    return total, {
        "latent_l1": float(latent_l1.item()),
        "image_l1": float(image_l1.item()),
        "lpips": float(perceptual.item()),
        "clip_loss": float(clip_loss.item()),
        "total_loss": float(total.item()),
    }


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(sum(metric[key] for metric in metrics) / len(metrics)) for key in keys}


def maybe_eval_official(
    *,
    real_images: list[torch.Tensor],
    fake_images: list[torch.Tensor],
    limit: int,
    device: torch.device,
) -> dict[str, float]:
    if limit <= 0 or not real_images:
        return {}
    real_tensor = torch.stack(real_images[:limit], dim=0)
    fake_tensor = torch.stack(fake_images[:limit], dim=0)
    return eval_images(real_images=real_tensor, fake_images=fake_tensor, device=device)


def train_one_epoch(
    model: PrototypeResidualModel,
    frozen_retrieval_model,
    loader,
    optimizer: torch.optim.Optimizer,
    full_latent_bank: TensorBank,
    candidate_semantic_bank: TensorBank | None,
    candidate_perceptual_bank: TensorBank | None,
    candidate_latent_bank: TensorBank,
    vae,
    lpips_model,
    clip_encode,
    device: torch.device,
    args: argparse.Namespace,
    alpha: float,
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []
    prototype_topk = 1 if args.prototype_mode == "top1" else args.prototype_topk

    for batch in loader:
        eeg = batch["eeg"].to(device)
        image_ids = list(batch["image_id"])
        target_latents = full_latent_bank.align(image_ids, device=device).float()
        target_images = load_image_batch(batch["image_path"], image_size=args.image_size, device=device)

        prototype_result = select_prototype_latents(
            frozen_retrieval_model,
            eeg,
            candidate_semantic_bank=candidate_semantic_bank,
            candidate_perceptual_bank=candidate_perceptual_bank,
            candidate_latent_bank=candidate_latent_bank,
            alpha=alpha,
            query_image_ids=image_ids,
            exclude_self=True,
            device=device,
            prototype_topk=prototype_topk,
            prototype_mode=args.prototype_mode,
        )
        prototype_latents = prototype_result["prototype_latents"]

        pred_latents = model(eeg, prototype_latents)
        pred_images = decode_latents(vae, pred_latents)
        loss, metric = compute_losses(
            pred_latents=pred_latents,
            target_latents=target_latents,
            pred_images=pred_images,
            target_images=target_images,
            lpips_model=lpips_model,
            clip_encode=clip_encode,
            latent_weight=args.latent_weight,
            image_weight=args.image_weight,
            lpips_weight=args.lpips_weight,
            clip_weight=args.clip_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append(metric)
    return average_metrics(metrics)


@torch.no_grad()
def evaluate(
    model: PrototypeResidualModel,
    frozen_retrieval_model,
    loader,
    full_latent_bank: TensorBank,
    candidate_semantic_bank: TensorBank | None,
    candidate_perceptual_bank: TensorBank | None,
    candidate_latent_bank: TensorBank,
    vae,
    lpips_model,
    clip_encode,
    device: torch.device,
    args: argparse.Namespace,
    alpha: float,
) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    real_images: list[torch.Tensor] = []
    fake_images: list[torch.Tensor] = []
    prototype_topk = 1 if args.prototype_mode == "top1" else args.prototype_topk

    for batch in loader:
        eeg = batch["eeg"].to(device)
        image_ids = list(batch["image_id"])
        target_latents = full_latent_bank.align(image_ids, device=device).float()
        target_images = load_image_batch(batch["image_path"], image_size=args.image_size, device=device)

        prototype_result = select_prototype_latents(
            frozen_retrieval_model,
            eeg,
            candidate_semantic_bank=candidate_semantic_bank,
            candidate_perceptual_bank=candidate_perceptual_bank,
            candidate_latent_bank=candidate_latent_bank,
            alpha=alpha,
            query_image_ids=image_ids,
            exclude_self=False,
            device=device,
            prototype_topk=prototype_topk,
            prototype_mode=args.prototype_mode,
        )
        prototype_latents = prototype_result["prototype_latents"]

        pred_latents = model(eeg, prototype_latents)
        pred_images = decode_latents(vae, pred_latents)
        _, metric = compute_losses(
            pred_latents=pred_latents,
            target_latents=target_latents,
            pred_images=pred_images,
            target_images=target_images,
            lpips_model=lpips_model,
            clip_encode=clip_encode,
            latent_weight=args.latent_weight,
            image_weight=args.image_weight,
            lpips_weight=args.lpips_weight,
            clip_weight=args.clip_weight,
        )
        metrics.append(metric)

        if len(real_images) < args.official_eval_limit:
            remaining = args.official_eval_limit - len(real_images)
            real_images.extend(list(target_images[:remaining].cpu()))
            fake_images.extend(list(pred_images[:remaining].cpu()))

    summary = average_metrics(metrics)
    summary.update(maybe_eval_official(real_images=real_images, fake_images=fake_images, limit=args.official_eval_limit, device=device))
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    frozen_retrieval_model, retrieval_config, retrieval_payload = build_retrieval_model(args.retrieval_checkpoint, device)
    full_semantic_bank = resolve_bank(
        args.semantic_bank,
        retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank"),
        name="Semantic",
    )
    full_perceptual_bank = resolve_bank(args.perceptual_bank, retrieval_config.get("perceptual_bank"), name="Perceptual")
    if full_semantic_bank is None and full_perceptual_bank is None:
        raise ValueError("At least one retrieval bank must be available for prototype selection.")
    if not args.latent_bank.exists():
        raise FileNotFoundError(f"Latent bank not found: {args.latent_bank}. Run scripts/cache_image_bank.py --bank-type vae --split train first.")

    full_latent_bank = TensorBank.load(args.latent_bank)
    alpha = resolve_alpha(
        args,
        retrieval_payload,
        has_semantic=full_semantic_bank is not None,
        has_perceptual=full_perceptual_bank is not None,
    )

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    candidate_semantic_bank = None if full_semantic_bank is None else full_semantic_bank.subset(train_ids)
    candidate_perceptual_bank = None if full_perceptual_bank is None else full_perceptual_bank.subset(train_ids)
    candidate_latent_bank = full_latent_bank.subset(train_ids)

    train_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=not args.train_no_avg_trials,
        image_ids=train_ids,
    )
    val_records = load_eeg_records(
        data_dir=args.data_dir,
        split="train",
        avg_trials=True,
        image_ids=val_ids,
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

    eeg_embedder = frozen_retrieval_model.build_primary_embedder()
    embedding_dim = frozen_retrieval_model.output_dim()
    model = PrototypeResidualModel(
        eeg_encoder=eeg_embedder,
        embedding_dim=embedding_dim,
        prototype_channels=int(full_latent_bank.values.shape[1]),
        hidden_dim=args.hidden_dim,
    ).to(device)

    vae = prepare_vae(args.vae_model, device)
    vae.requires_grad_(False)
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval().requires_grad_(False)
    clip_encode = clip_image_loss_model(device, model_name=args.clip_loss_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "reconstruction",
        "mode": "residual",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "image_size": args.image_size,
        "vae_model": args.vae_model,
        "clip_loss_model": args.clip_loss_model,
        "latent_weight": args.latent_weight,
        "image_weight": args.image_weight,
        "lpips_weight": args.lpips_weight,
        "clip_weight": args.clip_weight,
        "alpha": alpha,
        "prototype_topk": args.prototype_topk,
        "prototype_mode": args.prototype_mode,
        "prototype_channels": int(full_latent_bank.values.shape[1]),
        "train_avg_trials": not args.train_no_avg_trials,
        "official_eval_limit": args.official_eval_limit,
        "retrieval_checkpoint": str(args.retrieval_checkpoint),
        "semantic_bank": None if full_semantic_bank is None else str(
            args.semantic_bank or retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank")
        ),
        "perceptual_bank": None if full_perceptual_bank is None else str(args.perceptual_bank or retrieval_config.get("perceptual_bank")),
        "latent_bank": str(args.latent_bank),
        "retrieval_config": retrieval_config,
        "retrieval_primary_head": frozen_retrieval_model.primary_head(),
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            frozen_retrieval_model,
            train_loader,
            optimizer,
            full_latent_bank,
            candidate_semantic_bank,
            candidate_perceptual_bank,
            candidate_latent_bank,
            vae,
            lpips_model,
            clip_encode,
            device,
            args,
            alpha,
        )

        if args.official_eval_limit > 0 and (epoch % args.official_eval_every == 0 or epoch == args.epochs):
            val_metrics = evaluate(
                model,
                frozen_retrieval_model,
                val_loader,
                full_latent_bank,
                candidate_semantic_bank,
                candidate_perceptual_bank,
                candidate_latent_bank,
                vae,
                lpips_model,
                clip_encode,
                device,
                args,
                alpha,
            )
        else:
            val_metrics = evaluate(
                model,
                frozen_retrieval_model,
                val_loader,
                full_latent_bank,
                candidate_semantic_bank,
                candidate_perceptual_bank,
                candidate_latent_bank,
                vae,
                lpips_model,
                clip_encode,
                device,
                argparse.Namespace(**{**vars(args), "official_eval_limit": 0}),
                alpha,
            )

        scheduler.step()
        combined_metrics = {
            "epoch": float(epoch),
            "train_total_loss": train_metrics["total_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_latent_l1": val_metrics["latent_l1"],
            "val_image_l1": val_metrics["image_l1"],
            "val_lpips": val_metrics["lpips"],
            "val_clip_loss": val_metrics["clip_loss"],
            "alpha": float(alpha),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if "eval_ssim" in val_metrics:
            combined_metrics["eval_ssim"] = val_metrics["eval_ssim"]
            combined_metrics["eval_pixcorr"] = val_metrics["eval_pixcorr"]
            combined_metrics["eval_clip"] = val_metrics["eval_clip"]
            score = val_metrics["eval_ssim"] + val_metrics["eval_clip"]
        else:
            score = -val_metrics["total_loss"]

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
            f"alpha={alpha:.2f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
