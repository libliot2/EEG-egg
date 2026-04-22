#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, make_train_val_split, ordered_image_ids
from project1_eeg.retrieval import RetrievalModel, weighted_retrieval_loss
from project1_eeg.runtime import compute_retrieval_metrics, make_dataloader
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
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval with online DreamSim image encoder.")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_dreamsim_online")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--channel-dropout", type=float, default=0.1)
    parser.add_argument("--time-mask-ratio", type=float, default=0.1)
    parser.add_argument("--encoder-type", choices=["atm_small", "atm_base", "atm_large", "atm_spatial", "atm_multiscale", "eeg_conformer"], default="atm_large")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-eeg-perturbation", action="store_true")
    parser.add_argument("--use-perceptual-target-adapter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-adapter-hidden-dim", type=int, default=None)
    parser.add_argument("--target-adapter-dropout", type=float, default=None)
    parser.add_argument("--target-adapter-beta", type=float, default=None)
    parser.add_argument("--target-adapter-loss-weight", type=float, default=None)
    parser.add_argument("--retrieval-loss-type", choices=["clip", "neuroclip"], default="neuroclip")
    parser.add_argument("--soft-target-beta", type=float, default=10.0)
    parser.add_argument("--soft-target-source", choices=["blend", "image", "eeg"], default="blend")
    parser.add_argument("--clip-loss-coef", type=float, default=1.0)
    parser.add_argument("--soft-loss-coef", type=float, default=0.2)
    parser.add_argument("--relation-loss-coef", type=float, default=0.0)
    parser.add_argument("--eeg-to-image-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-to-eeg-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-loss-coef", type=float, default=0.0)
    parser.add_argument("--hard-negative-topk", type=int, default=0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default="visual17")
    parser.add_argument("--dreamsim-type", type=str, default="ensemble")
    parser.add_argument("--dreamsim-train-mode", choices=["lora_only", "full"], default="lora_only")
    parser.add_argument("--freeze-eeg-encoder", action="store_true")
    parser.add_argument("--freeze-retrieval-model", action="store_true")
    parser.add_argument("--unfreeze-eeg-projection", action="store_true")
    parser.add_argument("--unfreeze-eeg-perturbation", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--evaluate-init", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-candidates", type=int, default=None)
    return parser.parse_args()


def resolve_selected_channels(args: argparse.Namespace) -> list[str] | None:
    if args.selected_channels and args.channel_preset:
        raise ValueError("--selected-channels and --channel-preset cannot both be set.")
    if args.selected_channels:
        return [str(channel) for channel in args.selected_channels]
    if args.channel_preset:
        return list(CHANNEL_PRESETS[args.channel_preset])
    return None


def load_images(image_paths: list[str], preprocess, device: torch.device) -> torch.Tensor:
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensor = preprocess(image.convert("RGB"))
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0)
    return batch.to(device, non_blocking=device.type == "cuda")


def configure_dreamsim_trainability(model, *, train_mode: str) -> None:
    model.requires_grad_(False)
    base = model.get_base_model()
    if train_mode == "full":
        base.requires_grad_(True)
        return
    if train_mode != "lora_only":
        raise ValueError(f"Unsupported dreamsim train mode: {train_mode}")
    for name, parameter in base.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def apply_retrieval_freeze_options(
    model: RetrievalModel,
    *,
    freeze_eeg_encoder: bool,
    freeze_retrieval_model: bool,
    unfreeze_eeg_projection: bool,
    unfreeze_eeg_perturbation: bool,
) -> None:
    if freeze_retrieval_model:
        for parameter in model.parameters():
            parameter.requires_grad = False
        return
    if freeze_eeg_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        if unfreeze_eeg_projection and hasattr(model.encoder, "projection"):
            for parameter in model.encoder.projection.parameters():
                parameter.requires_grad = True
        if unfreeze_eeg_perturbation and hasattr(model.encoder, "eeg_perturbation"):
            for parameter in model.encoder.eeg_perturbation.parameters():
                parameter.requires_grad = True


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in keys}


def set_retrieval_train_mode(
    model: RetrievalModel,
    *,
    freeze_retrieval_model: bool,
    freeze_eeg_encoder: bool,
) -> None:
    model.train()
    if freeze_retrieval_model:
        model.eval()
        return
    if freeze_eeg_encoder:
        model.encoder.eval()


def resolve_model_hparams(args: argparse.Namespace, checkpoint_config: dict[str, object]) -> dict[str, object]:
    def get_value(name: str, default):
        value = getattr(args, name)
        if value is not None:
            return value
        return checkpoint_config.get(name, default)

    return {
        "use_perceptual_target_adapter": bool(get_value("use_perceptual_target_adapter", False)),
        "target_adapter_hidden_dim": int(get_value("target_adapter_hidden_dim", 1024)),
        "target_adapter_dropout": float(get_value("target_adapter_dropout", 0.1)),
        "target_adapter_beta": float(get_value("target_adapter_beta", 0.1)),
        "target_adapter_loss_weight": float(get_value("target_adapter_loss_weight", 0.0)),
    }


def evaluate_online(
    retrieval_model: RetrievalModel,
    dreamsim_model,
    preprocess,
    loader,
    *,
    candidate_records,
    candidate_batch_size: int,
    device: torch.device,
    max_val_candidates: int | None = None,
) -> dict[str, float]:
    retrieval_model.eval()
    dreamsim_model.eval()

    outputs: list[torch.Tensor] = []
    query_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            encoded = retrieval_model.encode_all(eeg)
            outputs.append(encoded.perceptual.float().cpu())
            query_ids.extend(list(batch["image_id"]))

        candidate_embeddings: list[torch.Tensor] = []
        candidate_ids: list[str] = []
        base_model = dreamsim_model.get_base_model()
        resolved_candidates = candidate_records if max_val_candidates is None else candidate_records[:max_val_candidates]
        for start in range(0, len(resolved_candidates), candidate_batch_size):
            chunk = resolved_candidates[start : start + candidate_batch_size]
            batch_images = load_images([record.image_path for record in chunk], preprocess, device)
            embeds = base_model.embed(batch_images).float()
            embeds = torch.nn.functional.normalize(embeds, dim=-1).cpu()
            candidate_embeddings.append(embeds)
            candidate_ids.extend([record.image_id for record in chunk])

    query_embeddings = torch.cat(outputs, dim=0).to(device)
    candidate_matrix = torch.cat(candidate_embeddings, dim=0).to(device)
    valid_mask = [image_id in set(candidate_ids) for image_id in query_ids]
    filtered_query_ids = [image_id for image_id, keep in zip(query_ids, valid_mask, strict=True) if keep]
    filtered_query_embeddings = query_embeddings[torch.tensor(valid_mask, dtype=torch.bool, device=query_embeddings.device)]
    logits = retrieval_model.similarity(filtered_query_embeddings, candidate_matrix, head="perceptual").cpu()
    metrics = compute_retrieval_metrics(logits, ordered_image_ids=filtered_query_ids, candidate_image_ids=candidate_ids)
    return {
        "top1_acc": float(metrics["top1_acc"]),
        "top5_acc": float(metrics["top5_acc"]),
        "selected_alpha": 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    selected_channels = resolve_selected_channels(args)

    from dreamsim import dreamsim

    dreamsim_model, preprocess = dreamsim(
        pretrained=True,
        device=str(device),
        dreamsim_type=args.dreamsim_type,
    )
    checkpoint_payload = None
    checkpoint_config: dict[str, object] = {}
    if args.init_checkpoint is not None:
        checkpoint_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = dict(checkpoint_payload.get("config", {}))

    train_ids, val_ids = make_train_val_split(data_dir=args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
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

    sample_eeg = train_records[0].eeg
    in_channels = int(sample_eeg.shape[0])
    perceptual_dim = int(dreamsim_model.get_base_model().embed(load_images([train_records[0].image_path], preprocess, device)).shape[-1])
    resolved_model_hparams = resolve_model_hparams(args, checkpoint_config)
    retrieval_model = RetrievalModel(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        channel_dropout=args.channel_dropout,
        time_mask_ratio=args.time_mask_ratio,
        encoder_type=args.encoder_type,
        semantic_dim=None,
        perceptual_dim=perceptual_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        use_eeg_perturbation=args.use_eeg_perturbation,
        use_perceptual_target_adapter=bool(resolved_model_hparams["use_perceptual_target_adapter"]),
        target_adapter_hidden_dim=int(resolved_model_hparams["target_adapter_hidden_dim"]),
        target_adapter_dropout=float(resolved_model_hparams["target_adapter_dropout"]),
        target_adapter_beta=float(resolved_model_hparams["target_adapter_beta"]),
    ).to(device)
    if args.init_checkpoint is not None:
        retrieval_model.load_state_dict(checkpoint_payload["model_state"], strict=False)
        if "dreamsim_state" in checkpoint_payload:
            dreamsim_model.load_state_dict(checkpoint_payload["dreamsim_state"], strict=False)

    configure_dreamsim_trainability(dreamsim_model, train_mode=args.dreamsim_train_mode)
    apply_retrieval_freeze_options(
        retrieval_model,
        freeze_eeg_encoder=args.freeze_eeg_encoder,
        freeze_retrieval_model=args.freeze_retrieval_model,
        unfreeze_eeg_projection=args.unfreeze_eeg_projection,
        unfreeze_eeg_perturbation=args.unfreeze_eeg_perturbation,
    )

    parameters = [parameter for parameter in list(retrieval_model.parameters()) + list(dreamsim_model.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume_optimizer and checkpoint_payload is not None and checkpoint_payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint_payload["optimizer_state"])

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "retrieval_dreamsim_online",
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "encoder_type": args.encoder_type,
        "in_channels": in_channels,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "channel_dropout": args.channel_dropout,
        "time_mask_ratio": args.time_mask_ratio,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "use_eeg_perturbation": bool(args.use_eeg_perturbation),
        "perceptual_dim": perceptual_dim,
        "selected_channels": selected_channels,
        "dreamsim_type": args.dreamsim_type,
        "dreamsim_train_mode": args.dreamsim_train_mode,
        "use_perceptual_target_adapter": bool(resolved_model_hparams["use_perceptual_target_adapter"]),
        "target_adapter_hidden_dim": int(resolved_model_hparams["target_adapter_hidden_dim"]),
        "target_adapter_dropout": float(resolved_model_hparams["target_adapter_dropout"]),
        "target_adapter_beta": float(resolved_model_hparams["target_adapter_beta"]),
        "target_adapter_loss_weight": float(resolved_model_hparams["target_adapter_loss_weight"]),
        "retrieval_loss_type": args.retrieval_loss_type,
        "soft_target_beta": args.soft_target_beta,
        "soft_target_source": args.soft_target_source,
        "clip_loss_coef": args.clip_loss_coef,
        "soft_loss_coef": args.soft_loss_coef,
        "relation_loss_coef": args.relation_loss_coef,
        "eeg_to_image_loss_weight": args.eeg_to_image_loss_weight,
        "image_to_eeg_loss_weight": args.image_to_eeg_loss_weight,
        "hard_negative_loss_coef": args.hard_negative_loss_coef,
        "hard_negative_topk": args.hard_negative_topk,
        "hard_negative_margin": args.hard_negative_margin,
        "freeze_eeg_encoder": bool(args.freeze_eeg_encoder),
        "freeze_retrieval_model": bool(args.freeze_retrieval_model),
        "unfreeze_eeg_projection": bool(args.unfreeze_eeg_projection),
        "unfreeze_eeg_perturbation": bool(args.unfreeze_eeg_perturbation),
    }
    save_json(config, run_dir / "config.json")
    save_json({"train_ids": train_ids, "val_ids": val_ids}, run_dir / "split.json")

    best_top1 = float("-inf")
    history: list[dict[str, float]] = []
    start_time = time.time()

    if args.evaluate_init:
        init_metrics = evaluate_online(
            retrieval_model,
            dreamsim_model,
            preprocess,
            val_loader,
            candidate_records=val_records,
            candidate_batch_size=args.candidate_batch_size,
            device=device,
            max_val_candidates=args.max_val_candidates,
        )
        init_payload = {
            "epoch": 0.0,
            "val_top1": float(init_metrics["top1_acc"]),
            "val_top5": float(init_metrics["top5_acc"]),
            "val_blend_top1_top5": float(0.5 * init_metrics["top1_acc"] + 0.5 * init_metrics["top5_acc"]),
            "val_selected_alpha": 0.0,
        }
        save_json(init_payload, run_dir / "init_eval.json")
        print(
            f"epoch=000 "
            f"val_top1={init_payload['val_top1']:.4f} "
            f"val_top5={init_payload['val_top5']:.4f} "
            f"blend={init_payload['val_blend_top1_top5']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        set_retrieval_train_mode(
            retrieval_model,
            freeze_retrieval_model=args.freeze_retrieval_model,
            freeze_eeg_encoder=args.freeze_eeg_encoder,
        )
        dreamsim_model.train()
        metrics: list[dict[str, float]] = []
        base_model = dreamsim_model.get_base_model()

        for batch_idx, batch in enumerate(train_loader, start=1):
            eeg = batch["eeg"].to(device)
            images = load_images(list(batch["image_path"]), preprocess, device)
            perceptual_targets = torch.nn.functional.normalize(base_model.embed(images).float(), dim=-1)
            outputs = retrieval_model.encode_all(eeg)
            loss, batch_metrics = weighted_retrieval_loss(
                retrieval_model,
                outputs,
                perceptual_targets=perceptual_targets,
                semantic_targets=None,
                semantic_weight=0.0,
                perceptual_weight=1.0,
                retrieval_loss_type=args.retrieval_loss_type,
                soft_target_beta=args.soft_target_beta,
                soft_target_source=args.soft_target_source,
                clip_loss_coef=args.clip_loss_coef,
                soft_loss_coef=args.soft_loss_coef,
                relation_loss_coef=args.relation_loss_coef,
                eeg_to_image_weight=args.eeg_to_image_loss_weight,
                image_to_eeg_weight=args.image_to_eeg_loss_weight,
                hard_negative_loss_coef=args.hard_negative_loss_coef,
                hard_negative_topk=args.hard_negative_topk,
                hard_negative_margin=args.hard_negative_margin,
                target_adapter_loss_weight=float(resolved_model_hparams["target_adapter_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics.append(batch_metrics)
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        train_metrics = average_metrics(metrics)
        val_metrics = evaluate_online(
            retrieval_model,
            dreamsim_model,
            preprocess,
            val_loader,
            candidate_records=val_records,
            candidate_batch_size=args.candidate_batch_size,
            device=device,
            max_val_candidates=args.max_val_candidates,
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_total_loss": float(train_metrics.get("total_loss", 0.0)),
            "val_top1": float(val_metrics["top1_acc"]),
            "val_top5": float(val_metrics["top5_acc"]),
            "val_blend_top1_top5": float(0.5 * val_metrics["top1_acc"] + 0.5 * val_metrics["top5_acc"]),
            "val_selected_alpha": 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        for key, value in train_metrics.items():
            if key == "total_loss":
                continue
            metric_key = f"train_{key}"
            if metric_key not in epoch_metrics:
                epoch_metrics[metric_key] = float(value)
        history.append(epoch_metrics)
        save_json({"history": history}, run_dir / "history.json")
        save_checkpoint(
            run_dir / "last.pt",
            model_state=retrieval_model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state={},
            config=config,
            metrics=epoch_metrics,
            extra_state={"dreamsim_state": dreamsim_model.state_dict()},
        )
        if epoch_metrics["val_top1"] > best_top1:
            best_top1 = epoch_metrics["val_top1"]
            save_checkpoint(
                run_dir / "best.pt",
                model_state=retrieval_model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state={},
                config=config,
                metrics=epoch_metrics,
                extra_state={"dreamsim_state": dreamsim_model.state_dict()},
            )

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"epoch={epoch:03d} "
            f"train_total={epoch_metrics['train_total_loss']:.4f} "
            f"val_top1={epoch_metrics['val_top1']:.4f} "
            f"val_top5={epoch_metrics['val_top5']:.4f} "
            f"blend={epoch_metrics['val_blend_top1_top5']:.4f} "
            f"elapsed={elapsed}"
        )


if __name__ == "__main__":
    main()
