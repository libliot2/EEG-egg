#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import eval_images
from project1_eeg.image_banks import TensorBank
from project1_eeg.reconstruction import PrototypeResidualModel
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import (
    aggregate_prototype_latents,
    compute_retrieval_logits,
    decode_latents,
    make_dataloader,
    prepare_vae,
    prototype_lookup_from_logits,
    save_image_batch,
)
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, load_image_batch, resolve_device, save_json


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
    parser = argparse.ArgumentParser(description="Generate reconstruction outputs for train/test splits.")
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--reconstruction-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--latent-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "reconstruction_predictions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--prototype-topk", type=int, default=None)
    parser.add_argument("--prototype-mode", choices=["top1", "score_weighted_topk"], default=None)
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    parser.add_argument("--channel-subset-name", type=str, default=None)
    return parser.parse_args()


def load_retrieval_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, payload


def load_reconstruction_model(checkpoint_path: Path, device: torch.device) -> tuple[PrototypeResidualModel, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    retrieval_model = build_retrieval_model_from_config(config["retrieval_config"])
    eeg_embedder = retrieval_model.build_primary_embedder()
    model = PrototypeResidualModel(
        eeg_encoder=eeg_embedder,
        embedding_dim=retrieval_model.output_dim(),
        prototype_channels=int(config.get("prototype_channels", 4)),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_alpha(
    args: argparse.Namespace,
    retrieval_payload: dict,
    reconstruction_config: dict | None,
    *,
    has_semantic: bool,
    has_perceptual: bool,
) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    if reconstruction_config is not None and "alpha" in reconstruction_config:
        return float(reconstruction_config["alpha"])
    metrics = retrieval_payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    return 0.5


def resolve_selected_channels(args: argparse.Namespace, retrieval_config: dict) -> list[str] | None:
    if args.selected_channels is not None:
        return list(args.selected_channels)
    if args.channel_preset is not None:
        return list(CHANNEL_PRESETS[args.channel_preset])
    if args.channel_subset_name is not None:
        selected_channels = retrieval_config.get("selected_channels")
        if not selected_channels:
            raise ValueError(
                f"--channel-subset-name={args.channel_subset_name} was provided, but the retrieval checkpoint "
                "does not store selected_channels."
            )
        return list(selected_channels)
    if retrieval_config.get("selected_channels") is not None:
        return list(retrieval_config["selected_channels"])
    return None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    retrieval_model, retrieval_config, retrieval_payload = load_retrieval_model(args.retrieval_checkpoint, device)

    recon_model = None
    recon_config = None
    vae = None
    if args.reconstruction_checkpoint is not None:
        recon_model, recon_config = load_reconstruction_model(args.reconstruction_checkpoint, device)
        vae = prepare_vae(recon_config["vae_model"], device)
        vae.requires_grad_(False)

    semantic_bank = resolve_bank(
        args.semantic_bank,
        retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank"),
        name="Semantic",
    )
    perceptual_bank = resolve_bank(args.perceptual_bank, retrieval_config.get("perceptual_bank"), name="Perceptual")
    if semantic_bank is None and perceptual_bank is None:
        raise ValueError("At least one retrieval bank must be available.")

    latent_bank = None
    if recon_model is not None:
        latent_fallback = None if recon_config is None else recon_config.get("latent_bank")
        latent_bank = resolve_bank(args.latent_bank, latent_fallback, name="Latent")
        if latent_bank is None:
            raise ValueError("Residual reconstruction requires a latent bank.")

    alpha = resolve_alpha(
        args,
        retrieval_payload,
        recon_config,
        has_semantic=semantic_bank is not None,
        has_perceptual=perceptual_bank is not None,
    )
    prototype_topk = args.prototype_topk
    if prototype_topk is None:
        prototype_topk = 1 if recon_config is None else int(recon_config.get("prototype_topk", 1))
    prototype_mode = args.prototype_mode
    if prototype_mode is None:
        prototype_mode = "top1" if recon_config is None else str(recon_config.get("prototype_mode", "top1"))

    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    selected_channels = resolve_selected_channels(args, retrieval_config)
    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        selected_channels=selected_channels,
        image_ids=selected_image_ids,
    )
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    saved_paths = []
    all_fake_images = []
    all_real_images = []

    effective_topk = 1 if recon_model is None or prototype_mode == "top1" else prototype_topk

    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.no_grad():
            output_bundle = retrieval_model.encode_all(eeg)
        output_dict = {
            name: tensor
            for name, tensor in {
                "semantic": output_bundle.semantic,
                "perceptual": output_bundle.perceptual,
                "legacy": output_bundle.legacy,
            }.items()
            if tensor is not None
        }
        logits, candidate_ids, _ = compute_retrieval_logits(
            retrieval_model,
            output_dict,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            alpha=alpha,
        )
        _, topk_ids, topk_scores = prototype_lookup_from_logits(
            logits,
            candidate_image_ids=candidate_ids,
            topk=effective_topk,
        )

        prototype_ids = [row[0] for row in topk_ids]
        prototype_paths = []
        source_bank = semantic_bank if semantic_bank is not None else perceptual_bank
        assert source_bank is not None
        for image_id in prototype_ids:
            prototype_paths.append(source_bank.image_paths[source_bank._index[image_id]])

        if recon_model is None:
            for query_id, ranked_ids, prototype_path in zip(batch["image_id"], topk_ids, prototype_paths, strict=True):
                dest = output_images_dir / f"{query_id}.png"
                with Image.open(prototype_path) as image:
                    image.convert("RGB").save(dest)
                saved_paths.append(str(dest))
                metadata.append(
                    {
                        "query_image_id": query_id,
                        "prototype_image_id": ranked_ids[0],
                        "prototype_topk_ids": ranked_ids,
                        "output_path": str(dest),
                        "mode": "nearest_neighbor",
                    }
                )
        else:
            assert latent_bank is not None
            prototype_latents, prototype_weights = aggregate_prototype_latents(
                latent_bank,
                topk_ids,
                topk_scores,
                device=device,
                prototype_mode=prototype_mode,
            )
            with torch.no_grad():
                pred_latents = recon_model(eeg, prototype_latents)
                pred_images = decode_latents(vae, pred_latents)
            batch_saved = save_image_batch(pred_images, list(batch["image_id"]), output_images_dir)
            saved_paths.extend(batch_saved)
            all_fake_images.extend(list(pred_images.cpu()))

            for query_id, ranked_ids, weights, output_path in zip(
                batch["image_id"],
                topk_ids,
                prototype_weights.cpu().tolist(),
                batch_saved,
                strict=True,
            ):
                metadata.append(
                    {
                        "query_image_id": query_id,
                        "prototype_image_id": ranked_ids[0],
                        "prototype_topk_ids": ranked_ids,
                        "prototype_topk_weights": weights,
                        "output_path": output_path,
                        "mode": "residual",
                    }
                )

        if args.evaluate:
            real_images = load_image_batch(batch["image_path"], image_size=args.image_size)
            all_real_images.extend(list(real_images.cpu()))
            if recon_model is None:
                fake_images = load_image_batch(prototype_paths, image_size=args.image_size)
                all_fake_images.extend(list(fake_images.cpu()))

    save_json(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "alpha": alpha,
            "prototype_topk": effective_topk,
            "prototype_mode": "top1" if recon_model is None else prototype_mode,
            "predictions": metadata,
        },
        args.output_dir / "reconstruction_metadata.json",
    )

    if args.evaluate:
        metrics = eval_images(
            real_images=torch.stack(all_real_images, dim=0),
            fake_images=torch.stack(all_fake_images, dim=0),
            device=device,
        )
        metrics["alpha"] = float(alpha)
        metrics["prototype_topk"] = float(effective_topk)
        save_json(metrics, args.output_dir / "reconstruction_metrics.json")
        print(metrics)
    else:
        print(f"saved reconstruction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
