#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.image_banks import TensorBank, default_bank_path
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, resolve_device, save_json


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
    parser = argparse.ArgumentParser(description="Run retrieval inference on train/test splits.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_predictions")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    return parser.parse_args()


def resolve_selected_channels(args: argparse.Namespace, config: dict) -> list[str] | None:
    if args.selected_channels and args.channel_preset:
        raise ValueError("--selected-channels and --channel-preset cannot both be set.")
    if args.selected_channels:
        return [str(channel) for channel in args.selected_channels]
    if args.channel_preset:
        return list(CHANNEL_PRESETS[args.channel_preset])
    configured = config.get("selected_channels")
    if configured is None:
        return None
    return [str(channel) for channel in configured]


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_alpha(args: argparse.Namespace, payload: dict, has_semantic: bool, has_perceptual: bool) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    metrics = payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    return 0.5


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]

    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    semantic_bank = resolve_bank(
        args.semantic_bank,
        config.get("semantic_bank") or config.get("clip_bank"),
        name="Semantic",
    )
    perceptual_bank = resolve_bank(args.perceptual_bank, config.get("perceptual_bank"), name="Perceptual")
    if semantic_bank is None and perceptual_bank is None:
        raise ValueError("At least one of --semantic-bank or --perceptual-bank must resolve to a bank.")

    selected_channels = resolve_selected_channels(args, config)
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
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
    outputs, query_image_ids, _ = compute_retrieval_outputs(model, loader, device)

    alpha = resolve_alpha(
        args,
        payload,
        has_semantic=semantic_bank is not None,
        has_perceptual=perceptual_bank is not None,
    )
    candidate_image_ids = query_image_ids if args.split == "train" else None
    logits, resolved_candidate_ids, component_logits = compute_retrieval_logits(
        model,
        outputs,
        semantic_bank=semantic_bank,
        perceptual_bank=perceptual_bank,
        candidate_image_ids=candidate_image_ids,
        alpha=alpha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "ordered_query_image_ids": query_image_ids,
            "candidate_image_ids": resolved_candidate_ids,
            "alpha": alpha,
            "logits": logits,
            "semantic_logits": component_logits.get("semantic"),
            "perceptual_logits": component_logits.get("perceptual"),
        },
        args.output_dir / "retrieval_logits.pt",
    )

    rankings = rank_candidate_ids(logits, resolved_candidate_ids)
    ranking_payload = [
        {
            "query_image_id": image_id,
            "ranked_candidate_ids": ranked_ids,
        }
        for image_id, ranked_ids in zip(query_image_ids, rankings, strict=True)
    ]
    save_json(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "alpha": alpha,
            "predictions": ranking_payload,
        },
        args.output_dir / "retrieval_rankings.json",
    )

    if args.evaluate:
        metrics = compute_retrieval_metrics(
            logits,
            ordered_image_ids=query_image_ids,
            candidate_image_ids=resolved_candidate_ids,
        )
        metrics["alpha"] = float(alpha)
        save_json(metrics, args.output_dir / "retrieval_metrics.json")
        print(metrics)
    else:
        print(f"saved retrieval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
