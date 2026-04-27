#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import (
    DEFAULT_CHANNEL_SUBSET_FILE,
    EEGImageDataset,
    load_eeg_records,
    load_split_image_ids,
    resolve_channel_subset,
)
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
    parser.add_argument(
        "--candidate-source",
        choices=["auto", "bank"],
        default="auto",
        help="Use query ids as train candidates by default; use bank to rank against the full embedding bank.",
    )
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    parser.add_argument("--channel-subset-name", type=str, default=None)
    parser.add_argument("--channel-subset-file", type=Path, default=DEFAULT_CHANNEL_SUBSET_FILE)
    parser.add_argument("--tta-trial-views", type=int, default=1)
    parser.add_argument("--tta-trial-k", type=int, default=None)
    parser.add_argument("--tta-trial-k-min", type=int, default=8)
    parser.add_argument("--tta-trial-k-max", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_selected_channels(args: argparse.Namespace, config: dict) -> tuple[str | None, Path, list[str] | None]:
    explicit_sources = sum(
        bool(value)
        for value in (args.selected_channels, args.channel_preset, args.channel_subset_name)
    )
    if explicit_sources > 1:
        raise ValueError("--selected-channels, --channel-preset, and --channel-subset-name are mutually exclusive.")
    if args.selected_channels:
        return None, args.channel_subset_file, [str(channel) for channel in args.selected_channels]
    if args.channel_preset:
        return None, args.channel_subset_file, list(CHANNEL_PRESETS[args.channel_preset])
    subset_name = args.channel_subset_name or config.get("channel_subset_name")
    subset_file = args.channel_subset_file
    config_subset_file = config.get("channel_subset_file")
    if config_subset_file and args.channel_subset_file == DEFAULT_CHANNEL_SUBSET_FILE:
        subset_file = Path(config_subset_file)
    if subset_name:
        return subset_name, subset_file, resolve_channel_subset(subset_name, subset_file=subset_file)
    configured = config.get("selected_channels")
    if configured is None:
        return None, subset_file, None
    return subset_name, subset_file, [str(channel) for channel in configured]


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_split_bank_path(path: Path, *, split: str) -> Path:
    suffix_map = {
        "train": ("_test.pt", "_train.pt"),
        "test": ("_train.pt", "_test.pt"),
    }
    from_suffix, to_suffix = suffix_map.get(split, (None, None))
    if from_suffix is None or not path.name.endswith(from_suffix):
        return path
    candidate = path.with_name(path.name[: -len(from_suffix)] + to_suffix)
    return candidate if candidate.exists() else path


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


def average_component_logits(
    accumulated: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    *,
    weight: float,
) -> dict[str, torch.Tensor]:
    for name, logits in current.items():
        value = logits.float() * weight
        if name in accumulated:
            accumulated[name] = accumulated[name] + value
        else:
            accumulated[name] = value
    return accumulated


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]

    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    semantic_fallback = config.get("semantic_bank") or config.get("clip_bank")
    perceptual_fallback = config.get("perceptual_bank")
    if args.semantic_bank is None and semantic_fallback is not None:
        semantic_fallback = str(resolve_split_bank_path(Path(str(semantic_fallback)), split=args.split))
    if args.perceptual_bank is None and perceptual_fallback is not None:
        perceptual_fallback = str(resolve_split_bank_path(Path(str(perceptual_fallback)), split=args.split))
    semantic_bank = resolve_bank(args.semantic_bank, semantic_fallback, name="Semantic")
    perceptual_bank = resolve_bank(args.perceptual_bank, perceptual_fallback, name="Perceptual")
    if semantic_bank is None and perceptual_bank is None:
        raise ValueError("At least one of --semantic-bank or --perceptual-bank must resolve to a bank.")

    channel_subset_name, channel_subset_file, selected_channels = resolve_selected_channels(args, config)
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    use_trial_tta = args.tta_trial_views > 1 or args.tta_trial_k is not None
    if args.tta_trial_views < 1:
        raise ValueError("--tta-trial-views must be >= 1.")
    if args.tta_trial_k is not None and args.tta_trial_k < 1:
        raise ValueError("--tta-trial-k must be >= 1.")

    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=not use_trial_tta,
        preserve_trials=use_trial_tta,
        selected_channels=selected_channels,
        image_ids=selected_image_ids,
    )

    alpha = resolve_alpha(
        args,
        payload,
        has_semantic=semantic_bank is not None,
        has_perceptual=perceptual_bank is not None,
    )
    logits_accum: torch.Tensor | None = None
    component_logits_accum: dict[str, torch.Tensor] = {}
    query_image_ids: list[str] | None = None
    resolved_candidate_ids: list[str] | None = None
    view_weight = 1.0 / float(args.tta_trial_views)

    for view_idx in range(args.tta_trial_views):
        if use_trial_tta:
            torch.manual_seed(int(args.seed) + view_idx)
        trial_k_min = args.tta_trial_k if args.tta_trial_k is not None else args.tta_trial_k_min
        trial_k_max = args.tta_trial_k if args.tta_trial_k is not None else args.tta_trial_k_max
        loader = make_dataloader(
            EEGImageDataset(
                records,
                trial_sampling="random_avg" if use_trial_tta else "none",
                trial_k_min=trial_k_min,
                trial_k_max=trial_k_max,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        outputs, view_query_image_ids, _ = compute_retrieval_outputs(model, loader, device)
        if query_image_ids is None:
            query_image_ids = view_query_image_ids
        elif query_image_ids != view_query_image_ids:
            raise ValueError("Query image ordering changed across trial-TTA views.")
        view_candidate_image_ids = query_image_ids if args.split == "train" and args.candidate_source == "auto" else None
        view_logits, view_candidate_ids, view_component_logits = compute_retrieval_logits(
            model,
            outputs,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            candidate_image_ids=view_candidate_image_ids,
            alpha=alpha,
        )
        if resolved_candidate_ids is None:
            resolved_candidate_ids = view_candidate_ids
        elif resolved_candidate_ids != view_candidate_ids:
            raise ValueError("Candidate image ordering changed across trial-TTA views.")
        weighted_logits = view_logits.float() * view_weight
        logits_accum = weighted_logits if logits_accum is None else logits_accum + weighted_logits
        component_logits_accum = average_component_logits(
            component_logits_accum,
            view_component_logits,
            weight=view_weight,
        )

    assert query_image_ids is not None
    assert resolved_candidate_ids is not None
    assert logits_accum is not None
    logits = logits_accum
    component_logits = component_logits_accum

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "candidate_source": args.candidate_source,
            "channel_subset_name": channel_subset_name,
            "channel_subset_file": str(channel_subset_file),
            "selected_channels": selected_channels,
            "tta_trial_views": int(args.tta_trial_views),
            "tta_trial_k": args.tta_trial_k,
            "tta_trial_k_min": int(args.tta_trial_k_min),
            "tta_trial_k_max": int(args.tta_trial_k_max),
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
            "candidate_source": args.candidate_source,
            "channel_subset_name": channel_subset_name,
            "channel_subset_file": str(channel_subset_file),
            "selected_channels": selected_channels,
            "tta_trial_views": int(args.tta_trial_views),
            "tta_trial_k": args.tta_trial_k,
            "tta_trial_k_min": int(args.tta_trial_k_min),
            "tta_trial_k_max": int(args.tta_trial_k_max),
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
        metrics["channel_subset_name"] = channel_subset_name
        metrics["tta_trial_views"] = int(args.tta_trial_views)
        metrics["tta_trial_k"] = args.tta_trial_k
        save_json(metrics, args.output_dir / "retrieval_metrics.json")
        print(metrics)
    else:
        print(f"saved retrieval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
