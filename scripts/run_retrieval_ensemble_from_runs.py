#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.utils import DEFAULT_OUTPUT_DIR, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate val/test retrieval logits for multiple runs, select ensemble weights on val, and evaluate once on test."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize", choices=["none", "zscore", "l2", "softmax"], default="zscore")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--test-semantic-bank", type=Path, default=None)
    parser.add_argument("--test-perceptual-bank", type=Path, default=DEFAULT_OUTPUT_DIR / "cache" / "dreamsim_test.pt")
    return parser.parse_args()


def normalize_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    values = logits.float()
    if mode == "none":
        return values
    if mode == "zscore":
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (values - mean) / std
    if mode == "l2":
        norm = values.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return values / norm
    if mode == "softmax":
        return torch.softmax(values, dim=1)
    raise ValueError(f"Unknown normalize mode: {mode}")


def resolve_test_bank(train_bank: str | None, override: Path | None, *, name: str) -> Path | None:
    if override is not None:
        return override
    if train_bank is None:
        return None

    train_path = Path(train_bank)
    if "_train" in train_path.name:
        candidate = train_path.with_name(train_path.name.replace("_train", "_test", 1))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer {name} test bank from {train_path}. Pass --test-{name}-bank explicitly."
    )


def run_predict(
    *,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str | None,
    semantic_bank: Path | None,
    perceptual_bank: Path | None,
    split_file: Path | None = None,
    image_id_source: str = "all",
) -> None:
    cmd = [
        sys.executable,
        "scripts/predict_retrieval.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--split",
        split,
        "--evaluate",
    ]
    if device is not None:
        cmd.extend(["--device", device])
    if semantic_bank is not None:
        cmd.extend(["--semantic-bank", str(semantic_bank)])
    if perceptual_bank is not None:
        cmd.extend(["--perceptual-bank", str(perceptual_bank)])
    if split_file is not None:
        cmd.extend(["--split-file", str(split_file), "--image-id-source", image_id_source])
    subprocess.run(cmd, check=True)


def ensure_logits(
    *,
    run_dir: Path,
    checkpoint_name: str,
    batch_size: int,
    num_workers: int,
    device: str | None,
    force_predict: bool,
    test_semantic_bank: Path | None,
    test_perceptual_bank: Path | None,
) -> tuple[Path, Path]:
    checkpoint = run_dir / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = load_json(run_dir / "config.json")
    split_file = run_dir / "split.json"
    checkpoint_stem = Path(checkpoint_name).stem
    val_output = run_dir / f"val_eval_{checkpoint_stem}"
    test_output = run_dir / f"test_eval_{checkpoint_stem}"

    val_logits = val_output / "retrieval_logits.pt"
    test_logits = test_output / "retrieval_logits.pt"

    if force_predict or not val_logits.exists():
        run_predict(
            checkpoint=checkpoint,
            output_dir=val_output,
            split="train",
            split_file=split_file,
            image_id_source="val_ids",
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            semantic_bank=None if config.get("semantic_bank") is None else Path(config["semantic_bank"]),
            perceptual_bank=None if config.get("perceptual_bank") is None else Path(config["perceptual_bank"]),
        )

    if force_predict or not test_logits.exists():
        run_predict(
            checkpoint=checkpoint,
            output_dir=test_output,
            split="test",
            split_file=None,
            image_id_source="all",
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            semantic_bank=test_semantic_bank,
            perceptual_bank=test_perceptual_bank,
        )

    return val_logits, test_logits


def load_logits(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def check_compatibility(payloads: list[dict], paths: list[Path]) -> tuple[list[str], list[str], str]:
    reference = payloads[0]
    query_ids = list(reference["ordered_query_image_ids"])
    candidate_ids = list(reference["candidate_image_ids"])
    split = str(reference.get("split", "unknown"))
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query ordering mismatch in {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate ordering mismatch in {path}")
        if str(payload.get("split", "unknown")) != split:
            raise ValueError(f"Split mismatch in {path}")
    return query_ids, candidate_ids, split


def search_two_way_weights(
    logits_list: list[torch.Tensor],
    *,
    ordered_image_ids: list[str],
    candidate_image_ids: list[str],
    step: float,
) -> tuple[list[float], dict[str, float], list[dict[str, float]]]:
    if len(logits_list) != 2:
        raise ValueError("This script currently supports exactly two runs.")
    if step <= 0.0 or step > 1.0:
        raise ValueError("--grid-step must be in (0, 1].")

    num_steps = int(round(1.0 / step))
    candidates = [round(index * step, 10) for index in range(num_steps + 1)]
    if candidates[-1] != 1.0:
        candidates.append(1.0)

    best_weights = [0.5, 0.5]
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    history: list[dict[str, float]] = []

    for alpha in candidates:
        weights = [float(alpha), float(1.0 - alpha)]
        fused = weights[0] * logits_list[0] + weights[1] * logits_list[1]
        metrics = compute_retrieval_metrics(
            fused,
            ordered_image_ids=ordered_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        history.append({"weight_0": weights[0], "weight_1": weights[1], **metrics})
        score = (
            float(metrics["top1_acc"]),
            float(metrics["top5_acc"]),
            -abs(weights[0] - 0.5),
        )
        if best_score is None or score > best_score:
            best_weights = weights
            best_metrics = metrics
            best_score = score

    assert best_metrics is not None
    return best_weights, best_metrics, history


def main() -> None:
    args = parse_args()
    if len(args.run_dirs) != 2:
        raise ValueError("--run-dirs currently requires exactly two runs.")

    first_config = load_json(args.run_dirs[0] / "config.json")
    test_semantic_bank = resolve_test_bank(
        first_config.get("semantic_bank"),
        args.test_semantic_bank,
        name="semantic",
    )
    test_perceptual_bank = resolve_test_bank(
        first_config.get("perceptual_bank"),
        args.test_perceptual_bank,
        name="perceptual",
    )

    val_logit_files: list[Path] = []
    test_logit_files: list[Path] = []
    for run_dir in args.run_dirs:
        val_logits, test_logits = ensure_logits(
            run_dir=run_dir,
            checkpoint_name=args.checkpoint_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            force_predict=args.force_predict,
            test_semantic_bank=test_semantic_bank,
            test_perceptual_bank=test_perceptual_bank,
        )
        val_logit_files.append(val_logits)
        test_logit_files.append(test_logits)

    val_payloads = [load_logits(path) for path in val_logit_files]
    val_query_ids, val_candidate_ids, val_split = check_compatibility(val_payloads, val_logit_files)
    val_logits = [normalize_logits(torch.as_tensor(payload["logits"]), args.normalize) for payload in val_payloads]
    weights, val_metrics, grid_history = search_two_way_weights(
        val_logits,
        ordered_image_ids=val_query_ids,
        candidate_image_ids=val_candidate_ids,
        step=args.grid_step,
    )

    test_payloads = [load_logits(path) for path in test_logit_files]
    test_query_ids, test_candidate_ids, test_split = check_compatibility(test_payloads, test_logit_files)
    test_logits = [normalize_logits(torch.as_tensor(payload["logits"]), args.normalize) for payload in test_payloads]
    fused_test = weights[0] * test_logits[0] + weights[1] * test_logits[1]
    test_metrics = compute_retrieval_metrics(
        fused_test,
        ordered_image_ids=test_query_ids,
        candidate_image_ids=test_candidate_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": test_split,
            "ordered_query_image_ids": test_query_ids,
            "candidate_image_ids": test_candidate_ids,
            "weights": weights,
            "normalize": args.normalize,
            "selection_split": val_split,
            "selection_metrics": val_metrics,
            "source_logit_files": [str(path) for path in test_logit_files],
            "selection_logit_files": [str(path) for path in val_logit_files],
            "logits": fused_test,
        },
        args.output_dir / "retrieval_logits.pt",
    )
    rankings = rank_candidate_ids(fused_test, test_candidate_ids)
    save_json(
        {
            "split": test_split,
            "weights": weights,
            "normalize": args.normalize,
            "selection_split": val_split,
            "selection_metrics": val_metrics,
            "source_logit_files": [str(path) for path in test_logit_files],
            "selection_logit_files": [str(path) for path in val_logit_files],
            "predictions": [
                {
                    "query_image_id": image_id,
                    "ranked_candidate_ids": ranked_ids,
                }
                for image_id, ranked_ids in zip(test_query_ids, rankings, strict=True)
            ],
        },
        args.output_dir / "retrieval_rankings.json",
    )
    save_json(
        {
            "split": test_split,
            "weights": weights,
            "normalize": args.normalize,
            "selection_split": val_split,
            "selection_metrics": val_metrics,
            "selection_grid_history": grid_history,
            "source_logit_files": [str(path) for path in test_logit_files],
            "selection_logit_files": [str(path) for path in val_logit_files],
            **test_metrics,
        },
        args.output_dir / "retrieval_metrics.json",
    )
    print(
        {
            "selection_split": val_split,
            "selection_metrics": val_metrics,
            "test_split": test_split,
            "weights": weights,
            "normalize": args.normalize,
            **test_metrics,
        }
    )


if __name__ == "__main__":
    main()
