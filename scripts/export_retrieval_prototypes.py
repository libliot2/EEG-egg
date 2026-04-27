#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import load_split_image_ids
from project1_eeg.image_banks import TensorBank
from project1_eeg.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-1 prototype images from retrieval logits.")
    parser.add_argument("--logit-files", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--normalize", choices=["none", "zscore", "l2", "softmax"], default="zscore")
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"ordered_query_image_ids", "candidate_image_ids", "logits"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")
    return payload


def normalize_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    values = logits.float()
    if mode == "none":
        return values
    if mode == "zscore":
        return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)
    if mode == "l2":
        return values / values.norm(dim=1, keepdim=True).clamp_min(1e-6)
    if mode == "softmax":
        return torch.softmax(values, dim=1)
    raise ValueError(f"Unknown normalization mode: {mode}")


def resolve_weights(weights: list[float] | None, count: int) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}.")
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        raise ValueError("--weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def resolve_image_path(bank: TensorBank, image_id: str, *, data_dir: Path | None = None) -> Path:
    path = Path(bank.image_paths[bank._index[image_id]])
    if path.exists():
        return path
    if data_dir is not None:
        for split_dir in ("training_images", "test_images"):
            root = data_dir / split_dir
            if not root.exists():
                continue
            matches = sorted(root.glob(f"*/{image_id}.*"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Prototype image path does not exist for {image_id}: {path}")


def main() -> None:
    args = parse_args()
    payloads = [load_payload(path) for path in args.logit_files]
    query_ids = list(payloads[0]["ordered_query_image_ids"])
    candidate_ids = list(payloads[0]["candidate_image_ids"])
    for path, payload in zip(args.logit_files[1:], payloads[1:], strict=True):
        if list(payload["ordered_query_image_ids"]) != query_ids:
            raise ValueError(f"Query ids mismatch: {path}")
        if list(payload["candidate_image_ids"]) != candidate_ids:
            raise ValueError(f"Candidate ids mismatch: {path}")

    selected_query_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    if selected_query_ids is not None:
        query_index = {image_id: idx for idx, image_id in enumerate(query_ids)}
        row_indices = torch.tensor([query_index[image_id] for image_id in selected_query_ids], dtype=torch.long)
        query_ids = list(selected_query_ids)
    else:
        row_indices = torch.arange(len(query_ids), dtype=torch.long)

    weights = resolve_weights(args.weights, len(payloads))
    fused = torch.zeros((len(row_indices), len(candidate_ids)), dtype=torch.float32)
    for weight, payload in zip(weights, payloads, strict=True):
        logits = torch.as_tensor(payload["logits"]).index_select(0, row_indices)
        fused += float(weight) * normalize_logits(logits, args.normalize)

    if args.exclude_self:
        candidate_index = {image_id: idx for idx, image_id in enumerate(candidate_ids)}
        for row_idx, image_id in enumerate(query_ids):
            col_idx = candidate_index.get(image_id)
            if col_idx is not None:
                fused[row_idx, col_idx] = -torch.inf

    top_indices = fused.argmax(dim=1).tolist()
    source_bank = TensorBank.load(args.source_bank)
    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for query_id, candidate_index in zip(query_ids, top_indices, strict=True):
        prototype_id = candidate_ids[candidate_index]
        source_path = resolve_image_path(source_bank, prototype_id, data_dir=args.data_dir)
        output_path = output_images_dir / f"{query_id}.png"
        with Image.open(source_path) as image:
            image.convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC).save(output_path)
        predictions.append(
            {
                "query_image_id": query_id,
                "prototype_image_id": prototype_id,
                "prototype_source_path": str(source_path),
                "prototype_score": float(fused[len(predictions), candidate_index]),
                "output_path": str(output_path),
            }
        )

    save_json(
        {
            "logit_files": [str(path) for path in args.logit_files],
            "weights": weights,
            "normalize": args.normalize,
            "source_bank": str(args.source_bank),
            "data_dir": None if args.data_dir is None else str(args.data_dir),
            "split_file": None if args.split_file is None else str(args.split_file),
            "image_id_source": args.image_id_source,
            "exclude_self": bool(args.exclude_self),
            "image_size": int(args.image_size),
            "predictions": predictions,
        },
        args.output_dir / "prototype_metadata.json",
    )
    print(f"exported {len(predictions)} prototypes to {output_images_dir}")


if __name__ == "__main__":
    main()
