#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import load_split_image_ids, make_train_val_split
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic subset manifest for evaluation runs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-split-file", type=Path, default=None)
    parser.add_argument("--source-image-id-source", choices=["train_ids", "val_ids"], default="val_ids")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--output-image-id-source",
        choices=["train_ids", "val_ids"],
        default=None,
        help="Key used in the output manifest. Defaults to the source image-id field.",
    )
    return parser.parse_args()


def resolve_source_image_ids(args: argparse.Namespace) -> list[str]:
    if args.source_split_file is not None:
        image_ids = load_split_image_ids(
            args.source_split_file,
            image_id_source=args.source_image_id_source,
        )
        assert image_ids is not None
        return image_ids

    train_ids, val_ids = make_train_val_split(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    if args.source_image_id_source == "train_ids":
        return train_ids
    return val_ids


def main() -> None:
    args = parse_args()
    source_image_ids = resolve_source_image_ids(args)
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.count > len(source_image_ids):
        raise ValueError(
            f"Requested {args.count} ids, but only {len(source_image_ids)} are available in "
            f"{args.source_image_id_source}."
        )

    rng = random.Random(args.seed)
    selected_ids = sorted(rng.sample(list(source_image_ids), args.count))
    output_image_id_source = args.output_image_id_source or args.source_image_id_source

    payload = {
        "train_ids": [],
        "val_ids": [],
        "count": len(selected_ids),
        "seed": args.seed,
        "source_count": len(source_image_ids),
        "source_image_id_source": args.source_image_id_source,
        "source_split_file": None if args.source_split_file is None else str(args.source_split_file.resolve()),
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
    }
    payload[output_image_id_source] = selected_ids

    save_json(payload, args.output or (DEFAULT_OUTPUT_DIR / "subsets" / "subset.json"))
    print(args.output)
    print(f"{output_image_id_source}={len(selected_ids)}")


if __name__ == "__main__":
    main()
