#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend two image directories by matching file names.")
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.first_dir.exists():
        raise FileNotFoundError(args.first_dir)
    if not args.second_dir.exists():
        raise FileNotFoundError(args.second_dir)
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError("--first-weight must be in [0, 1].")

    second_weight = 1.0 - float(args.first_weight)
    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for first_path in sorted(args.first_dir.glob("*.png")):
        second_path = args.second_dir / first_path.name
        if not second_path.exists():
            raise FileNotFoundError(f"Missing matching image: {second_path}")
        output_path = output_images_dir / first_path.name
        with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
            first = first_image.convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
            second = second_image.convert("RGB").resize((args.image_size, args.image_size), Image.BICUBIC)
            Image.blend(first, second, alpha=second_weight).save(output_path)
        entries.append(
            {
                "image_id": first_path.stem,
                "first_path": str(first_path),
                "second_path": str(second_path),
                "output_path": str(output_path),
            }
        )

    save_json(
        {
            "first_dir": str(args.first_dir),
            "second_dir": str(args.second_dir),
            "first_weight": float(args.first_weight),
            "second_weight": second_weight,
            "image_size": int(args.image_size),
            "count": len(entries),
            "images": entries,
        },
        args.output_dir / "blend_metadata.json",
    )
    print(f"blended {len(entries)} images to {output_images_dir}")


if __name__ == "__main__":
    main()
