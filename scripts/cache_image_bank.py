#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.image_banks import (
    build_clip_bank,
    build_clip_text_bank,
    build_dreamsim_bank,
    build_openclip_bank,
    build_vae_latent_bank,
    default_bank_path,
)
from project1_eeg.kandinsky import DEFAULT_KANDINSKY_PRIOR_MODEL, build_kandinsky_bank
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache image feature banks for retrieval or reconstruction.")
    parser.add_argument(
        "--bank-type",
        choices=["clip", "openclip", "clip_text", "dreamsim", "vae", "kandinsky"],
        required=True,
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--openclip-model", type=str, default="ViT-H-14")
    parser.add_argument("--openclip-pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument("--openclip-view", choices=["full", "crop", "contour", "object"], default="full")
    parser.add_argument("--openclip-layer", choices=["early", "mid", "late"], default="late")
    parser.add_argument("--dreamsim-type", type=str, default="ensemble")
    parser.add_argument("--dreamsim-view", choices=["full", "crop", "contour", "multiview"], default="full")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--kandinsky-prior-model", type=str, default=DEFAULT_KANDINSKY_PRIOR_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        output_path = args.output
    else:
        output_path = default_bank_path(DEFAULT_OUTPUT_DIR, args.bank_type, args.split)
        if args.bank_type == "openclip":
            output_path = output_path.with_name(
                f"openclip_{args.openclip_view}_{args.openclip_layer}_{args.split}.pt"
            )
        elif args.bank_type == "dreamsim" and args.dreamsim_view != "full":
            output_path = output_path.with_name(
                f"dreamsim_{args.dreamsim_view}_{args.split}.pt"
            )

    if args.bank_type == "clip":
        bank = build_clip_bank(
            data_dir=args.data_dir,
            split=args.split,
            model_name=args.clip_model,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    elif args.bank_type == "openclip":
        bank = build_openclip_bank(
            data_dir=args.data_dir,
            split=args.split,
            model_name=args.openclip_model,
            pretrained=args.openclip_pretrained,
            batch_size=args.batch_size,
            device=args.device,
            view_type=args.openclip_view,
            layer=args.openclip_layer,
        )
    elif args.bank_type == "clip_text":
        bank = build_clip_text_bank(
            data_dir=args.data_dir,
            split=args.split,
            model_name=args.clip_model,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.bank_type == "dreamsim":
        bank = build_dreamsim_bank(
            data_dir=args.data_dir,
            split=args.split,
            dreamsim_type=args.dreamsim_type,
            view_type=args.dreamsim_view,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    elif args.bank_type == "vae":
        bank = build_vae_latent_bank(
            data_dir=args.data_dir,
            split=args.split,
            model_name=args.vae_model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:
        bank = build_kandinsky_bank(
            data_dir=args.data_dir,
            split=args.split,
            model_name=args.kandinsky_prior_model,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
        )

    bank.save(output_path)
    print(f"saved {args.bank_type} bank for {args.split} to {output_path}")


if __name__ == "__main__":
    main()
