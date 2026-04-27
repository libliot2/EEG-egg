#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = [
    "eval_clip",
    "eval_ssim",
    "eval_pixcorr",
    "eval_alex2",
    "eval_alex5",
    "eval_inception",
    "eval_effnet",
    "eval_swav",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reconstruction campaign metrics.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    row = {
        "run": str(path.parent),
        "balanced": float(metrics.get("eval_clip", 0.0)) + 0.25 * float(metrics.get("eval_ssim", 0.0)),
        "clip_ssim": float(metrics.get("eval_clip", 0.0)) + float(metrics.get("eval_ssim", 0.0)),
    }
    for key in METRIC_KEYS:
        value = metrics.get(key)
        row[key] = None if value is None else float(value)
    return row


def format_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def render_table(title: str, rows: list[dict], *, sort_key: str, top_k: int) -> list[str]:
    selected = sorted(rows, key=lambda item: item.get(sort_key) or float("-inf"), reverse=True)[:top_k]
    lines = [f"## {title}", "", "| rank | run | balanced | clip+ssim | clip | ssim | pixcorr | alex5 | inception |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(selected, start=1):
        run_name = Path(row["run"]).name
        parent_name = Path(row["run"]).parent.name
        label = f"{parent_name}/{run_name}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    label,
                    format_value(row["balanced"]),
                    format_value(row["clip_ssim"]),
                    format_value(row["eval_clip"]),
                    format_value(row["eval_ssim"]),
                    format_value(row["eval_pixcorr"]),
                    format_value(row["eval_alex5"]),
                    format_value(row["eval_inception"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(args.root)
    rows = [load_metrics(path) for path in sorted(args.root.rglob("reconstruction_metrics.json"))]
    lines = [
        f"# Reconstruction Campaign Summary: {args.root}",
        "",
        f"Found {len(rows)} metric files.",
        "",
    ]
    if rows:
        lines.extend(render_table("Balanced Ranking", rows, sort_key="balanced", top_k=args.top_k))
        lines.extend(render_table("CLIP Ranking", rows, sort_key="eval_clip", top_k=args.top_k))
        lines.extend(render_table("SSIM Ranking", rows, sort_key="eval_ssim", top_k=args.top_k))
        lines.extend(render_table("PixCorr Ranking", rows, sort_key="eval_pixcorr", top_k=args.top_k))
    text = "\n".join(lines)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
