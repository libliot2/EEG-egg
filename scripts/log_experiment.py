#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.utils import ensure_dir, load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "EXPERIMENT_LOG.md"

BEST_RETRIEVAL_START = "<!-- BEST_RETRIEVAL_START -->"
BEST_RETRIEVAL_END = "<!-- BEST_RETRIEVAL_END -->"
BEST_RECONSTRUCTION_START = "<!-- BEST_RECONSTRUCTION_START -->"
BEST_RECONSTRUCTION_END = "<!-- BEST_RECONSTRUCTION_END -->"
OPEN_ISSUES_START = "<!-- OPEN_ISSUES_START -->"
OPEN_ISSUES_END = "<!-- OPEN_ISSUES_END -->"
ENTRIES_START = "<!-- LOG_ENTRIES_START -->"
ENTRIES_END = "<!-- LOG_ENTRIES_END -->"

DEFAULT_OPEN_ISSUES = [
    "Prototype-based reconstruction is qualitatively failing on test EEG because the prototype bank comes from `training_images`, while train/test concept overlap is zero.",
    "Closed-set retrieval test accuracy and reconstruction prototype selection are different tasks; retrieval `test_acc` must not be used as a proxy for reconstruction quality.",
    "The current `train prototype + residual VAE` reconstruction path is not a promising mainline and should only be kept as a baseline.",
]

AREA_CHOICES = ["retrieval", "reconstruction", "infra", "debug"]
KIND_CHOICES = ["train", "eval", "predict", "cache", "smoke", "ablation", "debug"]
STATUS_CHOICES = ["started", "success", "failed", "aborted"]
SCOPE_CHOICES = ["official_test", "test", "val", "smoke", "unknown"]

EMBEDDING_PROXY_KEYS = [
    "epoch",
    "val_subset_top1_acc",
    "val_subset_top5_acc",
    "avg_target_cosine",
    "full_train_top1_acc",
    "full_train_top5_acc",
]
DECODER_EVAL_KEYS = [
    "epoch",
    "eval_clip",
    "eval_inception",
    "eval_alex5",
    "eval_ssim",
    "eval_pixcorr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append structured experiment records to EXPERIMENT_LOG.md.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    start = subparsers.add_parser("start", description="Record the start of an experiment attempt.")
    add_common_entry_args(start, require_attempt_id=False, include_status=False)

    finish = subparsers.add_parser("finish", description="Record the terminal state of an experiment attempt.")
    add_common_entry_args(finish, require_attempt_id=True, include_status=True)
    finish.add_argument("--open-issue", action="append", default=[], help="Replace the Open Issues section with these items.")
    finish.add_argument("--clear-open-issues", action="store_true", help="Clear the Open Issues section.")

    return parser.parse_args()


def add_common_entry_args(
    parser: argparse.ArgumentParser,
    *,
    require_attempt_id: bool,
    include_status: bool,
) -> None:
    parser.add_argument("--attempt-id", type=str, required=require_attempt_id)
    parser.add_argument("--timestamp", type=str, default=None, help="ISO timestamp. Defaults to the current local time.")
    parser.add_argument("--area", choices=AREA_CHOICES, default=None)
    parser.add_argument("--kind", choices=KIND_CHOICES, default=None)
    if include_status:
        parser.add_argument("--status", choices=[value for value in STATUS_CHOICES if value != "started"], required=True)
    parser.add_argument("--goal", type=str, default=None)
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--key-input", action="append", default=[], help="Repeat to add multiple key inputs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric-scope", choices=SCOPE_CHOICES, default=None)
    parser.add_argument("--observation", action="append", default=[], help="Repeat to add multiple observations.")
    parser.add_argument("--next-action", type=str, default=None)
    parser.add_argument("--backfilled", action="store_true", help="Mark the entry as backfilled from existing artifacts.")


def ensure_log_file(path: Path) -> None:
    if path.exists():
        return

    ensure_dir(path.parent)
    path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                "This file is the project-wide experiment ledger. Use `scripts/log_experiment.py` to append new attempts.",
                "",
                "## Current Best Retrieval",
                BEST_RETRIEVAL_START,
                "- No successful retrieval experiments recorded yet.",
                BEST_RETRIEVAL_END,
                "",
                "## Current Best Reconstruction",
                BEST_RECONSTRUCTION_START,
                "- No successful reconstruction experiments recorded yet.",
                BEST_RECONSTRUCTION_END,
                "",
                "## Open Issues",
                OPEN_ISSUES_START,
                *[f"- {issue}" for issue in DEFAULT_OPEN_ISSUES],
                OPEN_ISSUES_END,
                "",
                "## Experiment Entries",
                ENTRIES_START,
                ENTRIES_END,
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_timestamp(raw_value: str | None) -> datetime:
    if raw_value is None:
        return datetime.now().astimezone()
    return datetime.fromisoformat(raw_value)


def timestamp_to_string(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


def generate_attempt_id(area: str, timestamp: datetime) -> str:
    return f"EXP-{timestamp.strftime('%Y%m%d-%H%M%S')}-{area}"


def infer_metric_scope(output_dir: Path | None, metrics: dict[str, Any] | None) -> str:
    if metrics:
        if "top1_acc" in metrics or "eval_clip" in metrics:
            if output_dir is not None and "smoke" in str(output_dir).lower():
                return "smoke"
            if output_dir is not None and "test" in str(output_dir).lower():
                return "test"
        if "val_top1" in metrics or "val_total_loss" in metrics:
            return "val"

    if output_dir is None:
        return "unknown"

    normalized = str(output_dir).lower()
    if "smoke" in normalized:
        return "smoke"
    if "test" in normalized:
        return "test"
    if "val" in normalized:
        return "val"
    return "unknown"


def pick_best_history_entry(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}

    if any("val_subset_top1_acc" in item for item in history):
        return max(
            history,
            key=lambda item: (
                float(item.get("val_subset_top1_acc", float("-inf"))),
                float(item.get("val_subset_top5_acc", float("-inf"))),
                float(item.get("avg_target_cosine", float("-inf"))),
                -float(item.get("val_total_loss", float("inf"))),
                float(item.get("epoch", float("-inf"))),
            ),
        )

    if any("val_top1" in item for item in history):
        return max(
            history,
            key=lambda item: (
                float(item.get("val_top1", float("-inf"))),
                float(item.get("val_top5", float("-inf"))),
                float(item.get("epoch", float("-inf"))),
            ),
        )

    if any("eval_clip" in item for item in history):
        return max(
            history,
            key=lambda item: (
                float(item.get("eval_clip", float("-inf"))),
                float(item.get("eval_ssim", float("-inf"))),
                float(item.get("eval_pixcorr", float("-inf"))),
                -float(item.get("val_total_loss", float("inf"))),
            ),
        )

    if any("val_total_loss" in item for item in history):
        return min(
            history,
            key=lambda item: (
                float(item.get("val_total_loss", float("inf"))),
                float(item.get("val_clip_loss", float("inf"))),
                float(item.get("epoch", float("inf"))),
            ),
        )

    return history[-1]


def pick_best_embedding_proxy_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not any("val_subset_top1_acc" in item for item in history):
        return None
    return max(
        history,
        key=lambda item: (
            float(item.get("val_subset_top1_acc", float("-inf"))),
            float(item.get("val_subset_top5_acc", float("-inf"))),
            float(item.get("avg_target_cosine", float("-inf"))),
            -float(item.get("val_total_loss", float("inf"))),
            float(item.get("epoch", float("-inf"))),
        ),
    )


def pick_best_decoder_eval_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not any("eval_clip" in item for item in history):
        return None
    return max(
        history,
        key=lambda item: (
            float(item.get("eval_clip", float("-inf"))),
            float(item.get("eval_inception", float("-inf"))),
            float(item.get("eval_alex5", float("-inf"))),
            float(item.get("eval_ssim", float("-inf"))),
            float(item.get("eval_pixcorr", float("-inf"))),
            -float(item.get("val_total_loss", float("inf"))),
            float(item.get("epoch", float("-inf"))),
        ),
    )


def summarize_metric_subset(metrics: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    subset = {key: metrics[key] for key in keys if key in metrics}
    if set(subset) == {"epoch"}:
        return {}
    return subset


def read_selection_summary(output_dir: Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {}

    summary: dict[str, Any] = {}
    best_checkpoint = output_dir / "best.pt"
    if best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        config = dict(payload.get("config", {}))
        metrics = dict(payload.get("metrics", {}))
        if "selection_metric" in config:
            summary["selection_metric"] = config["selection_metric"]
        if "epoch" in metrics:
            summary["selected_epoch"] = metrics["epoch"]
        selected_proxy = summarize_metric_subset(metrics, EMBEDDING_PROXY_KEYS)
        if selected_proxy:
            summary["selected_embedding_proxy"] = selected_proxy
        selected_decoder = summarize_metric_subset(metrics, DECODER_EVAL_KEYS)
        if selected_decoder:
            summary["selected_decoder_eval"] = selected_decoder

    reconstruction_metrics = output_dir / "reconstruction_metrics.json"
    if reconstruction_metrics.exists():
        metrics = load_json(reconstruction_metrics)
        best_decoder = summarize_metric_subset(metrics, DECODER_EVAL_KEYS[1:])
        if best_decoder:
            summary["best_decoder_eval"] = best_decoder

    history_path = output_dir / "history.json"
    if history_path.exists():
        history_payload = load_json(history_path)
        history = history_payload.get("history", [])
        if history:
            proxy_entry = pick_best_embedding_proxy_entry(history)
            if proxy_entry is not None:
                summary["best_embedding_proxy"] = summarize_metric_subset(proxy_entry, EMBEDDING_PROXY_KEYS)
            decoder_entry = pick_best_decoder_eval_entry(history)
            if decoder_entry is not None:
                summary["best_decoder_eval"] = summarize_metric_subset(decoder_entry, DECODER_EVAL_KEYS)

    return summary


def read_metrics(output_dir: Path | None) -> tuple[dict[str, Any], str | None, str | None]:
    if output_dir is None:
        return {}, None, None

    retrieval_metrics = output_dir / "retrieval_metrics.json"
    if retrieval_metrics.exists():
        metrics = load_json(retrieval_metrics)
        return metrics, "retrieval_metrics.json", infer_metric_scope(output_dir, metrics)

    reconstruction_metrics = output_dir / "reconstruction_metrics.json"
    if reconstruction_metrics.exists():
        metrics = load_json(reconstruction_metrics)
        return metrics, "reconstruction_metrics.json", infer_metric_scope(output_dir, metrics)

    best_checkpoint = output_dir / "best.pt"
    if best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        metrics = dict(payload.get("metrics", {}))
        if metrics:
            return metrics, "best.pt", infer_metric_scope(output_dir, metrics)

    history_path = output_dir / "history.json"
    if history_path.exists():
        history_payload = load_json(history_path)
        history = history_payload.get("history", [])
        if history:
            metrics = dict(pick_best_history_entry(history))
            return metrics, "history.json", infer_metric_scope(output_dir, metrics)

    return {}, None, None


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_list(items: list[str]) -> list[str]:
    if not items:
        return ["- None"]
    return [f"- {item}" for item in items]


def render_metrics(metrics: dict[str, Any]) -> list[str]:
    if not metrics:
        return ["- None"]
    ordered = sorted(metrics.items())
    return [f"- `{key}` = {format_metric_value(value)}" for key, value in ordered]


def render_metric_inline(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "None"
    ordered = sorted(metrics.items())
    return ", ".join(f"`{key}={format_metric_value(value)}`" for key, value in ordered)


def render_selection_summary(meta: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if meta.get("selection_metric") is not None:
        lines.append(f"- `selection_metric` = `{meta['selection_metric']}`")
    if meta.get("selected_epoch") is not None:
        lines.append(f"- `selected_epoch` = {format_metric_value(meta['selected_epoch'])}")
    if meta.get("selected_embedding_proxy"):
        lines.append(f"- Selected Embedding Proxy: {render_metric_inline(meta['selected_embedding_proxy'])}")
    if meta.get("best_embedding_proxy"):
        lines.append(f"- Best Embedding Proxy: {render_metric_inline(meta['best_embedding_proxy'])}")
    if meta.get("selected_decoder_eval"):
        lines.append(f"- Selected Decoder Eval: {render_metric_inline(meta['selected_decoder_eval'])}")
    if meta.get("best_decoder_eval"):
        lines.append(f"- Best Decoder Eval: {render_metric_inline(meta['best_decoder_eval'])}")
    return lines or ["- None"]


def render_entry(meta: dict[str, Any]) -> str:
    lines = [
        f"### {meta['attempt_id']} [{meta['status']}]",
        f"<!-- log-meta: {compact_json(meta)} -->",
        "",
        f"- Timestamp: {meta['timestamp']}",
        f"- Area: {meta['area']}",
        f"- Kind: {meta['kind']}",
        f"- Goal: {meta['goal']}",
        f"- Metric Scope: {meta['metric_scope']}",
        f"- Metric Source: {meta['metric_source'] or 'None'}",
        f"- Output Dir: {meta['output_dir'] or 'None'}",
        f"- Backfilled: {'yes' if meta['backfilled'] else 'no'}",
        "",
        "#### Command",
        "```bash",
        meta["command"] or "# none recorded",
        "```",
        "",
        "#### Key Inputs",
        *render_list(meta["key_inputs"]),
        "",
        "#### Metrics",
        *render_metrics(meta["metrics"]),
        "",
        "#### Selection Summary",
        *render_selection_summary(meta),
        "",
        "#### Observations",
        *render_list(meta["observations"]),
        "",
        "#### Next Action",
        meta["next_action"] or "None",
        "",
    ]
    return "\n".join(lines)


def insert_entry(log_text: str, entry_text: str) -> str:
    needle = f"{ENTRIES_START}\n"
    if needle not in log_text:
        raise ValueError("Experiment entry marker not found in log file.")
    return log_text.replace(needle, f"{needle}{entry_text}\n", 1)


def update_section(log_text: str, start_marker: str, end_marker: str, lines: list[str]) -> str:
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), flags=re.S)
    replacement = "\n".join([start_marker, *lines, end_marker])
    return pattern.sub(replacement, log_text, count=1)


def parse_meta_entries(log_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    pattern = re.compile(r"<!-- log-meta: (.*?) -->")
    for match in pattern.finditer(log_text):
        entries.append(json.loads(match.group(1)))
    return entries


def scope_rank(scope: str) -> int:
    ranking = {
        "official_test": 4,
        "test": 3,
        "val": 2,
        "unknown": 1,
        "smoke": 0,
    }
    return ranking.get(scope, 1)


def choose_best_retrieval(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_entry: dict[str, Any] | None = None
    best_score: tuple[int, float, float] | None = None

    for entry in entries:
        if entry.get("area") != "retrieval" or entry.get("status") != "success":
            continue
        metrics = entry.get("metrics", {})
        if "top1_acc" in metrics:
            score = (
                scope_rank(entry.get("metric_scope", "unknown")),
                float(metrics.get("top1_acc", 0.0)),
                float(metrics.get("top5_acc", 0.0)),
            )
        elif "val_top1" in metrics:
            score = (
                scope_rank(entry.get("metric_scope", "unknown")),
                float(metrics.get("val_top1", 0.0)),
                float(metrics.get("val_top5", 0.0)),
            )
        else:
            continue

        if best_score is None or score > best_score:
            best_entry = entry
            best_score = score

    return best_entry


def choose_best_reconstruction(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_entry: dict[str, Any] | None = None
    best_score: tuple[int, float, float, float] | None = None

    for entry in entries:
        if entry.get("area") != "reconstruction" or entry.get("status") != "success":
            continue
        metrics = entry.get("metrics", {})
        if "eval_clip" in metrics:
            score = (
                scope_rank(entry.get("metric_scope", "unknown")),
                float(metrics.get("eval_clip", 0.0)),
                float(metrics.get("eval_ssim", 0.0)),
                float(metrics.get("eval_pixcorr", 0.0)),
            )
        else:
            continue

        if best_score is None or score > best_score:
            best_entry = entry
            best_score = score

    return best_entry


def render_best_retrieval(entry: dict[str, Any] | None) -> list[str]:
    if entry is None:
        return ["- No successful retrieval experiments recorded yet."]

    metrics = entry["metrics"]
    lines = [
        f"- Attempt ID: `{entry['attempt_id']}`",
        f"- Scope: `{entry['metric_scope']}`",
        f"- Output Dir: `{entry['output_dir']}`",
    ]
    if "top1_acc" in metrics:
        lines.append(
            "- Metrics: "
            f"`top1_acc={format_metric_value(metrics.get('top1_acc'))}`, "
            f"`top5_acc={format_metric_value(metrics.get('top5_acc'))}`"
            + (
                f", `alpha={format_metric_value(metrics.get('alpha'))}`"
                if "alpha" in metrics
                else ""
            )
        )
    else:
        lines.append(
            "- Metrics: "
            f"`val_top1={format_metric_value(metrics.get('val_top1'))}`, "
            f"`val_top5={format_metric_value(metrics.get('val_top5'))}`"
        )
    lines.append(f"- Goal: {entry['goal']}")
    return lines


def render_best_reconstruction(entry: dict[str, Any] | None) -> list[str]:
    if entry is None:
        return ["- No successful reconstruction experiments recorded yet."]

    metrics = entry["metrics"]
    lines = [
        f"- Attempt ID: `{entry['attempt_id']}`",
        f"- Scope: `{entry['metric_scope']}`",
        f"- Output Dir: `{entry['output_dir']}`",
        "- Metrics: "
        f"`eval_clip={format_metric_value(metrics.get('eval_clip'))}`, "
        f"`eval_ssim={format_metric_value(metrics.get('eval_ssim'))}`, "
        f"`eval_pixcorr={format_metric_value(metrics.get('eval_pixcorr'))}`",
        f"- Goal: {entry['goal']}",
        "- Qualitative Caveat: see `Open Issues`; the current best quantitative run is still visually unreliable.",
    ]
    return lines


def replace_open_issues(log_text: str, issues: list[str]) -> str:
    lines = [f"- {issue}" for issue in issues] if issues else ["- None currently recorded."]
    return update_section(log_text, OPEN_ISSUES_START, OPEN_ISSUES_END, lines)


def refresh_summaries(log_text: str) -> str:
    entries = parse_meta_entries(log_text)
    log_text = update_section(
        log_text,
        BEST_RETRIEVAL_START,
        BEST_RETRIEVAL_END,
        render_best_retrieval(choose_best_retrieval(entries)),
    )
    log_text = update_section(
        log_text,
        BEST_RECONSTRUCTION_START,
        BEST_RECONSTRUCTION_END,
        render_best_reconstruction(choose_best_reconstruction(entries)),
    )
    return log_text


def find_attempt_context(entries: list[dict[str, Any]], attempt_id: str) -> dict[str, Any] | None:
    matches = [entry for entry in entries if entry.get("attempt_id") == attempt_id]
    if not matches:
        return None
    return matches[0]


def build_entry(args: argparse.Namespace, existing_context: dict[str, Any] | None) -> dict[str, Any]:
    timestamp = parse_timestamp(args.timestamp)
    attempt_id = args.attempt_id or generate_attempt_id(args.area, timestamp)

    area = args.area or (existing_context.get("area") if existing_context else None)
    kind = args.kind or (existing_context.get("kind") if existing_context else None)
    goal = args.goal or (existing_context.get("goal") if existing_context else None)
    command = args.command if args.command is not None else (existing_context.get("command") if existing_context else None)
    key_inputs = args.key_input or (existing_context.get("key_inputs") if existing_context else [])
    output_dir = str(args.output_dir.resolve()) if args.output_dir is not None else (existing_context.get("output_dir") if existing_context else None)

    if area is None or kind is None or goal is None:
        raise ValueError("area, kind, and goal must be provided on the first entry for an attempt.")

    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    metrics, metric_source, inferred_scope = read_metrics(resolved_output_dir)
    metric_scope = args.metric_scope or inferred_scope or (existing_context.get("metric_scope") if existing_context else "unknown")
    selection_summary = read_selection_summary(resolved_output_dir)

    return {
        "attempt_id": attempt_id,
        "timestamp": timestamp_to_string(timestamp),
        "area": area,
        "kind": kind,
        "status": "started" if args.subcommand == "start" else args.status,
        "goal": goal,
        "command": command,
        "key_inputs": key_inputs,
        "output_dir": output_dir,
        "metrics": metrics,
        "metric_source": metric_source,
        "metric_scope": metric_scope,
        "selection_metric": selection_summary.get("selection_metric"),
        "selected_epoch": selection_summary.get("selected_epoch"),
        "selected_embedding_proxy": selection_summary.get("selected_embedding_proxy"),
        "best_embedding_proxy": selection_summary.get("best_embedding_proxy"),
        "selected_decoder_eval": selection_summary.get("selected_decoder_eval"),
        "best_decoder_eval": selection_summary.get("best_decoder_eval"),
        "observations": args.observation,
        "next_action": args.next_action,
        "backfilled": bool(args.backfilled),
    }


def main() -> None:
    args = parse_args()
    ensure_log_file(args.log_path)

    log_text = args.log_path.read_text(encoding="utf-8")
    existing_entries = parse_meta_entries(log_text)
    existing_context = find_attempt_context(existing_entries, args.attempt_id) if args.attempt_id else None

    entry = build_entry(args, existing_context)
    rendered = render_entry(entry)
    log_text = insert_entry(log_text, rendered)

    if args.subcommand == "finish":
        if args.clear_open_issues:
            log_text = replace_open_issues(log_text, [])
        elif args.open_issue:
            log_text = replace_open_issues(log_text, args.open_issue)

    log_text = refresh_summaries(log_text)
    args.log_path.write_text(log_text, encoding="utf-8")

    print(entry["attempt_id"])
    print(args.log_path)


if __name__ == "__main__":
    main()
