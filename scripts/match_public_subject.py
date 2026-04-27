#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match the released course subject against public THINGS-EEG2 subjects.",
    )
    parser.add_argument(
        "--local-test",
        type=Path,
        required=True,
        help="Path to the released local test.pt file.",
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        required=True,
        help="Directory that contains downloaded sub-XX_ses-YY_raw_eeg_test.npy files.",
    )
    parser.add_argument(
        "--glob",
        default="sub-*_ses-01_raw_eeg_test.npy",
        help="Glob pattern for public files inside --public-dir.",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=[-200, -100, 0, 100, 200],
        help="Offsets in milliseconds at 1000 Hz relative to stimulus onset.",
    )
    parser.add_argument(
        "--local-session-index",
        type=int,
        default=0,
        help="Which local 20-trial session chunk to compare against.",
    )
    return parser.parse_args()


def zscore_time(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)


def unit_norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_local_avg(local_test_path: Path, session_idx: int) -> np.ndarray:
    local = torch.load(local_test_path, map_location="cpu", weights_only=False)
    eeg = np.asarray(local["eeg"], dtype=np.float32)
    start = session_idx * 20
    stop = start + 20
    eeg = zscore_time(eeg[:, start:stop])
    avg = eeg.mean(axis=1).reshape(len(eeg), -1)
    return unit_norm_rows(avg)


def extract_public_avg(public_path: Path, offset_ms: int) -> np.ndarray:
    obj = np.load(public_path, allow_pickle=True).item()
    raw = obj["raw_eeg_data"][:63].astype(np.float32, copy=False)
    stim = obj["raw_eeg_data"][63].astype(np.int64, copy=False)
    rising = np.flatnonzero((stim[:-1] == 0) & (stim[1:] > 0)) + 1

    per_code: dict[int, list[np.ndarray]] = {code: [] for code in range(1, 201)}
    for onset in rising:
        code = int(stim[onset])
        if not 1 <= code <= 200:
            continue
        if len(per_code[code]) >= 20:
            continue
        start = onset + offset_ms
        end = start + 1000
        if start < 0 or end > raw.shape[1]:
            continue
        per_code[code].append(raw[:, start:end][:, ::4])

    counts = [len(per_code[code]) for code in range(1, 201)]
    if min(counts) < 20:
        raise RuntimeError(f"{public_path.name}: missing trials after offset {offset_ms}, min={min(counts)}")

    eeg = np.stack([np.stack(per_code[code], axis=0) for code in range(1, 201)], axis=0)
    eeg = zscore_time(eeg)
    avg = eeg.mean(axis=1).reshape(len(eeg), -1)
    return unit_norm_rows(avg)


def compute_metrics(local_avg: np.ndarray, public_avg: np.ndarray) -> dict[str, float]:
    sims = np.matmul(local_avg, public_avg.T)
    order = np.argsort(-sims, axis=1)
    targets = np.arange(len(sims))

    top1 = float((order[:, 0] == targets).mean())
    top5 = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    diag = float(np.diag(sims).mean())
    offdiag = float((sims.sum() - np.trace(sims)) / (sims.size - len(sims)))

    local_rdm = 1.0 - np.matmul(local_avg, local_avg.T)
    public_rdm = 1.0 - np.matmul(public_avg, public_avg.T)
    iu = np.triu_indices(len(sims), k=1)

    local_rdm_u = local_rdm[iu]
    public_rdm_u = public_rdm[iu]
    local_rdm_u_z = (local_rdm_u - local_rdm_u.mean()) / (local_rdm_u.std() + 1e-8)
    public_rdm_u_z = (public_rdm_u - public_rdm_u.mean()) / (public_rdm_u.std() + 1e-8)
    rdm_pearson = float((local_rdm_u_z * public_rdm_u_z).mean())

    local_rank = rankdata(local_rdm_u)
    public_rank = rankdata(public_rdm_u)
    local_rank = (local_rank - local_rank.mean()) / (local_rank.std() + 1e-8)
    public_rank = (public_rank - public_rank.mean()) / (public_rank.std() + 1e-8)
    rdm_spearman = float((local_rank * public_rank).mean())

    return {
        "top1": top1,
        "top5": top5,
        "diag_gap": diag - offdiag,
        "rdm_pearson": rdm_pearson,
        "rdm_spearman": rdm_spearman,
    }


def main() -> None:
    args = parse_args()
    local_avg = load_local_avg(args.local_test, args.local_session_index)
    rows: list[dict[str, float | int | str]] = []

    for public_path in sorted(args.public_dir.glob(args.glob)):
        best_score = None
        best_offset = None
        best_metrics = None
        for offset in args.offsets:
            try:
                public_avg = extract_public_avg(public_path, offset)
            except Exception:
                continue
            metrics = compute_metrics(local_avg, public_avg)
            score = (
                metrics["top1"] * 100
                + metrics["top5"] * 20
                + metrics["rdm_spearman"] * 10
                + metrics["diag_gap"] * 1000
            )
            if best_score is None or score > best_score:
                best_score = score
                best_offset = offset
                best_metrics = metrics
        if best_metrics is None:
            continue
        rows.append(
            {
                "subject": public_path.name.split("_")[0],
                "best_offset": int(best_offset),
                **best_metrics,
            }
        )

    print("RANK_BY_TOP")
    for row in sorted(
        rows,
        key=lambda x: (x["top1"], x["top5"], x["rdm_spearman"], x["diag_gap"]),
        reverse=True,
    ):
        print(json.dumps(row, ensure_ascii=False))

    print("RANK_BY_RDM")
    for row in sorted(
        rows,
        key=lambda x: (x["rdm_spearman"], x["rdm_pearson"], x["top5"], x["top1"]),
        reverse=True,
    ):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
