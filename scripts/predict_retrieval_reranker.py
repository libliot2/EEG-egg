#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.image_banks import TensorBank
from project1_eeg.reranker import (
    TopKRetrievalReranker,
    build_shortlist_indices,
    reorder_logits_within_shortlist,
)
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval inference with a second-stage top-k reranker.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_reranker_predictions")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--rerank-weight", type=float, default=None)
    parser.add_argument("--shortlist-topk", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def compute_base_query_embeddings(
    base_model,
    records,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, list[str]]:
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    outputs, image_ids, _ = compute_retrieval_outputs(base_model, loader, device)
    if "perceptual" not in outputs:
        raise ValueError("Base checkpoint does not expose a perceptual retrieval head.")
    return outputs["perceptual"].float(), image_ids


@torch.no_grad()
def compute_rerank_scores(
    reranker: TopKRetrievalReranker,
    query_embeddings: torch.Tensor,
    base_shortlist_scores: torch.Tensor,
    shortlist_indices: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    candidate_embeddings = candidate_embeddings.float()
    for start in range(0, len(query_embeddings), batch_size):
        query_chunk = query_embeddings[start : start + batch_size].to(device)
        base_score_chunk = base_shortlist_scores[start : start + batch_size].to(device)
        shortlist_chunk = shortlist_indices[start : start + batch_size]
        candidate_chunk = candidate_embeddings[shortlist_chunk].to(device)
        scores.append(
            reranker.score_shortlist(
                query_chunk,
                candidate_chunk,
                base_shortlist_scores=base_score_chunk,
            ).cpu()
        )
    return torch.cat(scores, dim=0)


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    reranker_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reranker_config = dict(reranker_payload["config"])
    base_checkpoint = args.base_checkpoint or Path(str(reranker_payload.get("base_checkpoint") or reranker_config["base_checkpoint"]))
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_config = dict(base_payload["config"])

    base_model = build_retrieval_model_from_config(base_config).to(device)
    base_model.load_state_dict(base_payload["model_state"], strict=True)
    base_model.eval().requires_grad_(False)

    feature_dim = int(base_config["perceptual_dim"])
    reranker = TopKRetrievalReranker(
        feature_dim,
        scorer_type=str(reranker_config.get("scorer_type", "cosine")),
        hidden_dim=int(reranker_config["adapter_hidden_dim"]),
        score_hidden_dim=None if reranker_config.get("score_hidden_dim") is None else int(reranker_config["score_hidden_dim"]),
        dropout=float(reranker_config["adapter_dropout"]),
        beta=float(reranker_config["adapter_beta"]),
        share_adapters=bool(reranker_config["share_adapters"]),
        use_top1_head=bool(reranker_config.get("use_top1_head", False)),
        top1_head_hidden_dim=(
            None
            if reranker_config.get("top1_head_hidden_dim") is None
            else int(reranker_config["top1_head_hidden_dim"])
        ),
        top1_score_coef=float(reranker_config.get("top1_score_coef", 0.25)),
    ).to(device)
    reranker.load_state_dict(reranker_payload["model_state"], strict=True)
    reranker.eval()

    bank_path = args.perceptual_bank
    if bank_path is None:
        fallback = reranker_config.get("perceptual_bank") or base_config.get("perceptual_bank")
        if fallback is None:
            raise ValueError("--perceptual-bank is required when the checkpoint config does not define one.")
        bank_path = Path(str(fallback))
        bank_path = resolve_split_bank_path(bank_path, split=args.split)
    perceptual_bank = TensorBank.load(bank_path)

    selected_channels = base_config.get("selected_channels")
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        selected_channels=selected_channels,
        image_ids=selected_image_ids,
    )
    query_embeddings, query_image_ids = compute_base_query_embeddings(
        base_model,
        records,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    candidate_image_ids = list(perceptual_bank.image_ids)
    candidate_embeddings = perceptual_bank.values.float()
    base_logits = base_model.similarity(
        query_embeddings.to(device),
        candidate_embeddings.to(device),
        head="perceptual",
    ).cpu()
    shortlist_topk = args.shortlist_topk or int(reranker_config["shortlist_topk"])
    shortlist_indices, _ = build_shortlist_indices(
        base_logits,
        candidate_image_ids=candidate_image_ids,
        shortlist_topk=shortlist_topk,
        ensure_positive=False,
    )
    shortlist_base_scores = base_logits.gather(1, shortlist_indices)
    rerank_scores = compute_rerank_scores(
        reranker,
        query_embeddings.cpu(),
        shortlist_base_scores,
        shortlist_indices,
        candidate_embeddings,
        device=device,
        batch_size=args.batch_size,
    )
    rerank_weight = args.rerank_weight
    if rerank_weight is None:
        rerank_weight = float(reranker_payload["metrics"].get("val_selected_rerank_weight", 1.0))
    order_scores = shortlist_base_scores + float(rerank_weight) * rerank_scores
    logits = reorder_logits_within_shortlist(base_logits, shortlist_indices, order_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "ordered_query_image_ids": query_image_ids,
            "candidate_image_ids": candidate_image_ids,
            "rerank_weight": float(rerank_weight),
            "shortlist_topk": int(shortlist_topk),
            "logits": logits,
            "base_logits": base_logits,
            "shortlist_indices": shortlist_indices,
            "rerank_scores": rerank_scores,
        },
        args.output_dir / "retrieval_logits.pt",
    )

    rankings = rank_candidate_ids(logits, candidate_image_ids)
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
            "rerank_weight": float(rerank_weight),
            "shortlist_topk": int(shortlist_topk),
            "predictions": ranking_payload,
        },
        args.output_dir / "retrieval_rankings.json",
    )

    if args.evaluate:
        metrics = compute_retrieval_metrics(
            logits,
            ordered_image_ids=query_image_ids,
            candidate_image_ids=candidate_image_ids,
        )
        metrics["rerank_weight"] = float(rerank_weight)
        metrics["shortlist_topk"] = int(shortlist_topk)
        save_json(metrics, args.output_dir / "retrieval_metrics.json")
        print(metrics)
    else:
        print(f"saved retrieval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
