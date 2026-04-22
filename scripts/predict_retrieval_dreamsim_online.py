#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import compute_retrieval_metrics, rank_candidate_ids
from project1_eeg.retrieval import RetrievalModel
from project1_eeg.runtime import compute_retrieval_outputs, make_dataloader
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, resolve_device, save_json


CHANNEL_PRESETS: dict[str, list[str]] = {
    "visual17": [
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    ]
}


def load_images(image_paths: list[str], preprocess, device: torch.device) -> torch.Tensor:
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensor = preprocess(image.convert("RGB"))
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0)
    return batch.to(device, non_blocking=device.type == "cuda")


def resolve_model_hparams(config: dict[str, object]) -> dict[str, object]:
    return {
        "use_perceptual_target_adapter": bool(config.get("use_perceptual_target_adapter", False)),
        "target_adapter_hidden_dim": int(config.get("target_adapter_hidden_dim", 1024)),
        "target_adapter_dropout": float(config.get("target_adapter_dropout", 0.1)),
        "target_adapter_beta": float(config.get("target_adapter_beta", 0.1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal retrieval inference with online DreamSim targets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "retrieval_predictions_online")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--selected-channels", nargs="+", default=None)
    parser.add_argument("--channel-preset", choices=sorted(CHANNEL_PRESETS.keys()), default=None)
    parser.add_argument("--dreamsim-type", type=str, default=None)
    return parser.parse_args()


def resolve_selected_channels(args: argparse.Namespace, config: dict) -> list[str] | None:
    if args.selected_channels and args.channel_preset:
        raise ValueError("--selected-channels and --channel-preset cannot both be set.")
    if args.selected_channels:
        return [str(channel) for channel in args.selected_channels]
    if args.channel_preset:
        return list(CHANNEL_PRESETS[args.channel_preset])
    configured = config.get("selected_channels")
    if configured is None:
        return None
    return [str(channel) for channel in configured]


def build_model(
    *,
    checkpoint_payload: dict,
    records,
    dreamsim_model,
    preprocess,
    device: torch.device,
) -> RetrievalModel:
    config = dict(checkpoint_payload.get("config", {}))
    resolved_model_hparams = resolve_model_hparams(config)
    sample_eeg = records[0].eeg
    in_channels = int(config.get("in_channels", sample_eeg.shape[0]))
    sample_image = load_images([records[0].image_path], preprocess, device)
    perceptual_dim = int(config.get("perceptual_dim", dreamsim_model.get_base_model().embed(sample_image).shape[-1]))
    model = RetrievalModel(
        in_channels=in_channels,
        hidden_dim=int(config.get("hidden_dim", 256)),
        embedding_dim=int(config.get("embedding_dim", 768)),
        channel_dropout=float(config.get("channel_dropout", 0.1)),
        time_mask_ratio=float(config.get("time_mask_ratio", 0.1)),
        encoder_type=str(config.get("encoder_type", "atm_large")),
        semantic_dim=None,
        perceptual_dim=perceptual_dim,
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_heads=int(config.get("transformer_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_eeg_perturbation=bool(config.get("use_eeg_perturbation", False)),
        use_perceptual_target_adapter=bool(resolved_model_hparams["use_perceptual_target_adapter"]),
        target_adapter_hidden_dim=int(resolved_model_hparams["target_adapter_hidden_dim"]),
        target_adapter_dropout=float(resolved_model_hparams["target_adapter_dropout"]),
        target_adapter_beta=float(resolved_model_hparams["target_adapter_beta"]),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state"], strict=False)
    model.eval()
    return model


def compute_candidate_embeddings(
    dreamsim_model,
    preprocess,
    candidate_records,
    *,
    candidate_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    base_model = dreamsim_model.get_base_model()
    embeddings: list[torch.Tensor] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for start in range(0, len(candidate_records), candidate_batch_size):
            chunk = candidate_records[start : start + candidate_batch_size]
            batch_images = load_images([record.image_path for record in chunk], preprocess, device)
            chunk_embeddings = base_model.embed(batch_images).float()
            chunk_embeddings = torch.nn.functional.normalize(chunk_embeddings, dim=-1).cpu()
            embeddings.append(chunk_embeddings)
            image_ids.extend([record.image_id for record in chunk])
    return torch.cat(embeddings, dim=0), image_ids


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    selected_channels = resolve_selected_channels(args, config)
    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)

    dreamsim_type = args.dreamsim_type or str(config.get("dreamsim_type", "ensemble"))
    from dreamsim import dreamsim

    dreamsim_model, preprocess = dreamsim(
        pretrained=True,
        device=str(device),
        dreamsim_type=dreamsim_type,
    )
    if "dreamsim_state" in payload:
        dreamsim_model.load_state_dict(payload["dreamsim_state"], strict=False)
    dreamsim_model.eval()

    query_records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        selected_channels=selected_channels,
        image_ids=selected_image_ids,
    )
    candidate_records = query_records
    if args.split == "test":
        candidate_records = load_eeg_records(
            data_dir=args.data_dir,
            split=args.split,
            avg_trials=True,
            selected_channels=selected_channels,
            image_ids=None,
        )

    model = build_model(
        checkpoint_payload=payload,
        records=query_records,
        dreamsim_model=dreamsim_model,
        preprocess=preprocess,
        device=device,
    )
    loader = make_dataloader(
        EEGImageDataset(query_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    outputs, query_image_ids, _ = compute_retrieval_outputs(model, loader, device)
    query_embeddings = outputs["perceptual"]
    candidate_embeddings, candidate_image_ids = compute_candidate_embeddings(
        dreamsim_model,
        preprocess,
        candidate_records,
        candidate_batch_size=args.candidate_batch_size,
        device=device,
    )
    logits = model.similarity(query_embeddings.to(device), candidate_embeddings.to(device), head="perceptual").cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "ordered_query_image_ids": query_image_ids,
            "candidate_image_ids": candidate_image_ids,
            "alpha": 0.0,
            "logits": logits,
            "semantic_logits": None,
            "perceptual_logits": logits,
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
            "alpha": 0.0,
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
        metrics["alpha"] = 0.0
        save_json(metrics, args.output_dir / "retrieval_metrics.json")
        print(metrics)
    else:
        print(f"saved retrieval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
