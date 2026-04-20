#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import eval_images
from project1_eeg.image_banks import TensorBank, default_bank_path
from project1_eeg.kandinsky import (
    DEFAULT_KANDINSKY_DECODER_MODEL,
    DEFAULT_KANDINSKY_PRIOR_MODEL,
    generate_best_images,
    load_kandinsky_decoder_pipeline,
    load_kandinsky_prior_pipeline,
    negative_image_embed_from_bank,
)
from project1_eeg.reconstruction import EEGEmbeddingRegressor, GatedRetrievalResidualRegressor
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, make_dataloader, prototype_lookup_from_logits, save_image_batch
from project1_eeg.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    load_image_batch,
    resolve_device,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reconstructions from EEG via Kandinsky image embeddings.")
    parser.add_argument("--reconstruction-checkpoint", type=Path, default=None)
    parser.add_argument("--retrieval-checkpoint", type=Path, default=None)
    parser.add_argument("--model-type", choices=["regression", "rag_residual"], default=None)
    parser.add_argument(
        "--embedding-bank",
        type=Path,
        default=None,
        help="Defaults to the bank recorded in the reconstruction checkpoint config.",
    )
    parser.add_argument("--retrieval-bank", type=Path, default=None)
    parser.add_argument("--text-bank", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "reconstruction_predictions_embed",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument(
        "--embedding-source",
        choices=["predicted", "ground_truth", "retrieval_top1"],
        default="predicted",
    )
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--prior-model", type=str, default=None)
    parser.add_argument("--decoder-model", type=str, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--decoder-steps", type=int, default=None)
    parser.add_argument("--decoder-guidance-scale", type=float, default=None)
    parser.add_argument("--decoder-height", type=int, default=None)
    parser.add_argument("--decoder-width", type=int, default=None)
    parser.add_argument("--retrieval-topk", type=int, default=None)
    parser.add_argument("--use-text-context", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_artifact_path(path: str | Path | None, *, base_paths: Iterable[Path] = ()) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for base_path in base_paths:
        resolved = base_path / candidate
        if resolved.exists():
            return resolved
    return candidate


def resolve_bank(path: Path | None, fallback: str | None, *, name: str, base_paths: Iterable[Path]) -> TensorBank | None:
    resolved = path or resolve_artifact_path(fallback, base_paths=base_paths)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(f"{name} bank not found: {resolved}")
    return TensorBank.load(resolved)


def resolve_alpha(args: argparse.Namespace, payload: dict, *, has_semantic: bool, has_perceptual: bool) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    metrics = payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    return 0.5


def resolve_embedding_bank_path(args: argparse.Namespace, config: dict | None, checkpoint_path: Path | None) -> Path:
    if args.embedding_bank is not None:
        return args.embedding_bank
    if config is not None and config.get("embedding_bank"):
        resolved = resolve_artifact_path(
            config["embedding_bank"],
            base_paths=[PROJECT_ROOT, checkpoint_path.parent if checkpoint_path is not None else PROJECT_ROOT],
        )
        if resolved is not None:
            return resolved
    default_path = default_bank_path(DEFAULT_OUTPUT_DIR, "kandinsky", "train")
    if default_path.exists():
        return default_path
    raise ValueError(
        "Unable to resolve an embedding bank. Pass --embedding-bank or provide a reconstruction checkpoint "
        "whose config contains `embedding_bank`."
    )


def resolve_generation_config(
    args: argparse.Namespace,
    config: dict | None,
    embedding_bank_path: Path,
) -> dict[str, object]:
    def coalesce(*values):
        for value in values:
            if value is not None:
                return value
        return None

    return {
        "embedding_bank": str(embedding_bank_path),
        "prior_model": coalesce(
            args.prior_model,
            config.get("prior_model") if config is not None else None,
            DEFAULT_KANDINSKY_PRIOR_MODEL,
        ),
        "decoder_model": coalesce(
            args.decoder_model,
            config.get("decoder_model") if config is not None else None,
            DEFAULT_KANDINSKY_DECODER_MODEL,
        ),
        "num_candidates": int(coalesce(args.num_candidates, config.get("num_candidates") if config else None, 4)),
        "decoder_steps": int(coalesce(args.decoder_steps, config.get("decoder_steps") if config else None, 50)),
        "decoder_guidance_scale": float(
            coalesce(args.decoder_guidance_scale, config.get("decoder_guidance_scale") if config else None, 4.0)
        ),
        "decoder_height": int(coalesce(args.decoder_height, config.get("decoder_height") if config else None, 512)),
        "decoder_width": int(coalesce(args.decoder_width, config.get("decoder_width") if config else None, 512)),
    }


def load_embedding_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    retrieval_model = build_retrieval_model_from_config(config["retrieval_config"])
    eeg_encoder = copy.deepcopy(retrieval_model.encoder)
    model_type = config.get("model_type", "regression")

    if model_type == "rag_residual":
        model = GatedRetrievalResidualRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=int(config["backbone_dim"]),
            target_dim=int(config["embedding_dim"]),
            retrieval_dim=int(config.get("retrieval_bank_dim") or config["retrieval_config"].get("perceptual_dim") or config["retrieval_config"].get("semantic_dim") or config["embedding_dim"]),
            text_dim=config.get("text_dim"),
            hidden_dim=int(config.get("head_hidden_dim", 1024)),
            attention_heads=int(config.get("attention_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    else:
        model = EEGEmbeddingRegressor(
            eeg_encoder=eeg_encoder,
            backbone_dim=int(config["backbone_dim"]),
            target_dim=int(config["embedding_dim"]),
            hidden_dim=int(config.get("head_hidden_dim", 1024)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, model_type


def load_retrieval_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, payload


def build_context_batch(
    query_embeddings: torch.Tensor,
    *,
    retrieval_bank: TensorBank,
    embedding_bank: TensorBank,
    device: torch.device,
    topk: int,
    text_bank: TensorBank | None = None,
) -> dict[str, object]:
    effective_topk = min(topk, len(retrieval_bank.image_ids))
    logits = query_embeddings.float().cpu() @ retrieval_bank.values.float().T.cpu()
    _, topk_ids, topk_scores = prototype_lookup_from_logits(
        logits,
        candidate_image_ids=list(retrieval_bank.image_ids),
        topk=effective_topk,
    )
    flat_ids = [image_id for row in topk_ids for image_id in row]
    retrieved_embeddings = embedding_bank.align(flat_ids, device=device).float()
    retrieved_embeddings = retrieved_embeddings.reshape(query_embeddings.shape[0], effective_topk, -1)
    retrieved_text_embeddings = None
    if text_bank is not None:
        retrieved_text_embeddings = text_bank.align(flat_ids, device=device).float()
        retrieved_text_embeddings = retrieved_text_embeddings.reshape(query_embeddings.shape[0], effective_topk, -1)
    topk_probs = torch.softmax(topk_scores.float(), dim=-1)
    confidence = topk_probs[:, 0].to(device)
    return {
        "topk_ids": topk_ids,
        "topk_scores": topk_scores,
        "topk_probs": topk_probs,
        "confidence": confidence,
        "retrieved_embeddings": retrieved_embeddings,
        "retrieved_text_embeddings": retrieved_text_embeddings,
    }


def resolve_conditioning_embeddings(
    *,
    args: argparse.Namespace,
    batch: dict[str, object],
    device: torch.device,
    embedding_bank: TensorBank,
    embedding_model,
    embedding_config: dict | None,
    model_type: str | None,
    retrieval_bank: TensorBank | None,
    text_bank: TensorBank | None,
    retrieval_model: torch.nn.Module | None,
    semantic_bank: TensorBank | None,
    perceptual_bank: TensorBank | None,
    alpha: float | None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    query_ids = list(batch["image_id"])
    eeg = batch["eeg"].to(device)

    if args.embedding_source == "ground_truth":
        target = embedding_bank.align(query_ids, device=device).float()
        context = [{"conditioning_source": "ground_truth", "conditioning_image_id": image_id} for image_id in query_ids]
        return target, context

    if args.embedding_source == "retrieval_top1":
        if retrieval_model is None or alpha is None:
            raise ValueError("Retrieval conditioning requires a retrieval checkpoint and a resolved alpha.")

        with torch.no_grad():
            output_bundle = retrieval_model.encode_all(eeg)
        output_dict = {
            name: tensor
            for name, tensor in {
                "semantic": output_bundle.semantic,
                "perceptual": output_bundle.perceptual,
                "legacy": output_bundle.legacy,
            }.items()
            if tensor is not None
        }
        logits, candidate_ids, _ = compute_retrieval_logits(
            retrieval_model,
            output_dict,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            alpha=alpha,
        )
        _, top1_ids, top1_scores = prototype_lookup_from_logits(
            logits,
            candidate_image_ids=candidate_ids,
            topk=1,
        )
        conditioning_image_ids = [row[0] for row in top1_ids]
        embeddings = embedding_bank.align(conditioning_image_ids, device=device).float()
        context = [
            {
                "conditioning_source": "retrieval_top1",
                "conditioning_image_id": image_id,
                "conditioning_score": float(score),
            }
            for image_id, score in zip(conditioning_image_ids, top1_scores[:, 0].tolist(), strict=True)
        ]
        return embeddings, context

    if embedding_model is None:
        raise ValueError("--reconstruction-checkpoint is required when --embedding-source=predicted.")

    effective_model_type = args.model_type or model_type or (embedding_config.get("model_type") if embedding_config else "regression")
    if effective_model_type == "rag_residual":
        if retrieval_bank is None:
            raise ValueError("rag_residual prediction requires a retrieval bank.")
        with torch.no_grad():
            shared = embedding_model.encode_backbone(eeg)
            base_outputs = embedding_model.project_shared(shared)
            effective_topk = args.retrieval_topk or int(embedding_config.get("retrieval_topk", 4))
            use_text_context = args.use_text_context or bool(embedding_config.get("use_text_context", False))
            context_batch = build_context_batch(
                base_outputs.retrieval_embedding,
                retrieval_bank=retrieval_bank,
                embedding_bank=embedding_bank,
                device=device,
                topk=effective_topk,
                text_bank=text_bank if use_text_context else None,
            )
            final_outputs = embedding_model.project_shared(
                shared,
                retrieved_embeddings=context_batch["retrieved_embeddings"],
                retrieved_text_embeddings=context_batch["retrieved_text_embeddings"],
                retrieval_confidence=context_batch["confidence"],
            )
        conditioning_context = []
        for topk_ids, topk_scores, topk_probs, gate, conf in zip(
            context_batch["topk_ids"],
            context_batch["topk_scores"].tolist(),
            context_batch["topk_probs"].tolist(),
            final_outputs.gate.squeeze(-1).tolist(),
            context_batch["confidence"].tolist(),
            strict=True,
        ):
            conditioning_context.append(
                {
                    "conditioning_source": "predicted",
                    "conditioning_image_id": None,
                    "retrieval_topk_ids": list(topk_ids),
                    "retrieval_topk_scores": [float(score) for score in topk_scores],
                    "retrieval_topk_probs": [float(prob) for prob in topk_probs],
                    "retrieval_confidence": float(conf),
                    "residual_gate": float(gate),
                }
            )
        return final_outputs.final_embedding, conditioning_context

    with torch.no_grad():
        predicted = embedding_model(eeg)
    context = [{"conditioning_source": "predicted", "conditioning_image_id": None} for _ in query_ids]
    return predicted, context


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    embedding_model = None
    embedding_config = None
    checkpoint_model_type = None
    if args.reconstruction_checkpoint is not None:
        embedding_model, embedding_config, checkpoint_model_type = load_embedding_model(args.reconstruction_checkpoint, device)

    if args.embedding_source == "predicted" and embedding_model is None:
        raise ValueError("--reconstruction-checkpoint is required when --embedding-source=predicted.")

    embedding_bank_path = resolve_embedding_bank_path(args, embedding_config, args.reconstruction_checkpoint)
    embedding_bank = TensorBank.load(embedding_bank_path)
    generation_config = resolve_generation_config(args, embedding_config, embedding_bank_path)

    effective_model_type = args.model_type or checkpoint_model_type or (embedding_config.get("model_type") if embedding_config else "regression")
    base_paths = [PROJECT_ROOT]
    if args.reconstruction_checkpoint is not None:
        base_paths.extend([args.reconstruction_checkpoint.parent, args.reconstruction_checkpoint.parent.parent])

    retrieval_bank = resolve_bank(
        args.retrieval_bank,
        embedding_config.get("retrieval_bank") if embedding_config is not None else None,
        name="Retrieval",
        base_paths=base_paths,
    )
    text_bank = resolve_bank(
        args.text_bank,
        embedding_config.get("text_bank") if embedding_config is not None else None,
        name="Text",
        base_paths=base_paths,
    )

    retrieval_model = None
    retrieval_config = None
    retrieval_checkpoint = args.retrieval_checkpoint
    if retrieval_checkpoint is None and embedding_config is not None and embedding_config.get("retrieval_checkpoint"):
        retrieval_checkpoint = resolve_artifact_path(
            embedding_config["retrieval_checkpoint"],
            base_paths=base_paths,
        )

    semantic_bank = None
    perceptual_bank = None
    alpha = None
    if args.embedding_source == "retrieval_top1":
        if retrieval_checkpoint is None:
            raise ValueError(
                "--retrieval-checkpoint is required when --embedding-source=retrieval_top1 "
                "and no retrieval checkpoint is stored in the reconstruction config."
            )
        retrieval_model, retrieval_config, retrieval_payload = load_retrieval_model(retrieval_checkpoint, device)
        semantic_bank = resolve_bank(
            args.semantic_bank,
            retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank"),
            name="Semantic",
            base_paths=[PROJECT_ROOT, retrieval_checkpoint.parent, retrieval_checkpoint.parent.parent],
        )
        perceptual_bank = resolve_bank(
            args.perceptual_bank,
            retrieval_config.get("perceptual_bank"),
            name="Perceptual",
            base_paths=[PROJECT_ROOT, retrieval_checkpoint.parent, retrieval_checkpoint.parent.parent],
        )
        if semantic_bank is None and perceptual_bank is None:
            raise ValueError("At least one retrieval bank must be available for retrieval_top1 conditioning.")
        alpha = resolve_alpha(
            args,
            retrieval_payload,
            has_semantic=semantic_bank is not None,
            has_perceptual=perceptual_bank is not None,
        )

    prior_pipe = load_kandinsky_prior_pipeline(
        model_name=str(generation_config["prior_model"]),
        device=device,
        local_files_only=args.local_files_only,
    )
    decoder_pipe = load_kandinsky_decoder_pipeline(
        model_name=str(generation_config["decoder_model"]),
        device=device,
        local_files_only=args.local_files_only,
    )

    negative_image_embed = negative_image_embed_from_bank(
        embedding_bank,
        batch_size=1,
        device=device,
        dtype=next(decoder_pipe.unet.parameters()).dtype,
    )

    selected_image_ids = load_split_image_ids(args.split_file, image_id_source=args.image_id_source)
    records = load_eeg_records(
        data_dir=args.data_dir,
        split=args.split,
        avg_trials=True,
        image_ids=selected_image_ids,
    )
    loader = make_dataloader(
        EEGImageDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    all_fake_images: list[torch.Tensor] = []
    all_real_images: list[torch.Tensor] = []

    for batch in loader:
        conditioning_embeddings, conditioning_context = resolve_conditioning_embeddings(
            args=args,
            batch=batch,
            device=device,
            embedding_bank=embedding_bank,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
            model_type=effective_model_type,
            retrieval_bank=retrieval_bank,
            text_bank=text_bank,
            retrieval_model=retrieval_model,
            semantic_bank=semantic_bank,
            perceptual_bank=perceptual_bank,
            alpha=alpha,
        )
        generation = generate_best_images(
            predicted_embeddings=conditioning_embeddings,
            prior_pipe=prior_pipe,
            decoder_pipe=decoder_pipe,
            negative_image_embeds=negative_image_embed,
            num_candidates=int(generation_config["num_candidates"]),
            num_inference_steps=int(generation_config["decoder_steps"]),
            guidance_scale=float(generation_config["decoder_guidance_scale"]),
            height=int(generation_config["decoder_height"]),
            width=int(generation_config["decoder_width"]),
        )
        selected_images = generation["selected_images"]
        saved_paths = save_image_batch(selected_images, list(batch["image_id"]), output_images_dir)
        all_fake_images.extend(list(selected_images))

        if args.evaluate:
            real_images = load_image_batch(list(batch["image_path"]), image_size=int(generation_config["decoder_height"]))
            all_real_images.extend(list(real_images))

        candidate_scores = generation["candidate_scores"].tolist()
        selected_indices = generation["selected_candidate_indices"].tolist()
        selected_scores = generation["selected_scores"].tolist()
        candidate_seeds = generation["candidate_seeds"]
        selected_seeds = generation["selected_candidate_seeds"]

        for query_id, output_path, selected_index, selected_seed, selected_score, scores, context in zip(
            batch["image_id"],
            saved_paths,
            selected_indices,
            selected_seeds,
            selected_scores,
            candidate_scores,
            conditioning_context,
            strict=True,
        ):
            item = {
                "query_image_id": query_id,
                "output_path": output_path,
                "mode": "embedding_decoder",
                "model_type": effective_model_type,
                "embedding_source": args.embedding_source,
                "selection_mode": "max_cosine_to_conditioning_embedding",
                "candidate_seeds": candidate_seeds,
                "selected_candidate_index": int(selected_index),
                "selected_candidate_seed": int(selected_seed),
                "selected_cosine": float(selected_score),
                "candidate_scores": [float(score) for score in scores],
            }
            item.update(context)
            metadata.append(item)

    save_json(
        {
            "split": args.split,
            "split_file": None if args.split_file is None else str(args.split_file.resolve()),
            "image_id_source": args.image_id_source,
            "embedding_source": args.embedding_source,
            "embedding_bank": str(embedding_bank_path),
            "retrieval_bank": None if retrieval_bank is None else retrieval_bank.bank_type,
            "text_bank": None if text_bank is None else text_bank.bank_type,
            "mode": "embedding_decoder",
            "model_type": effective_model_type,
            "decoder_model": str(generation_config["decoder_model"]),
            "prior_model": str(generation_config["prior_model"]),
            "num_candidates": float(generation_config["num_candidates"]),
            "decoder_steps": float(generation_config["decoder_steps"]),
            "decoder_guidance_scale": float(generation_config["decoder_guidance_scale"]),
            "decoder_height": float(generation_config["decoder_height"]),
            "decoder_width": float(generation_config["decoder_width"]),
            "reconstruction_checkpoint": None
            if args.reconstruction_checkpoint is None
            else str(args.reconstruction_checkpoint.resolve()),
            "retrieval_checkpoint": None if retrieval_checkpoint is None else str(retrieval_checkpoint.resolve()),
            "alpha": None if alpha is None else float(alpha),
            "local_files_only": bool(args.local_files_only),
            "predictions": metadata,
        },
        args.output_dir / "reconstruction_metadata.json",
    )

    if args.evaluate:
        metrics = eval_images(
            real_images=torch.stack(all_real_images, dim=0),
            fake_images=torch.stack(all_fake_images, dim=0),
            device=device,
        )
        metrics["num_candidates"] = float(generation_config["num_candidates"])
        metrics["decoder_steps"] = float(generation_config["decoder_steps"])
        metrics["decoder_guidance_scale"] = float(generation_config["decoder_guidance_scale"])
        save_json(metrics, args.output_dir / "reconstruction_metrics.json")
        print(metrics)
    else:
        print(f"saved reconstruction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
