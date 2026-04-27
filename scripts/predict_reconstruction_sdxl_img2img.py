#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project1_eeg.data import EEGImageDataset, list_release_images, load_eeg_records, load_split_image_ids
from project1_eeg.evaluation import eval_images
from project1_eeg.image_banks import TensorBank
from project1_eeg.retrieval import build_retrieval_model_from_config
from project1_eeg.runtime import compute_retrieval_logits, make_dataloader, prototype_lookup_from_logits, save_image_batch
from project1_eeg.sdxl import DEFAULT_SDXL_IMG2IMG_MODEL, generate_sdxl_img2img_images, load_sdxl_img2img_pipeline
from project1_eeg.utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, load_image_batch, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SDXL img2img reconstruction feasibility outputs from retrieval prototypes."
    )
    parser.add_argument("--retrieval-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--semantic-bank", type=Path, default=None)
    parser.add_argument("--perceptual-bank", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "reconstruction_predictions_sdxl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-image-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--image-id-source", choices=["all", "train_ids", "val_ids"], default="all")
    parser.add_argument("--prototype-topk", type=int, default=1)
    parser.add_argument(
        "--prompt-source",
        choices=["prototype_text", "prototype_image_id", "query_text", "none"],
        default="prototype_text",
    )
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdxl-model", type=str, default=DEFAULT_SDXL_IMG2IMG_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--init-image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--strength", type=float, default=0.4)
    return parser.parse_args()


def load_retrieval_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_retrieval_model_from_config(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, config, payload


def resolve_bank(path: Path | None, fallback: str | None, *, name: str) -> TensorBank | None:
    bank_path = path or (Path(fallback) if fallback else None)
    if bank_path is None:
        return None
    if not bank_path.exists():
        raise FileNotFoundError(f"{name} bank not found: {bank_path}")
    return TensorBank.load(bank_path)


def resolve_alpha(
    args: argparse.Namespace,
    retrieval_payload: dict,
    *,
    has_semantic: bool,
    has_perceptual: bool,
) -> float:
    if args.alpha is not None:
        return float(args.alpha)
    metrics = retrieval_payload.get("metrics", {})
    if "val_selected_alpha" in metrics:
        return float(metrics["val_selected_alpha"])
    if not has_semantic:
        return 0.0
    if not has_perceptual:
        return 1.0
    return 0.5


def concept_text_lookup(
    *,
    data_dir: Path,
    split: str,
) -> dict[str, str]:
    records = load_eeg_records(data_dir=data_dir, split=split, avg_trials=True)
    lookup: dict[str, str] = {}
    for record in records:
        text = record.concept_text.strip()
        lookup.setdefault(record.image_id, text or record.image_id.replace("_", " "))
    return lookup


def image_id_to_prompt(image_id: str) -> str:
    return image_id.replace("_", " ")


def resolve_prompts(
    *,
    prompt_source: str,
    query_ids: list[str],
    query_texts: list[str],
    prototype_ids: list[str],
    prototype_text_lookup_map: dict[str, str],
) -> list[str]:
    prompts: list[str] = []
    for query_id, query_text, prototype_id in zip(query_ids, query_texts, prototype_ids, strict=True):
        if prompt_source == "none":
            prompt = ""
        elif prompt_source == "query_text":
            prompt = query_text.strip() or image_id_to_prompt(query_id)
        elif prompt_source == "prototype_image_id":
            prompt = image_id_to_prompt(prototype_id)
        else:
            prompt = prototype_text_lookup_map.get(prototype_id, "") or image_id_to_prompt(prototype_id)
        prompts.append(prompt)
    return prompts


def build_train_image_path_lookup(data_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in list_release_images(data_dir, "train"):
        lookup[path.stem] = str(path)
    return lookup


def resolve_prototype_paths(
    prototype_ids: list[str],
    source_bank: TensorBank,
    *,
    train_image_lookup: dict[str, str],
) -> list[str]:
    prototype_paths: list[str] = []
    for image_id in prototype_ids:
        bank_path = Path(source_bank.image_paths[source_bank._index[image_id]])
        if bank_path.exists():
            prototype_paths.append(str(bank_path))
            continue
        fallback_path = train_image_lookup.get(image_id)
        if fallback_path is None:
            raise FileNotFoundError(f"Could not resolve prototype image path for {image_id}.")
        prototype_paths.append(fallback_path)
    return prototype_paths


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    retrieval_model, retrieval_config, retrieval_payload = load_retrieval_model(args.retrieval_checkpoint, device)

    semantic_bank = resolve_bank(
        args.semantic_bank,
        retrieval_config.get("semantic_bank") or retrieval_config.get("clip_bank"),
        name="Semantic",
    )
    perceptual_bank = resolve_bank(args.perceptual_bank, retrieval_config.get("perceptual_bank"), name="Perceptual")
    if semantic_bank is None and perceptual_bank is None:
        raise ValueError("At least one retrieval bank must be available.")

    alpha = resolve_alpha(
        args,
        retrieval_payload,
        has_semantic=semantic_bank is not None,
        has_perceptual=perceptual_bank is not None,
    )
    source_bank = semantic_bank if semantic_bank is not None else perceptual_bank
    assert source_bank is not None

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

    prototype_texts = concept_text_lookup(data_dir=args.data_dir, split="train")
    train_image_lookup = build_train_image_path_lookup(args.data_dir)
    pipe = load_sdxl_img2img_pipeline(
        model_name=args.sdxl_model,
        device=device,
        local_files_only=args.local_files_only,
    )

    output_images_dir = args.output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    all_fake_images = []
    all_real_images = []

    for batch_index, batch in enumerate(loader):
        eeg = batch["eeg"].to(device)
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
        _, topk_ids, topk_scores = prototype_lookup_from_logits(
            logits,
            candidate_image_ids=candidate_ids,
            topk=args.prototype_topk,
        )

        prototype_ids = [row[0] for row in topk_ids]
        prototype_paths = resolve_prototype_paths(
            prototype_ids,
            source_bank,
            train_image_lookup=train_image_lookup,
        )
        prompts = resolve_prompts(
            prompt_source=args.prompt_source,
            query_ids=list(batch["image_id"]),
            query_texts=list(batch["concept_text"]),
            prototype_ids=prototype_ids,
            prototype_text_lookup_map=prototype_texts,
        )

        init_images = load_image_batch(
            prototype_paths,
            image_size=args.init_image_size,
        )
        fake_images = generate_sdxl_img2img_images(
            pipe,
            init_images=init_images,
            prompts=prompts,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed + batch_index * max(1, len(prompts)),
        )
        batch_saved = save_image_batch(fake_images, list(batch["image_id"]), output_images_dir)

        for query_id, prompt, ranked_ids, ranked_scores, output_path in zip(
            batch["image_id"],
            prompts,
            topk_ids,
            topk_scores.cpu().tolist(),
            batch_saved,
            strict=True,
        ):
            metadata.append(
                {
                    "query_image_id": query_id,
                    "prototype_image_id": ranked_ids[0],
                    "prototype_topk_ids": ranked_ids,
                    "prototype_topk_scores": ranked_scores,
                    "prompt": prompt,
                    "prompt_source": args.prompt_source,
                    "output_path": output_path,
                    "mode": "sdxl_img2img_feasibility",
                }
            )

        if args.evaluate:
            real_images = load_image_batch(batch["image_path"], image_size=args.eval_image_size)
            all_real_images.extend(list(real_images.cpu()))
            all_fake_images.extend(list(fake_images.cpu()))

    save_json(
        {
            "split": args.split,
            "image_id_source": args.image_id_source,
            "alpha": alpha,
            "prototype_topk": int(args.prototype_topk),
            "prompt_source": args.prompt_source,
            "sdxl_model": args.sdxl_model,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "strength": float(args.strength),
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
        metrics["alpha"] = float(alpha)
        metrics["prototype_topk"] = float(args.prototype_topk)
        metrics["num_inference_steps"] = float(args.num_inference_steps)
        metrics["guidance_scale"] = float(args.guidance_scale)
        metrics["strength"] = float(args.strength)
        save_json(metrics, args.output_dir / "reconstruction_metrics.json")
        print(metrics)
    else:
        print(f"saved SDXL feasibility outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
