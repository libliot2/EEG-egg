from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from .data import list_release_images
from .image_banks import TensorBank
from .utils import DEFAULT_DATA_DIR, resolve_device


DEFAULT_KANDINSKY_PRIOR_MODEL = "kandinsky-community/kandinsky-2-2-prior"
DEFAULT_KANDINSKY_DECODER_MODEL = "kandinsky-community/kandinsky-2-2-decoder"
LOCAL_HF_MODEL_ROOTS = (
    Path("/data/xiaoh/DeepLearning_storage/hf_models"),
    Path("/data1/xiaoh/DeepLearning_storage/hf_models"),
)


@dataclass
class KandinskyImageEmbedder:
    image_encoder: object
    image_processor: object

    @property
    def device(self) -> torch.device:
        return next(self.image_encoder.parameters()).device

    def get_zero_embed(self, batch_size: int = 1, device: str | torch.device | None = None) -> torch.Tensor:
        device_obj = resolve_device(device) if device is not None else self.device
        dtype = next(self.image_encoder.parameters()).dtype
        image_size = int(self.image_encoder.config.image_size)
        zero_img = torch.zeros(1, 3, image_size, image_size, device=device_obj, dtype=dtype)
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        return zero_image_emb.repeat(batch_size, 1)


def kandinsky_torch_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def resolve_local_hf_model_path(model_name: str | Path, *, local_files_only: bool) -> str | Path:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate
    if not local_files_only:
        return model_name
    model_basename = str(model_name).rstrip("/").split("/")[-1]
    for root in LOCAL_HF_MODEL_ROOTS:
        resolved = root / model_basename
        if resolved.exists():
            return resolved
    return model_name


def load_kandinsky_prior_pipeline(
    *,
    model_name: str = DEFAULT_KANDINSKY_PRIOR_MODEL,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    device_obj = resolve_device(device)
    resolved_model_name = resolve_local_hf_model_path(model_name, local_files_only=local_files_only)
    image_processor = CLIPImageProcessor.from_pretrained(
        resolved_model_name,
        subfolder="image_processor",
        local_files_only=local_files_only,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        resolved_model_name,
        subfolder="image_encoder",
        torch_dtype=kandinsky_torch_dtype(device_obj),
        local_files_only=local_files_only,
    )
    image_encoder = image_encoder.to(device_obj)
    image_encoder.eval().requires_grad_(False)
    return KandinskyImageEmbedder(
        image_encoder=image_encoder,
        image_processor=image_processor,
    )


def load_kandinsky_decoder_pipeline(
    *,
    model_name: str = DEFAULT_KANDINSKY_DECODER_MODEL,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
):
    from diffusers import KandinskyV22Pipeline

    device_obj = resolve_device(device)
    resolved_model_name = resolve_local_hf_model_path(model_name, local_files_only=local_files_only)
    pipe = KandinskyV22Pipeline.from_pretrained(
        resolved_model_name,
        torch_dtype=kandinsky_torch_dtype(device_obj),
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device_obj)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_kandinsky_img2img_decoder_pipeline(
    *,
    model_name: str = DEFAULT_KANDINSKY_DECODER_MODEL,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
):
    from diffusers import KandinskyV22Img2ImgPipeline

    device_obj = resolve_device(device)
    resolved_model_name = resolve_local_hf_model_path(model_name, local_files_only=local_files_only)
    pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
        resolved_model_name,
        torch_dtype=kandinsky_torch_dtype(device_obj),
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device_obj)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _encoder_device_dtype(pipe) -> tuple[torch.device, torch.dtype]:
    parameter = next(pipe.image_encoder.parameters())
    return parameter.device, parameter.dtype


@torch.no_grad()
def encode_images_with_prior(pipe, images: list[Image.Image] | list[torch.Tensor]) -> torch.Tensor:
    device, dtype = _encoder_device_dtype(pipe)
    pixel_values = pipe.image_processor(images, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
    image_embeds = pipe.image_encoder(pixel_values)["image_embeds"]
    return image_embeds.float()


def build_kandinsky_bank(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str,
    model_name: str = DEFAULT_KANDINSKY_PRIOR_MODEL,
    batch_size: int = 16,
    device: str | None = None,
    local_files_only: bool = False,
) -> TensorBank:
    device_obj = resolve_device(device)
    pipe = load_kandinsky_prior_pipeline(
        model_name=model_name,
        device=device_obj,
        local_files_only=local_files_only,
    )
    image_paths = list_release_images(data_dir, split)

    all_ids: list[str] = []
    all_paths: list[str] = []
    all_values: list[torch.Tensor] = []

    negative_image_embed = pipe.get_zero_embed(batch_size=1, device=device_obj).squeeze(0).float().cpu()

    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"kandinsky-{split}"):
        chunk = image_paths[start : start + batch_size]
        images: list[Image.Image] = []
        for path in chunk:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        features = encode_images_with_prior(pipe, images).cpu()
        all_ids.extend([path.stem for path in chunk])
        all_paths.extend([str(path) for path in chunk])
        all_values.append(features)

    values = torch.cat(all_values, dim=0)
    return TensorBank(
        bank_type="kandinsky",
        image_ids=all_ids,
        image_paths=all_paths,
        values=values,
        metadata={
            "model_name": model_name,
            "split": split,
            "embedding_dim": int(values.shape[1]),
            "negative_image_embed": negative_image_embed,
        },
    )


def negative_image_embed_from_bank(
    bank: TensorBank,
    *,
    batch_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    raw = bank.metadata.get("negative_image_embed")
    if raw is None:
        raise KeyError("Kandinsky bank metadata does not contain `negative_image_embed`.")

    values = torch.as_tensor(raw).view(1, -1)
    if device is not None:
        values = values.to(device)
    if dtype is not None:
        values = values.to(dtype=dtype)
    return values.repeat(batch_size, 1)


@torch.no_grad()
def generate_candidate_images(
    decoder_pipe,
    image_embeds: torch.Tensor,
    negative_image_embeds: torch.Tensor,
    *,
    candidate_seeds: Iterable[int],
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    init_images: torch.Tensor | None = None,
    strength: float = 0.3,
) -> torch.Tensor:
    candidate_seeds = list(candidate_seeds)
    if not candidate_seeds:
        raise ValueError("candidate_seeds must not be empty.")

    device = next(decoder_pipe.unet.parameters()).device
    dtype = next(decoder_pipe.unet.parameters()).dtype
    condition = image_embeds.to(device=device, dtype=dtype)
    negatives = negative_image_embeds.to(device=device, dtype=dtype)
    init_batch = None
    if init_images is not None:
        init_batch = init_images.to(device=device, dtype=dtype)

    candidate_images: list[torch.Tensor] = []
    generator_device = device.type if device.type == "cuda" else "cpu"
    for seed in candidate_seeds:
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        decoder_kwargs = {
            "image_embeds": condition,
            "negative_image_embeds": negatives,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pt",
        }
        if init_batch is not None:
            decoder_kwargs["image"] = init_batch
            decoder_kwargs["strength"] = strength
        output = decoder_pipe(
            **decoder_kwargs,
        )
        candidate_images.append(output.images.float().cpu())

    return torch.stack(candidate_images, dim=0)


@torch.no_grad()
def select_best_candidate_images(
    prior_pipe,
    predicted_embeddings: torch.Tensor,
    candidate_images: torch.Tensor,
    *,
    init_images: torch.Tensor | None = None,
    selection_mode: str = "semantic",
    lowlevel_weight: float = 0.0,
    lowlevel_metric: str = "pixel_cosine",
    score_normalization: str = "per_query_minmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if candidate_images.ndim != 5:
        raise ValueError("candidate_images must have shape [num_candidates, batch, channels, height, width].")
    if selection_mode not in {"semantic", "semantic_lowlevel"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    if lowlevel_metric not in {"pixel_cosine", "neg_mse"}:
        raise ValueError(f"Unsupported lowlevel_metric: {lowlevel_metric}")
    if score_normalization not in {"none", "per_query_minmax", "per_query_zscore"}:
        raise ValueError(f"Unsupported score_normalization: {score_normalization}")

    num_candidates, batch_size = candidate_images.shape[:2]
    flattened = candidate_images.reshape(num_candidates * batch_size, *candidate_images.shape[2:])
    pil_images = [to_pil_image(image) for image in flattened]
    candidate_embeds = encode_images_with_prior(prior_pipe, pil_images)
    candidate_embeds = candidate_embeds.view(num_candidates, batch_size, -1).permute(1, 0, 2)

    pred_norm = F.normalize(predicted_embeddings.to(candidate_embeds.device).float(), dim=-1)
    cand_norm = F.normalize(candidate_embeds.float(), dim=-1)
    semantic_scores = (cand_norm * pred_norm.unsqueeze(1)).sum(dim=-1)
    candidate_scores = semantic_scores
    lowlevel_scores = None

    if selection_mode == "semantic_lowlevel":
        if init_images is None:
            raise ValueError("selection_mode='semantic_lowlevel' requires init_images.")
        lowlevel_scores = compute_lowlevel_candidate_scores(
            candidate_images,
            init_images,
            metric=lowlevel_metric,
        ).to(semantic_scores.device)
        candidate_scores = combine_candidate_scores(
            semantic_scores,
            lowlevel_scores,
            lowlevel_weight=lowlevel_weight,
            normalization=score_normalization,
        )
    selected_indices = candidate_scores.argmax(dim=1)

    selected_images = torch.stack(
        [candidate_images[idx, row].cpu() for row, idx in enumerate(selected_indices.tolist())],
        dim=0,
    )
    selected_scores = candidate_scores[torch.arange(batch_size, device=candidate_scores.device), selected_indices].cpu()
    return selected_images, selected_indices.cpu(), selected_scores, candidate_scores.cpu(), semantic_scores.cpu(), (
        None if lowlevel_scores is None else lowlevel_scores.cpu()
    )


def normalize_candidate_scores(scores: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "none":
        return scores
    if mode == "per_query_minmax":
        min_values = scores.min(dim=1, keepdim=True).values
        max_values = scores.max(dim=1, keepdim=True).values
        return (scores - min_values) / (max_values - min_values).clamp_min(1e-6)
    if mode == "per_query_zscore":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False)
        return (scores - mean) / std.clamp_min(1e-6)
    raise ValueError(f"Unsupported score normalization mode: {mode}")


def combine_candidate_scores(
    semantic_scores: torch.Tensor,
    lowlevel_scores: torch.Tensor,
    *,
    lowlevel_weight: float,
    normalization: str,
) -> torch.Tensor:
    semantic = normalize_candidate_scores(semantic_scores.float(), mode=normalization)
    lowlevel = normalize_candidate_scores(lowlevel_scores.float(), mode=normalization)
    return semantic + float(lowlevel_weight) * lowlevel


def compute_lowlevel_candidate_scores(
    candidate_images: torch.Tensor,
    init_images: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    if init_images.ndim != 4:
        raise ValueError("init_images must have shape [batch, channels, height, width].")
    num_candidates, batch_size = candidate_images.shape[:2]
    if init_images.shape[0] != batch_size:
        raise ValueError(
            f"init_images batch size {init_images.shape[0]} does not match candidate batch size {batch_size}."
        )

    candidates = candidate_images.float()
    init = init_images.float().cpu()
    if candidates.shape[-2:] != init.shape[-2:]:
        init = F.interpolate(init, size=candidates.shape[-2:], mode="bilinear", align_corners=False)
    if candidates.shape[-1] > 128 or candidates.shape[-2] > 128:
        target_size = (128, 128)
        channels = candidates.shape[2]
        candidates = F.interpolate(
            candidates.reshape(num_candidates * batch_size, *candidates.shape[2:]),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(num_candidates, batch_size, channels, *target_size)
        init = F.interpolate(init, size=target_size, mode="bilinear", align_corners=False)

    if metric == "pixel_cosine":
        flat_candidates = candidates.flatten(start_dim=2)
        flat_init = init.flatten(start_dim=1)
        flat_candidates = F.normalize(flat_candidates, dim=-1)
        flat_init = F.normalize(flat_init, dim=-1)
        return (flat_candidates * flat_init.unsqueeze(0)).sum(dim=-1).transpose(0, 1)
    if metric == "neg_mse":
        return -((candidates - init.unsqueeze(0)) ** 2).mean(dim=(2, 3, 4)).transpose(0, 1)
    raise ValueError(f"Unsupported lowlevel metric: {metric}")


@torch.no_grad()
def generate_best_images(
    *,
    predicted_embeddings: torch.Tensor,
    prior_pipe,
    decoder_pipe,
    negative_image_embeds: torch.Tensor,
    num_candidates: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    height: int = 512,
    width: int = 512,
    init_images: torch.Tensor | None = None,
    img2img_strength: float = 0.3,
    candidate_seed_offset: int = 0,
    selection_mode: str = "semantic",
    candidate_lowlevel_weight: float = 0.0,
    candidate_lowlevel_metric: str = "pixel_cosine",
    candidate_score_normalization: str = "per_query_minmax",
) -> dict[str, torch.Tensor | list[int]]:
    candidate_seeds = list(range(candidate_seed_offset, candidate_seed_offset + num_candidates))
    negatives = negative_image_embeds
    if negatives.shape[0] == 1 and predicted_embeddings.shape[0] > 1:
        negatives = negatives.repeat(predicted_embeddings.shape[0], 1)

    candidate_images = generate_candidate_images(
        decoder_pipe,
        predicted_embeddings,
        negatives,
        candidate_seeds=candidate_seeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        init_images=init_images,
        strength=img2img_strength,
    )
    (
        selected_images,
        selected_indices,
        selected_scores,
        candidate_scores,
        candidate_semantic_scores,
        candidate_lowlevel_scores,
    ) = select_best_candidate_images(
        prior_pipe,
        predicted_embeddings,
        candidate_images,
        init_images=init_images,
        selection_mode=selection_mode,
        lowlevel_weight=candidate_lowlevel_weight,
        lowlevel_metric=candidate_lowlevel_metric,
        score_normalization=candidate_score_normalization,
    )
    selected_seeds = [candidate_seeds[index] for index in selected_indices.tolist()]
    return {
        "selected_images": selected_images,
        "candidate_images": candidate_images,
        "selected_candidate_indices": selected_indices,
        "selected_candidate_seeds": selected_seeds,
        "selected_scores": selected_scores,
        "candidate_scores": candidate_scores,
        "candidate_semantic_scores": candidate_semantic_scores,
        "candidate_lowlevel_scores": candidate_lowlevel_scores,
        "candidate_seeds": candidate_seeds,
        "generation_mode": "img2img" if init_images is not None else "text2img",
        "selection_mode": selection_mode,
    }
