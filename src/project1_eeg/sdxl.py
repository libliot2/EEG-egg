from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from .kandinsky import kandinsky_torch_dtype, resolve_local_hf_model_path
from .utils import resolve_device


DEFAULT_SDXL_IMG2IMG_MODEL = "stabilityai/sdxl-turbo"


def load_sdxl_img2img_pipeline(
    *,
    model_name: str | Path = DEFAULT_SDXL_IMG2IMG_MODEL,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
):
    from diffusers import AutoPipelineForImage2Image

    device_obj = resolve_device(device)
    resolved_model_name = resolve_local_hf_model_path(model_name, local_files_only=local_files_only)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        resolved_model_name,
        torch_dtype=kandinsky_torch_dtype(device_obj),
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device_obj)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_sdxl_img2img_images(
    pipe,
    *,
    init_images: torch.Tensor,
    prompts: list[str],
    negative_prompt: str | None = None,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int = 0,
) -> torch.Tensor:
    if init_images.ndim != 4:
        raise ValueError(f"Expected init_images to have shape [B, C, H, W], got {tuple(init_images.shape)}")
    if len(prompts) != init_images.shape[0]:
        raise ValueError("prompts length must match init_images batch size.")
    if not (0.0 < strength <= 1.0):
        raise ValueError("strength must be in (0, 1].")

    pil_images = [to_pil_image(image.clamp(0.0, 1.0).cpu()) for image in init_images]

    module = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
    parameter = next(module.parameters())
    generator_device = parameter.device.type if parameter.device.type == "cuda" else "cpu"
    generators = [
        torch.Generator(device=generator_device).manual_seed(int(seed) + index)
        for index in range(len(prompts))
    ]

    negative_prompts = None
    if negative_prompt is not None:
        negative_prompts = [negative_prompt] * len(prompts)

    output = pipe(
        prompt=prompts,
        image=pil_images,
        negative_prompt=negative_prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generators,
        output_type="pt",
    )
    return torch.as_tensor(output.images).float().cpu()
