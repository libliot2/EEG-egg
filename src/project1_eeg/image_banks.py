from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import list_release_images, load_eeg_records
from .utils import DEFAULT_DATA_DIR, ensure_dir, load_image_batch, resolve_device


@dataclass
class TensorBank:
    bank_type: str
    image_ids: list[str]
    image_paths: list[str]
    values: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    _index: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {image_id: idx for idx, image_id in enumerate(self.image_ids)}

    def align(self, image_ids: Iterable[str], *, device: torch.device | None = None) -> torch.Tensor:
        indices = [self._index[image_id] for image_id in image_ids]
        values = self.values[indices]
        if device is not None:
            values = values.to(device)
        return values

    def subset(self, image_ids: Iterable[str]) -> "TensorBank":
        image_ids = list(image_ids)
        indices = [self._index[image_id] for image_id in image_ids]
        return TensorBank(
            bank_type=self.bank_type,
            image_ids=image_ids,
            image_paths=[self.image_paths[idx] for idx in indices],
            values=self.values[indices],
            metadata=dict(self.metadata),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "bank_type": self.bank_type,
                "image_ids": self.image_ids,
                "image_paths": self.image_paths,
                "values": self.values.cpu(),
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "TensorBank":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            bank_type=payload["bank_type"],
            image_ids=list(payload["image_ids"]),
            image_paths=list(payload["image_paths"]),
            values=torch.as_tensor(payload["values"]),
            metadata=dict(payload.get("metadata", {})),
        )


def default_bank_path(output_dir: str | Path, bank_type: str, split: str) -> Path:
    return Path(output_dir) / "cache" / f"{bank_type}_{split}.pt"


def build_clip_bank(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str,
    model_name: str = "ViT-L/14",
    batch_size: int = 32,
    device: str | None = None,
) -> TensorBank:
    import clip
    from PIL import Image

    device_obj = resolve_device(device)
    image_paths = list_release_images(data_dir, split)
    model, preprocess = clip.load(model_name, device=device_obj, jit=False)
    model.eval()

    all_ids: list[str] = []
    all_paths: list[str] = []
    all_values: list[torch.Tensor] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"clip-{split}"):
        chunk = image_paths[start : start + batch_size]
        images = []
        for path in chunk:
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB"))
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                images.append(tensor)
        batch = torch.stack(images, dim=0).to(device_obj)
        with torch.no_grad():
            features = model.encode_image(batch).float()
            features = F.normalize(features, dim=-1).cpu()
        all_ids.extend([path.stem for path in chunk])
        all_paths.extend([str(path) for path in chunk])
        all_values.append(features)

    return TensorBank(
        bank_type="clip",
        image_ids=all_ids,
        image_paths=all_paths,
        values=torch.cat(all_values, dim=0),
        metadata={"model_name": model_name, "split": split},
    )


def build_clip_text_bank(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str,
    model_name: str = "ViT-L/14",
    batch_size: int = 128,
    device: str | None = None,
) -> TensorBank:
    import clip

    device_obj = resolve_device(device)
    image_paths = list_release_images(data_dir, split)
    records = load_eeg_records(data_dir=data_dir, split=split, avg_trials=True)
    text_by_image_id: dict[str, str] = {}
    path_by_image_id: dict[str, str] = {}
    for record in records:
        text = record.concept_text.strip() or record.image_id.replace("_", " ")
        text_by_image_id.setdefault(record.image_id, text)
        path_by_image_id.setdefault(record.image_id, record.image_path)

    model, _ = clip.load(model_name, device=device_obj, jit=False)
    model.eval()

    all_ids: list[str] = []
    all_paths: list[str] = []
    all_texts: list[str] = []
    all_values: list[torch.Tensor] = []

    ordered_items: list[tuple[str, str, str]] = []
    for path in image_paths:
        image_id = path.stem
        ordered_items.append(
            (
                image_id,
                path_by_image_id.get(image_id, str(path)),
                text_by_image_id.get(image_id, image_id.replace("_", " ")),
            )
        )

    for start in tqdm(range(0, len(ordered_items), batch_size), desc=f"clip-text-{split}"):
        chunk = ordered_items[start : start + batch_size]
        tokens = clip.tokenize([item[2] for item in chunk], truncate=True).to(device_obj)
        with torch.no_grad():
            features = model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1).cpu()
        all_ids.extend([item[0] for item in chunk])
        all_paths.extend([item[1] for item in chunk])
        all_texts.extend([item[2] for item in chunk])
        all_values.append(features)

    return TensorBank(
        bank_type="clip_text",
        image_ids=all_ids,
        image_paths=all_paths,
        values=torch.cat(all_values, dim=0),
        metadata={"model_name": model_name, "split": split, "texts": all_texts},
    )


def build_dreamsim_bank(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str,
    dreamsim_type: str = "ensemble",
    batch_size: int = 16,
    device: str | None = None,
) -> TensorBank:
    from PIL import Image
    from dreamsim import dreamsim

    device_obj = resolve_device(device)
    image_paths = list_release_images(data_dir, split)
    model, preprocess = dreamsim(pretrained=True, device=str(device_obj), dreamsim_type=dreamsim_type)
    model.eval()

    all_ids: list[str] = []
    all_paths: list[str] = []
    all_values: list[torch.Tensor] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"dreamsim-{split}"):
        chunk = image_paths[start : start + batch_size]
        images = []
        for path in chunk:
            with Image.open(path) as image:
                tensor = preprocess(image.convert("RGB"))
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                images.append(tensor)
        batch = torch.stack(images, dim=0).to(device_obj)
        with torch.no_grad():
            features = model.embed(batch)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = torch.as_tensor(features).float().flatten(1)
            features = F.normalize(features, dim=-1).cpu()
        all_ids.extend([path.stem for path in chunk])
        all_paths.extend([str(path) for path in chunk])
        all_values.append(features)

    return TensorBank(
        bank_type="dreamsim",
        image_ids=all_ids,
        image_paths=all_paths,
        values=torch.cat(all_values, dim=0),
        metadata={"dreamsim_type": dreamsim_type, "split": split},
    )


def build_vae_latent_bank(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str,
    model_name: str = "stabilityai/sd-vae-ft-mse",
    batch_size: int = 16,
    image_size: int = 256,
    device: str | None = None,
) -> TensorBank:
    from diffusers import AutoencoderKL

    device_obj = resolve_device(device)
    image_paths = list_release_images(data_dir, split)
    vae = AutoencoderKL.from_pretrained(model_name)
    vae = vae.to(device_obj)
    vae.eval()

    all_ids: list[str] = []
    all_paths: list[str] = []
    all_values: list[torch.Tensor] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"vae-{split}"):
        chunk = image_paths[start : start + batch_size]
        images = load_image_batch(chunk, image_size=image_size, device=device_obj)
        images = images * 2.0 - 1.0
        with torch.no_grad():
            posterior = vae.encode(images).latent_dist
            latents = posterior.mode() * vae.config.scaling_factor
        all_ids.extend([path.stem for path in chunk])
        all_paths.extend([str(path) for path in chunk])
        all_values.append(latents.cpu().to(torch.float16))

    return TensorBank(
        bank_type="vae",
        image_ids=all_ids,
        image_paths=all_paths,
        values=torch.cat(all_values, dim=0),
        metadata={"model_name": model_name, "split": split, "image_size": image_size},
    )
