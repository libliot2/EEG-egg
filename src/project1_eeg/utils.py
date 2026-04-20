from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = REPO_ROOT / "image-eeg-data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_checkpoint(
    path: str | Path,
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    scheduler_state: dict[str, Any] | None,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def image_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )


def load_image_tensor(path: str | Path, image_size: int = 256) -> torch.Tensor:
    transform = image_transform(image_size=image_size)
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def load_image_batch(
    paths: list[str | Path],
    *,
    image_size: int = 256,
    device: torch.device | None = None,
) -> torch.Tensor:
    batch = torch.stack([load_image_tensor(path, image_size=image_size) for path in paths], dim=0)
    if device is not None:
        batch = batch.to(device)
    return batch


def cosine_warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step >= warmup_steps:
        return 1.0
    return float(step + 1) / float(max(1, warmup_steps))


def summarize_metric_list(metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metrics:
        return {}
    summary: dict[str, dict[str, float]] = {}
    keys = metrics[0].keys()
    for key in keys:
        values = np.asarray([metric[key] for metric in metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(math.ceil(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
