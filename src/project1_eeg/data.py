from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch
from torch.utils.data import Dataset

from .utils import DEFAULT_DATA_DIR, load_json


SplitName = Literal["train", "test"]
ImageIdSource = Literal["all", "train_ids", "val_ids"]
TrialSamplingMode = Literal["none", "random_avg"]


@dataclass(frozen=True)
class EEGRecord:
    eeg: torch.Tensor
    image_id: str
    image_path: str
    label: int
    concept_text: str
    split: SplitName


def _selected_channel_indices_from_jsonl(
    selected_channels: str | Sequence[str],
    eeg_channel_jsonl: str | Path,
) -> list[int]:
    if isinstance(selected_channels, str):
        selected_channels = [selected_channels]

    channel_names: list[str] = []
    with open(eeg_channel_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            name = item.get("name") or item.get("channel_name") or item.get("label")
            if name is None:
                raise KeyError(
                    "Each EEG channel record must contain 'name', 'channel_name', or 'label'."
                )
            channel_names.append(str(name))

    name_to_index = {name: idx for idx, name in enumerate(channel_names)}
    missing = [channel for channel in selected_channels if channel not in name_to_index]
    if missing:
        raise ValueError(f"Unknown EEG channels: {missing}")

    return [name_to_index[channel] for channel in selected_channels]


def resolve_image_path(data_dir: str | Path, raw_path: str, split: SplitName) -> Path:
    data_dir = Path(data_dir)
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("train_images/"):
        normalized = normalized.replace("train_images/", "training_images/", 1)
    elif normalized.startswith("training_images/"):
        normalized = normalized
    elif normalized.startswith("test_images/"):
        normalized = normalized
    else:
        folder = "training_images" if split == "train" else "test_images"
        normalized = f"{folder}/{normalized.lstrip('./')}"
    return data_dir / normalized


def _filter_by_image_ids(records: list[EEGRecord], image_ids: Iterable[str] | None) -> list[EEGRecord]:
    if image_ids is None:
        return records
    allowed = set(image_ids)
    return [record for record in records if record.image_id in allowed]


def load_eeg_records(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: SplitName,
    avg_trials: bool = True,
    preserve_trials: bool = False,
    selected_channels: str | Sequence[str] | None = None,
    eeg_channel_jsonl: str | Path | None = None,
    image_ids: Iterable[str] | None = None,
) -> list[EEGRecord]:
    data_dir = Path(data_dir)
    eeg_channel_jsonl = eeg_channel_jsonl or data_dir / "EEG_CHANNELS.jsonl"
    loaded = torch.load(data_dir / f"{split}.pt", weights_only=False)

    eeg = torch.as_tensor(loaded["eeg"]).float()
    labels = torch.as_tensor(loaded["label"]).long()
    images = loaded["img"]
    texts = loaded.get("text")

    if eeg.ndim == 4:
        if avg_trials and preserve_trials:
            raise ValueError("avg_trials and preserve_trials cannot both be enabled.")
        if avg_trials:
            eeg = eeg.mean(dim=1)
            labels = labels[:, 0] if labels.ndim == 2 else labels
            images = images[:, 0] if getattr(images, "ndim", 1) == 2 else images
            if texts is not None and getattr(texts, "ndim", 1) == 2:
                texts = texts[:, 0]
        elif preserve_trials:
            labels = labels[:, 0] if labels.ndim == 2 else labels
            images = images[:, 0] if getattr(images, "ndim", 1) == 2 else images
            if texts is not None and getattr(texts, "ndim", 1) == 2:
                texts = texts[:, 0]
        else:
            eeg = eeg.reshape(-1, eeg.shape[-2], eeg.shape[-1])
            labels = labels.reshape(-1)
            images = images.reshape(-1)
            if texts is not None:
                texts = texts.reshape(-1)
    elif eeg.ndim != 3:
        raise ValueError(f"Unexpected EEG tensor shape: {tuple(eeg.shape)}")

    if selected_channels is not None:
        channel_indices = _selected_channel_indices_from_jsonl(selected_channels, eeg_channel_jsonl)
        if eeg.ndim == 3:
            eeg = eeg[:, channel_indices, :]
        elif eeg.ndim == 4:
            eeg = eeg[:, :, channel_indices, :]
        else:
            raise ValueError(f"Unexpected EEG tensor shape while selecting channels: {tuple(eeg.shape)}")

    if texts is None:
        texts = [""] * len(images)
    else:
        texts = [str(item) for item in texts.tolist()]

    records: list[EEGRecord] = []
    for signal, raw_path, label, concept_text in zip(eeg, images.tolist(), labels.tolist(), texts, strict=True):
        image_path = resolve_image_path(data_dir, str(raw_path), split)
        records.append(
            EEGRecord(
                eeg=signal.contiguous(),
                image_id=image_path.stem,
                image_path=str(image_path),
                label=int(label),
                concept_text=concept_text,
                split=split,
            )
        )

    return _filter_by_image_ids(records, image_ids)


def list_release_images(data_dir: str | Path, split: SplitName) -> list[Path]:
    data_dir = Path(data_dir)
    root = data_dir / ("training_images" if split == "train" else "test_images")
    return sorted(root.rglob("*.jpg"))


def make_train_val_split(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    base_records = load_eeg_records(data_dir=data_dir, split="train", avg_trials=True)
    image_ids = sorted({record.image_id for record in base_records})
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    val_count = max(1, int(round(len(image_ids) * val_ratio)))
    val_ids = sorted(image_ids[:val_count])
    train_ids = sorted(image_ids[val_count:])
    return train_ids, val_ids


def load_split_image_ids(
    split_file: str | Path | None,
    *,
    image_id_source: ImageIdSource = "all",
) -> list[str] | None:
    if image_id_source == "all":
        return None
    if split_file is None:
        raise ValueError("split_file is required when image_id_source is not 'all'.")

    payload = load_json(split_file)
    image_ids = payload.get(image_id_source)
    if not isinstance(image_ids, list):
        raise KeyError(f"Expected '{image_id_source}' to be present in {split_file}.")
    return [str(image_id) for image_id in image_ids]


class EEGImageDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        records: Sequence[EEGRecord],
        *,
        trial_sampling: TrialSamplingMode = "none",
        trial_k_min: int = 1,
        trial_k_max: int | None = None,
    ) -> None:
        self.records = list(records)
        self.trial_sampling = trial_sampling
        self.trial_k_min = max(1, int(trial_k_min))
        self.trial_k_max = None if trial_k_max is None else max(1, int(trial_k_max))

    def _sample_trials(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or self.trial_sampling == "none":
            return eeg
        if self.trial_sampling != "random_avg":
            raise ValueError(f"Unknown trial_sampling mode: {self.trial_sampling}")

        num_trials = eeg.shape[0]
        k_min = min(self.trial_k_min, num_trials)
        k_max = num_trials if self.trial_k_max is None else min(self.trial_k_max, num_trials)
        if k_max < k_min:
            k_max = k_min

        k = int(torch.randint(low=k_min, high=k_max + 1, size=(1,)).item())
        trial_indices = torch.randperm(num_trials)[:k]
        return eeg[trial_indices].mean(dim=0)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        eeg = self._sample_trials(record.eeg).contiguous()
        return {
            "eeg": eeg,
            "image_id": record.image_id,
            "image_path": record.image_path,
            "label": record.label,
            "concept_text": record.concept_text,
        }


def ordered_image_ids(records: Sequence[EEGRecord]) -> list[str]:
    return [record.image_id for record in records]


def ordered_image_paths(records: Sequence[EEGRecord]) -> list[str]:
    return [record.image_path for record in records]
