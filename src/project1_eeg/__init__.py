"""Project1 EEG package."""

from .data import EEGImageDataset, load_eeg_records, make_train_val_split
from .evaluation import compute_retrieval_metrics

__all__ = [
    "EEGImageDataset",
    "compute_retrieval_metrics",
    "load_eeg_records",
    "make_train_val_split",
]
