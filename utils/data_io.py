from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SignalArrays:
    clean_train: np.ndarray
    noisy_train: np.ndarray
    clean_test: np.ndarray
    noisy_test: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


def load_signal_arrays(data_dir: str | Path, config: dict[str, Any] | None = None) -> SignalArrays:
    data_dir = Path(data_dir)
    names = {
        "clean_train": "s_train.npy",
        "noisy_train": "y_train.npy",
        "clean_test": "s_test.npy",
        "noisy_test": "y_test.npy",
    }
    if config:
        names.update(
            {
                "clean_train": str(config.get("clean_train", names["clean_train"])),
                "noisy_train": str(config.get("noisy_train", names["noisy_train"])),
                "clean_test": str(config.get("clean_test", names["clean_test"])),
                "noisy_test": str(config.get("noisy_test", names["noisy_test"])),
            }
        )
    arrays = {
        key: np.load(data_dir / filename).astype(np.float32)
        for key, filename in names.items()
    }
    _validate_array_lengths(arrays)
    return SignalArrays(
        clean_train=arrays["clean_train"],
        noisy_train=arrays["noisy_train"],
        clean_test=arrays["clean_test"],
        noisy_test=arrays["noisy_test"],
    )


def load_dataset_metadata(data_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(data_dir) / "dataset_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def normalization_from_noisy_train(noisy_train: np.ndarray) -> NormalizationStats:
    mean = float(np.mean(noisy_train))
    std = float(np.std(noisy_train))
    if std <= 0.0:
        raise ValueError("Cannot normalize with zero standard deviation")
    return NormalizationStats(mean=mean, std=std)


def make_lag_windows(noisy: np.ndarray, clean: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")
    noisy = np.asarray(noisy, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    if noisy.ndim != 1 or clean.ndim != 1:
        raise ValueError("Expected one-dimensional clean and noisy signals")
    if len(noisy) != len(clean):
        raise ValueError(f"Signal length mismatch: noisy={len(noisy)}, clean={len(clean)}")
    if len(noisy) < order:
        raise ValueError(f"Signal length {len(noisy)} is shorter than order {order}")

    windows = np.lib.stride_tricks.sliding_window_view(noisy, order)
    windows = np.ascontiguousarray(windows[:, ::-1], dtype=np.float32)
    targets = np.ascontiguousarray(clean[order - 1 :], dtype=np.float32)
    return windows, targets


class WindowedSignalDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray, order: int) -> None:
        self.windows, self.targets = make_lag_windows(noisy, clean, order)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[index]),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


def _validate_array_lengths(arrays: dict[str, np.ndarray]) -> None:
    train_lengths = {len(arrays["clean_train"]), len(arrays["noisy_train"])}
    test_lengths = {len(arrays["clean_test"]), len(arrays["noisy_test"])}
    if len(train_lengths) != 1:
        raise ValueError("Training clean/noisy arrays must have the same length")
    if len(test_lengths) != 1:
        raise ValueError("Test clean/noisy arrays must have the same length")

