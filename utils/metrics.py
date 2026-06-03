from __future__ import annotations

import numpy as np


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError(f"NMSE shape mismatch: reference={reference.shape}, estimate={estimate.shape}")
    denominator = float(np.sum(reference**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum((reference - estimate) ** 2) / denominator)


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    return float(np.mean((reference - estimate) ** 2))


def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    clean_power = float(np.sum(np.asarray(clean, dtype=np.float64) ** 2))
    noise_power = float(np.sum(np.asarray(noise, dtype=np.float64) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(clean_power / noise_power))

