from __future__ import annotations

import numpy as np

from utils.data_io import make_lag_windows


def biased_autocorrelation(signal: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Expected one-dimensional signal")
    if max_lag <= 0:
        raise ValueError(f"max_lag must be positive, got {max_lag}")
    n = len(signal)
    lags = np.arange(-(max_lag - 1), max_lag)
    values = []
    for lag in lags:
        shift = abs(int(lag))
        if shift >= n:
            values.append(0.0)
        elif lag >= 0:
            values.append(float(np.dot(signal[shift:], signal[: n - shift]) / n))
        else:
            values.append(float(np.dot(signal[: n - shift], signal[shift:]) / n))
    return lags, np.asarray(values, dtype=np.float64)


def welch_psd(
    signal: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("scipy is required for Welch PSD estimation") from exc

    frequencies, psd = welch(
        np.asarray(signal, dtype=np.float64),
        fs=float(sampling_rate_hz),
        window=window,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        detrend="constant",
        scaling="density",
    )
    return frequencies, psd


def estimate_lmmse_filter(
    clean_train: np.ndarray,
    noisy_train: np.ndarray,
    order: int,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, targets = make_lag_windows(noisy_train, clean_train, order)
    design = windows.astype(np.float64)
    target = targets.astype(np.float64)
    ryy = design.T @ design / len(target)
    rsy = design.T @ target / len(target)
    if ridge > 0.0:
        ryy = ryy + float(ridge) * np.eye(order)
    weights = np.linalg.solve(ryy, rsy)
    return weights.astype(np.float64), ryy, rsy


def apply_lmmse_filter(noisy: np.ndarray, clean_reference: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = int(len(weights))
    windows, targets = make_lag_windows(noisy, clean_reference, order)
    estimate = windows.astype(np.float64) @ np.asarray(weights, dtype=np.float64)
    return targets.astype(np.float64), estimate.astype(np.float64)

