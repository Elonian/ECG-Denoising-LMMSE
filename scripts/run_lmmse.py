from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.data_io import load_dataset_metadata, load_signal_arrays
from utils.logging_utils import log_runtime_context, setup_logging
from utils.metrics import nmse, snr_db
from utils.signal_processing import apply_lmmse_filter, biased_autocorrelation, estimate_lmmse_filter, welch_psd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LMMSE analysis for ECG denoising.")
    parser.add_argument("--config", default="configs/lmmse.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(resolve_path(config["output"]["log_dir"], project_root), config["experiment"]["name"])
    log_runtime_context(logger, config, output_dir)

    data_dir = resolve_path(config["data"]["data_dir"], project_root)
    arrays = load_signal_arrays(data_dir, config["data"])
    metadata = load_dataset_metadata(data_dir)
    sampling_rate_hz = float(metadata.get("signal", {}).get("sampling_rate_hz", 360.0))
    noise_train = arrays.noisy_train - arrays.clean_train

    max_lag = int(config["lmmse"]["autocorrelation_max_lag"])
    clean_lags, clean_autocorr = biased_autocorrelation(arrays.clean_train, max_lag=max_lag)
    noise_lags, noise_autocorr = biased_autocorrelation(noise_train, max_lag=max_lag)
    np.savez(
        output_dir / "autocorrelation_sequences.npz",
        clean_lags=clean_lags,
        clean_autocorrelation=clean_autocorr,
        noise_lags=noise_lags,
        noise_autocorrelation=noise_autocorr,
    )

    welch_cfg = config["lmmse"]["welch"]
    clean_freq, clean_psd = welch_psd(
        arrays.clean_train,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=int(welch_cfg["nperseg"]),
        noverlap=int(welch_cfg["noverlap"]),
        window=str(welch_cfg.get("window", "hann")),
    )
    noise_freq, noise_psd = welch_psd(
        noise_train,
        sampling_rate_hz=sampling_rate_hz,
        nperseg=int(welch_cfg["nperseg"]),
        noverlap=int(welch_cfg["noverlap"]),
        window=str(welch_cfg.get("window", "hann")),
    )
    np.savez(
        output_dir / "welch_psd.npz",
        clean_frequency=clean_freq,
        clean_psd=clean_psd,
        noise_frequency=noise_freq,
        noise_psd=noise_psd,
    )

    results = []
    filters = {}
    predictions = {}
    ridge = float(config["lmmse"].get("ridge", 0.0))
    for order in config["lmmse"]["orders"]:
        order = int(order)
        weights, ryy, rsy = estimate_lmmse_filter(
            clean_train=arrays.clean_train,
            noisy_train=arrays.noisy_train,
            order=order,
            ridge=ridge,
        )
        reference, estimate = apply_lmmse_filter(arrays.noisy_test, arrays.clean_test, weights)
        order_nmse = nmse(reference, estimate)
        filters[f"P{order}"] = weights
        predictions[f"P{order}_reference"] = reference
        predictions[f"P{order}_estimate"] = estimate
        np.savez(output_dir / f"lmmse_order_{order}.npz", weights=weights, ryy=ryy, rsy=rsy)
        results.append(
            {
                "order": order,
                "nmse": order_nmse,
                "filter_l2_norm": float(np.linalg.norm(weights)),
                "test_samples_aligned": int(len(reference)),
            }
        )
        logger.info("P=%s NMSE=%.8f aligned_samples=%s", order, order_nmse, len(reference))

    payload = {
        "orders": [int(order) for order in config["lmmse"]["orders"]],
        "train_noise_snr_db": snr_db(arrays.clean_train, noise_train),
        "results": results,
        "metadata": metadata,
    }
    (output_dir / "lmmse_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez(output_dir / "lmmse_filters.npz", **filters)
    np.savez(output_dir / "lmmse_predictions.npz", **predictions)
    logger.info("Wrote LMMSE outputs to %s", output_dir)


if __name__ == "__main__":
    main()
