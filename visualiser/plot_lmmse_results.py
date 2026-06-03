from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.data_io import load_signal_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LMMSE ECG denoising outputs.")
    parser.add_argument("--config", default="configs/visualiser.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    data_dir = resolve_path(config["inputs"]["data_dir"], project_root)
    lmmse_dir = resolve_path(config["inputs"]["lmmse_dir"], project_root)
    output_dir = resolve_path(config["output"]["output_dir"], project_root) / "lmmse"
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_signal_arrays(data_dir)
    plot_signal_preview(arrays.clean_test, arrays.noisy_test, output_dir / "signal_preview.png")
    plot_autocorrelation(lmmse_dir / "autocorrelation_sequences.npz", output_dir / "autocorrelation.png")
    plot_psd(lmmse_dir / "welch_psd.npz", output_dir / "welch_psd.png")
    plot_nmse(lmmse_dir / "lmmse_results.json", output_dir / "lmmse_nmse.png")
    plot_filters(lmmse_dir / "lmmse_filters.npz", output_dir / "lmmse_filters.png")
    print(f"Wrote LMMSE figures to {output_dir}")


def plot_signal_preview(clean: np.ndarray, noisy: np.ndarray, output_path: Path, sample_count: int = 1800) -> None:
    samples = np.arange(min(sample_count, len(clean)))
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(samples, clean[: len(samples)], label="Clean s[n]", linewidth=1.2)
    ax.plot(samples, noisy[: len(samples)], label="Noisy y[n]", linewidth=1.0, alpha=0.75)
    ax.set_title("ECG Test Signal Preview")
    ax.set_xlabel("Sample")
    ax.set_ylabel("mV")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_autocorrelation(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(data["clean_lags"], data["clean_autocorrelation"], label="Clean ECG")
    ax.plot(data["noise_lags"], data["noise_autocorrelation"], label="Noise")
    ax.set_title("Biased Autocorrelation Estimates")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_psd(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.semilogy(data["clean_frequency"], data["clean_psd"], label="Clean ECG")
    ax.semilogy(data["noise_frequency"], data["noise_psd"], label="Noise")
    ax.set_title("Welch PSD Estimates")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_nmse(results_path: Path, output_path: Path) -> None:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    orders = [row["order"] for row in payload["results"]]
    values = [row["nmse"] for row in payload["results"]]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(orders, values, marker="o", linewidth=2.0)
    ax.set_title("LMMSE Test NMSE By Filter Order")
    ax.set_xlabel("Filter order P")
    ax.set_ylabel("NMSE")
    ax.set_xscale("log", base=2)
    ax.set_xticks(orders, labels=[str(order) for order in orders])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_filters(filters_path: Path, output_path: Path) -> None:
    filters = np.load(filters_path)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    for key in sorted(filters.files, key=lambda item: int(item[1:])):
        weights = filters[key]
        ax.plot(np.arange(len(weights)), weights, label=key)
    ax.set_title("Estimated LMMSE Filters")
    ax.set_xlabel("Lag k")
    ax.set_ylabel("h[k]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
