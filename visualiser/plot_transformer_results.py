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
    parser = argparse.ArgumentParser(description="Plot Transformer training and inference results.")
    parser.add_argument("--config", default="configs/visualiser.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    transformer_dir = resolve_path(config["inputs"]["transformer_dir"], project_root)
    data_dir = resolve_path(config["inputs"]["data_dir"], project_root)
    lmmse_dir = resolve_path(config["inputs"]["lmmse_dir"], project_root)
    output_dir = resolve_path(config["output"]["output_dir"], project_root) / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(transformer_dir)
    if not runs:
        print(f"No Transformer metrics found under {transformer_dir}")
        return
    plot_train_validation_losses(runs, output_dir / "transformer_train_validation_losses.png")
    plot_nmse_summary(runs, output_dir / "transformer_nmse_summary.png")
    plot_grouped_losses(runs, output_dir)
    plot_architecture_nmse(runs, output_dir / "architecture_nmse_comparison.png")
    plot_window_nmse_comparison(runs, lmmse_dir, output_dir / "window_nmse_vs_lmmse.png")
    plot_validation_gap_summary(runs, output_dir / "validation_gap_summary.png")
    plot_best_inference_waveform(runs, data_dir, output_dir / "transformer_inference_waveform.png")
    print(f"Wrote Transformer figures to {output_dir}")


def collect_runs(transformer_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(transformer_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        payload["run_dir"] = str(run_dir)
        payload["run_dir_path"] = run_dir
        payload["run_name"] = payload.get("run_name") or run_dir.name
        for key, filename in [("train_losses", "train_losses.npy"), ("val_losses", "val_losses.npy")]:
            path = run_dir / filename
            if path.exists():
                payload[key] = np.load(path)
        runs.append(payload)
    return runs


def plot_train_validation_losses(runs: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    for run in runs:
        train_losses = run.get("train_losses")
        if train_losses is None:
            continue
        x_train = np.arange(1, len(train_losses) + 1)
        line = ax.plot(x_train, train_losses, linewidth=2.0, label=f"{run['run_name']} train")[0]
        val_losses = run.get("val_losses")
        if val_losses is not None and len(val_losses) > 0:
            x_val = np.arange(1, len(val_losses) + 1)
            ax.plot(
                x_val,
                val_losses,
                linestyle="--",
                linewidth=2.0,
                color=line.get_color(),
                label=f"{run['run_name']} validation",
            )
    ax.set_title("Transformer Train and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grouped_losses(runs: list[dict], output_dir: Path) -> None:
    groups = [
        ("architecture", [run for run in runs if run["run_name"].startswith("arch_")]),
        ("window", [run for run in runs if run["run_name"].startswith("window_P")]),
    ]
    for name, group in groups:
        if group:
            plot_train_validation_losses(group, output_dir / f"{name}_train_validation_losses.png")


def plot_nmse_summary(runs: list[dict], output_path: Path) -> None:
    ranked = sorted(runs, key=lambda run: float(run["metrics"]["nmse"]))
    labels = [run["run_name"] for run in ranked]
    values = np.asarray([run["metrics"]["nmse"] for run in ranked], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8), dpi=150)
    if len(values) == 1:
        ax.scatter([1], values, s=180, color="#1f77b4")
        ax.hlines(values[0], 0.75, 1.25, colors="#1f77b4", linewidth=4, alpha=0.35)
        ax.text(1, values[0], f"  NMSE {values[0]:.4f}", va="center", fontsize=11)
        ax.set_xlim(0.65, 1.55)
        ax.set_xticks([1], labels=labels)
    else:
        x = np.arange(len(labels))
        ax.plot(x, values, marker="o", linewidth=2.0)
        ax.fill_between(x, values, values.min(), alpha=0.12)
        ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_title("Transformer Test NMSE Summary")
    ax.set_ylabel("NMSE, lower is better")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_architecture_nmse(runs: list[dict], output_path: Path) -> None:
    architecture_runs = sorted(
        [run for run in runs if run["run_name"].startswith("arch_")],
        key=lambda run: float(run["metrics"]["nmse"]),
    )
    if not architecture_runs:
        return
    labels = [run["run_name"].replace("arch_", "").replace("_P64", "") for run in architecture_runs]
    nmse_values = [float(run["metrics"]["nmse"]) for run in architecture_runs]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    bars = ax.bar(np.arange(len(labels)), nmse_values, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(labels)])
    for bar, value in zip(bars, nmse_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Architecture Sweep Test NMSE, P=64")
    ax.set_ylabel("NMSE, lower is better")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_window_nmse_comparison(runs: list[dict], lmmse_dir: Path, output_path: Path) -> None:
    window_runs = [run for run in runs if run["run_name"].startswith("window_P")]
    if not window_runs:
        return
    transformer_by_order = {int(run["order"]): float(run["metrics"]["nmse"]) for run in window_runs}
    lmmse_by_order = load_lmmse_nmse(lmmse_dir)
    orders = sorted(set(transformer_by_order) | set(lmmse_by_order))
    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    if lmmse_by_order:
        ax.plot(
            orders,
            [lmmse_by_order.get(order, np.nan) for order in orders],
            marker="o",
            linewidth=2.0,
            label="LMMSE",
        )
    ax.plot(
        orders,
        [transformer_by_order.get(order, np.nan) for order in orders],
        marker="s",
        linewidth=2.0,
        label="Transformer",
    )
    ax.set_title("Window Size Sweep: Transformer vs LMMSE")
    ax.set_xlabel("Filter/window order P")
    ax.set_ylabel("Test NMSE, lower is better")
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_validation_gap_summary(runs: list[dict], output_path: Path) -> None:
    rows = []
    for run in runs:
        summary = run.get("training_summary", {})
        train = summary.get("final_train_loss")
        validation = summary.get("final_validation_loss")
        if train is None or validation is None:
            continue
        rows.append((run["run_name"], float(train), float(validation), float(validation) - float(train)))
    if not rows:
        return
    rows.sort(key=lambda row: row[3])
    labels = [row[0] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.75), 4.8), dpi=150)
    ax.bar(x, [row[3] for row in rows], color="#9467bd")
    ax.set_title("Final Validation Gap By Run")
    ax.set_ylabel("Validation loss minus train loss")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_lmmse_nmse(lmmse_dir: Path) -> dict[int, float]:
    results_path = lmmse_dir / "lmmse_results.json"
    if not results_path.exists():
        return {}
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return {int(row["order"]): float(row["nmse"]) for row in payload.get("results", [])}


def plot_best_inference_waveform(runs: list[dict], data_dir: Path, output_path: Path) -> None:
    ranked = sorted(runs, key=lambda run: float(run["metrics"]["nmse"]))
    for run in ranked:
        run_dir = Path(run["run_dir_path"])
        estimate_path = run_dir / "inference" / "transformer_estimate.npy"
        reference_path = run_dir / "inference" / "transformer_reference.npy"
        if estimate_path.exists() and reference_path.exists():
            estimate = np.load(estimate_path)
            reference = np.load(reference_path)
            order = int(run["order"])
            noisy = load_signal_arrays(data_dir).noisy_test[order - 1 : order - 1 + len(reference)]
            plot_waveform_panel(run, noisy, reference, estimate, output_path)
            return


def plot_waveform_panel(run: dict, noisy: np.ndarray, reference: np.ndarray, estimate: np.ndarray, output_path: Path) -> None:
    samples = min(1200, len(reference), len(estimate), len(noisy))
    x = np.arange(samples)
    residual = reference[:samples] - estimate[:samples]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True, height_ratios=[2.2, 1.0])
    axes[0].plot(x, noisy[:samples], color="#7f7f7f", linewidth=0.9, alpha=0.75, label="Noisy ECG")
    axes[0].plot(x, reference[:samples], color="#111111", linewidth=1.4, label="Clean reference")
    axes[0].plot(x, estimate[:samples], color="#1f77b4", linewidth=1.2, label="Transformer estimate")
    axes[0].set_title(f"Transformer Denoising Waveform, {run['run_name']}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].plot(x, residual, color="#d62728", linewidth=1.0)
    axes[1].axhline(0.0, color="#111111", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Aligned test sample")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
