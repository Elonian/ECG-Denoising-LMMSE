from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.logging_utils import log_runtime_context, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and summarize LMMSE ECG denoising outputs.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(resolve_path(config["output"]["log_dir"], project_root), "evaluate_lmmse")
    log_runtime_context(logger, config, output_dir)
    summary = evaluate_lmmse(config, project_root)
    (output_dir / "lmmse_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / "lmmse_evaluation.md")
    logger.info("Wrote LMMSE evaluation to %s", output_dir)


def evaluate_lmmse(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    lmmse_dir = resolve_path(config["inputs"]["lmmse_dir"], project_root)
    results_path = lmmse_dir / config["evaluation"]["lmmse_results_file"]
    autocorr_path = lmmse_dir / "autocorrelation_sequences.npz"
    psd_path = lmmse_dir / "welch_psd.npz"
    filters_path = lmmse_dir / config["evaluation"]["lmmse_filter_file"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    autocorr = np.load(autocorr_path)
    psd = np.load(psd_path)
    filters = np.load(filters_path)

    order_rows = sorted(results["results"], key=lambda row: row["order"])
    best_row = min(order_rows, key=lambda row: row["nmse"])
    noise_lags = autocorr["noise_lags"]
    noise_autocorr = autocorr["noise_autocorrelation"]
    zero_idx = int(np.where(noise_lags == 0)[0][0])
    zero_lag = float(abs(noise_autocorr[zero_idx]))
    off_zero = np.delete(noise_autocorr, zero_idx)
    whiteness_ratio = float(np.max(np.abs(off_zero)) / max(zero_lag, 1e-12))

    clean_peak_frequency = float(psd["clean_frequency"][int(np.argmax(psd["clean_psd"]))])
    noise_peak_frequency = float(psd["noise_frequency"][int(np.argmax(psd["noise_psd"]))])
    filter_norms = {
        key: float(np.linalg.norm(filters[key]))
        for key in sorted(filters.files, key=lambda item: int(item[1:]))
    }
    return {
        "orders": [row["order"] for row in order_rows],
        "order_results": order_rows,
        "best_order": best_row["order"],
        "best_nmse": best_row["nmse"],
        "noise_autocorrelation_whiteness_ratio": whiteness_ratio,
        "noise_white_interpretation": (
            "not well approximated as white"
            if whiteness_ratio > 0.1
            else "approximately white by off-zero autocorrelation"
        ),
        "clean_peak_frequency_hz": clean_peak_frequency,
        "noise_peak_frequency_hz": noise_peak_frequency,
        "filter_l2_norms": filter_norms,
        "source_files": {
            "results": str(results_path),
            "autocorrelation": str(autocorr_path),
            "psd": str(psd_path),
            "filters": str(filters_path),
        },
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# LMMSE Evaluation",
        "",
        f"Best LMMSE order: **P = {summary['best_order']}**",
        f"Best LMMSE NMSE: **{summary['best_nmse']:.8f}**",
        "",
        "## NMSE By Order",
        "",
        "| P | NMSE | Aligned samples | Filter norm |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary["order_results"]:
        norm = summary["filter_l2_norms"][f"P{row['order']}"]
        lines.append(f"| {row['order']} | {row['nmse']:.8f} | {row['test_samples_aligned']} | {norm:.6f} |")
    lines.extend(
        [
            "",
            "## Signal Statistics",
            "",
            f"- Noise autocorrelation whiteness ratio: `{summary['noise_autocorrelation_whiteness_ratio']:.6f}`",
            f"- Interpretation: {summary['noise_white_interpretation']}.",
            f"- Clean PSD peak frequency: `{summary['clean_peak_frequency_hz']:.3f}` Hz.",
            f"- Noise PSD peak frequency: `{summary['noise_peak_frequency_hz']:.3f}` Hz.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
