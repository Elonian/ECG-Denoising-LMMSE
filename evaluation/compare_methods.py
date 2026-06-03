from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.logging_utils import log_runtime_context, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LMMSE filters with Transformer metrics and attention.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(resolve_path(config["output"]["log_dir"], project_root), "compare_methods")
    log_runtime_context(logger, config, output_dir)
    summary = compare_methods(config, project_root, output_dir)
    (output_dir / "method_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / "method_comparison.md")
    logger.info("Wrote method comparison to %s", output_dir)


def compare_methods(config: dict[str, Any], project_root: Path, output_dir: Path) -> dict[str, Any]:
    lmmse_dir = resolve_path(config["inputs"]["lmmse_dir"], project_root)
    transformer_dir = resolve_path(config["inputs"]["transformer_dir"], project_root)
    lmmse_payload = json.loads((lmmse_dir / config["evaluation"]["lmmse_results_file"]).read_text(encoding="utf-8"))
    lmmse_by_order = {int(row["order"]): row for row in lmmse_payload["results"]}
    transformer_runs = []
    for metrics_path in sorted(transformer_dir.glob(config["evaluation"]["transformer_run_glob"])):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        order = int(payload.get("order", payload.get("model", {}).get("window_size", -1)))
        transformer_runs.append(
            {
                "run_name": payload.get("run_name", metrics_path.parent.name),
                "order": order,
                "nmse": float(payload["metrics"]["nmse"]),
                "mse": float(payload["metrics"]["mse"]),
            }
        )
    comparison_rows = []
    for run in transformer_runs:
        lmmse_row = lmmse_by_order.get(run["order"])
        comparison_rows.append(
            {
                "order": run["order"],
                "transformer_run": run["run_name"],
                "transformer_nmse": run["nmse"],
                "lmmse_nmse": lmmse_row["nmse"] if lmmse_row else None,
                "transformer_minus_lmmse_nmse": run["nmse"] - lmmse_row["nmse"] if lmmse_row else None,
            }
        )
    attention_summary = build_attention_comparison(config, project_root, output_dir)
    if comparison_rows:
        plot_nmse_comparison(comparison_rows, output_dir / "method_nmse_comparison.png")
    return {
        "comparison_rows": comparison_rows,
        "attention_summary": attention_summary,
    }


def build_attention_comparison(config: dict[str, Any], project_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    transformer_dir = resolve_path(config["inputs"]["transformer_dir"], project_root)
    filters = np.load(resolve_path(config["inputs"]["lmmse_dir"], project_root) / config["evaluation"]["lmmse_filter_file"])
    rows = []
    for metrics_path in sorted(transformer_dir.glob(config["evaluation"]["transformer_run_glob"])):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        order = int(payload.get("order", payload.get("model", {}).get("window_size", -1)))
        attention_path = metrics_path.parent / config["evaluation"]["attention_filename"]
        filter_key = f"P{order}"
        if not attention_path.exists() or filter_key not in filters:
            continue
        attention = np.load(attention_path).astype(np.float64)
        weights = np.abs(filters[filter_key].astype(np.float64))
        if attention.shape[0] != weights.shape[0]:
            continue
        if attention.sum() > 0:
            attention = attention / attention.sum()
        if weights.sum() > 0:
            weights = weights / weights.sum()
        correlation = float(np.corrcoef(attention, weights)[0, 1]) if len(attention) > 1 else 0.0
        rows.append({"run_name": payload.get("run_name", metrics_path.parent.name), "order": order, "attention_filter_correlation": correlation})
        plot_attention_vs_filter(attention, weights, output_dir / f"attention_vs_lmmse_P{order}.png", order)
    return rows


def plot_nmse_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    valid = [row for row in rows if row["lmmse_nmse"] is not None]
    labels = [row["transformer_run"] for row in valid]
    x = np.arange(len(valid))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.9), 4.5), dpi=150)
    ax.bar(x - width / 2, [row["lmmse_nmse"] for row in valid], width, label="LMMSE")
    ax.bar(x + width / 2, [row["transformer_nmse"] for row in valid], width, label="Transformer")
    ax.set_title("LMMSE vs Transformer Test NMSE")
    ax.set_ylabel("NMSE")
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_attention_vs_filter(attention: np.ndarray, weights: np.ndarray, output_path: Path, order: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(np.arange(order), weights, marker="o", label="|LMMSE h[k]| normalized")
    ax.plot(np.arange(order), attention, marker="s", label="Transformer CLS attention")
    ax.set_title(f"Attention vs LMMSE Filter, P={order}")
    ax.set_xlabel("Lag k")
    ax.set_ylabel("Normalized importance")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Method Comparison",
        "",
        "| Transformer run | P | LMMSE NMSE | Transformer NMSE | Difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["comparison_rows"]:
        lmmse = row["lmmse_nmse"]
        diff = row["transformer_minus_lmmse_nmse"]
        if lmmse is not None:
            lines.append(
                f"| {row['transformer_run']} | {row['order']} | {lmmse:.8f} | "
                f"{row['transformer_nmse']:.8f} | {diff:.8f} |"
            )
        else:
            lines.append(
                f"| {row['transformer_run']} | {row['order']} | n/a | "
                f"{row['transformer_nmse']:.8f} | n/a |"
            )
    if summary["attention_summary"]:
        lines.extend(["", "## Attention vs LMMSE", "", "| Run | P | Correlation |", "| --- | ---: | ---: |"])
        for row in summary["attention_summary"]:
            lines.append(f"| {row['run_name']} | {row['order']} | {row['attention_filter_correlation']:.6f} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
