from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.logging_utils import log_runtime_context, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation scripts and build an ECG denoising results summary.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--skip-subprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(resolve_path(config["output"]["log_dir"], project_root), "results_summary")
    log_runtime_context(logger, config, output_dir)
    if not args.skip_subprocess:
        for script in ["evaluate_lmmse.py", "evaluate_transformer.py", "compare_methods.py"]:
            subprocess.run([sys.executable, f"evaluation/{script}", "--config", args.config], cwd=project_root, check=True)
    write_results_summary(output_dir / "results_summary.md", output_dir)
    logger.info("Wrote results summary to %s", output_dir / "results_summary.md")


def write_results_summary(output_path: Path, output_dir: Path) -> None:
    lmmse = json.loads((output_dir / "lmmse_evaluation.json").read_text(encoding="utf-8"))
    transformer = json.loads((output_dir / "transformer_evaluation.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "method_comparison.json").read_text(encoding="utf-8"))
    run_names = {row["run_name"] for row in transformer.get("runs", [])}
    architecture_runs = sorted(name for name in run_names if name.startswith("arch_"))
    window_runs = sorted(name for name in run_names if name.startswith("window_P"))
    lines = [
        "# ECG Denoising Results Summary",
        "",
        "## Coverage",
        "",
        "| PDF part | Implemented by | Output files |",
        "| --- | --- | --- |",
        "| Problem 1(a) autocorrelation | `scripts/run_lmmse.py`, `visualiser/plot_lmmse_results.py` | `outputs/lmmse/autocorrelation_sequences.npz`, `outputs/visualiser/lmmse/autocorrelation.png` |",
        "| Problem 1(b) Welch PSD | `scripts/run_lmmse.py`, `visualiser/plot_lmmse_results.py` | `outputs/lmmse/welch_psd.npz`, `outputs/visualiser/lmmse/welch_psd.png` |",
        "| Problem 1(c) LMMSE P sweep | `scripts/run_lmmse.py`, `evaluation/evaluate_lmmse.py` | `outputs/lmmse/lmmse_results.json`, `outputs/evaluation/lmmse_evaluation.md` |",
        "| Problem 2(a) Transformer architecture sweep | `scripts/run_transformer_sweep.py` | `outputs/transformer/<run>/metrics.json`, `train_losses.npy`, `val_losses.npy`, `history.csv` |",
        "| Problem 2(b) Transformer P sweep | `scripts/run_transformer_sweep.py`, `evaluation/evaluate_transformer.py` | `outputs/evaluation/transformer_evaluation.md` |",
        "| Problem 2(c) attention vs LMMSE | `scripts/infer_transformer.py --save-attention`, `evaluation/compare_methods.py` | `outputs/evaluation/attention_vs_lmmse_P*.png` |",
        "",
        "## Current Results Snapshot",
        "",
        f"- Best LMMSE order: P={lmmse['best_order']} with NMSE={lmmse['best_nmse']:.8f}.",
    ]
    if transformer["best_run"]:
        best = transformer["best_run"]
        lines.append(f"- Best Transformer run currently present: {best['run_name']} with NMSE={best['metrics']['nmse']:.8f}.")
    else:
        lines.append("- No Transformer runs are present yet.")
    lines.append(f"- Method comparison rows present: {len(comparison['comparison_rows'])}.")
    lines.append(f"- Architecture sweep runs present: {', '.join(architecture_runs) if architecture_runs else 'none'}.")
    lines.append(f"- Window sweep runs present: {', '.join(window_runs) if window_runs else 'none'}.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The current summary was rebuilt from the completed LMMSE sweep, Transformer architecture sweep, Transformer window sweep, inference outputs, and attention-vs-filter comparisons.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
