from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.logging_utils import log_runtime_context, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Transformer denoising run metrics.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(resolve_path(config["output"]["log_dir"], project_root), "evaluate_transformer")
    log_runtime_context(logger, config, output_dir)
    summary = evaluate_transformer(config, project_root)
    (output_dir / "transformer_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / "transformer_evaluation.md")
    logger.info("Wrote Transformer evaluation to %s", output_dir)


def evaluate_transformer(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    transformer_dir = resolve_path(config["inputs"]["transformer_dir"], project_root)
    metric_paths = sorted(transformer_dir.glob(config["evaluation"]["transformer_run_glob"]))
    runs = []
    for metrics_path in metric_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        checkpoint = run_dir / "Final_Transformer.pt"
        attention_path = run_dir / config["evaluation"]["attention_filename"]
        history_path = run_dir / "history.csv"
        runs.append(
            {
                "run_name": payload.get("run_name", run_dir.name),
                "run_dir": str(run_dir),
                "order": int(payload.get("order", payload.get("model", {}).get("window_size", -1))),
                "model": payload.get("model", {}),
                "metrics": payload.get("metrics", {}),
                "training_summary": payload.get("training_summary", {}),
                "has_checkpoint": checkpoint.exists(),
                "has_attention": attention_path.exists(),
                "has_history": history_path.exists(),
                "checkpoint": str(checkpoint) if checkpoint.exists() else None,
                "attention": str(attention_path) if attention_path.exists() else None,
                "history": str(history_path) if history_path.exists() else None,
            }
        )
    best = min(runs, key=lambda row: row["metrics"].get("nmse", float("inf"))) if runs else None
    return {
        "run_count": len(runs),
        "runs": runs,
        "best_run": best,
        "source_dir": str(transformer_dir),
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = ["# Transformer Evaluation", ""]
    if summary["best_run"]:
        best = summary["best_run"]
        lines.extend(
            [
                f"Best run: **{best['run_name']}**",
                f"Best NMSE: **{best['metrics'].get('nmse', float('nan')):.8f}**",
                "",
            ]
        )
    lines.extend(
        [
            "| Run | P | Final train loss | Final validation loss | NMSE | MSE | Checkpoint | Attention |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for run in summary["runs"]:
        metrics = run["metrics"]
        training = run.get("training_summary", {})
        lines.append(
            f"| {run['run_name']} | {run['order']} | "
            f"{format_float(training.get('final_train_loss'))} | "
            f"{format_float(training.get('final_validation_loss'))} | "
            f"{metrics.get('nmse', float('nan')):.8f} | {metrics.get('mse', float('nan')):.8f} | "
            f"{'yes' if run['has_checkpoint'] else 'no'} | {'yes' if run['has_attention'] else 'no'} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8f}"


if __name__ == "__main__":
    main()
