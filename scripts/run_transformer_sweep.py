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
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Transformer architecture and window-size sweeps.")
    parser.add_argument("--config", default="configs/transformer_sweep.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config = load_yaml_config(args.config)
    project_root = project_root_from_config(sweep_config)
    logger, _ = setup_logging(project_root / "logs", "transformer_sweep")
    base_config = resolve_path(sweep_config["base_config"], project_root)
    commands = []

    for architecture in sweep_config.get("architecture_sweep", []):
        run_name = f"arch_{architecture['name']}_P64"
        commands.append(build_command(base_config, run_name, order=64, architecture=architecture, sweep_config=sweep_config))

    architecture_name = str(sweep_config["window_sweep"]["architecture_name"])
    architectures = {item["name"]: item for item in sweep_config.get("architecture_sweep", [])}
    selected_architecture = architectures[architecture_name]
    for order in sweep_config["window_sweep"]["orders"]:
        run_name = f"window_P{order}_{architecture_name}"
        commands.append(build_command(base_config, run_name, order=int(order), architecture=selected_architecture, sweep_config=sweep_config))

    summary = {"commands": [" ".join(command) for command in commands]}
    output_path = project_root / "outputs" / "transformer_sweep_commands.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for command in commands:
        logger.info("Command: %s", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=project_root, check=True)


def build_command(
    base_config: Path,
    run_name: str,
    order: int,
    architecture: dict,
    sweep_config: dict,
) -> list[str]:
    training = sweep_config.get("training", {})
    return [
        sys.executable,
        "scripts/train_transformer.py",
        "--config",
        str(base_config),
        "--run-name",
        run_name,
        "--P",
        str(order),
        "--d-model",
        str(architecture["d_model"]),
        "--nhead",
        str(architecture["nhead"]),
        "--num-layers",
        str(architecture["num_layers"]),
        "--dim-feedforward",
        str(architecture["dim_feedforward"]),
        "--dropout",
        str(architecture.get("dropout", 0.1)),
        "--epochs",
        str(training.get("epochs", 200)),
        "--batch-size",
        str(training.get("batch_size", 256)),
        "--lr",
        str(training.get("learning_rate", 1.0e-3)),
        "--validation-fraction",
        str(training.get("validation_fraction", 0.15)),
        "--seed",
        str(training.get("seed", 0)),
    ]


if __name__ == "__main__":
    main()
