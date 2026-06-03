from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all available ECG denoising figures.")
    parser.add_argument("--config", default="configs/visualiser.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        [sys.executable, "visualiser/plot_lmmse_results.py", "--config", args.config],
        [sys.executable, "visualiser/plot_transformer_results.py", "--config", args.config],
    ]
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
