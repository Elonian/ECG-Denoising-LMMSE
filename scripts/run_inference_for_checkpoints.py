from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Transformer inference for saved ECG denoising checkpoints.")
    parser.add_argument("--transformer-dir", default="outputs/transformer")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-glob", default="*/Final_Transformer.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--max-attention-samples", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformer_dir = resolve_path(args.transformer_dir)
    data_dir = resolve_path(args.data_dir)
    checkpoints = sorted(transformer_dir.glob(args.checkpoint_glob))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {transformer_dir / args.checkpoint_glob}")

    completed = 0
    for checkpoint in checkpoints:
        output_dir = checkpoint.parent / "inference"
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not args.overwrite:
            continue
        command = [
            sys.executable,
            "scripts/infer_transformer.py",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--max-attention-samples",
            str(args.max_attention_samples),
        ]
        if args.save_attention:
            command.append("--save-attention")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        completed += 1
    print(f"Ran inference for {completed} checkpoints")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


if __name__ == "__main__":
    main()
