from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_denoiser import ECGTransformerDenoiser
from utils.data_io import NormalizationStats, WindowedSignalDataset, load_signal_arrays
from utils.metrics import nmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Transformer ECG denoising inference.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--max-attention-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = ECGTransformerDenoiser(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_payload = checkpoint["normalization"]
    norm = NormalizationStats(mean=float(norm_payload["mean"]), std=float(norm_payload["std"]))
    order = int(checkpoint["order"])

    arrays = load_signal_arrays(Path(args.data_dir))
    noisy_test = norm.normalize(arrays.noisy_test)
    clean_test = norm.normalize(arrays.clean_test)
    dataset = WindowedSignalDataset(noisy_test, clean_test, order=order)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    attention_accumulator = []
    consumed_attention = 0
    with torch.no_grad():
        for windows, _ in loader:
            windows = windows.to(device)
            if args.save_attention and consumed_attention < int(args.max_attention_samples):
                limit = min(windows.size(0), int(args.max_attention_samples) - consumed_attention)
                batch_prediction, attentions = model(windows[:limit], return_attention=True)
                predictions.append(batch_prediction.detach().cpu().numpy())
                if windows.size(0) > limit:
                    predictions.append(model(windows[limit:]).detach().cpu().numpy())
                last_attention = attentions[-1].detach().cpu().numpy()
                cls_attention = last_attention[:, :, 0, 1:]
                attention_accumulator.append(cls_attention)
                consumed_attention += limit
            else:
                predictions.append(model(windows).detach().cpu().numpy())

    normalized_estimate = np.concatenate(predictions).astype(np.float32)
    estimate = norm.denormalize(normalized_estimate)
    reference = arrays.clean_test[order - 1 :].astype(np.float32)
    np.save(output_dir / "transformer_estimate.npy", estimate)
    np.save(output_dir / "transformer_reference.npy", reference)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "order": order,
        "nmse": nmse(reference, estimate),
        "mse": float(np.mean((reference - estimate) ** 2)),
        "samples": int(len(reference)),
    }
    if attention_accumulator:
        attention = np.concatenate(attention_accumulator, axis=0)
        mean_attention = attention.mean(axis=(0, 1))
        np.save(output_dir / "cls_attention_mean.npy", mean_attention.astype(np.float32))
        metrics["attention_samples"] = int(attention.shape[0])
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()

