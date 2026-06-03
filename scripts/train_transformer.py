from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_denoiser import ECGTransformerDenoiser
from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.data_io import WindowedSignalDataset, load_signal_arrays, normalization_from_noisy_train
from utils.logging_utils import log_runtime_context, setup_logging
from utils.metrics import nmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer ECG denoiser.")
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--P", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_cli_overrides(config, args)
    train_from_config(config, run_name=args.run_name)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.P is not None:
        config["window"]["order"] = args.P
    for arg_name, config_name in [
        ("d_model", "d_model"),
        ("nhead", "nhead"),
        ("num_layers", "num_layers"),
        ("dim_feedforward", "dim_feedforward"),
        ("dropout", "dropout"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config["model"][config_name] = value
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.validation_fraction is not None:
        config["training"]["validation_fraction"] = args.validation_fraction
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device


def train_from_config(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    project_root = project_root_from_config(config)
    order = int(config["window"]["order"])
    run_name = run_name or str(config["experiment"].get("name") or f"transformer_P{order}")
    base_output_dir = resolve_path(config["output"]["output_dir"], project_root)
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logging(resolve_path(config["output"]["log_dir"], project_root), run_name)
    log_runtime_context(logger, config, output_dir)

    seed = int(config["experiment"].get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    arrays = load_signal_arrays(resolve_path(config["data"]["data_dir"], project_root), config["data"])
    norm = normalization_from_noisy_train(arrays.noisy_train)
    full_train_dataset = WindowedSignalDataset(
        noisy=norm.normalize(arrays.noisy_train),
        clean=norm.normalize(arrays.clean_train),
        order=order,
    )
    train_dataset, validation_dataset = split_train_validation(
        full_train_dataset,
        validation_fraction=float(config["training"].get("validation_fraction", 0.15)),
    )
    test_dataset = WindowedSignalDataset(
        noisy=norm.normalize(arrays.noisy_test),
        clean=norm.normalize(arrays.clean_test),
        order=order,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    model = ECGTransformerDenoiser(window_size=order, **config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []
    history = []
    epochs = int(config["training"]["epochs"])
    log_every = max(int(config["training"].get("log_every_n_steps", 25)), 1)
    logger.info(
        "Training %s for %s epochs with %s train windows and %s validation windows",
        model.__class__.__name__,
        epochs,
        len(train_dataset),
        len(validation_dataset) if validation_dataset is not None else 0,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for step, (windows, targets) in enumerate(train_loader, start=1):
            windows = windows.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(windows)
            loss = loss_fn(prediction, targets)
            loss.backward()
            grad_clip = config["training"].get("grad_clip_norm")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            if step % log_every == 0:
                logger.info("epoch=%s step=%s train_loss=%.8f", epoch, step, total_loss / max(total_examples, 1))
        avg_loss = total_loss / max(total_examples, 1)
        avg_val_loss = evaluate_loss(model, validation_loader, loss_fn, device) if validation_loader is not None else None
        train_losses.append(avg_loss)
        if avg_val_loss is not None:
            val_losses.append(avg_val_loss)
            logger.info("epoch=%s train_loss=%.8f val_loss=%.8f", epoch, avg_loss, avg_val_loss)
        else:
            logger.info("epoch=%s train_loss=%.8f", epoch, avg_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "validation_loss": avg_val_loss,
            }
        )

    metrics = evaluate_model(model, test_loader, arrays.clean_test[order - 1 :], norm, device)
    logger.info("test_NMSE=%.8f test_MSE=%.8f", metrics["nmse"], metrics["mse"])
    np.save(output_dir / "train_losses.npy", np.asarray(train_losses, dtype=np.float32))
    np.save(output_dir / "val_losses.npy", np.asarray(val_losses, dtype=np.float32))
    write_history_files(output_dir, history)
    metrics_payload = {
        "run_name": run_name,
        "order": order,
        "model": model.config_dict(),
        "training": config["training"],
        "training_summary": {
            "train_windows": int(len(train_dataset)),
            "validation_windows": int(len(validation_dataset) if validation_dataset is not None else 0),
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "final_validation_loss": float(val_losses[-1]) if val_losses else None,
        },
        "normalization": {"mean": norm.mean, "std": norm.std},
        "metrics": metrics,
        "log_path": str(log_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = output_dir / str(config["output"].get("checkpoint_name", "Final_Transformer.pt"))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "normalization": {"mean": norm.mean, "std": norm.std},
            "order": order,
            "config": config,
            "metrics": metrics,
            "training_history": history,
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return metrics_payload


def split_train_validation(
    dataset: WindowedSignalDataset,
    validation_fraction: float,
) -> tuple[Subset, Subset | None]:
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}")
    total = len(dataset)
    validation_count = int(round(total * validation_fraction))
    if validation_count == 0:
        return Subset(dataset, list(range(total))), None
    train_count = total - validation_count
    if train_count <= 0:
        raise ValueError("Validation split leaves no training windows")
    indices = list(range(total))
    return Subset(dataset, indices[:train_count]), Subset(dataset, indices[train_count:])


@torch.no_grad()
def evaluate_loss(
    model: ECGTransformerDenoiser,
    loader: DataLoader | None,
    loss_fn: nn.Module,
    device: torch.device,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)
        prediction = model(windows)
        loss = loss_fn(prediction, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        return None
    return total_loss / total_examples


def write_history_files(output_dir: Path, history: list[dict[str, float | int | None]]) -> None:
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def evaluate_model(
    model: ECGTransformerDenoiser,
    loader: DataLoader,
    original_reference: np.ndarray,
    norm,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    for windows, _ in loader:
        prediction = model(windows.to(device)).detach().cpu().numpy()
        predictions.append(prediction)
    normalized_estimate = np.concatenate(predictions).astype(np.float32)
    estimate = norm.denormalize(normalized_estimate)
    reference = np.asarray(original_reference, dtype=np.float32)
    return {
        "nmse": nmse(reference, estimate),
        "mse": float(np.mean((reference - estimate) ** 2)),
        "samples": int(len(reference)),
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
