from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


NSTDB_BASE_URL = "https://physionet.org/files/nstdb/1.0.0"
MITDB_BASE_URL = "https://physionet.org/files/mitdb/1.0.0"


def download_ecg_records(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    nstdb_dir = data_dir / "nstdb" / "1.0.0"
    mitdb_dir = data_dir / "mitdb" / "1.0.0"
    nstdb_dir.mkdir(parents=True, exist_ok=True)
    mitdb_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, Path]] = [(f"{NSTDB_BASE_URL}/RECORDS", nstdb_dir / "RECORDS")]
    for record in ["118e06"]:
        for extension in ["hea", "dat", "atr", "xws"]:
            files.append((f"{NSTDB_BASE_URL}/{record}.{extension}", nstdb_dir / f"{record}.{extension}"))
    for record in ["bw", "em", "ma"]:
        for extension in ["hea", "dat", "xws"]:
            files.append((f"{NSTDB_BASE_URL}/{record}.{extension}", nstdb_dir / f"{record}.{extension}"))
    files.append((f"{MITDB_BASE_URL}/RECORDS", mitdb_dir / "RECORDS"))
    for record in ["118"]:
        for extension in ["hea", "dat", "atr", "xws"]:
            files.append((f"{MITDB_BASE_URL}/{record}.{extension}", mitdb_dir / f"{record}.{extension}"))

    manifest = []
    for url, output_path in files:
        status = "exists"
        if not output_path.exists() or output_path.stat().st_size == 0:
            try:
                with urllib.request.urlopen(url, timeout=60) as response, output_path.open("wb") as handle:
                    handle.write(response.read())
                status = "downloaded"
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and output_path.suffix == ".xws":
                    status = "missing_optional"
                else:
                    raise
        manifest.append(
            {
                "url": url,
                "path": str(output_path.relative_to(data_dir.parent)),
                "bytes": output_path.stat().st_size if output_path.exists() else 0,
                "status": status,
            }
        )
    return manifest


def create_ecg_train_test_arrays(
    data_dir: str | Path,
    start_sample: int = 108000,
    train_samples: int = 21000,
    test_samples: int = 9000,
    channel_index: int = 0,
    downloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        import wfdb
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("wfdb is required to create ECG arrays from PhysioNet records") from exc

    data_dir = Path(data_dir)
    clean_record = data_dir / "mitdb" / "1.0.0" / "118"
    noisy_record = data_dir / "nstdb" / "1.0.0" / "118e06"
    clean = wfdb.rdrecord(str(clean_record), physical=False)
    noisy = wfdb.rdrecord(str(noisy_record), physical=False)
    if clean.fs != noisy.fs:
        raise RuntimeError(f"Sampling rates differ: clean={clean.fs}, noisy={noisy.fs}")

    gain = float(clean.adc_gain[channel_index])
    clean_adc_zero = int(clean.adc_zero[channel_index])
    noisy_adc_zero = int(noisy.adc_zero[channel_index])
    clean_digital = clean.d_signal[:, channel_index].astype(np.float64)
    noisy_digital = noisy.d_signal[:, channel_index].astype(np.float64)

    clean_signal = ((clean_digital - clean_adc_zero) / gain).astype(np.float32)
    noisy_signal = ((noisy_digital - noisy_adc_zero) / gain).astype(np.float32)

    end = start_sample + train_samples + test_samples
    if end > len(clean_signal) or end > len(noisy_signal):
        raise RuntimeError("Requested split extends past the available record length")

    arrays = {
        "s_train.npy": clean_signal[start_sample : start_sample + train_samples],
        "y_train.npy": noisy_signal[start_sample : start_sample + train_samples],
        "s_test.npy": clean_signal[start_sample + train_samples : end],
        "y_test.npy": noisy_signal[start_sample + train_samples : end],
    }
    for filename, array in arrays.items():
        np.save(data_dir / filename, array.astype(np.float32))

    lead_diff = noisy_signal[:30000] - clean_signal[:30000]
    train_noise = arrays["y_train.npy"] - arrays["s_train.npy"]
    test_noise = arrays["y_test.npy"] - arrays["s_test.npy"]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "nstdb": "https://physionet.org/content/nstdb/1.0.0/",
            "mitdb": "https://physionet.org/content/mitdb/1.0.0/",
        },
        "records": {
            "clean": "mitdb/1.0.0/118",
            "noisy": "nstdb/1.0.0/118e06",
            "noise_context_records": ["nstdb/1.0.0/bw", "nstdb/1.0.0/em", "nstdb/1.0.0/ma"],
        },
        "signal": {
            "sampling_rate_hz": float(clean.fs),
            "channel_index": int(channel_index),
            "channel_name_clean": clean.sig_name[channel_index],
            "channel_name_noisy": noisy.sig_name[channel_index],
            "adc_gain": gain,
            "clean_adc_zero": clean_adc_zero,
            "noisy_adc_zero": noisy_adc_zero,
        },
        "split": {
            "start_sample": int(start_sample),
            "start_time_seconds": float(start_sample / clean.fs),
            "train_samples": int(train_samples),
            "test_samples": int(test_samples),
            "train_range": [int(start_sample), int(start_sample + train_samples)],
            "test_range": [int(start_sample + train_samples), int(end)],
            "note": (
                "Noise in NSTDB starts after the first five minutes, then alternates in noisy and clean sections. "
                "This split starts at sample 108000 to use the first noisy segment and keeps 21000 train samples "
                "and 9000 test samples."
            ),
        },
        "arrays": {
            filename: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for filename, array in arrays.items()
        },
        "sanity_checks": {
            "clean_lead_in_diff_mean_first_30000": float(lead_diff.mean()),
            "clean_lead_in_diff_std_first_30000": float(lead_diff.std()),
            "train_noise_mean": float(train_noise.mean()),
            "train_noise_std": float(train_noise.std()),
            "test_noise_mean": float(test_noise.mean()),
            "test_noise_std": float(test_noise.std()),
        },
        "downloads": downloads or [],
    }
    (data_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
