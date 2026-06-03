from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config, project_root_from_config, resolve_path
from utils.logging_utils import setup_logging
from utils.physionet import create_ecg_train_test_arrays, download_ecg_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PhysioNet records and create ECG train/test arrays.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded WFDB files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = project_root_from_config(config)
    data_cfg = config["data"]
    data_dir = resolve_path(data_cfg["data_dir"], project_root)
    logger, _ = setup_logging(project_root / "logs", "create_training_test_set")
    downloads = [] if args.skip_download else download_ecg_records(data_dir)
    split = data_cfg.get("split", {})
    metadata = create_ecg_train_test_arrays(
        data_dir=data_dir,
        start_sample=int(split.get("start_sample", 108000)),
        train_samples=int(split.get("train_samples", 21000)),
        test_samples=int(split.get("test_samples", 9000)),
        channel_index=int(data_cfg.get("channel_index", 0)),
        downloads=downloads,
    )
    logger.info("Created arrays in %s", data_dir)
    logger.info("Split metadata: %s", metadata["split"])


if __name__ == "__main__":
    main()
