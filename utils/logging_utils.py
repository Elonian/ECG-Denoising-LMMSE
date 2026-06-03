from __future__ import annotations

import json
import logging
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str | Path, run_name: str, level: str = "INFO") -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{timestamp}.log"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger, log_path


def log_runtime_context(logger: logging.Logger, config: dict[str, Any], output_dir: str | Path) -> None:
    payload = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "config_path": config.get("_config_path"),
        "output_dir": str(output_dir),
    }
    logger.info("Runtime context:\n%s", json.dumps(payload, indent=2, sort_keys=True))

