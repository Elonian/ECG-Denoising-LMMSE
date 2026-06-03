#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m pip install -r requirements.txt
python scripts/run_inference_for_checkpoints.py \
  --transformer-dir outputs/transformer \
  --data-dir data \
  --save-attention \
  --overwrite \
  "$@"
