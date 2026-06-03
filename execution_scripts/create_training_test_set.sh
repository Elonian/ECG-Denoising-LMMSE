#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m pip install -r requirements.txt
python scripts/create_training_test_set.py --config configs/data.yaml "$@"
