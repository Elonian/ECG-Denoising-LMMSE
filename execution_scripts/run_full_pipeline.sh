#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m pip install -r requirements.txt
python scripts/create_training_test_set.py --config configs/data.yaml --skip-download
python scripts/run_lmmse.py --config configs/lmmse.yaml
python scripts/train_transformer.py --config configs/transformer.yaml
python scripts/run_inference_for_checkpoints.py --transformer-dir outputs/transformer --data-dir data --save-attention --overwrite
python visualiser/build_figures.py --config configs/visualiser.yaml
python evaluation/summarize_results.py --config configs/evaluation.yaml
