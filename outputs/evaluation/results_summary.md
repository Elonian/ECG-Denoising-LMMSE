# ECG Denoising Results Summary

## Coverage

| PDF part | Implemented by | Output files |
| --- | --- | --- |
| Problem 1(a) autocorrelation | `scripts/run_lmmse.py`, `visualiser/plot_lmmse_results.py` | `outputs/lmmse/autocorrelation_sequences.npz`, `outputs/visualiser/lmmse/autocorrelation.png` |
| Problem 1(b) Welch PSD | `scripts/run_lmmse.py`, `visualiser/plot_lmmse_results.py` | `outputs/lmmse/welch_psd.npz`, `outputs/visualiser/lmmse/welch_psd.png` |
| Problem 1(c) LMMSE P sweep | `scripts/run_lmmse.py`, `evaluation/evaluate_lmmse.py` | `outputs/lmmse/lmmse_results.json`, `outputs/evaluation/lmmse_evaluation.md` |
| Problem 2(a) Transformer architecture sweep | `scripts/run_transformer_sweep.py` | `outputs/transformer/<run>/metrics.json`, `train_losses.npy`, `val_losses.npy`, `history.csv` |
| Problem 2(b) Transformer P sweep | `scripts/run_transformer_sweep.py`, `evaluation/evaluate_transformer.py` | `outputs/evaluation/transformer_evaluation.md` |
| Problem 2(c) attention vs LMMSE | `scripts/infer_transformer.py --save-attention`, `evaluation/compare_methods.py` | `outputs/evaluation/attention_vs_lmmse_P*.png` |

## Current Results Snapshot

- Best LMMSE order: P=128 with NMSE=0.37433756.
- Best Transformer run currently present: window_P128_medium with NMSE=0.12366345.
- Method comparison rows present: 9.
- Architecture sweep runs present: arch_deep_P64, arch_medium_P64, arch_small_P64.
- Window sweep runs present: window_P128_medium, window_P16_medium, window_P32_medium, window_P64_medium, window_P8_medium.

## Notes

The current summary was rebuilt from the completed LMMSE sweep, Transformer architecture sweep, Transformer window sweep, inference outputs, and attention-vs-filter comparisons.
