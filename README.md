# ECG-Denoising-LMMSE

Author: Varun Moparthi

## Abstract

ECG-Denoising-LMMSE implements a complete ECG noise-removal study on paired PhysioNet signals, using clean ECG from MIT-BIH record `118` and noisy ECG from NSTDB record `118e06`. The project constructs aligned train/test arrays, characterizes the signal and noise with autocorrelation and Welch spectral estimates, derives a sample-wise LMMSE estimator from lagged noisy ECG windows, and compares that linear estimator against Transformer encoder denoisers trained on the same causal window formulation.

The denoising task is treated as clean-sample reconstruction from a local noisy context, where each model receives `y_P[n] = [y[n], y[n-1], ..., y[n-P+1]]^T` and predicts `s[n]`. The LMMSE branch solves the empirical normal equation `R_yy h = r_sy` for multiple filter orders, while the Transformer branch learns a nonlinear CLS-token representation over the same window sizes. The final analysis includes order sweeps, architecture sweeps, train/validation loss curves, test NMSE tables, denoised waveform views, and attention-vs-filter comparisons.

On the completed test split, the best LMMSE filter reaches NMSE `0.37433756` at `P=128`, while the best Transformer run reaches NMSE `0.12366345` with `window_P128_medium`. The result shows that learned nonlinear temporal context modeling provides a much stronger denoising fit than the linear LMMSE estimator for this ECG/noise pairing, while still keeping the experiment interpretable through matched window orders and filter/attention diagnostics.

## Highlights

- Builds aligned clean/noisy ECG training and test arrays from PhysioNet MIT-BIH record `118` and NSTDB record `118e06`.
- Estimates signal and noise autocorrelation up to lag `256`, plus Welch PSD using a Hann window of length `512` with `50%` overlap.
- Trains and evaluates LMMSE filters for `P in {8, 16, 32, 64, 128}`.
- Trains Transformer encoder denoisers with both architecture sweeps and window-size sweeps.
- Stores train and validation loss curves for every Transformer run.
- Extracts CLS attention weights and compares them against normalized LMMSE filter magnitudes.
- Tracks generated logs, metrics, summaries, NumPy outputs, and visualizations; model checkpoint `.pt` files remain ignored.

## Data Source

The data pipeline creates four one-dimensional NumPy arrays:

| File | Meaning | Samples |
| --- | --- | ---: |
| `data/s_train.npy` | clean training ECG | 21000 |
| `data/y_train.npy` | noisy training ECG | 21000 |
| `data/s_test.npy` | clean test ECG | 9000 |
| `data/y_test.npy` | noisy test ECG | 9000 |

The clean ECG comes from PhysioNet MIT-BIH Arrhythmia Database record `118`. The noisy ECG comes from MIT-BIH Noise Stress Test Database record `118e06`. The split begins at sample `108000`, corresponding to `300` seconds at `360 Hz`, so the selected segment lies in the noisy portion of the NSTDB record. The project stores metadata in `data/dataset_metadata.json`, but the data arrays and downloaded WFDB records are ignored by Git so the repository stays light.

## Method

The denoising problem is modeled as

```text
y[n] = s[n] + w[n],
```

where `s[n]` is the clean ECG, `w[n]` is additive noise, and `y[n]` is the observed noisy ECG. Both the LMMSE and Transformer methods use a causal lag window of length `P`:

```text
y_P[n] = [y[n], y[n-1], ..., y[n-P+1]]^T.
```

The target is the clean sample aligned with the newest value in the window, `s[n]`.

### Signal Statistics

The code estimates biased autocorrelation for the clean signal and training noise:

```text
r_x[k] = (1 / N) sum_{n=0}^{N-|k|-1} x[n+|k|] x[n].
```

It also estimates the power spectral density with Welch's method. For each segment `x_m[n]`, the periodogram is averaged as

```text
S_xx[f] = average_m |FFT{a[n] x_m[n]}|^2,
```

where `a[n]` is the Hann analysis window. The generated evaluation found that the training noise is not well approximated as white: the noise autocorrelation whiteness ratio is `0.998270`, the clean PSD peak is near `6.328 Hz`, and the noise PSD peak is near `0.703 Hz`.

### LMMSE Filter

The LMMSE filter predicts the clean sample with a linear estimator:

```text
s_hat[n] = h^T y_P[n].
```

The optimal filter minimizes mean squared error:

```text
h* = argmin_h E[(s[n] - h^T y_P[n])^2].
```

Setting the gradient to zero gives the normal equation:

```text
R_yy h = r_sy,
```

with

```text
R_yy = E[y_P[n] y_P[n]^T]
r_sy = E[y_P[n] s[n]].
```

The implemented empirical estimates are

```text
R_yy_hat = (1 / M) Y^T Y
r_sy_hat = (1 / M) Y^T s,
```

where each row of `Y` is one lagged noisy ECG window. The filter is solved as

```text
h_hat = (R_yy_hat + lambda I)^(-1) r_sy_hat,
```

using a small ridge value `lambda = 1e-8` for numerical stability.

### Transformer Denoiser

The Transformer uses the same lag window, but learns a nonlinear mapping from the noisy context to the clean sample. Each scalar window value is projected into an embedding, a trainable CLS token is prepended, sinusoidal positional encoding is added, and the sequence is passed through a Transformer encoder stack:

```text
e_i = W_in x_i + b_in
Z_0 = [c, e_1, ..., e_P] + PE
Z_l = TransformerEncoderLayer_l(Z_{l-1})
s_hat[n] = W_out LayerNorm(Z_L[CLS]) + b_out.
```

Training minimizes mean squared error on normalized ECG samples:

```text
L(theta) = (1 / M) sum_n (s_norm[n] - f_theta(y_P_norm[n]))^2.
```

Normalization is computed from the noisy training signal:

```text
x_norm = (x - mean(y_train)) / std(y_train).
```

After inference, predictions are mapped back to the original ECG amplitude scale before computing metrics.

### Evaluation Metric

All denoising results are evaluated using normalized mean squared error:

```text
NMSE = sum_n (s[n] - s_hat[n])^2 / sum_n s[n]^2.
```

Lower NMSE means the denoised signal is closer to the clean reference.

## Results

### LMMSE Sweep

| P | Test NMSE | Aligned samples | Filter norm |
| ---: | ---: | ---: | ---: |
| 8 | 0.45455394 | 8993 | 0.378638 |
| 16 | 0.45427845 | 8985 | 0.440204 |
| 32 | 0.43762223 | 8969 | 0.699592 |
| 64 | 0.40329022 | 8937 | 0.697101 |
| 128 | 0.37433756 | 8873 | 0.555837 |

The LMMSE trend improves as the filter order increases. The best LMMSE result is `P=128`, but the method remains limited because the ECG/noise relationship is not fully captured by a linear stationary filter.

![LMMSE NMSE by filter order](outputs/visualiser/lmmse/lmmse_nmse.png)

### Transformer Sweep

| Run | P | Final train loss | Final validation loss | Test NMSE | Test MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arch_small_P64` | 64 | 0.01151857 | 0.05396836 | 0.13872611 | 0.14016187 |
| `arch_medium_P64` | 64 | 0.01061656 | 0.04986817 | 0.14324441 | 0.14472693 |
| `arch_deep_P64` | 64 | 0.01074144 | 0.05094075 | 0.15095983 | 0.15252221 |
| `window_P8_medium` | 8 | 0.06404387 | 0.06708255 | 0.18213470 | 0.18368720 |
| `window_P16_medium` | 16 | 0.03254132 | 0.05668371 | 0.15638778 | 0.15775682 |
| `window_P32_medium` | 32 | 0.01465691 | 0.04484816 | 0.14647637 | 0.14783353 |
| `window_P64_medium` | 64 | 0.01061656 | 0.04986817 | 0.14324441 | 0.14472693 |
| `window_P128_medium` | 128 | 0.00788836 | 0.04227321 | 0.12366345 | 0.12527506 |

The best architecture sweep run is `arch_small_P64`. In the window sweep, increasing context improves test NMSE, with `window_P128_medium` giving the strongest result. The validation curves show that the models learn quickly and that the final validation loss is tracked for every run, not only the final test score.

![Transformer train and validation loss curves](outputs/visualiser/transformer/window_train_validation_losses.png)

![Transformer architecture comparison](outputs/visualiser/transformer/architecture_nmse_comparison.png)

### LMMSE vs Transformer

| Transformer run | P | LMMSE NMSE | Transformer NMSE | Difference |
| --- | ---: | ---: | ---: | ---: |
| `arch_small_P64` | 64 | 0.40329022 | 0.13872611 | -0.26456411 |
| `arch_medium_P64` | 64 | 0.40329022 | 0.14324441 | -0.26004581 |
| `arch_deep_P64` | 64 | 0.40329022 | 0.15095983 | -0.25233039 |
| `window_P8_medium` | 8 | 0.45455394 | 0.18213470 | -0.27241924 |
| `window_P16_medium` | 16 | 0.45427845 | 0.15638778 | -0.29789066 |
| `window_P32_medium` | 32 | 0.43762223 | 0.14647637 | -0.29114585 |
| `window_P64_medium` | 64 | 0.40329022 | 0.14324441 | -0.26004581 |
| `window_P128_medium` | 128 | 0.37433756 | 0.12366345 | -0.25067412 |

Across every comparable order, the Transformer has a lower NMSE than the LMMSE filter. The gap is especially clear in the window sweep, where the Transformer curve remains far below the LMMSE curve for all tested `P`.

![Window sweep comparison](outputs/visualiser/transformer/window_nmse_vs_lmmse.png)

![Method comparison](outputs/evaluation/method_nmse_comparison.png)

### Attention vs LMMSE Filter

For interpretability, the pipeline extracts the mean CLS attention distribution from Transformer inference and compares it to the normalized absolute LMMSE filter coefficients. This does not require the Transformer attention to match the linear filter exactly; instead, it gives a diagnostic view of whether the learned model emphasizes similar temporal positions.

| Run | P | Attention-filter correlation |
| --- | ---: | ---: |
| `window_P8_medium` | 8 | -0.373772 |
| `window_P16_medium` | 16 | 0.203514 |
| `window_P32_medium` | 32 | 0.031165 |
| `window_P64_medium` | 64 | 0.232308 |
| `window_P128_medium` | 128 | 0.124198 |

![Attention vs LMMSE, P=128](outputs/evaluation/attention_vs_lmmse_P128.png)

## Output Gallery

| Figure | Path |
| --- | --- |
| Clean/noisy ECG preview | `outputs/visualiser/lmmse/signal_preview.png` |
| Autocorrelation | `outputs/visualiser/lmmse/autocorrelation.png` |
| Welch PSD | `outputs/visualiser/lmmse/welch_psd.png` |
| LMMSE filters | `outputs/visualiser/lmmse/lmmse_filters.png` |
| Transformer all-run loss curves | `outputs/visualiser/transformer/transformer_train_validation_losses.png` |
| Transformer window loss curves | `outputs/visualiser/transformer/window_train_validation_losses.png` |
| Transformer inference waveform | `outputs/visualiser/transformer/transformer_inference_waveform.png` |
| Validation gap summary | `outputs/visualiser/transformer/validation_gap_summary.png` |

## Repository Layout

```text
configs/             YAML configuration for data, LMMSE, Transformer, sweeps, visualizers, and evaluation
data/                Local PhysioNet arrays and metadata; ignored except data/.gitkeep
evaluation/          Metric summaries, method comparison, and attention-vs-filter analysis
execution_scripts/   Shell entry points for reproducible pipeline runs
logs/                Tracked run logs from data creation, LMMSE, Transformer training, and evaluation
models/              ECG Transformer denoiser definition
modules/             Transformer encoder blocks and positional encoding
outputs/             Tracked generated metrics, curves, NumPy artifacts, markdown summaries, and figures
scripts/             Data creation, LMMSE execution, Transformer training, sweep, and inference scripts
utils/               Config loading, PhysioNet utilities, metrics, logging, and signal processing helpers
visualiser/          Plot builders for LMMSE, Transformer, and final figures
```

## Reproducing The Pipeline

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create or refresh the train/test arrays:

```bash
./execution_scripts/create_training_test_set.sh
```

Run the full core pipeline:

```bash
./execution_scripts/run_full_pipeline.sh
```

Run the full Transformer sweep:

```bash
./execution_scripts/run_transformer_sweep.sh
```

Regenerate inference outputs, figures, and summaries:

```bash
./execution_scripts/run_transformer_inference.sh
./execution_scripts/build_figures.sh
./execution_scripts/run_evaluation.sh
```

The shell scripts install `requirements.txt` before running their Python commands. This makes each entry point usable from a fresh environment as long as Python and the required system libraries are available.

## Key Artifacts

| Artifact | Purpose |
| --- | --- |
| `outputs/lmmse/lmmse_results.json` | LMMSE order sweep metrics |
| `outputs/lmmse/lmmse_filters.npz` | Estimated filters for each `P` |
| `outputs/transformer/*/history.csv` | Per-epoch train and validation losses |
| `outputs/transformer/*/metrics.json` | Transformer test metrics and model config |
| `outputs/transformer/*/inference/cls_attention_mean.npy` | Mean CLS attention for attention-vs-filter plots |
| `outputs/evaluation/results_summary.md` | Coverage and final result summary |
| `outputs/evaluation/method_comparison.md` | LMMSE vs Transformer comparison table |

## Git Tracking Notes

The repository tracks logs and generated visual/evaluation outputs because they document the completed runs. The following are intentionally ignored:

- `data/*`, except `data/.gitkeep`
- `*.pt` model checkpoints
- Python cache files
- `cache_weights/`
- `papers/`

This keeps the repository reproducible and inspectable without committing heavy raw data or checkpoint files.

## License

This project is released under the MIT License.

Copyright (c) 2026 Varun Moparthi
