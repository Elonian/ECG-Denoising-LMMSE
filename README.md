# ECG Denoising LMMSE

Author: Varun Moparthi

## Overview

This repository implements a complete ECG denoising study for ECE 251A Numerical Assignment 2. It compares a classical linear minimum mean-square error estimator with Transformer encoder denoisers on paired clean and noisy ECG signals.

The clean ECG comes from MIT-BIH Arrhythmia Database record `118`. The noisy ECG comes from MIT-BIH Noise Stress Test Database record `118e06`, sampled at `360 Hz`. The denoising task is:

```text
y[n] = s[n] + w[n]
```

where `y[n]` is the noisy ECG, `s[n]` is the clean ECG, and `w[n]` is additive noise. Both methods estimate `s[n]` from a causal window of noisy samples:

```text
y_P[n] = [y[n], y[n-1], ..., y[n-P+1]]
```

On the completed test split, the best LMMSE result is NMSE `0.37433756` at `P=128`. The best Transformer result is NMSE `0.12366345` from `window_P128_medium`.

## Output Gallery

The best qualitative denoising example comes from `window_P128_medium`. The grey trace is the noisy ECG, the black trace is the clean reference, and the blue trace is the Transformer estimate.

![Transformer denoising waveform](outputs/visualiser/transformer/transformer_inference_waveform.png)

Additional generated figures:

| Figure | Path |
| --- | --- |
| Clean and noisy ECG preview | `outputs/visualiser/lmmse/signal_preview.png` |
| Autocorrelation | `outputs/visualiser/lmmse/autocorrelation.png` |
| Welch PSD | `outputs/visualiser/lmmse/welch_psd.png` |
| LMMSE filters | `outputs/visualiser/lmmse/lmmse_filters.png` |
| Transformer train and validation losses | `outputs/visualiser/transformer/window_train_validation_losses.png` |
| Transformer architecture comparison | `outputs/visualiser/transformer/architecture_nmse_comparison.png` |
| LMMSE and Transformer NMSE comparison | `outputs/visualiser/transformer/window_nmse_vs_lmmse.png` |
| Attention and LMMSE filter comparison | `outputs/evaluation/attention_vs_lmmse_P128.png` |

## Data

The data pipeline creates four one-dimensional NumPy arrays:

| File | Meaning | Samples |
| --- | --- | ---: |
| `data/s_train.npy` | Clean training ECG | 21000 |
| `data/y_train.npy` | Noisy training ECG | 21000 |
| `data/s_test.npy` | Clean test ECG | 9000 |
| `data/y_test.npy` | Noisy test ECG | 9000 |

The split starts at sample `108000`, which is `300` seconds into the record at `360 Hz`. The training range is `[108000, 129000]`, and the test range is `[129000, 138000]`.

The repository tracks `data/.gitkeep`, but the generated arrays and downloaded WFDB records are intended to be local data artifacts.

## Method

### LMMSE Estimator

The LMMSE branch uses a linear filter of order `P`:

```text
s_hat[n] = h^T y_P[n]
```

The filter is fitted from training windows by solving the empirical normal equation:

```text
R_yy h = r_sy
```

where:

```text
R_yy = average(y_P[n] y_P[n]^T)
r_sy = average(y_P[n] s[n])
```

The implementation adds a small ridge term, `lambda = 1e-8`, before solving:

```text
h = inverse(R_yy + lambda I) r_sy
```

The project evaluates LMMSE orders:

```text
P = 8, 16, 32, 64, 128
```

The signal statistics required by the assignment are also computed:

| Statistic | Implementation | Output |
| --- | --- | --- |
| Biased autocorrelation | `utils/signal_processing.py` | `outputs/lmmse/autocorrelation_sequences.npz` |
| Welch PSD | `utils/signal_processing.py` | `outputs/lmmse/welch_psd.npz` |
| LMMSE filters | `utils/signal_processing.py` | `outputs/lmmse/lmmse_filters.npz` |

The generated evaluation reports:

| Quantity | Value |
| --- | ---: |
| Train noise SNR | `-1.51833277 dB` |
| Noise autocorrelation whiteness ratio | `0.998270` |
| Clean PSD peak | `6.328 Hz` |
| Noise PSD peak | `0.703 Hz` |

The high off-zero autocorrelation ratio indicates that the selected noise segment is not well approximated as white noise.

### Transformer Denoiser

The Transformer branch uses the same input window and target convention, but learns a nonlinear mapping:

```text
s_hat[n] = f_theta(y_P[n])
```

Before training, both inputs and targets are normalized using the noisy training statistics:

```text
x_norm = (x - mean(y_train)) / std(y_train)
```

The model architecture is implemented in `models/transformer_denoiser.py`:

| Component | Description |
| --- | --- |
| Scalar projection | Projects each noisy ECG sample into a token embedding |
| CLS token | Prepended learned token used for regression |
| Positional encoding | Sinusoidal position encoding |
| Transformer encoder | Custom encoder stack with self-attention and feedforward blocks |
| Regression head | Layer normalization plus linear scalar output |

Training uses mean squared error on normalized clean ECG targets. Each run trains for `200` epochs with Adam, learning rate `1e-3`, batch size `256`, dropout `0.1`, validation fraction `0.15`, and gradient clipping at norm `1.0`.

The architecture sweep fixes `P=64`:

| Name | `d_model` | Heads | Layers | Feedforward |
| --- | ---: | ---: | ---: | ---: |
| `small` | 32 | 4 | 2 | 128 |
| `medium` | 64 | 4 | 3 | 256 |
| `deep` | 64 | 8 | 4 | 256 |

The window sweep fixes the `medium` architecture and evaluates:

```text
P = 8, 16, 32, 64, 128
```

### Evaluation

The primary metric is normalized mean squared error:

```text
NMSE = sum((s[n] - s_hat[n])^2) / sum(s[n]^2)
```

The code also stores MSE:

```text
MSE = average((s[n] - s_hat[n])^2)
```

For interpretability, Transformer inference stores the average CLS attention distribution. The evaluation compares this attention vector with normalized absolute LMMSE filter coefficients for the same order `P`.

## Results

### LMMSE Sweep

| P | Test NMSE | Aligned samples | Filter norm |
| ---: | ---: | ---: | ---: |
| 8 | 0.45455394 | 8993 | 0.378638 |
| 16 | 0.45427845 | 8985 | 0.440204 |
| 32 | 0.43762223 | 8969 | 0.699592 |
| 64 | 0.40329022 | 8937 | 0.697101 |
| 128 | 0.37433756 | 8873 | 0.555837 |

The LMMSE filter improves as `P` increases, with the best result at `P=128`.

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

The best architecture sweep run is `arch_small_P64`. In the window sweep, longer context improves performance, and `window_P128_medium` gives the strongest result.

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

Across every comparable order, the Transformer has a lower NMSE than the LMMSE filter.

![Window sweep comparison](outputs/visualiser/transformer/window_nmse_vs_lmmse.png)

![Method comparison](outputs/evaluation/method_nmse_comparison.png)

### Attention vs LMMSE Filter

| Run | P | Attention and filter correlation |
| --- | ---: | ---: |
| `window_P8_medium` | 8 | -0.373772 |
| `window_P16_medium` | 16 | 0.203514 |
| `window_P32_medium` | 32 | 0.031165 |
| `window_P64_medium` | 64 | 0.232308 |
| `window_P128_medium` | 128 | 0.124198 |

![Attention vs LMMSE, P=128](outputs/evaluation/attention_vs_lmmse_P128.png)

## Reproducing The Pipeline

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create or refresh the data arrays:

```bash
python scripts/create_training_test_set.py --config configs/data.yaml --skip-download
```

Run LMMSE:

```bash
python scripts/run_lmmse.py --config configs/lmmse.yaml
```

Train the default Transformer:

```bash
python scripts/train_transformer.py --config configs/transformer.yaml
```

Run the full Transformer sweep:

```bash
python scripts/run_transformer_sweep.py --config configs/transformer_sweep.yaml
```

Run inference for saved checkpoints:

```bash
python scripts/run_inference_for_checkpoints.py --transformer-dir outputs/transformer --data-dir data --save-attention --overwrite
```

Build figures and summaries:

```bash
python visualiser/build_figures.py --config configs/visualiser.yaml
python evaluation/summarize_results.py --config configs/evaluation.yaml
```

## Repository Layout

```text
configs/             YAML configuration for data, LMMSE, Transformer, sweeps, visualizers, and evaluation
data/                Local PhysioNet arrays and metadata; ignored except data/.gitkeep
evaluation/          Metric summaries, method comparison, and attention and filter analysis
execution_scripts/   Shell entry points for reproducible pipeline runs
logs/                Tracked run logs from data creation, LMMSE, Transformer training, and evaluation
models/              ECG Transformer denoiser definition
modules/             Transformer encoder blocks and positional encoding
outputs/             Tracked generated metrics, curves, NumPy artifacts, markdown summaries, and figures
scripts/             Data creation, LMMSE execution, Transformer training, sweep, and inference scripts
utils/               Config loading, PhysioNet utilities, metrics, logging, and signal processing helpers
visualiser/          Plot builders for LMMSE, Transformer, and final figures
```

## License

This project is released under the MIT License.

Copyright (c) 2026 Varun Moparthi
