# LMMSE Evaluation

Best LMMSE order: **P = 128**
Best LMMSE NMSE: **0.37433756**

## NMSE By Order

| P | NMSE | Aligned samples | Filter norm |
| ---: | ---: | ---: | ---: |
| 8 | 0.45455394 | 8993 | 0.378638 |
| 16 | 0.45427845 | 8985 | 0.440204 |
| 32 | 0.43762223 | 8969 | 0.699592 |
| 64 | 0.40329022 | 8937 | 0.697101 |
| 128 | 0.37433756 | 8873 | 0.555837 |

## Signal Statistics

- Noise autocorrelation whiteness ratio: `0.998270`
- Interpretation: not well approximated as white.
- Clean PSD peak frequency: `6.328` Hz.
- Noise PSD peak frequency: `0.703` Hz.
