# Method Comparison

| Transformer run | P | LMMSE NMSE | Transformer NMSE | Difference |
| --- | ---: | ---: | ---: | ---: |
| arch_deep_P64 | 64 | 0.40329022 | 0.15095983 | -0.25233039 |
| arch_medium_P64 | 64 | 0.40329022 | 0.14324441 | -0.26004581 |
| arch_small_P64 | 64 | 0.40329022 | 0.13872611 | -0.26456411 |
| transformer_p64 | 64 | 0.40329022 | 0.13872611 | -0.26456411 |
| window_P128_medium | 128 | 0.37433756 | 0.12366345 | -0.25067412 |
| window_P16_medium | 16 | 0.45427845 | 0.15638778 | -0.29789066 |
| window_P32_medium | 32 | 0.43762223 | 0.14647637 | -0.29114585 |
| window_P64_medium | 64 | 0.40329022 | 0.14324441 | -0.26004581 |
| window_P8_medium | 8 | 0.45455394 | 0.18213470 | -0.27241924 |

## Attention vs LMMSE

| Run | P | Correlation |
| --- | ---: | ---: |
| arch_deep_P64 | 64 | 0.649871 |
| arch_medium_P64 | 64 | 0.232308 |
| arch_small_P64 | 64 | 0.148786 |
| transformer_p64 | 64 | 0.148786 |
| window_P128_medium | 128 | 0.124198 |
| window_P16_medium | 16 | 0.203514 |
| window_P32_medium | 32 | 0.031165 |
| window_P64_medium | 64 | 0.232308 |
| window_P8_medium | 8 | -0.373772 |
