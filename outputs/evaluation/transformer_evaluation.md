# Transformer Evaluation

Best run: **window_P128_medium**
Best NMSE: **0.12366345**

| Run | P | Final train loss | Final validation loss | NMSE | MSE | Checkpoint | Attention |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| arch_deep_P64 | 64 | 0.01074144 | 0.05094075 | 0.15095983 | 0.15252221 | yes | yes |
| arch_medium_P64 | 64 | 0.01061656 | 0.04986817 | 0.14324441 | 0.14472693 | yes | yes |
| arch_small_P64 | 64 | 0.01151857 | 0.05396836 | 0.13872611 | 0.14016187 | yes | yes |
| transformer_p64 | 64 | 0.01151857 | 0.05396836 | 0.13872611 | 0.14016187 | yes | yes |
| window_P128_medium | 128 | 0.00788836 | 0.04227321 | 0.12366345 | 0.12527506 | yes | yes |
| window_P16_medium | 16 | 0.03254132 | 0.05668371 | 0.15638778 | 0.15775682 | yes | yes |
| window_P32_medium | 32 | 0.01465691 | 0.04484816 | 0.14647637 | 0.14783353 | yes | yes |
| window_P64_medium | 64 | 0.01061656 | 0.04986817 | 0.14324441 | 0.14472693 | yes | yes |
| window_P8_medium | 8 | 0.06404387 | 0.06708255 | 0.18213470 | 0.18368720 | yes | yes |
