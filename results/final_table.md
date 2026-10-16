Test set: 225 images, seeds [1, 2, 3]. Latency: batch 1, 1 CPU thread.

| Model | Val acc (%) | Test acc (%) | Params | Size (MB) | Latency (ms/img) | Train (s) |
|---|---|---|---|---|---|---|
| logreg | 37.8 ± 0.0 | 34.7 ± 0.0 | 15,365 | 0.2 | 0.16 | 1 |
| random_forest | 44.4 ± 1.2 | 39.9 ± 1.1 | 121,610 | 12.8 | 6.81 | 7 |
| ffn | 42.2 ± 2.8 | 38.4 ± 4.9 | 1,573,637 | 6.3 | 0.06 | 2 |
| resnet_extract | 75.6 ± 1.2 | 73.3 ± 1.2 | 11,179,077 | 44.7 | 9.27 | 110 |
| resnet_finetune | 78.7 ± 1.9 | 76.4 ± 2.2 | 11,179,077 | 44.7 | 9.01 | 161 |
