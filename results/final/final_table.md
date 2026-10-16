Artist-disjoint splits, seeds [1, 2, 3, 4, 5] (one split per seed, 217-237 test images each). Mean ± std over runs; [95 % CI of the mean].
Latency: end-to-end (resize + features + model), batch 1, 1 CPU thread.

| Model | Val acc (%) | Test acc (%) | 95 % CI | Gap to fine-tuned ResNet (pts, paired) | Params | Size (MB) | Latency (ms/img) | Train (s) |
|---|---|---|---|---|---|---|---|---|
| logreg_pixels | 40.7 ± 2.2 | 42.1 ± 3.6 | [37.6, 46.5] | -35.0 [-39.3, -30.7] | 15,365 | 0.1 | 0.61 | 2 |
| random_forest_pixels | 40.0 ± 5.3 | 39.3 ± 4.7 | [33.5, 45.2] | -37.7 [-42.0, -33.4] | 124,272 nodes | 12.7 | 12.73 | 7 |
| logreg_handcrafted | 55.0 ± 3.7 | 53.5 ± 4.0 | [48.4, 58.5] | -23.6 [-26.5, -20.7] | 4,235 | 0.0 | 1.44 | 0 |
| svm_handcrafted | 51.8 ± 2.3 | 52.2 ± 4.4 | [46.8, 57.6] | -24.9 [-27.9, -21.8] | 874,650 | 6.9 | 1.93 | 2 |
| ffn | 42.7 ± 1.7 | 40.7 ± 4.1 | [35.6, 45.8] | -36.4 [-40.4, -32.4] | 1,573,637 | 6.3 | 0.47 | 4 |
| resnet_features_logreg | 73.0 ± 1.7 | 74.8 ± 2.9 | [71.3, 78.4] | -2.3 [-6.5, +2.0] | 2,565 + 11.2 M frozen | 44.7 | 12.47 | 26 |
| resnet_extract | 72.4 ± 3.1 | 73.7 ± 2.1 | [71.0, 76.3] | -3.4 [-5.1, -1.7] | 11,179,077 | 44.7 | 15.16 | 252 |
| resnet_finetune | 75.5 ± 4.0 | 77.1 ± 2.4 | [74.1, 80.1] | reference | 11,179,077 | 44.7 | 15.37 | 273 |
