#!/usr/bin/env bash
# Re-run every experiment of the report, in order (about 45 min on a laptop CPU).
# Needs the data: see "Data" in README.md. Outputs go to results/<topic>/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

# 0. Data checks
$PY src/explore.py
$PY src/check_artists.py

# 1. Classical baselines (validation only)
mkdir -p results/baselines
$PY src/baseline.py | tee results/baselines/baseline_log.txt

# 2. Feed-forward network: effect of each regularisation (seed 42 split)
$PY src/ffn.py --tag ffn_base
$PY src/ffn.py --augment --tag ffn_aug
$PY src/ffn.py --dropout 0.5 --wd 1e-3 --tag ffn_reg
$PY src/ffn.py --dropout 0.5 --wd 1e-3 --augment --epochs 80 --patience 8 --tag ffn_all

# 3. Monitoring and failure diagnosis
D="--subdir diagnosis"
$PY src/ffn.py --overfit-batch --epochs 60 --tag sanity_overfit_batch $D
$PY src/ffn.py --lr 0.5 --epochs 30 --tag fail_lr_high $D
$PY src/ffn.py --lr 1e-6 --epochs 30 --tag fail_lr_low $D
$PY src/ffn.py --activation none --epochs 30 --tag fail_linear $D
$PY src/ffn.py --no-normalise --epochs 30 --tag fail_no_normalise $D
$PY src/ffn.py --depth 8 --activation sigmoid --optimizer sgd --lr 0.05 --epochs 30 --tag fail_vanishing_sigmoid $D
$PY src/ffn.py --depth 8 --activation relu --optimizer sgd --lr 0.05 --epochs 30 --tag control_deep_relu $D

# 4. Pretrained ResNet-18 (learning curves, seed 42 split)
$PY src/resnet.py --mode extract
$PY src/resnet.py --mode finetune

# 5. Final comparison on the test set (5 artist-disjoint splits) and error analysis
$PY src/final.py --seeds 1 2 3 4 5 | tee results/final/final_log.txt
$PY src/confusion.py | tee results/resnet/confusion_log.txt
