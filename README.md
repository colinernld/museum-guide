# Museum guide: artwork genre classification

Neural-network components of an intelligent museum guide (fictional French fine-arts museum).
A visitor points a phone or kiosk camera at a painting; the guide recognises its **genre**
(cityscape, landscape, portrait, religious painting, still life) so it can tell the story of that tradition.

We build, diagnose and compare:

1. a **feed-forward network** in PyTorch with a hand-written training loop;
2. **training monitoring and failure diagnosis** (gradient norms, update sizes, dead / saturated units, provoked failures);
3. a **pretrained ResNet-18** (frozen features and fine-tuning);
4. **classical baselines** (raw pixels and hand-crafted colour / shape / texture features) to answer the
   director's question: *is a neural network worth it on modest kiosk CPUs?*

Every number below is produced by the scripts of this repository and stored in `results/`.

## Key results

| | Test accuracy (5 splits, painters never seen in training) |
|---|---|
| Chance | 20 % |
| Feed-forward network on pixels | 40.7 % ± 4.1 |
| Best classical model (hand-crafted features + logistic regression) | 53.5 % ± 4.0 |
| ResNet-18 frozen features + logistic regression | 74.8 % ± 2.9 |
| ResNet-18 fine-tuned | **77.1 % ± 2.4** |

- The pretrained CNN beats the best classical model by **23.6 points** (95 % CI 20.7 to 26.5, paired over 5 splits).
- Fine-tuning is **not significantly better** than a logistic regression on frozen ResNet features
  (+2.3 points, 95 % CI -2.0 to +6.5) and takes 10x longer to train.
- The feed-forward network on raw pixels does not beat a logistic regression on the same pixels.

## Data

WikiArt subset (`huggan/wikiart`, pinned revision), 5 genres x 300 images = 1 500 images, 128x128 px (centre crop).
Exact command, dataset revision and shards: `results/data/dataset_info.json`.

Images are not committed (WikiArt licence unknown). Rebuild them with (about 2.3 GB streamed, a few minutes):

    pip install -r requirements.txt
    python make_wikiart_subset.py --label genre \
        --classes portrait landscape cityscape still_life religious_painting \
        --per-class 300 --size 128 --out data/wikiart_genre5 --retries 10

### Artist-disjoint split

WikiArt is ordered by painter. With a random split, **53 % of the test images were painted by an artist also
present in training** (`src/check_artists.py`, `results/data/artist_leakage.json`): the model can recognise
the painter's hand instead of the genre. A quick check (frozen ResNet features + logistic regression) showed
about 7 points of optimism from this leak.

All experiments therefore use an **artist-disjoint split** (`src/data.py`, `get_splits`): a named painter is
in exactly one of train / validation / test; the 717 "Unknown Artist" images (a catch-all label, not one
painter) are split at random with stratification. Painters are assigned class by class (by their most frequent
genre) so that classes stay balanced. About 70 / 15 / 15 %, i.e. about 1 050 / 225 / 225 images; sizes vary
by a few images from one seed to the other. Seed 42 is the development split; seeds 1 to 5 are the five
evaluation splits of `src/final.py`.

## Repository layout

    make_wikiart_subset.py   data download script (provided by the course)
    run_all.sh               re-runs every experiment of this README, in order (~1 h on a laptop CPU)
    src/data.py              paths, artist-disjoint split (get_splits), output folders
    src/explore.py           class counts, image sizes, sample grid (grid not committed: WikiArt images)
    src/check_artists.py     artist leakage: random split vs artist-disjoint split
    src/features.py          features for classical models: raw pixels, hand-crafted (HSV histogram, HOG, LBP)
    src/baseline.py          classical models, first look on validation (C chosen on validation)
    src/ffn.py               feed-forward network, hand-written loop, monitoring, flags for every diagnosis experiment
    src/resnet.py            ResNet-18 learning curves: --mode extract | finetune
    src/final.py             final comparison of 8 models on 5 splits, test set, CIs, end-to-end CPU latency
    src/confusion.py         confusion matrix, per-class recall, confidence threshold (fine-tuned ResNet)
    results/
      data/                  dataset_info.json, artist_leakage.json
      baselines/             baseline_log.txt
      ffn/                   regularisation runs (*_curves.png, *_history.json)
      diagnosis/             sanity check and provoked failures (*_curves.png, *_history.json)
      resnet/                learning curves, confusion matrix, confusion_log.txt
      final/                 final_table.md, final_results.json (one row per model and split), final_log.txt

## How to run

Python 3.12, CPU only. Every script can be run from any folder.

    pip install -r requirements.txt
    ./run_all.sh                                   # everything (or PYTHON=/path/to/python ./run_all.sh)

    python src/ffn.py --dropout 0.5 --wd 1e-3 --augment --patience 8 --tag my_run   # one FFN run
    python src/ffn.py --help                       # all flags (depth, activation, optimizer, sanity check...)
    python src/final.py --seeds 1 2 3 4 5          # final comparison (~35 min)
    python src/final.py --table-only               # rebuild final_table.md from final_results.json

## 1. Feed-forward network

`Flatten -> Linear(12 288, 128) -> ReLU -> Dropout -> Linear(128, 5)`, 1.57 M parameters, 64x64 input
standardised with training statistics, Adam (lr 1e-3), cross-entropy, mini-batches of 64.
The model kept is the one of the epoch with the lowest validation loss. Development split (seed 42):

| Run | Best epoch | Val acc | Train acc at best epoch | Max train acc |
|---|---|---|---|---|
| base | 3 / 40 | 41.4 % | 75.4 % | 100 % |
| + augmentation (horizontal flip) | 3 / 40 | 41.4 % | 65.3 % | 100 % |
| + dropout 0.5, weight decay 1e-3 | 6 / 40 | 40.6 % | 75.0 % | 98.2 % |
| all combined + early stopping (patience 8) | 11 / 19 | 42.6 % | 66.3 % | 79.5 % |

Without regularisation the network memorises the training set in a few epochs (100 % train accuracy) while
validation stays around 41 %. Regularisation reduces the train / validation gap and delays the best epoch,
but the validation ceiling does not move: a dense network on 12 288 pixels has no notion of locality or
translation, and 1 050 images are far too few to learn it. Over the 5 final splits it reaches
40.7 % ± 4.1, the same level as a logistic regression on pixels (42.1 % ± 3.6).

## 2. Training monitoring and failure diagnosis

`src/ffn.py` logs at every epoch, for each linear layer: the mean gradient norm, the relative size of the
weight update ||dW|| / ||W||, and the fraction of hidden units that are dead (ReLU never active on the
validation set) or saturated (sigmoid output below 0.01 or above 0.99 for 90 % of the images).
The curves are in `results/diagnosis/*_curves.png` (loss, accuracy, gradient norms, update sizes).

| Experiment | Setting | What the monitoring shows | Diagnosis |
|---|---|---|---|
| Sanity check | train on one batch of 64 images | train accuracy 100 % by epoch 5, loss 0 | the loop, loss and gradients are wired correctly (**PASS**) |
| Learning rate too high | Adam, lr 0.5 | val loss up to 1e5, last-layer gradient 5e3 at epoch 1, train acc never above 52 % | divergence: the steps overshoot |
| Learning rate too low | Adam, lr 1e-6 | train acc 48 %, still improving at epoch 30, best epoch = last epoch | under-training; it still reaches 44 % val: the FFN is limited by its input, not by optimisation |
| No activation | ReLU removed | train 91 % at best epoch then 99.5 %, val 37 % | the network collapses to a linear model, which overfits the pixels |
| No input normalisation | raw [0, 1] pixels | **79 % of the ReLU units dead**, val accuracy still 42 % | silent failure: only the unit-health metric reveals it |
| Vanishing gradient | 8 sigmoid layers, SGD lr 0.05 | gradient 2e-6 in the first layer vs 0.8 in the last, train accuracy at chance for 30 epochs | the gradient shrinks at every sigmoid layer and never reaches the first ones |
| Control for the above | same depth with ReLU | first-layer gradient 1e-2 at start, train accuracy rising to 67 % by epoch 29 | changing only the activation lets the gradient flow |

## 3. Pretrained ResNet-18

ImageNet weights, last layer replaced by a 5-class linear layer, 128x128 input with ImageNet normalisation,
horizontal-flip augmentation, early stopping on validation loss (patience 4).

- **Feature extraction**: backbone frozen (BatchNorm in eval mode), only the head is trained (Adam lr 1e-3).
  Development split: best epoch 8, val 71.5 %, train 87.9 %.
- **Fine-tuning**: all layers trained (Adam lr 1e-4). Best epoch 5, val 77.1 %, **train 99.8 %**: it overfits
  within a few epochs, early stopping is essential.
- **Frozen features + logistic regression**: the 512 ResNet features of every image are computed once, then a
  scikit-learn logistic regression is trained on them in seconds.

Error analysis of the fine-tuned model (split 1, `results/resnet/confusion_log.txt`, test accuracy 75.2 %):

| Class | Recall |
|---|---|
| landscape | 88.5 % |
| still life | 85.0 % |
| cityscape | 80.0 % |
| religious painting | 61.9 % |
| portrait | 59.6 % |

Most errors are **portrait -> religious painting (15 images) and religious painting -> portrait (9)**: many
religious paintings are portraits of saints, so the two labels genuinely overlap. Then cityscape -> landscape (8).

A confidence threshold lets the guide abstain instead of guessing: with threshold 0.7 it answers on 76 % of the
paintings with 83.1 % accuracy; with 0.9, on 53 % with 89.1 %.

## 4. Final comparison

`src/final.py`: 5 artist-disjoint splits (seeds 1 to 5), test set used once per model and split.
The regularisation strength C of the scikit-learn models is chosen on validation. Gap = paired difference with
the fine-tuned ResNet on the same split. Latency is **end-to-end** (resize + features + model) for one image,
batch 1, 1 CPU thread, measured on the development laptop.

| Model | Test acc (%) | 95 % CI | Gap to fine-tuned ResNet (pts) | Size (MB) | Latency (ms/img) | Train (s) |
|---|---|---|---|---|---|---|
| logreg, pixels 32x32 | 42.1 ± 3.6 | [37.6, 46.5] | -35.0 [-39.3, -30.7] | 0.1 | 0.6 | 2 |
| random forest, pixels 32x32 | 39.3 ± 4.7 | [33.5, 45.2] | -37.7 [-42.0, -33.4] | 12.7 | 12.7 | 7 |
| logreg, hand-crafted features | 53.5 ± 4.0 | [48.4, 58.5] | -23.6 [-26.5, -20.7] | < 0.1 | 1.4 | < 1 |
| SVM (RBF), hand-crafted features | 52.2 ± 4.4 | [46.8, 57.6] | -24.9 [-27.9, -21.8] | 6.9 | 1.9 | 2 |
| feed-forward network, pixels 64x64 | 40.7 ± 4.1 | [35.6, 45.8] | -36.4 [-40.4, -32.4] | 6.3 | 0.5 | 4 |
| ResNet-18 features + logreg | 74.8 ± 2.9 | [71.3, 78.4] | -2.3 [-6.5, +2.0] | 44.7 | 12.5 | 26 |
| ResNet-18, frozen + trained head | 73.7 ± 2.1 | [71.0, 76.3] | -3.4 [-5.1, -1.7] | 44.7 | 15.2 | 252 |
| ResNet-18, fine-tuned | **77.1 ± 2.4** | [74.1, 80.1] | reference | 44.7 | 15.4 | 273 |

Full table with validation accuracy and parameter counts: `results/final/final_table.md`;
one row per model and split: `results/final/final_results.json`.

## Recommendation to the director

- **A pretrained neural network is worth it.** On paintings by artists it has never seen, a ResNet-18 is right
  about 3 times out of 4, against about 1 time out of 2 for the best classical pipeline (+24 points, an interval
  that stays far from zero). The cost is about 12-15 ms per photo on a single CPU thread and a 45 MB model:
  compatible with a kiosk.
- **No need to train a network from scratch, nor to fine-tune it.** A frozen ResNet-18 used as a feature
  extractor, followed by a logistic regression, is within the noise of the fine-tuned model, trains in about
  30 s on a laptop CPU, and can be retrained by a small team when the collection changes.
- **A feed-forward network on pixels is not worth it**: it does no better than a logistic regression.
- **Use a confidence threshold** (e.g. 0.7) so that the guide says "I am not sure" rather than giving a wrong
  story; and be aware that portraits and religious paintings will be the most frequent confusion.

## Limitations

- Latency was measured on a development laptop, not on a kiosk CPU. It must be re-measured on the target
  hardware (quantisation or a smaller network such as MobileNet are options if it is too slow).
- 1 500 images, 5 clean genres, 128 px, about 225 test images per split: each split's accuracy carries about
  ±6 points of sampling noise, which is why we report 5 splits and paired differences.
- Hyper-parameters of the FFN and the ResNet (learning rates, dropout, epochs) were chosen by the team on the
  earlier random split and were not re-tuned on the artist-disjoint split.
- `src/confusion.py` retrains the fine-tuned model for split 1; CPU non-determinism gives 75.2 % there vs
  76.5 % in `final.py` for the same split.
- Genre labels overlap (portrait / religious painting), which caps the accuracy any model can reach.
- Only image recognition is covered; answering questions in natural language is out of the scope of this deliverable.

## Team and git workflow

Team: Coline Renauld, Ariane Boulet, Téo Desquatrevaux.

All parts of the project (data, feed-forward network, diagnosis, ResNet, final comparison,
recommendation) were designed, run and discussed together in working sessions, with every
member contributing to each part. Code was mostly written on one machine during these sessions; the
experiments and their interpretation are a joint work.
