# Museum guide: artwork genre classification

Neural-network components of an intelligent museum guide (fictional French fine-arts museum).
A visitor points a phone or kiosk camera at a painting; the guide recognises its **genre**
(portrait, landscape, cityscape, still life, religious painting) so it can tell the story of that tradition.

We build, diagnose and compare:

1. a **feed-forward network** (PyTorch, hand-written training loop) on raw pixels;
2. **training monitoring and failure diagnosis** (learning rate too high / too low, missing non-linearity, overfitting);
3. a **pretrained ResNet-18** (feature extraction and fine-tuning);
4. **classical baselines** (logistic regression, random forest) to answer the director's question:
   *is a neural network worth it on modest kiosk CPUs?*

## Data

WikiArt subset (`huggan/wikiart`, pinned revision), 5 genres x 300 images = 1 500 images, 128x128 px (centre crop).
Stratified split 70 / 15 / 15 (seed 42): 1 050 train, 225 validation, 225 test. Chance level = 20 %.

Images are not committed (WikiArt licence unknown). Rebuild them with:

    pip install -r requirements.txt
    python make_wikiart_subset.py --label genre \
        --classes portrait landscape cityscape still_life religious_painting \
        --per-class 300 --size 128 --out data/wikiart_genre5 --retries 10

The exact command, dataset revision and shards used are in `results/dataset_info.json`.

## Repository layout

    make_wikiart_subset.py   data download script (provided)
    src/data.py              dataset loading and stratified split
    src/explore.py           class counts, image sizes, sample grid
    src/baseline.py          logistic regression and random forest (first look, validation only)
    src/ffn.py               feed-forward network, hand-written training loop, CLI flags for every experiment
    src/resnet.py            ResNet-18: --mode extract | finetune
    src/final.py             final comparison of all models, several seeds, test set, CPU benchmark
    src/confusion.py         confusion matrix and confidence-threshold analysis (fine-tuned ResNet, test set)
    results/                 curves (*_curves.png), histories (*_history.json), final table and log

Run every script from the repository root, e.g. `python src/ffn.py --dropout 0.5 --wd 1e-3 --augment --patience 8 --tag ffn_all`.

## Reproduce the final results

    python src/final.py --seeds 1 2 3     # writes results/final_table.md, final_results.json
    python src/confusion.py               # writes results/confusion_resnet_finetune.png

## Results

### Final comparison (test set, 225 images, 3 seeds, mean ± std)

Latency: batch of 1, 1 CPU thread, model only (image resizing not included).

| Model | Val acc (%) | Test acc (%) | Size (MB) | Latency (ms/img) | Train (s) |
|---|---|---|---|---|---|
| Logistic regression (32x32 pixels) | 37.8 | 34.7 | 0.2 | 0.16 | 1 |
| Random forest (32x32 pixels) | 44.4 ± 1.2 | 39.9 ± 1.1 | 12.8 | 6.81 | 7 |
| Feed-forward network (64x64 pixels) | 42.2 ± 2.8 | 38.4 ± 4.9 | 6.3 | 0.06 | 2 |
| ResNet-18, features extracted (128 px) | 75.6 ± 1.2 | 73.3 ± 1.2 | 44.7 | 9.27 | 110 |
| ResNet-18, fine-tuned (128 px) | 78.7 ± 1.9 | 76.4 ± 2.2 | 44.7 | 9.01 | 161 |

Logistic regression is deterministic, so its spread across seeds is zero and says nothing about uncertainty.
With 225 test images, one accuracy figure carries roughly ±6 points of sampling uncertainty.

### Feed-forward network

Flatten -> Linear(12 288, 128) -> ReLU -> Dropout -> Linear(128, 5), 1.57 M parameters, Adam.
Best epoch chosen on validation loss (seed 42, single runs):

| Run | Best epoch | Val acc | Train acc at best epoch |
|---|---|---|---|
| base | 5 | 38.2 % | 82.1 % |
| + augmentation (horizontal flip) | 9 | 39.6 % | 81.8 % |
| + dropout 0.5, weight decay 1e-3 | 8 | 39.1 % | 76.3 % |
| all combined + early stopping | 9 | 43.1 % | 63.8 % |

Regularisation shrinks the train/val gap but the ceiling stays low: a dense network on raw pixels
has no notion of local structure.

### Failure diagnosis

| Experiment | Setting | Observation |
|---|---|---|
| Learning rate too high | lr = 0.5 | Validation loss explodes (best val loss 8 133), accuracy near chance |
| Learning rate too low | lr = 1e-6 | Slow, still improving at epoch 30 (train 50 %, val 41 %): under-trained |
| No activation | ReLU removed (linear model) | Train 90 % at best epoch, then ~98 %, but val stays at 30-37 %: the model memorises the training set |

See `results/fail_*_curves.png`.

### ResNet-18 (ImageNet weights, new 5-class head)

- **Extraction**: only the last layer is trained, BatchNorm kept in eval mode.
- **Fine-tuning**: all layers trained, lr 1e-4. Reaches 99.6 % train accuracy at its best epoch (4) for 79.6 % validation: it overfits quickly.
- Errors of the fine-tuned model (seed 1): mostly **portrait <-> religious painting** (10 + 7 images out of 90), a pair whose labels overlap in art history. See `results/confusion_resnet_finetune.png`.

## Recommendation to the director

On this data, a pretrained CNN is worth it: about **+35 points** of accuracy over the best classical
baseline (76 % vs 40 %), at **about 9 ms per image on one CPU thread** and 45 MB. That is fast enough for
a kiosk, and training takes minutes on a laptop CPU, so the budget and team-size constraints are met.
Training a feed-forward network from scratch on pixels is **not** worth it: it does not beat a random forest
and its results vary a lot with the seed.

Caveats we are aware of:

- Latency was measured on a development laptop, not on a kiosk CPU, and excludes image pre-processing.
- The classical baselines use raw 32x32 pixels. Hand-crafted features (colour histograms, HOG) were not tried,
  so the gap is an upper bound on what a *tuned* classical model would lose.
- The split is random: a painter may appear in train and test (the data script caps images per artist but does not separate them).
  Accuracy on unseen artists may be lower.
- 5 genres, 1 500 images, 128 px: a small, clean setting; the real collection will be harder.
- The confidence threshold analysis (`src/confusion.py`) shows how the guide can abstain on uncertain images.

## Team and git workflow

<!-- TODO: fill in who did what (one line per member) -->

## Notes

- All numbers in this README come from `results/`; test accuracy is computed once per model at the end.
- Dependencies are listed in `requirements.txt` (not pinned).
