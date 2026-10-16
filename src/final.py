"""Final comparison of every model on the test set, repeated over several artist-disjoint splits.

For each seed s: split = get_splits(seed=s) (different painters in test each time) and weight
init / shuffling seeded with s. Hyper-parameters of sklearn models (C) are chosen on validation.
The test set of each split is used once per model.

Reported: mean and std over runs, 95 % confidence interval of the mean (Student t), and the paired
difference with the fine-tuned ResNet (same split for both models, so split difficulty cancels out).
Latency is END-TO-END for one 128x128 image as stored on disk (resize + features/normalisation +
model), batch of 1, 1 CPU thread: what a kiosk would pay per photo.
"""
import argparse
import copy
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import get_splits, out_path
from features import handcrafted, pixels

ALL = ["logreg_pixels", "random_forest_pixels", "logreg_handcrafted", "svm_handcrafted", "ffn",
       "resnet_features_logreg", "resnet_extract", "resnet_finetune"]
REF = "resnet_finetune"

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--only", nargs="+", default=None, choices=ALL)
p.add_argument("--table-only", action="store_true",
               help="only rebuild results/final/final_table.md from final_results.json")
a = p.parse_args()
WANTED = a.only or ALL


# ---------------- aggregate ----------------
def ci(v):
    """mean, std and half-width of the 95 % confidence interval of the mean (Student t)."""
    v = np.asarray(v) * 100
    if len(v) < 2:
        return v.mean(), 0.0, float("nan")
    s = v.std(ddof=1)
    return v.mean(), s, stats.t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))


def write_table(rows):
    seeds = sorted({r["seed"] for r in rows})
    names = [n for n in ALL if any(r["model"] == n for r in rows)]
    by = {n: sorted([r for r in rows if r["model"] == n], key=lambda r: r["seed"]) for n in names}
    ref = {r["seed"]: r["test_acc"] for r in by.get(REF, [])}
    n_test = [len(get_splits(seed=s)[3]) for s in seeds]
    lines = [f"Artist-disjoint splits, seeds {seeds} (one split per seed, {min(n_test)}-{max(n_test)} "
             f"test images each). Mean ± std over runs; [95 % CI of the mean].",
             "Latency: end-to-end (resize + features + model), batch 1, 1 CPU thread.", "",
             "| Model | Val acc (%) | Test acc (%) | 95 % CI | Gap to fine-tuned ResNet (pts, paired) "
             "| Params | Size (MB) | Latency (ms/img) | Train (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        r = by[name]
        vm, vs, _ = ci([x["val_acc"] for x in r])
        tm, ts, th = ci([x["test_acc"] for x in r])
        if ref and name != REF:
            dm, _, dh = ci([x["test_acc"] - ref[x["seed"]] for x in r])
            gap = f"{dm:+.1f} [{dm - dh:+.1f}, {dm + dh:+.1f}]"
        else:
            gap = "reference" if name == REF else "-"
        params = f"{r[0]['params']:,}" + (" nodes" if r[0]["params_kind"] == "tree nodes" else "")
        if name == "resnet_features_logreg":
            params += " + 11.2 M frozen"
        lines.append(f"| {name} | {vm:.1f} ± {vs:.1f} | {tm:.1f} ± {ts:.1f} | [{tm - th:.1f}, {tm + th:.1f}] "
                     f"| {gap} | {params} | {np.mean([x['size_mb'] for x in r]):.1f} "
                     f"| {np.mean([x['latency_ms'] for x in r]):.2f} "
                     f"| {np.mean([x['train_s'] for x in r]):.0f} |")
    open(out_path("final", "final_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if a.table_only:      # rebuild the table from saved results, no training
    write_table(json.load(open(out_path("final", "final_results.json"))))
    raise SystemExit

ds, *_ = get_splits()
y = np.array(ds.targets)
yt = torch.tensor(y)
n_classes = len(ds.classes)
images = [ds[i][0].convert("RGB") for i in range(len(ds))]
crit = nn.CrossEntropyLoss()

# ---------------- preprocessing (image -> model input), shared by training and latency ----------------
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(img, size):
    arr = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)


def prep_resnet(img):
    return (to_tensor(img, 128) - IMNET_MEAN) / IMNET_STD


X_pix = np.stack([pixels(im) for im in images])
X_hand = np.stack([handcrafted(im) for im in images])
X_rn = torch.cat([prep_resnet(im) for im in images])
X64 = torch.cat([to_tensor(im, 64) for im in images])

backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()
backbone.eval()
t0 = time.perf_counter()
with torch.no_grad():
    X_rnfeat = torch.cat([backbone(X_rn[i:i + 100]) for i in range(0, len(X_rn), 100)]).numpy()
BACKBONE_S = time.perf_counter() - t0      # one pass over all images, added to its training time


# ---------------- helpers ----------------
def evaluate(model, X, Y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), 64):
            correct += (model(X[i:i + 64]).argmax(1) == Y[i:i + 64]).sum().item()
    return correct / len(X)


def fit_torch(model, X, tr, va, lr, epochs, patience, wd=0.0, batch=64, eval_mode=False):
    """Hand-written loop, horizontal-flip augmentation, early stopping on validation loss."""
    Xtr, ytr, Xva, yva = X[tr], yt[tr], X[va], yt[va]
    opt = torch.optim.Adam([w for w in model.parameters() if w.requires_grad], lr=lr,
                           weight_decay=wd)
    best_loss, best_state, bad = float("inf"), None, 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.eval() if eval_mode else model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            flip = torch.rand(len(b)) < 0.5
            xb = torch.where(flip[:, None, None, None], Xtr[b].flip(3), Xtr[b])
            opt.zero_grad()
            crit(model(xb), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(Xva[i:i + 64]), yva[i:i + 64]).item() * len(Xva[i:i + 64])
                     for i in range(0, len(Xva), 64)) / len(Xva)
        if vl < best_loss:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return time.perf_counter() - t0


def tune_sklearn(make, grid, X, tr, va):
    """Fit one model per value of C, keep the best on validation, report its total fit time."""
    best, t0 = None, time.perf_counter()
    for C in grid:
        m = make(C).fit(X[tr], y[tr])
        acc = m.score(X[va], y[va])
        if best is None or acc > best[0]:
            best = (acc, C, m)
    return best[2], best[1], time.perf_counter() - t0


def latency_ms(predict_one, test_idx, n=50):
    """End-to-end ms per image, batch 1, one CPU thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    imgs = [images[i] for i in test_idx[:n]]
    with torch.no_grad():
        for im in imgs[:5]:
            predict_one(im)
        t0 = time.perf_counter()
        for im in imgs:
            predict_one(im)
    torch.set_num_threads(threads)
    return (time.perf_counter() - t0) / len(imgs) * 1000


def n_params_sklearn(model):
    last = model[-1] if hasattr(model, "steps") else model
    if hasattr(last, "coef_"):
        return int(last.coef_.size + last.intercept_.size)
    if hasattr(last, "support_vectors_"):
        return int(last.support_vectors_.size + last.dual_coef_.size)
    return int(sum(t.tree_.node_count for t in last.estimators_))   # tree nodes, not weights


def row(name, seed, test_acc, val_acc, n_params, params_kind, size_mb, lat, train_s, **extra):
    return dict(model=name, seed=seed, val_acc=val_acc, test_acc=test_acc, params=n_params,
                params_kind=params_kind, size_mb=size_mb, latency_ms=lat, train_s=train_s, **extra)


# ---------------- runs ----------------
SK = {  # name: (features, extractor for one image, model factory, grid of C)
    "logreg_pixels": (X_pix, pixels, lambda C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000)), [0.001, 0.01, 0.1]),
    "random_forest_pixels": (X_pix, pixels, lambda C: RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=1), [None]),
    "logreg_handcrafted": (X_hand, handcrafted, lambda C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000)), [0.001, 0.01, 0.1]),
    "svm_handcrafted": (X_hand, handcrafted, lambda C: make_pipeline(
        StandardScaler(), SVC(C=C)), [0.3, 1, 3, 10]),
    "resnet_features_logreg": (X_rnfeat, None, lambda C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000)), [0.001, 0.01, 0.1]),
}

rows = []
for seed in a.seeds:
    _, tr, va, te = get_splits(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n=== seed {seed}: train {len(tr)}, val {len(va)}, test {len(te)} ===")
    for name in [n for n in WANTED if n in SK]:
        X, extract, make, grid = SK[name]
        model, C, train_s = tune_sklearn(make, grid, X, tr, va)
        if extract is None:      # ResNet features: the backbone runs at inference too
            def one(im, m=model):
                with torch.no_grad():
                    return m.predict(backbone(prep_resnet(im)).numpy())
            size = len(pickle.dumps(model)) / 1e6 + 44.7      # + frozen ResNet-18 backbone
            train_s += BACKBONE_S
        else:
            def one(im, m=model, f=extract):
                return m.predict(f(im)[None])
            size = len(pickle.dumps(model)) / 1e6
        kind = "tree nodes" if "forest" in name else "weights"
        rows.append(row(name, seed, model.score(X[te], y[te]), model.score(X[va], y[va]),
                        n_params_sklearn(model), kind, size, latency_ms(one, te), train_s, C=C))
    if "ffn" in WANTED:
        mean, std = X64[tr].mean(), X64[tr].std()          # train statistics only
        Xf = (X64 - mean) / std
        m = nn.Sequential(nn.Flatten(), nn.Linear(Xf[0].numel(), 128), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(128, n_classes))
        t = fit_torch(m, Xf, tr, va, lr=1e-3, epochs=80, patience=8, wd=1e-3)
        def one(im, m=m, mu=mean, sd=std):
            return m((to_tensor(im, 64) - mu) / sd).argmax(1)
        n = sum(w.numel() for w in m.parameters())
        rows.append(row("ffn", seed, evaluate(m, Xf[te], yt[te]), evaluate(m, Xf[va], yt[va]),
                        n, "weights", n * 4 / 1e6, latency_ms(one, te), t))
    for mode in ["extract", "finetune"]:
        name = f"resnet_{mode}"
        if name not in WANTED:
            continue
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        if mode == "extract":
            for n_, w in m.named_parameters():
                w.requires_grad = n_.startswith("fc.")
        t = fit_torch(m, X_rn, tr, va, lr=1e-3 if mode == "extract" else 1e-4, epochs=15,
                      patience=4, batch=32, eval_mode=(mode == "extract"))
        def one(im, m=m):
            return m(prep_resnet(im)).argmax(1)
        n = sum(w.numel() for w in m.parameters())
        rows.append(row(name, seed, evaluate(m, X_rn[te], yt[te]), evaluate(m, X_rn[va], yt[va]),
                        n, "weights", n * 4 / 1e6, latency_ms(one, te), t))
    for r in [r for r in rows if r["seed"] == seed]:
        print(f"seed {seed} | {r['model']:22s} val {r['val_acc']:.3f} test {r['test_acc']:.3f} "
              f"| {r['latency_ms']:.2f} ms | train {r['train_s']:.0f}s")
    json.dump(rows, open(out_path("final", "final_results.json"), "w"), indent=1)

write_table(rows)
