import argparse
import copy
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import get_splits

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[42])
p.add_argument("--only", nargs="+", default=None,
               help="subset of: logreg random_forest ffn resnet_extract resnet_finetune")
a = p.parse_args()
WANTED = a.only or ["logreg", "random_forest", "ffn", "resnet_extract", "resnet_finetune"]

ds, tr, va, te = get_splits()          # split seed is fixed (42): same images for every run
y = np.array(ds.targets)
yt = torch.tensor(y)
n_classes = len(ds.classes)
crit = nn.CrossEntropyLoss()


def load(size):
    return np.stack([
        np.asarray(ds[i][0].convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        for i in range(len(ds))])


def to_tensor(raw):
    return torch.tensor(raw).permute(0, 3, 1, 2).contiguous()


def split(X):
    return (X[tr], yt[tr]), (X[va], yt[va]), (X[te], yt[te])


# ---------------- torch helpers ----------------
def evaluate(model, X, Y):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), 64):
            out = model(X[i:i + 64])
            loss += crit(out, Y[i:i + 64]).item() * len(out)
            correct += (out.argmax(1) == Y[i:i + 64]).sum().item()
    return loss / len(X), correct / len(X)


def fit_torch(model, data, lr, epochs, patience, augment, wd=0.0, batch=64, eval_mode=False):
    (Xtr, ytr), (Xva, yva), _ = data
    params = [w for w in model.parameters() if w.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    best_loss, best_state, bad = float("inf"), None, 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.eval() if eval_mode else model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            xb = Xtr[b]
            if augment:
                flip = torch.rand(len(b)) < 0.5
                xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            opt.zero_grad()
            crit(model(xb), ytr[b]).backward()
            opt.step()
        vl, _ = evaluate(model, Xva, yva)
        if vl < best_loss:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return train_s


def latency_torch(model, x1):
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(100):
            model(x1)
        ms = (time.perf_counter() - t0) / 100 * 1000
    torch.set_num_threads(n)
    return ms


def row_torch(name, seed, model, data, train_s):
    (_, _), (Xva, yva), (Xte, yte) = data
    _, val_acc = evaluate(model, Xva, yva)
    _, test_acc = evaluate(model, Xte, yte)          # test set: used once per model
    n_params = sum(w.numel() for w in model.parameters())
    return dict(model=name, seed=seed, val_acc=val_acc, test_acc=test_acc,
                params=n_params, size_mb=n_params * 4 / 1e6,
                latency_ms=latency_torch(model, Xte[:1]), train_s=train_s)


def row_sklearn(name, seed, model, Xc):
    t0 = time.perf_counter()
    model.fit(Xc[tr], y[tr])
    train_s = time.perf_counter() - t0
    val_acc = model.score(Xc[va], y[va])
    test_acc = model.score(Xc[te], y[te])             # test set: used once per model
    x1 = Xc[te][:1]
    model.predict(x1)
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict(x1)
    ms = (time.perf_counter() - t0) / 200 * 1000
    last = model[-1] if hasattr(model, "steps") else model
    if hasattr(last, "coef_"):
        n_params = last.coef_.size + last.intercept_.size
    else:
        n_params = sum(t.tree_.node_count for t in last.estimators_)   # tree nodes
    return dict(model=name, seed=seed, val_acc=val_acc, test_acc=test_acc,
                params=int(n_params), size_mb=len(pickle.dumps(model)) / 1e6,
                latency_ms=ms, train_s=train_s)


# ---------------- data ----------------
need_sk = any(m in WANTED for m in ["logreg", "random_forest"])
need_ffn = "ffn" in WANTED
need_rn = any(m.startswith("resnet") for m in WANTED)
if need_sk:
    Xc = load(32).reshape(len(ds), -1)
if need_ffn:
    X64 = to_tensor(load(64))
    X64 = (X64 - X64[tr].mean()) / X64[tr].std()      # train statistics only
    data_ffn = split(X64)
if need_rn:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    data_rn = split((to_tensor(load(128)) - mean) / std)

# ---------------- runs ----------------
rows = []
for seed in a.seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "logreg" in WANTED:
        rows.append(row_sklearn("logreg", seed, make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.01)), Xc))
    if "random_forest" in WANTED:
        rows.append(row_sklearn("random_forest", seed, RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=1), Xc))
    if "ffn" in WANTED:
        m = nn.Sequential(nn.Flatten(), nn.Linear(X64[0].numel(), 128), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(128, n_classes))
        t = fit_torch(m, data_ffn, lr=1e-3, epochs=80, patience=8, augment=True, wd=1e-3)
        rows.append(row_torch("ffn", seed, m, data_ffn, t))
    for mode in ["extract", "finetune"]:
        name = f"resnet_{mode}"
        if name not in WANTED:
            continue
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        if mode == "extract":
            for n_, w in m.named_parameters():
                w.requires_grad = n_.startswith("fc.")
        t = fit_torch(m, data_rn, lr=1e-3 if mode == "extract" else 1e-4, epochs=15,
                      patience=4, augment=True, batch=32, eval_mode=(mode == "extract"))
        rows.append(row_torch(name, seed, m, data_rn, t))
    for r in rows[-5:]:
        if r["seed"] == seed:
            print(f"seed {seed} | {r['model']:15s} val {r['val_acc']:.3f} "
                  f"test {r['test_acc']:.3f} | {r['latency_ms']:.2f} ms")

# ---------------- aggregate ----------------
json.dump(rows, open("results/final_results.json", "w"), indent=1)


def ms(v):
    v = np.array(v)
    return f"{100 * v.mean():.1f} ± {100 * (v.std(ddof=1) if len(v) > 1 else 0):.1f}"


lines = [f"Test set: {len(te)} images, seeds {a.seeds}. Latency: batch 1, 1 CPU thread.", "",
         "| Model | Val acc (%) | Test acc (%) | Params | Size (MB) | Latency (ms/img) | Train (s) |",
         "|---|---|---|---|---|---|---|"]
for name in WANTED:
    r = [x for x in rows if x["model"] == name]
    lines.append(f"| {name} | {ms([x['val_acc'] for x in r])} | {ms([x['test_acc'] for x in r])} "
                 f"| {r[0]['params']:,} | {np.mean([x['size_mb'] for x in r]):.1f} "
                 f"| {np.mean([x['latency_ms'] for x in r]):.2f} "
                 f"| {np.mean([x['train_s'] for x in r]):.0f} |")
open("results/final_table.md", "w").write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))
