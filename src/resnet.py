import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

from data import SEED, get_splits

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["extract", "finetune"], required=True)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--size", type=int, default=128)
p.add_argument("--patience", type=int, default=4)
p.add_argument("--seed", type=int, default=SEED)
p.add_argument("--tag", default=None)
a = p.parse_args()
lr = a.lr or (1e-3 if a.mode == "extract" else 1e-4)
tag = a.tag or f"resnet_{a.mode}"
torch.manual_seed(a.seed)
np.random.seed(a.seed)

# ---- Data: ImageNet normalisation (the network was pretrained with it) ----
ds, tr, va, te = get_splits()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_array(img):
    img = img.convert("RGB").resize((a.size, a.size))
    return np.asarray(img, dtype=np.float32) / 255.0


X = torch.tensor(np.stack([to_array(ds[i][0]) for i in range(len(ds))]))
X = (X.permute(0, 3, 1, 2).contiguous() - mean) / std
y = torch.tensor(ds.targets)
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

# ---- Model: pretrained ResNet-18, new last layer for our 5 classes ----
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
if a.mode == "extract":
    for name, w in model.named_parameters():
        w.requires_grad = name.startswith("fc.")
n_total = sum(w.numel() for w in model.parameters())
n_train = sum(w.numel() for w in model.parameters() if w.requires_grad)
print(f"ResNet-18 | mode {a.mode} | lr {lr} | params {n_total:,} "
      f"| trainable {n_train:,} | input {a.size}x{a.size}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([w for w in model.parameters() if w.requires_grad], lr=lr)


def evaluate(Xs, ys):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(Xs), 64):
            out = model(Xs[i:i + 64])
            loss_sum += criterion(out, ys[i:i + 64]).item() * len(out)
            correct += (out.argmax(1) == ys[i:i + 64]).sum().item()
    return loss_sum / len(Xs), correct / len(Xs)


# ---- Training loop (written by hand) ----
hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
best = {"val_loss": float("inf"), "epoch": 0, "state": None}
bad = 0
t_start = time.perf_counter()
for epoch in range(1, a.epochs + 1):
    # frozen backbone: keep BatchNorm statistics fixed
    model.eval() if a.mode == "extract" else model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), a.batch):
        b = perm[i:i + a.batch]
        xb = Xtr[b]
        flip = torch.rand(len(b)) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        optimizer.zero_grad()
        criterion(model(xb), ytr[b]).backward()
        optimizer.step()
    tl, ta = evaluate(Xtr, ytr)
    vl, vacc = evaluate(Xva, yva)
    for k, v in zip(hist, [tl, vl, ta, vacc]):
        hist[k].append(v)
    print(f"epoch {epoch:3d} | train loss {tl:.3f} acc {ta:.3f} "
          f"| val loss {vl:.3f} acc {vacc:.3f}")
    if vl < best["val_loss"]:
        best = {"val_loss": vl, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        bad = 0
    else:
        bad += 1
        if bad >= a.patience:
            print(f"Early stopping at epoch {epoch}")
            break
train_s = time.perf_counter() - t_start

# ---- Best epoch, then CPU benchmark ----
model.load_state_dict(best["state"])
bl, bacc = evaluate(Xva, yva)
model.eval()
with torch.no_grad():
    x1 = Xva[:1]
    for _ in range(5):
        model(x1)
    t0 = time.perf_counter()
    for _ in range(50):
        model(x1)
    lat1 = (time.perf_counter() - t0) / 50 * 1000
    xb = Xva[:32]
    model(xb)
    t0 = time.perf_counter()
    for _ in range(5):
        model(xb)
    lat_b = (time.perf_counter() - t0) / (5 * 32) * 1000
size_mb = n_total * 4 / 1e6

print(f"\nBEST epoch {best['epoch']} | val loss {bl:.3f} | val acc {bacc:.3f} "
      f"| train acc {hist['train_acc'][best['epoch'] - 1]:.3f}")
print(f"train time {train_s:.0f}s | latency {lat1:.1f} ms/img (batch 1), "
      f"{lat_b:.1f} ms/img (batch 32) | size {size_mb:.1f} MB")

json.dump({"args": vars(a), "params": n_total, "trainable": n_train,
           "best_epoch": best["epoch"], "best_val_loss": bl, "best_val_acc": bacc,
           "train_seconds": train_s, "latency_ms_batch1": lat1,
           "latency_ms_batch32": lat_b, "size_mb": size_mb, **hist},
          open(f"results/{tag}_history.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["train_loss"], label="train")
ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title("Loss")
ax[1].plot(hist["train_acc"], label="train")
ax[1].plot(hist["val_acc"], label="val")
ax[1].axhline(0.2, ls="--", c="gray", label="chance")
ax[1].set_title("Accuracy")
for x in ax:
    x.axvline(best["epoch"] - 1, ls=":", c="k")
    x.set_xlabel("epoch")
    x.legend()
plt.tight_layout()
plt.savefig(f"results/{tag}_curves.png", dpi=120)
