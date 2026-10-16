import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import SEED, get_splits

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--size", type=int, default=64)
p.add_argument("--no-activation", action="store_true",
               help="remove the ReLU (diagnosis experiment)")
p.add_argument("--tag", default="ffn", help="name of the output files")
a = p.parse_args()

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Data: resize, flatten, normalise with TRAIN statistics only ----
ds, tr, va, te = get_splits()


def to_array(img):
    return np.asarray(img.resize((a.size, a.size)), dtype=np.float32) / 255.0


X = torch.tensor(np.stack([to_array(ds[i][0]) for i in range(len(ds))]))
X = X.reshape(len(ds), -1)                     # (1500, size*size*3)
y = torch.tensor(ds.targets)
mean, std = X[tr].mean(), X[tr].std()
X = (X - mean) / std
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

# ---- Model: Linear -> ReLU -> Linear ----
d_in, n_classes = X.shape[1], len(ds.classes)
model = nn.Sequential(
    nn.Linear(d_in, a.hidden),
    nn.Identity() if a.no_activation else nn.ReLU(),
    nn.Linear(a.hidden, n_classes),
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {d_in}-{a.hidden}-{n_classes} | parameters: {n_params:,} "
      f"| activation: {'none' if a.no_activation else 'ReLU'}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)


def evaluate(Xs, ys):
    model.eval()
    with torch.no_grad():
        logits = model(Xs)
        return criterion(logits, ys).item(), (logits.argmax(1) == ys).float().mean().item()


# ---- Training loop (written by hand) ----
hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
for epoch in range(1, a.epochs + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), a.batch):
        b = perm[i:i + a.batch]
        optimizer.zero_grad()
        loss = criterion(model(Xtr[b]), ytr[b])
        loss.backward()
        optimizer.step()
    tl, ta = evaluate(Xtr, ytr)
    vl, vacc = evaluate(Xva, yva)
    for k, v in zip(hist, [tl, vl, ta, vacc]):
        hist[k].append(v)
    print(f"epoch {epoch:3d} | train loss {tl:.3f} acc {ta:.3f} "
          f"| val loss {vl:.3f} acc {vacc:.3f}")

# ---- Save curves and numbers ----
json.dump({"args": vars(a), "params": n_params, **hist},
          open(f"results/{a.tag}_history.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["train_loss"], label="train")
ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title("Loss")
ax[1].plot(hist["train_acc"], label="train")
ax[1].plot(hist["val_acc"], label="val")
ax[1].axhline(0.2, ls="--", c="gray", label="chance")
ax[1].set_title("Accuracy")
for x in ax:
    x.set_xlabel("epoch")
    x.legend()
plt.tight_layout()
plt.savefig(f"results/{a.tag}_curves.png", dpi=120)
print(f"Saved results/{a.tag}_curves.png and results/{a.tag}_history.json")
