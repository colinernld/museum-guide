import argparse
import copy
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
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--wd", type=float, default=0.0, help="weight decay")
p.add_argument("--augment", action="store_true", help="random horizontal flip")
p.add_argument("--patience", type=int, default=0,
               help="early stopping on val loss (0 = off)")
p.add_argument("--seed", type=int, default=SEED)
p.add_argument("--no-activation", action="store_true",
               help="remove the ReLU (diagnosis experiment)")
p.add_argument("--tag", default="ffn", help="name of the output files")
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)

# ---- Data: resize, normalise with TRAIN statistics only ----
ds, tr, va, te = get_splits()


def to_array(img):
    return np.asarray(img.resize((a.size, a.size)), dtype=np.float32) / 255.0


X = torch.tensor(np.stack([to_array(ds[i][0]) for i in range(len(ds))]))
X = X.permute(0, 3, 1, 2).contiguous()          # (N, 3, S, S)
y = torch.tensor(ds.targets)
mean, std = X[tr].mean(), X[tr].std()
X = (X - mean) / std
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

# ---- Model: Flatten -> Linear -> ReLU -> Dropout -> Linear ----
d_in, n_classes = X[0].numel(), len(ds.classes)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(d_in, a.hidden),
    nn.Identity() if a.no_activation else nn.ReLU(),
    nn.Dropout(a.dropout),
    nn.Linear(a.hidden, n_classes),
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {d_in}-{a.hidden}-{n_classes} | parameters: {n_params:,} "
      f"| activation: {'none' if a.no_activation else 'ReLU'} "
      f"| dropout {a.dropout} | wd {a.wd} | augment {a.augment}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)


def evaluate(Xs, ys):
    model.eval()
    with torch.no_grad():
        logits = model(Xs)
        return criterion(logits, ys).item(), (logits.argmax(1) == ys).float().mean().item()


# ---- Training loop (written by hand) ----
hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
best = {"val_loss": float("inf"), "epoch": 0, "state": None}
bad_epochs = 0
for epoch in range(1, a.epochs + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), a.batch):
        b = perm[i:i + a.batch]
        xb = Xtr[b]
        if a.augment:
            flip = torch.rand(len(b)) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        optimizer.zero_grad()
        loss = criterion(model(xb), ytr[b])
        loss.backward()
        optimizer.step()
    tl, ta = evaluate(Xtr, ytr)
    vl, vacc = evaluate(Xva, yva)
    for k, v in zip(hist, [tl, vl, ta, vacc]):
        hist[k].append(v)
    print(f"epoch {epoch:3d} | train loss {tl:.3f} acc {ta:.3f} "
          f"| val loss {vl:.3f} acc {vacc:.3f}")
    if vl < best["val_loss"]:
        best = {"val_loss": vl, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if a.patience and bad_epochs >= a.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ---- Summary at the epoch with the lowest validation loss ----
model.load_state_dict(best["state"])
bl, bacc = evaluate(Xva, yva)
print(f"\nBEST epoch {best['epoch']} | val loss {bl:.3f} | val acc {bacc:.3f} "
      f"| train acc {hist['train_acc'][best['epoch'] - 1]:.3f}")

json.dump({"args": vars(a), "params": n_params, "best_epoch": best["epoch"],
           "best_val_loss": bl, "best_val_acc": bacc, **hist},
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
    x.axvline(best["epoch"] - 1, ls=":", c="k")
    x.set_xlabel("epoch")
    x.legend()
plt.tight_layout()
plt.savefig(f"results/{a.tag}_curves.png", dpi=120)
print(f"Saved results/{a.tag}_curves.png and results/{a.tag}_history.json")
