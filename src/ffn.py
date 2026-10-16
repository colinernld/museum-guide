import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import SEED, get_splits, out_path

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--depth", type=int, default=1, help="number of hidden layers")
p.add_argument("--activation", choices=["relu", "sigmoid", "none"], default="relu")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--size", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--wd", type=float, default=0.0, help="weight decay")
p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
               help="sgd (momentum 0.9) makes gradient problems visible: Adam rescales small gradients")
p.add_argument("--augment", action="store_true", help="random horizontal flip")
p.add_argument("--patience", type=int, default=0,
               help="early stopping on val loss (0 = off)")
p.add_argument("--seed", type=int, default=SEED)
p.add_argument("--no-activation", action="store_true",
               help="same as --activation none (diagnosis experiment)")
p.add_argument("--no-normalise", action="store_true",
               help="keep raw [0, 1] pixels instead of standardising (diagnosis experiment)")
p.add_argument("--overfit-batch", action="store_true",
               help="sanity check: train on ONE batch only; train accuracy must reach 100%%")
p.add_argument("--tag", default="ffn", help="name of the output files")
p.add_argument("--subdir", default="ffn", help="results/<subdir>/ receives the outputs")
a = p.parse_args()
if a.no_activation:
    a.activation = "none"

torch.manual_seed(a.seed)
np.random.seed(a.seed)

# ---- Data: resize, normalise with TRAIN statistics only ----
ds, tr, va, te = get_splits()


def to_array(img):
    return np.asarray(img.resize((a.size, a.size)), dtype=np.float32) / 255.0


X = torch.tensor(np.stack([to_array(ds[i][0]) for i in range(len(ds))]))
X = X.permute(0, 3, 1, 2).contiguous()          # (N, 3, S, S)
y = torch.tensor(ds.targets)
if not a.no_normalise:
    mean, std = X[tr].mean(), X[tr].std()
    X = (X - mean) / std
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
if a.overfit_batch:
    Xtr, ytr = Xtr[:a.batch], ytr[:a.batch]

# ---- Model: Flatten -> [Linear -> activation -> Dropout] x depth -> Linear ----
d_in, n_classes = X[0].numel(), len(ds.classes)
ACT = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "none": nn.Identity}[a.activation]
layers, width = [nn.Flatten()], d_in
for _ in range(a.depth):
    layers += [nn.Linear(width, a.hidden), ACT(), nn.Dropout(a.dropout)]
    width = a.hidden
layers.append(nn.Linear(width, n_classes))
model = nn.Sequential(*layers)
linears = [m for m in model if isinstance(m, nn.Linear)]
acts = [m for m in model if isinstance(m, ACT)] if a.activation != "none" else []
n_params = sum(p.numel() for p in model.parameters())
print(f"Architecture: {d_in}-" + "-".join([str(a.hidden)] * a.depth) + f"-{n_classes} "
      f"| parameters: {n_params:,} | activation: {a.activation} | dropout {a.dropout} "
      f"| wd {a.wd} | augment {a.augment} | normalise {not a.no_normalise} | {a.optimizer}"
      + (" | OVERFIT ONE BATCH" if a.overfit_batch else ""))

criterion = nn.CrossEntropyLoss()
if a.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)


def evaluate(Xs, ys):
    model.eval()
    with torch.no_grad():
        logits = model(Xs)
        return criterion(logits, ys).item(), (logits.argmax(1) == ys).float().mean().item()


def unit_health(Xs):
    """Per hidden layer: fraction of units that are dead (ReLU never > 0 on Xs)
    or saturated (sigmoid output < 0.01 or > 0.99 on more than 90 % of Xs)."""
    out = []
    hooks = [m.register_forward_hook(lambda m, i, o: out.append(o.detach())) for m in acts]
    model.eval()
    with torch.no_grad():
        model(Xs)
    for h in hooks:
        h.remove()
    if a.activation == "relu":
        return [(o <= 0).all(0).float().mean().item() for o in out]
    return [((o < 0.01) | (o > 0.99)).float().mean(0).gt(0.9).float().mean().item() for o in out]


# ---- Training loop (written by hand), with monitoring ----
hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
mon = {"grad_norm": [], "weight_norm": [], "update_ratio": [], "unit_health": []}
best = {"val_loss": float("inf"), "epoch": 0, "state": None}
bad_epochs = 0
for epoch in range(1, a.epochs + 1):
    model.train()
    w_before = [l.weight.detach().clone() for l in linears]
    grad_sum, n_steps = [0.0] * len(linears), 0
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
        for k, l in enumerate(linears):
            grad_sum[k] += l.weight.grad.norm().item()
        n_steps += 1
        optimizer.step()
    # monitoring: mean gradient norm, weight norm, relative size of this epoch's update
    mon["grad_norm"].append([g / n_steps for g in grad_sum])
    mon["weight_norm"].append([l.weight.norm().item() for l in linears])
    mon["update_ratio"].append([((l.weight.detach() - w0).norm() / w0.norm()).item()
                                for l, w0 in zip(linears, w_before)])
    mon["unit_health"].append(unit_health(Xva) if acts else [])
    tl, ta = evaluate(Xtr, ytr)
    vl, vacc = evaluate(Xva, yva)
    for k, v in zip(hist, [tl, vl, ta, vacc]):
        hist[k].append(v)
    g = mon["grad_norm"][-1]
    health = (f" | {'dead' if a.activation == 'relu' else 'saturated'} "
              + "/".join(f"{h:.0%}" for h in mon["unit_health"][-1])) if acts else ""
    print(f"epoch {epoch:3d} | train loss {tl:.3f} acc {ta:.3f} "
          f"| val loss {vl:.3f} acc {vacc:.3f} | grad first/last {g[0]:.2e}/{g[-1]:.2e}{health}")
    if not np.isfinite(tl):
        print("Loss is not finite: training diverged, stopping.")
        break
    if vl < best["val_loss"]:
        best = {"val_loss": vl, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if a.patience and bad_epochs >= a.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ---- Summary at the epoch with the lowest validation loss ----
if best["state"] is not None:
    model.load_state_dict(best["state"])
bl, bacc = evaluate(Xva, yva)
print(f"\nBEST epoch {best['epoch']} | val loss {bl:.3f} | val acc {bacc:.3f} "
      f"| train acc {hist['train_acc'][max(best['epoch'], 1) - 1]:.3f}")
if a.overfit_batch:
    ok = max(hist["train_acc"]) == 1.0
    print(f"Sanity check (one batch of {len(Xtr)}): max train acc {max(hist['train_acc']):.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")

json.dump({"args": vars(a), "params": n_params, "best_epoch": best["epoch"],
           "best_val_loss": bl, "best_val_acc": bacc, **hist, **mon},
          open(out_path(a.subdir, f"{a.tag}_history.json"), "w"), indent=1)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax = ax.ravel()
ax[0].plot(hist["train_loss"], label="train")
ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title("Loss")
ax[1].plot(hist["train_acc"], label="train")
ax[1].plot(hist["val_acc"], label="val")
ax[1].axhline(0.2, ls="--", c="gray", label="chance")
ax[1].set_title("Accuracy")
gn = np.array(mon["grad_norm"])
for k in range(gn.shape[1]):
    ax[2].plot(gn[:, k], label=f"Linear {k + 1}")
ax[2].set_yscale("log")
ax[2].set_title("Mean gradient norm per layer (log)")
ur = np.array(mon["update_ratio"])
for k in range(ur.shape[1]):
    ax[3].plot(ur[:, k], label=f"Linear {k + 1}")
ax[3].set_yscale("log")
ax[3].set_title("Relative weight update per epoch  ||dW|| / ||W|| (log)")
for x in ax:
    if best["epoch"]:
        x.axvline(best["epoch"] - 1, ls=":", c="k")
    x.set_xlabel("epoch")
    x.legend(fontsize=7)
plt.tight_layout()
plt.savefig(out_path(a.subdir, f"{a.tag}_curves.png"), dpi=120)
print(f"Saved results/{a.subdir}/{a.tag}_curves.png and {a.tag}_history.json")
