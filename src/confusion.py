import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix

from data import get_splits, out_path

SEED = 1                                  # same seed as the first run of final.py
torch.manual_seed(SEED)
np.random.seed(SEED)

ds, tr, va, te = get_splits(seed=SEED)      # same split as the seed-1 run of final.py
y = torch.tensor(ds.targets)
classes = ds.classes

raw = np.stack([np.asarray(ds[i][0].convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
                for i in range(len(ds))])
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
X = (torch.tensor(raw).permute(0, 3, 1, 2).contiguous() - mean) / std
Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


def logits_of(Xs):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(Xs[i:i + 64]) for i in range(0, len(Xs), 64)])


best_loss, best_state, bad = float("inf"), None, 0
for epoch in range(1, 16):                # same recipe as final.py (finetune)
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 32):
        b = perm[i:i + 32]
        xb = Xtr[b]
        flip = torch.rand(len(b)) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        opt.zero_grad()
        crit(model(xb), ytr[b]).backward()
        opt.step()
    vl = crit(logits_of(Xva), yva).item()
    print(f"epoch {epoch:2d} | val loss {vl:.3f}")
    if vl < best_loss:
        best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
    else:
        bad += 1
        if bad >= 4:
            break
model.load_state_dict(best_state)

# ---- Test set (used once, for the final report) ----
probs = torch.softmax(logits_of(Xte), 1)
conf, pred = probs.max(1)
true = yte.numpy()
pred = pred.numpy()
conf = conf.numpy()
print(f"\nTest accuracy: {(pred == true).mean():.3f}  ({len(true)} images)")

cm = confusion_matrix(true, pred)
print("\nConfusion matrix (rows = true, cols = predicted):")
print("columns:", classes)
print(cm)
print("\nPer-class recall:")
for k, c in enumerate(classes):
    print(f"  {c:20s} {cm[k, k] / cm[k].sum():.3f}  ({cm[k, k]}/{cm[k].sum()})")
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = sorted([(off[i, j], classes[i], classes[j]) for i in range(len(classes))
                for j in range(len(classes)) if i != j], reverse=True)[:3]
print("\nTop confusions (true -> predicted):")
for n, t, p_ in pairs:
    print(f"  {t} -> {p_}: {n} images")

print("\nConfidence threshold (fixed in advance):")
for th in [0.5, 0.7, 0.9]:
    keep = conf >= th
    acc = (pred[keep] == true[keep]).mean() if keep.any() else float("nan")
    print(f"  >= {th}: answers on {keep.mean():.0%} of images, accuracy on those {acc:.1%}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
ax.set_yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("ResNet-18 fine-tuned, test set")
plt.tight_layout()
plt.savefig(out_path("resnet", "confusion_resnet_finetune.png"), dpi=130)
print("Saved results/resnet/confusion_resnet_finetune.png")
