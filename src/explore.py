import collections
import numpy as np
import matplotlib.pyplot as plt
from data import get_splits, out_path

ds, tr, va, te = get_splits()
y = np.array(ds.targets)

print("Classes:", ds.classes)
print("Total images:", len(ds))
for name, idx in [("train", tr), ("val", va), ("test", te)]:
    counts = collections.Counter(y[idx])
    print(name, len(idx), {ds.classes[k]: v for k, v in sorted(counts.items())})

sizes = collections.Counter(ds[i][0].size for i in range(len(ds)))
print("Image sizes (top 5):", sizes.most_common(5))

# Grid: 5 random images per class
rng = np.random.default_rng(0)
fig, axes = plt.subplots(len(ds.classes), 5, figsize=(10, 10))
for c, cname in enumerate(ds.classes):
    picks = rng.choice(np.where(y == c)[0], 5, replace=False)
    for j, i in enumerate(picks):
        axes[c, j].imshow(ds[i][0])
        axes[c, j].axis("off")
    axes[c, 0].set_title(cname, fontsize=8, loc="left")
plt.tight_layout()
plt.savefig(out_path("data", "grid.png"), dpi=120)
print("Saved results/data/grid.png")
