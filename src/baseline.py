"""Classical baselines, first look (validation set only; the test set is used in final.py)."""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import SEED, get_splits
from features import handcrafted, pixels

ds, tr, va, te = get_splits()
y = np.array(ds.targets)
feats = {name: np.stack([f(ds[i][0]) for i in range(len(ds))])
         for name, f in [("pixels", pixels), ("handcrafted", handcrafted)]}
print("Classes:", ds.classes, "|", {k: v.shape for k, v in feats.items()})

# C (inverse regularisation strength) is chosen on the validation set
candidates = {
    "logreg_pixels": ("pixels", lambda C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000)), [0.001, 0.01, 0.1]),
    "random_forest_pixels": ("pixels", lambda C: RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1), [None]),
    "logreg_handcrafted": ("handcrafted", lambda C: make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=3000)), [0.001, 0.01, 0.1]),
    "svm_handcrafted": ("handcrafted", lambda C: make_pipeline(
        StandardScaler(), SVC(C=C)), [0.3, 1, 3, 10]),
}

for name, (feat, make, grid) in candidates.items():
    X = feats[feat]
    print(f"\n== {name} ==")
    best = None
    for C in grid:
        t0 = time.time()
        model = make(C).fit(X[tr], y[tr])
        fit_s = time.time() - t0
        acc = accuracy_score(y[va], model.predict(X[va]))
        print(f"  C={C}: train acc {model.score(X[tr], y[tr]):.3f} | val acc {acc:.3f} "
              f"| fit {fit_s:.1f}s")
        if best is None or acc > best[0]:
            best = (acc, C, model)
    print(f"best C={best[1]} | val acc {best[0]:.3f}")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y[va], best[2].predict(X[va])))
