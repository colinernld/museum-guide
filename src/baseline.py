import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SEED, get_splits


def features(img):
    """32x32 RGB pixels, scaled to [0, 1], flattened (3072 values)."""
    small = np.asarray(img.resize((32, 32)), dtype=np.float32) / 255.0
    return small.ravel()


ds, tr, va, te = get_splits()
X = np.stack([features(ds[i][0]) for i in range(len(ds))])
y = np.array(ds.targets)
print("Features:", X.shape, "| classes:", ds.classes)

models = {
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.01)),
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1),
}

for name, model in models.items():
    t0 = time.time()
    model.fit(X[tr], y[tr])
    fit_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(X[va])
    ms_per_img = (time.time() - t0) / len(va) * 1000

    print(f"\n== {name} ==")
    print(f"train acc: {accuracy_score(y[tr], model.predict(X[tr])):.3f}")
    print(f"val acc:   {accuracy_score(y[va], pred):.3f}")
    print(f"fit time: {fit_s:.1f}s | inference: {ms_per_img:.3f} ms/image")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y[va], pred))
