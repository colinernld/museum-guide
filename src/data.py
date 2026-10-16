from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "wikiart_genre5"
RESULTS_DIR = ROOT / "results"
SEED = 42
UNKNOWN_ARTIST = "Unknown Artist"


def out_path(subdir, name):
    """results/<subdir>/<name>, creating the folder if needed (scripts run from any directory)."""
    d = RESULTS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def artists(ds, data_dir=DATA_DIR):
    """Artist of every image of the ImageFolder, in dataset order (from metadata.csv)."""
    meta = pd.read_csv(Path(data_dir) / "metadata.csv").set_index("path")
    rel = [Path(s).relative_to(data_dir).as_posix() for s, _ in ds.samples]
    return meta.loc[rel, "artist"].to_numpy()


def get_splits(data_dir=DATA_DIR, seed=SEED, by_artist=True):
    """About 70/15/15 train/val/test split, stratified by class. Returns the dataset and three index arrays.

    by_artist=True (default): a named painter never appears in two splits, so test accuracy measures
    how the guide does on painters it has never seen. "Unknown Artist" is a catch-all label, not one
    painter: those images are split freely. Split sizes then vary by a few images.
    by_artist=False: the original random stratified split (painters shared between splits).
    """
    ds = datasets.ImageFolder(data_dir)
    idx = np.arange(len(ds))
    y = np.array(ds.targets)
    if not by_artist:
        train_idx, tmp_idx = train_test_split(
            idx, test_size=0.30, stratify=y, random_state=seed)
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.50, stratify=y[tmp_idx], random_state=seed)
        return ds, train_idx, val_idx, test_idx
    art = artists(ds, data_dir)
    unknown = art == UNKNOWN_ARTIST
    # Named painters, grouped by their most frequent class (keeps the classes balanced):
    # shuffle them, then fill test (15 % of the group's images), val (15 %), train (rest)
    rng = np.random.default_rng(seed)
    named = idx[~unknown]
    main_class = {p: np.bincount(y[named][art[named] == p]).argmax() for p in np.unique(art[named])}
    test_p, val_p = set(), set()
    for c in np.unique(y):
        painters = rng.permutation(sorted(p for p, k in main_class.items() if k == c))
        counts = np.array([(art[named] == p).sum() for p in painters])
        mid = (np.cumsum(counts) - counts / 2) / counts.sum()   # where each painter "sits"
        test_p |= set(painters[mid < 0.15])
        val_p |= set(painters[(mid >= 0.15) & (mid < 0.30)])
    named_test = named[np.isin(art[named], list(test_p))]
    named_val = named[np.isin(art[named], list(val_p))]
    named_train = np.setdiff1d(named, np.concatenate([named_test, named_val]))
    # Unknown Artist: ordinary stratified 70/15/15
    unk = idx[unknown]
    unk_train, unk_tmp = train_test_split(unk, test_size=0.30, stratify=y[unk], random_state=seed)
    unk_val, unk_test = train_test_split(unk_tmp, test_size=0.50, stratify=y[unk_tmp],
                                         random_state=seed)
    return (ds, np.sort(np.concatenate([named_train, unk_train])),
            np.sort(np.concatenate([named_val, unk_val])),
            np.sort(np.concatenate([named_test, unk_test])))
