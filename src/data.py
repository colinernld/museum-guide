import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split

DATA_DIR = "data/wikiart_genre5"
SEED = 42


def get_splits(data_dir=DATA_DIR, seed=SEED):
    """Stratified 70/15/15 split. Returns the dataset and three index arrays."""
    ds = datasets.ImageFolder(data_dir)
    idx = np.arange(len(ds))
    y = np.array(ds.targets)
    train_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=y[tmp_idx], random_state=seed)
    return ds, train_idx, val_idx, test_idx
