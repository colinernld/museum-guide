"""Feature extractors for the classical models (one PIL image -> one 1-D vector).

pixels:      32x32 RGB pixels in [0, 1] (3 072 values), the original baseline.
handcrafted: what a classical computer-vision pipeline would use, without any learning:
             - colour: HSV histogram 8x8x8 (512 values): palette, light, saturation;
             - shape:  HOG on the 64x64 grey image (324 values): edges, orientations, layout;
             - texture: uniform LBP histogram (10 values): brushwork, fine texture.
"""
import numpy as np
from skimage.feature import hog, local_binary_pattern


def pixels(img):
    small = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    return small.ravel()


def handcrafted(img):
    rgb = img.convert("RGB").resize((64, 64))
    hsv = np.asarray(rgb.convert("HSV")).reshape(-1, 3)
    colour, _ = np.histogramdd(hsv, bins=(8, 8, 8), range=((0, 256),) * 3)
    colour = colour.ravel() / len(hsv)
    grey = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    shape = hog(grey, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    lbp = local_binary_pattern((grey * 255).astype(np.uint8), P=8, R=1, method="uniform")
    texture, _ = np.histogram(lbp, bins=10, range=(0, 10))
    texture = texture / lbp.size
    return np.concatenate([colour, shape, texture]).astype(np.float32)
