"""Multi-octave value-noise fields used for glare, haze transmission,
inundation masks, and grain. Resolution-independent: the field's spatial
structure is defined relative to image size, not absolute pixels."""

from __future__ import annotations

import cv2
import numpy as np


def value_noise(shape: tuple[int, int], rng: np.random.Generator,
                octaves: int = 4, base_res: int = 4,
                persistence: float = 0.55) -> np.ndarray:
    """Fractal value noise in [0, 1], shape (H, W) float32.

    base_res is the grid resolution of the first octave along the longer
    image side; each octave doubles it. Bicubic upsampling of small random
    grids gives smooth large-scale structure without directional artifacts.
    """
    h, w = shape
    long_side = max(h, w)
    out = np.zeros((h, w), np.float32)
    amp_total = 0.0
    amp = 1.0
    for o in range(octaves):
        res_long = base_res * (2 ** o)
        gh = max(2, round(res_long * h / long_side) + 1)
        gw = max(2, round(res_long * w / long_side) + 1)
        grid = rng.random((gh, gw), dtype=np.float32)
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        amp_total += amp
        amp *= persistence
    out /= amp_total
    lo, hi = float(out.min()), float(out.max())
    if hi - lo > 1e-8:
        out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def coverage_mask(field: np.ndarray, coverage: float,
                  feather_sigma: float = 0.0) -> np.ndarray:
    """Soft mask in [0, 1] covering ~`coverage` fraction of the image: the
    field is thresholded at its (1 - coverage) quantile, then feathered."""
    if coverage <= 0.0:
        return np.zeros_like(field)
    if coverage >= 1.0:
        return np.ones_like(field)
    thresh = float(np.quantile(field, 1.0 - coverage))
    mask = (field >= thresh).astype(np.float32)
    if feather_sigma > 0:
        k = int(feather_sigma * 3) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), feather_sigma)
    return np.clip(mask, 0.0, 1.0)
