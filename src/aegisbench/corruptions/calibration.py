"""Measurable image statistics that define each corruption's severity ladder.

The severity levels in configs/corruptions.yaml are calibrated so the named
statistic moves monotonically from severity 1 to 3; tests and
scripts/phase3_calibrate.py enforce this on real data rather than trusting
the parameter values.
"""

from __future__ import annotations

import cv2
import numpy as np


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def rms_contrast(img: np.ndarray) -> float:
    """Std of luminance — the classic global contrast measure."""
    return float(_luma(img).std())


def mean_luminance(img: np.ndarray) -> float:
    return float(_luma(img).mean())


def saturated_fraction(img: np.ndarray, thresh: float = 0.94) -> float:
    """Fraction of near-saturated pixels (glare metric)."""
    return float((_luma(img) > thresh).mean())


def edge_strength(img: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of luminance (blur metric)."""
    luma = _luma(img).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def red_blue_ratio(img: np.ndarray) -> float:
    """Mean R / mean B — color-temperature shift metric."""
    return float(img[..., 0].mean() / max(img[..., 2].mean(), 1e-6))


def occluded_fraction(clean: np.ndarray, corrupted: np.ndarray,
                      delta: float = 0.08) -> float:
    """Fraction of pixels whose luminance changed by more than `delta` —
    proxy for the area covered by an occluding overlay."""
    return float((np.abs(_luma(corrupted) - _luma(clean)) > delta).mean())


# stat name -> (fn, needs_clean_reference)
STATS = {
    "rms_contrast": (rms_contrast, False),
    "mean_luminance": (mean_luminance, False),
    "saturated_fraction": (saturated_fraction, False),
    "edge_strength": (edge_strength, False),
    "red_blue_ratio": (red_blue_ratio, False),
    "occluded_fraction": (occluded_fraction, True),
    # streak_density has no closed-form image statistic; rain severity is
    # audited via edge_strength/mean_luminance side effects and visual review.
    "streak_density": (edge_strength, False),
}


def measure(stat_name: str, corrupted: np.ndarray,
            clean: np.ndarray | None = None) -> float:
    fn, needs_clean = STATS[stat_name]
    if needs_clean:
        if clean is None:
            raise ValueError(f"stat '{stat_name}' needs the clean image")
        return fn(clean, corrupted)
    return fn(corrupted)
