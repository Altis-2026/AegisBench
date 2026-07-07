"""Corruption base class and shared photometric helpers.

All corruptions operate on float32 RGB images in [0, 1] and are pure
functions of (image, severity, rng): identical inputs produce identical
outputs on any machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_CONFIG_CACHE: dict[str, dict] = {}


def load_corruption_config(path: str | Path) -> dict:
    key = str(Path(path).resolve())
    if key not in _CONFIG_CACHE:
        with open(path) as f:
            _CONFIG_CACHE[key] = yaml.safe_load(f)
    return _CONFIG_CACHE[key]


def px(value_per_1000px: float, img: np.ndarray) -> float:
    """Convert a per-1000px parameter to absolute pixels for this image."""
    return value_per_1000px * max(img.shape[:2]) / 1000.0


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def screen_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - base) * (1.0 - np.clip(layer, 0.0, 1.0))


def scale_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale chroma about the luminance axis (Rec.601 luma)."""
    luma = (0.299 * img[..., 0] + 0.587 * img[..., 1]
            + 0.114 * img[..., 2])[..., None]
    return np.clip(luma + factor * (img - luma), 0.0, 1.0)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


class Corruption(ABC):
    """A named, parameterized corruption with 3 severity levels."""

    name: str = ""
    family: str = ""

    def __init__(self, severities: dict[int, dict[str, Any]],
                 fixed: dict[str, Any]):
        self.severities = {int(k): v for k, v in severities.items()}
        self.fixed = fixed

    @classmethod
    def from_config(cls, config: dict) -> "Corruption":
        spec = config["corruptions"][cls.name]
        return cls(spec["severities"], spec.get("fixed") or {})

    def params(self, severity: int) -> dict[str, Any]:
        if severity not in self.severities:
            raise ValueError(
                f"{self.name}: severity must be one of "
                f"{sorted(self.severities)}, got {severity}")
        return {**self.fixed, **self.severities[severity]}

    def __call__(self, img: np.ndarray, severity: int,
                 rng: np.random.Generator) -> np.ndarray:
        img = to_float(img)
        out = self._apply(img, self.params(severity), rng)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @abstractmethod
    def _apply(self, img: np.ndarray, p: dict[str, Any],
               rng: np.random.Generator) -> np.ndarray:
        ...
