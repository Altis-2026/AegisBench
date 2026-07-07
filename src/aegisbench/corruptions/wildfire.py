"""Wildfire-family corruptions: smoke haze and fire-warm tint.

Both build on the Koschmieder atmospheric scattering model
I = J * t + A * (1 - t), the standard physical model for haze/smoke, with a
spatially varying transmission map t.
"""

from __future__ import annotations

import numpy as np

from .base import Corruption
from .noise_fields import value_noise


def transmission_map(shape, rng, t_mean: float, variation_amp: float,
                     octaves: int, base_res: int) -> np.ndarray:
    """Spatially varying transmission with mean ~t_mean, structure from
    multi-octave noise."""
    field = value_noise(shape, rng, octaves=octaves, base_res=base_res)
    t = t_mean + variation_amp * (field - 0.5) * 2.0
    return np.clip(t, 0.02, 1.0)[..., None]


def koschmieder(img: np.ndarray, t: np.ndarray,
                airlight_rgb) -> np.ndarray:
    airlight = np.asarray(airlight_rgb, np.float32)
    return img * t + airlight * (1.0 - t)


class SmokeHaze(Corruption):
    """Gray wildfire smoke: neutral airlight, low-frequency transmission."""

    name = "smoke_haze"
    family = "wildfire"

    def _apply(self, img, p, rng):
        t = transmission_map(img.shape[:2], rng, p["t_mean"],
                             p["t_variation_amp"], p["noise_octaves"],
                             p["noise_base_res"])
        return koschmieder(img, t, p["airlight_rgb"])


class FireWarmTint(Corruption):
    """Fire-proximate illumination: warm white-balance shift plus lighter,
    wispier warm-airlight haze."""

    name = "fire_warm_tint"
    family = "wildfire"

    def _apply(self, img, p, rng):
        out = img.copy()
        out[..., 0] *= p["r_gain"]
        out[..., 2] *= p["b_gain"]
        out = np.clip(out, 0.0, 1.0)
        t = transmission_map(img.shape[:2], rng, p["t_mean"],
                             p["t_variation_amp"], p["noise_octaves"],
                             p["noise_base_res"])
        return koschmieder(out, t, p["airlight_rgb"])
