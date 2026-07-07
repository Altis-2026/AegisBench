"""Earthquake / post-disaster dust corruption.

Distinct from wildfire smoke on three physically grounded axes: brown
mineral-particle airlight (vs. neutral gray), patchier higher-frequency
transmission structure (turbulent plumes vs. diffuse smoke layers), and
coarse near-lens particulate grain.
"""

from __future__ import annotations

import numpy as np

from .base import Corruption
from .noise_fields import value_noise
from .wildfire import koschmieder, transmission_map


class DustHaze(Corruption):
    name = "dust_haze"
    family = "earthquake"

    def _apply(self, img, p, rng):
        t = transmission_map(img.shape[:2], rng, p["t_mean"],
                             p["t_variation_amp"], p["noise_octaves"],
                             p["noise_base_res"])
        out = koschmieder(img, t, p["airlight_rgb"])

        # Coarse granular texture from suspended particles near the lens.
        grain = value_noise(img.shape[:2], rng, octaves=2,
                            base_res=p["grain_base_res"])
        out = out + p["grain_amp"] * (grain[..., None] - 0.5) * 2.0
        return out
