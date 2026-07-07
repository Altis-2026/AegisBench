"""Flood-family corruptions: water glare, turbidity cast, inundation."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Corruption, gaussian_blur, px, scale_saturation, screen_blend
from .noise_fields import coverage_mask, value_noise


class WaterGlare(Corruption):
    """Specular sun-glint on floodwater: thresholded low-frequency glint
    cores, screen-blended with a Gaussian bloom halo."""

    name = "water_glare"
    family = "flood"

    def _apply(self, img, p, rng):
        h, w = img.shape[:2]
        field = value_noise((h, w), rng, octaves=p["noise_octaves"],
                            base_res=p["noise_base_res"])
        core = coverage_mask(field, p["coverage"])
        bloom = gaussian_blur(core, px(p["bloom_sigma_per_1000px"], img))
        glare = np.clip(core + 0.6 * bloom, 0.0, 1.0) * p["intensity"]
        color = np.asarray(p["glare_color_rgb"], np.float32)
        return screen_blend(img, glare[..., None] * color)


class TurbidityCast(Corruption):
    """Muddy sediment-laden water: blend toward mud chromaticity, contrast
    compression toward the scene mean, saturation loss."""

    name = "turbidity_cast"
    family = "flood"

    def _apply(self, img, p, rng):
        mud = np.asarray(p["mud_color_rgb"], np.float32)
        out = (1.0 - p["mud_alpha"]) * img + p["mud_alpha"] * mud
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = mean + p["contrast_factor"] * (out - mean)
        return scale_saturation(out, p["saturation_factor"])


class Inundation(Corruption):
    """Semi-transparent standing water over part of the scene: smooth noise
    mask -> ripple warp + murky water blend + specular sparkle inside it."""

    name = "inundation"
    family = "flood"

    def _apply(self, img, p, rng):
        h, w = img.shape[:2]
        field = value_noise((h, w), rng, octaves=p["noise_octaves"],
                            base_res=p["noise_base_res"])
        mask = coverage_mask(field, p["coverage"],
                             feather_sigma=px(p["mask_feather_per_1000px"],
                                              img))[..., None]

        # Sinusoidal ripple displacement of the submerged content.
        amp = px(p["ripple_amp_per_1000px"], img)
        wavelength = max(px(p["ripple_wavelength_per_1000px"], img), 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xs + amp * np.sin(2.0 * np.pi * ys / wavelength + phase)
        map_y = ys + amp * np.cos(2.0 * np.pi * xs / wavelength + phase)
        rippled = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

        water = np.asarray(p["water_color_rgb"], np.float32)
        submerged = (1.0 - p["opacity"]) * rippled + p["opacity"] * water

        # Small specular sparkle on the water surface.
        sparkle_field = value_noise((h, w), rng, octaves=2, base_res=180)
        sparkle = coverage_mask(sparkle_field, 0.02)[..., None]
        submerged = screen_blend(submerged, sparkle * p["sparkle_intensity"])

        return (1.0 - mask) * img + mask * submerged
