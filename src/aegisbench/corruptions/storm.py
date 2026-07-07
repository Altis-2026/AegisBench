"""Storm-family corruptions: rain streaks, wind motion blur, low light."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Corruption, gaussian_blur, px, scale_saturation, screen_blend


class RainStreaks(Corruption):
    """Heavy rain: a veil (blur + desaturation + overcast darkening) plus
    additive elongated streaks sharing one wind direction per image."""

    name = "rain_streaks"
    family = "storm"

    def _apply(self, img, p, rng):
        h, w = img.shape[:2]

        # Rain veil first: rain scatters light before streaks are resolved.
        out = gaussian_blur(img, px(p["veil_blur_sigma_per_1000px"], img))
        out = out * p["brightness_factor"]
        out = scale_saturation(out, p["saturation_factor"])

        # One coherent wind direction per image, jittered per streak.
        angle_img = p["angle_mean_deg"] + rng.uniform(
            -p["angle_jitter_image_deg"], p["angle_jitter_image_deg"])
        n_streaks = int(round(p["streaks_per_megapixel"] * h * w / 1e6))
        length = max(2.0, px(p["length_per_1000px"], img))
        lo, hi = p["streak_intensity_range"]

        layer = np.zeros((h, w), np.float32)
        xs = rng.uniform(0, w, n_streaks)
        ys = rng.uniform(0, h, n_streaks)
        angles = np.deg2rad(angle_img + rng.uniform(
            -p["angle_jitter_streak_deg"], p["angle_jitter_streak_deg"],
            n_streaks))
        lengths = length * rng.uniform(0.7, 1.3, n_streaks)
        intensities = rng.uniform(lo, hi, n_streaks)
        for x0, y0, a, ln, it in zip(xs, ys, angles, lengths, intensities):
            x1 = int(round(x0 + ln * np.cos(a)))
            y1 = int(round(y0 + ln * np.sin(a)))
            cv2.line(layer, (int(round(x0)), int(round(y0))), (x1, y1),
                     float(it), 1, cv2.LINE_AA)

        # Soften streaks slightly so they read as rain, not scratches.
        layer = cv2.GaussianBlur(layer, (3, 3), 0.6)
        return screen_blend(out, layer[..., None])


class MotionBlur(Corruption):
    """Wind-induced platform shake: linear motion blur, one random direction
    per image, kernel length scaled to image size."""

    name = "motion_blur"
    family = "storm"

    def _apply(self, img, p, rng):
        length = max(3, int(round(px(p["kernel_len_per_1000px"], img))))
        angle = rng.uniform(0.0, 180.0)
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2),
                                      angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


class LowLight(Corruption):
    """Dusk / heavy overcast, modeled in an approximately linear photometric
    domain: photon scaling, twilight blue-shift, signal-dependent shot noise
    plus read noise, then re-encode to display gamma."""

    name = "low_light"
    family = "storm"

    def _apply(self, img, p, rng):
        g = p["decode_gamma"]
        linear = np.power(img, g)
        linear *= p["linear_gain"]
        linear[..., 0] *= p["wb_r_gain"]
        linear[..., 2] *= p["wb_b_gain"]

        shot_sigma = np.sqrt(np.clip(linear, 0.0, 1.0) * p["shot_noise_coef"])
        noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
        linear = linear + noise * (shot_sigma + p["read_noise_sigma"])

        return np.power(np.clip(linear, 0.0, 1.0), 1.0 / g)
