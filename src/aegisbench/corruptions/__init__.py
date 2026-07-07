"""Corruption registry: name -> class, plus a convenience applier that
handles config loading and deterministic seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..seeding import GLOBAL_SEED_DEFAULT, rng_for
from .base import Corruption, load_corruption_config, to_float, to_uint8
from .dust import DustHaze
from .flood import Inundation, TurbidityCast, WaterGlare
from .storm import LowLight, MotionBlur, RainStreaks
from .wildfire import FireWarmTint, SmokeHaze

CORRUPTION_CLASSES: dict[str, type[Corruption]] = {
    cls.name: cls
    for cls in (WaterGlare, TurbidityCast, Inundation, SmokeHaze,
                FireWarmTint, RainStreaks, MotionBlur, LowLight, DustHaze)
}

SEVERITIES = (1, 2, 3)

DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parents[3]
                       / "configs" / "corruptions.yaml")


class CorruptionSuite:
    """All corruptions instantiated from one config file."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config = load_corruption_config(config_path)
        self.global_seed = int(self.config.get("global_seed",
                                               GLOBAL_SEED_DEFAULT))
        self.corruptions: dict[str, Corruption] = {}
        for name in self.config["corruptions"]:
            if name not in CORRUPTION_CLASSES:
                raise KeyError(f"config names unknown corruption '{name}'")
            self.corruptions[name] = CORRUPTION_CLASSES[name].from_config(
                self.config)

    def names(self) -> list[str]:
        return list(self.corruptions)

    def family(self, name: str) -> str:
        return self.config["corruptions"][name]["family"]

    def calibration(self, name: str) -> dict:
        return self.config["corruptions"][name]["calibration"]

    def apply(self, img: np.ndarray, name: str, severity: int,
              image_id: str) -> np.ndarray:
        """Apply one corruption deterministically. Accepts uint8 or float32
        RGB; returns the same dtype it was given."""
        was_uint8 = img.dtype == np.uint8
        rng = rng_for(image_id, name, severity, global_seed=self.global_seed)
        out = self.corruptions[name](to_float(img), severity, rng)
        return to_uint8(out) if was_uint8 else out
