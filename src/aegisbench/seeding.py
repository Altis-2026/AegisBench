"""Deterministic per-(image, corruption, severity) random generators.

Every stochastic element in the corruption engine draws from a generator
seeded by a stable hash of (image_id, corruption_name, severity, global_seed),
so the corrupted benchmark is a fixed dataset: regenerating it on any machine
produces bit-identical images.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

GLOBAL_SEED_DEFAULT = 20260707


def stable_seed(*parts: object, global_seed: int = GLOBAL_SEED_DEFAULT) -> int:
    """Map arbitrary key parts to a stable uint32 seed (SHA-256 based;
    independent of PYTHONHASHSEED and platform)."""
    key = "\x1f".join(str(p) for p in parts) + f"\x1f{global_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(image_id: str, corruption: str, severity: int,
            global_seed: int = GLOBAL_SEED_DEFAULT) -> np.random.Generator:
    return np.random.default_rng(
        stable_seed(image_id, corruption, severity, global_seed=global_seed)
    )


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and (if installed) torch for training runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
