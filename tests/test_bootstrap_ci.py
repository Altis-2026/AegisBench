"""bootstrap_one must reproduce the exact (unresampled) point estimate as
its bootstrap mean on a trivial, noise-free case, and must return valid
probability-bounded, ordered CIs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase5_bootstrap_ci import bootstrap_one


def _gt(image_id, box):
    return {"image_id": image_id, "boxes": np.array([box], np.float32)}


def _dt(image_id, box, score):
    return {"image_id": image_id, "boxes": np.array([box], np.float32),
           "scores": np.array([score], np.float32)}


def test_perfect_detector_gives_recall_one_with_zero_width_ci():
    gt = [_gt(f"img{i}", [0, 0, 10, 10]) for i in range(20)]
    dt = [_dt(f"img{i}", [0, 0, 10, 10], 0.9) for i in range(20)]
    stats = bootstrap_one(gt, dt, conf_thresh=0.5, n_boot=200, seed=1)
    assert stats["recall_mean"] == 1.0
    assert stats["recall_ci_lo"] == 1.0
    assert stats["recall_ci_hi"] == 1.0


def test_partial_detector_ci_brackets_the_point_estimate():
    # 20 images, detector only finds the object in the first 10.
    gt = [_gt(f"img{i}", [0, 0, 10, 10]) for i in range(20)]
    dt = ([_dt(f"img{i}", [0, 0, 10, 10], 0.9) for i in range(10)]
         + [_dt(f"img{i}", [0, 0, 0, 0], 0.9) for i in range(10, 20)])
    stats = bootstrap_one(gt, dt, conf_thresh=0.5, n_boot=500, seed=1)
    assert 0.3 < stats["recall_mean"] < 0.7
    assert stats["recall_ci_lo"] <= stats["recall_mean"] <= stats["recall_ci_hi"]
    assert 0.0 <= stats["recall_ci_lo"]
    assert stats["recall_ci_hi"] <= 1.0


def test_deterministic_given_same_seed():
    gt = [_gt(f"img{i}", [0, 0, 10, 10]) for i in range(15)]
    dt = [_dt(f"img{i}", [0, 0, 10, 10], 0.9) for i in range(8)]
    a = bootstrap_one(gt, dt, conf_thresh=0.5, n_boot=100, seed=42)
    b = bootstrap_one(gt, dt, conf_thresh=0.5, n_boot=100, seed=42)
    assert a == b
