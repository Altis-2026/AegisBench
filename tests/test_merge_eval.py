"""Tile-merge NMS and the shared evaluator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.evaluation.coco_eval import (coco_map, pr_at_threshold,
                                             select_operating_point)
from aegisbench.evaluation.merge import merge_tile_detections, nms


def test_nms_dedupes_overlapping():
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 240, 240]],
                     np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.5)
    assert set(keep.tolist()) == {0, 2}


def test_merge_across_tiles():
    """The same person seen by two overlapping tiles collapses to one box
    in full-image coordinates."""
    per_tile = [
        {"origin_x": 0, "origin_y": 0,
         "boxes": np.array([[900.0, 500.0, 940.0, 560.0]]),
         "scores": np.array([0.85])},
        {"origin_x": 768, "origin_y": 0,
         "boxes": np.array([[132.0, 500.0, 172.0, 560.0]]),  # same person
         "scores": np.array([0.90])},
    ]
    merged = merge_tile_detections(per_tile)
    assert len(merged["boxes"]) == 1
    assert np.allclose(merged["boxes"][0], [900, 500, 940, 560], atol=1e-3)
    assert merged["scores"][0] == 0.90


def _records():
    gt = [{"image_id": "im1", "width": 1000, "height": 1000,
           "boxes": np.array([[100, 100, 130, 150],
                              [500, 500, 530, 550]], np.float32)}]
    dt = [{"image_id": "im1",
           "boxes": np.array([[102, 101, 131, 152],      # hit
                              [700, 700, 730, 750]], np.float32),  # FP
           "scores": np.array([0.9, 0.6], np.float32)}]
    return gt, dt


def test_pr_at_threshold():
    gt, dt = _records()
    m = pr_at_threshold(gt, dt, conf_thresh=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["n_gt"] == 2
    assert abs(m["recall"] - 0.5) < 1e-9
    assert abs(m["precision"] - 0.5) < 1e-9
    # Both GT boxes are "small" by COCO area (30x50 = 1500 < 1024)? No:
    # 1500 px^2 > 32^2 = 1024 -> medium. Check strata bookkeeping adds up.
    assert m["n_gt_small"] + m["n_gt_medium"] + m["n_gt_large"] == 2


def test_higher_threshold_removes_fp():
    gt, dt = _records()
    m = pr_at_threshold(gt, dt, conf_thresh=0.7)
    assert m["fp"] == 0 and m["precision"] == 1.0


def test_operating_point_prefers_fp_free_threshold():
    gt, dt = _records()
    t = select_operating_point(gt, dt)
    m = pr_at_threshold(gt, dt, t)
    assert m["f1"] >= 2 / 3 - 1e-9


def test_coco_map_perfect_predictions():
    gt, _ = _records()
    dt = [{"image_id": "im1", "boxes": gt[0]["boxes"].copy(),
           "scores": np.array([0.95, 0.9], np.float32)}]
    m = coco_map(gt, dt)
    assert m["map50"] > 0.99
