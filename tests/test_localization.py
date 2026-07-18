import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.evaluation.localization import (iou_matrix,
                                                localization_stability,
                                                match_gt_to_pred)


def test_iou_matrix_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-6
    b = np.array([[100, 100, 110, 110]], np.float32)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_matrix_half_overlap():
    a = np.array([[0, 0, 10, 10]], np.float32)      # area 100
    b = np.array([[5, 0, 15, 10]], np.float32)      # area 100, inter 50
    assert abs(iou_matrix(a, b)[0, 0] - 50 / 150) < 1e-6


def test_match_gt_to_pred_basic():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    dt = np.array([[1, 1, 11, 11], [101, 101, 111, 111]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    m = match_gt_to_pred(gt, dt, scores, 0.5)
    assert m[0] == 0 and m[1] == 1


def test_match_respects_score_order():
    """Two predictions contest one gt; the higher-scoring one wins it."""
    gt = np.array([[0, 0, 10, 10]], np.float32)
    dt = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.6, 0.95], np.float32)      # pred 1 scores higher
    m = match_gt_to_pred(gt, dt, scores, 0.5)
    assert m[0] == 1


def _gt_records():
    return [{"image_id": "im1",
             "boxes": np.array([[0, 0, 20, 40], [200, 200, 220, 240]],
                               np.float32)}]


def test_perfect_stability_when_boxes_unchanged():
    gt = _gt_records()
    dt = [{"image_id": "im1",
           "boxes": np.array([[1, 1, 21, 41], [201, 201, 221, 241]],
                             np.float32),
           "scores": np.array([0.9, 0.9], np.float32)}]
    m = localization_stability(gt, dt, dt, conf_thresh=0.5)
    assert m["n_common"] == 2
    assert abs(m["loc_stability_iou"] - 1.0) < 1e-6
    assert abs(m["loc_center_shift"]) < 1e-6
    assert abs(m["loc_iou_drop"]) < 1e-6


def test_drift_lowers_stability_and_shifts_center():
    gt = _gt_records()
    clean = [{"image_id": "im1",
              "boxes": np.array([[0, 0, 20, 40]], np.float32),
              "scores": np.array([0.9], np.float32)}]
    # Same survivor detected, but the corrupted box is shifted by 6 px.
    corrupt = [{"image_id": "im1",
                "boxes": np.array([[6, 0, 26, 40]], np.float32),
                "scores": np.array([0.9], np.float32)}]
    m = localization_stability(gt, clean, corrupt, conf_thresh=0.5)
    assert m["n_common"] == 1
    assert m["loc_stability_iou"] < 1.0
    assert m["loc_center_shift"] > 0.0
    # corrupted box fits gt worse than the clean box -> positive drop
    assert m["loc_iou_drop"] > 0.0


def test_only_common_instances_counted():
    """A survivor lost under corruption must not enter the averages."""
    gt = _gt_records()
    clean = [{"image_id": "im1",
              "boxes": np.array([[0, 0, 20, 40], [200, 200, 220, 240]],
                                np.float32),
              "scores": np.array([0.9, 0.9], np.float32)}]
    corrupt = [{"image_id": "im1",       # second survivor now missed
                "boxes": np.array([[0, 0, 20, 40]], np.float32),
                "scores": np.array([0.9], np.float32)}]
    m = localization_stability(gt, clean, corrupt, conf_thresh=0.5)
    assert m["n_common"] == 1


def test_center_shift_is_scale_normalized():
    """The same absolute pixel shift is a larger normalized shift on a
    smaller person."""
    small_gt = [{"image_id": "im1",
                 "boxes": np.array([[0, 0, 10, 10]], np.float32)}]
    large_gt = [{"image_id": "im1",
                 "boxes": np.array([[0, 0, 100, 100]], np.float32)}]
    clean_s = [{"image_id": "im1", "boxes": np.array([[0, 0, 10, 10]],
                np.float32), "scores": np.array([0.9], np.float32)}]
    corrupt_s = [{"image_id": "im1", "boxes": np.array([[3, 0, 13, 10]],
                  np.float32), "scores": np.array([0.9], np.float32)}]
    clean_l = [{"image_id": "im1", "boxes": np.array([[0, 0, 100, 100]],
                np.float32), "scores": np.array([0.9], np.float32)}]
    corrupt_l = [{"image_id": "im1", "boxes": np.array([[3, 0, 103, 100]],
                  np.float32), "scores": np.array([0.9], np.float32)}]
    ms = localization_stability(small_gt, clean_s, corrupt_s, 0.5)
    ml = localization_stability(large_gt, clean_l, corrupt_l, 0.5)
    assert ms["loc_center_shift"] > ml["loc_center_shift"]
