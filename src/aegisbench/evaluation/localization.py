"""Localization stability: a second robustness axis beyond recall.

Recall answers "did the detector still FIND the survivor under corruption?"
Localization stability answers a different, complementary question: "for the
survivors it still finds, does the predicted box stay planted on the same
place, or does its aim get shakier as corruption worsens?"

Because the ground truth is identical across the clean and corrupted runs
(same pixels, same boxes — only appearance changed), we can match each
condition's detections to the SAME gt instances and, for every survivor
detected in BOTH conditions, measure:

  * loc_stability_iou   IoU(clean_pred_box, corrupted_pred_box)
                        1.0 = the box did not move at all; lower = drift.
  * loc_center_shift    ||center(clean_pred) - center(corrupted_pred)||
                        normalized by the gt box diagonal (scale-invariant,
                        so a 5 px wobble on a tiny distant person counts
                        more than on a large near one).
  * loc_iou_clean       IoU(clean_pred, gt)      — box fit, clean.
  * loc_iou_corrupt     IoU(corrupted_pred, gt)  — box fit, corrupted.
  * loc_iou_drop        loc_iou_clean - loc_iou_corrupt (localization
                        quality lost while the object was still detected).

Only commonly-detected instances enter these averages, so the metric is not
confounded by recall: it isolates *where* the surviving detections landed.
`n_common` is reported alongside so a small, unstable sample is visible
rather than hidden.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4), xyxy -> (N, M)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_gt_to_pred(gt: np.ndarray, dt_boxes: np.ndarray,
                     dt_scores: np.ndarray, iou_thresh: float = 0.5
                     ) -> np.ndarray:
    """Greedy score-ordered matching (same rule as the P/R evaluator),
    returning for each gt box the index of its matched prediction, or -1.

    A prediction matches at most one gt; higher-score predictions choose
    first, taking the highest-IoU still-unclaimed gt above threshold.
    """
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    dt_boxes = np.asarray(dt_boxes, np.float32).reshape(-1, 4)
    dt_scores = np.asarray(dt_scores, np.float32).reshape(-1)
    gt_to_pred = np.full(len(gt), -1, int)
    if len(gt) == 0 or len(dt_boxes) == 0:
        return gt_to_pred
    ious = iou_matrix(dt_boxes, gt)          # (P, G)
    order = dt_scores.argsort()[::-1]
    gt_claimed = np.zeros(len(gt), bool)
    for p in order:
        row = ious[p].copy()
        row[gt_claimed] = -1.0
        g = int(row.argmax())
        if row[g] >= iou_thresh:
            gt_claimed[g] = True
            gt_to_pred[g] = int(p)
    return gt_to_pred


def _centers(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                     (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _diagonals(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.sqrt(np.maximum(boxes[:, 2] - boxes[:, 0], 0) ** 2
                   + np.maximum(boxes[:, 3] - boxes[:, 1], 0) ** 2)


def localization_stability(gt_records: list[dict], clean_dt: list[dict],
                           corrupt_dt: list[dict], conf_thresh: float,
                           iou_thresh: float = 0.5) -> dict:
    """Aggregate localization-stability metrics over survivors detected in
    BOTH the clean and corrupted conditions.

    gt_records:  [{'image_id', 'boxes' (N,4)}]
    clean_dt / corrupt_dt: [{'image_id', 'boxes', 'scores'}] — full-image
        detections for the clean and one corrupted condition respectively.
    conf_thresh: the model's frozen operating point (same one used for P/R),
        applied to both conditions so the comparison is apples-to-apples.
    """
    clean_by = {str(r["image_id"]): r for r in clean_dt}
    corrupt_by = {str(r["image_id"]): r for r in corrupt_dt}

    stab_iou, center_shift, iou_clean, iou_corrupt = [], [], [], []
    for rec in gt_records:
        gt = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        if len(gt) == 0:
            continue
        c = clean_by.get(str(rec["image_id"]))
        k = corrupt_by.get(str(rec["image_id"]))
        if c is None or k is None:
            continue

        def _thr(d):
            b = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
            s = np.asarray(d["scores"], np.float32).reshape(-1)
            keep = s >= conf_thresh
            return b[keep], s[keep]

        cb, cs = _thr(c)
        kb, ks = _thr(k)
        g2c = match_gt_to_pred(gt, cb, cs, iou_thresh)
        g2k = match_gt_to_pred(gt, kb, ks, iou_thresh)

        diag = _diagonals(gt)
        for gi in range(len(gt)):
            if g2c[gi] < 0 or g2k[gi] < 0:
                continue                       # not detected in both
            cbox = cb[g2c[gi]]
            kbox = kb[g2k[gi]]
            stab_iou.append(float(iou_matrix(cbox[None], kbox[None])[0, 0]))
            shift = np.linalg.norm(_centers(cbox[None])[0]
                                   - _centers(kbox[None])[0])
            center_shift.append(float(shift / max(diag[gi], 1e-6)))
            iou_clean.append(float(iou_matrix(cbox[None], gt[gi][None])[0, 0]))
            iou_corrupt.append(float(iou_matrix(kbox[None],
                                                gt[gi][None])[0, 0]))

    n = len(stab_iou)
    if n == 0:
        return {"loc_stability_iou": float("nan"),
                "loc_center_shift": float("nan"),
                "loc_iou_clean": float("nan"),
                "loc_iou_corrupt": float("nan"),
                "loc_iou_drop": float("nan"), "n_common": 0}
    lc, kc = float(np.mean(iou_clean)), float(np.mean(iou_corrupt))
    return {"loc_stability_iou": float(np.mean(stab_iou)),
            "loc_center_shift": float(np.mean(center_shift)),
            "loc_iou_clean": lc, "loc_iou_corrupt": kc,
            "loc_iou_drop": lc - kc, "n_common": n}
