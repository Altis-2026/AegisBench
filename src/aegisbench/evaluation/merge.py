"""Merging tile-level detections back into full-image detections.

Overlapping tiles see the same person twice; after mapping detections to
full-image coordinates we deduplicate with standard greedy NMS. Evaluation
is then always against original full-image ground truth.
"""

from __future__ import annotations

import numpy as np

from ..tiling import tile_to_full


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy NMS. boxes (N,4) xyxy, scores (N,). Returns kept indices in
    descending-score order."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, int)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, int)


def merge_tile_detections(per_tile: list[dict],
                          iou_thresh: float = 0.5) -> dict:
    """per_tile: [{'origin_x', 'origin_y', 'boxes' (N,4 tile coords),
    'scores' (N,)}]. Returns {'boxes' (M,4 full coords), 'scores' (M,)}."""
    all_boxes, all_scores = [], []
    for t in per_tile:
        if len(t["boxes"]) == 0:
            continue
        all_boxes.append(tile_to_full(t["boxes"], t["origin_x"],
                                      t["origin_y"]))
        all_scores.append(np.asarray(t["scores"], np.float32).reshape(-1))
    if not all_boxes:
        return {"boxes": np.zeros((0, 4), np.float32),
                "scores": np.zeros(0, np.float32)}
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(boxes, scores, iou_thresh)
    return {"boxes": boxes[keep], "scores": scores[keep]}
