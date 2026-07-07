"""One evaluator for every detector, so metric implementations can never
differ between models.

Metrics:
  * mAP@0.5 and mAP@[.5:.95] via pycocotools on serialized predictions.
  * Precision / recall / F1 at IoU 0.5 at a FIXED confidence operating
    point. The operating point is chosen once per model on the CLEAN
    validation split (max F1) and then frozen for every corruption run —
    otherwise recall drops would be confounded by threshold effects.
  * Size-stratified recall (COCO small/medium/large area breaks) for the
    failure-mode analysis: survivors are almost always small in frame.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

SMALL_MAX_AREA = 32.0 ** 2
MEDIUM_MAX_AREA = 96.0 ** 2


def to_coco_gt(records: list[dict], category_name: str = "person") -> dict:
    """records: [{'image_id', 'width', 'height', 'boxes' (N,4 xyxy)}]."""
    images, annotations = [], []
    ann_id = 1
    for idx, rec in enumerate(records, start=1):
        images.append({"id": idx, "file_name": str(rec["image_id"]),
                       "width": int(rec["width"]),
                       "height": int(rec["height"])})
        for x1, y1, x2, y2 in np.asarray(rec["boxes"],
                                         np.float32).reshape(-1, 4):
            w, h = float(x2 - x1), float(y2 - y1)
            annotations.append({
                "id": ann_id, "image_id": idx, "category_id": 1,
                "bbox": [float(x1), float(y1), w, h], "area": w * h,
                "iscrowd": 0})
            ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": category_name}]}


def to_coco_dt(records: list[dict], name_to_id: dict[str, int]) -> list[dict]:
    """records: [{'image_id', 'boxes' (N,4 xyxy), 'scores' (N,)}]."""
    dets = []
    for rec in records:
        img_id = name_to_id[str(rec["image_id"])]
        boxes = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        scores = np.asarray(rec["scores"], np.float32).reshape(-1)
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            dets.append({"image_id": img_id, "category_id": 1,
                         "bbox": [float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})
    return dets


def _greedy_match(gt: np.ndarray, dt: np.ndarray, iou_thresh: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Match detections (descending score order assumed) to GT at IoU
    threshold. Returns (dt_matched bool, gt_matched bool)."""
    gt_matched = np.zeros(len(gt), bool)
    dt_matched = np.zeros(len(dt), bool)
    if len(gt) == 0 or len(dt) == 0:
        return dt_matched, gt_matched
    gx1, gy1, gx2, gy2 = gt.T
    g_area = np.maximum(gx2 - gx1, 0) * np.maximum(gy2 - gy1, 0)
    for j, (x1, y1, x2, y2) in enumerate(dt):
        ix1 = np.maximum(x1, gx1)
        iy1 = np.maximum(y1, gy1)
        ix2 = np.minimum(x2, gx2)
        iy2 = np.minimum(y2, gy2)
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        d_area = max((x2 - x1), 0) * max((y2 - y1), 0)
        iou = inter / np.maximum(g_area + d_area - inter, 1e-9)
        iou[gt_matched] = -1.0
        best = int(iou.argmax())
        if iou[best] >= iou_thresh:
            gt_matched[best] = True
            dt_matched[j] = True
    return dt_matched, gt_matched


def pr_at_threshold(gt_records: list[dict], dt_records: list[dict],
                    conf_thresh: float, iou_thresh: float = 0.5) -> dict:
    """Precision/recall/F1 at a fixed operating point, plus size-stratified
    recall (COCO area breaks on GT boxes)."""
    dt_by_img = {str(r["image_id"]): r for r in dt_records}
    tp = fp = n_gt = 0
    size_tp = {"small": 0, "medium": 0, "large": 0}
    size_n = {"small": 0, "medium": 0, "large": 0}

    def size_of(box):
        a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
        if a <= SMALL_MAX_AREA:
            return "small"
        return "medium" if a <= MEDIUM_MAX_AREA else "large"

    for rec in gt_records:
        gt = np.asarray(rec["boxes"], np.float32).reshape(-1, 4)
        d = dt_by_img.get(str(rec["image_id"]),
                          {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)})
        boxes = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
        scores = np.asarray(d["scores"], np.float32).reshape(-1)
        keep = scores >= conf_thresh
        boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        dt_matched, gt_matched = _greedy_match(gt, boxes, iou_thresh)
        tp += int(dt_matched.sum())
        fp += int((~dt_matched).sum())
        n_gt += len(gt)
        for g, matched in zip(gt, gt_matched):
            s = size_of(g)
            size_n[s] += 1
            size_tp[s] += int(matched)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    out = {"precision": precision, "recall": recall, "f1": f1,
           "tp": tp, "fp": fp, "n_gt": n_gt,
           "conf_thresh": conf_thresh, "iou_thresh": iou_thresh}
    for s in ("small", "medium", "large"):
        out[f"recall_{s}"] = size_tp[s] / max(size_n[s], 1)
        out[f"n_gt_{s}"] = size_n[s]
    return out


def select_operating_point(gt_records: list[dict], dt_records: list[dict],
                           iou_thresh: float = 0.5,
                           grid: np.ndarray | None = None) -> float:
    """Confidence threshold maximizing F1 on (clean) validation data."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1 = pr_at_threshold(gt_records, dt_records, float(t),
                             iou_thresh)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def coco_map(gt_records: list[dict], dt_records: list[dict]) -> dict:
    """mAP@0.5 and mAP@[.5:.95] via pycocotools (single 'person' class)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = to_coco_gt(gt_records)
    name_to_id = {img["file_name"]: img["id"] for img in gt_dict["images"]}
    dt_list = to_coco_dt(dt_records, name_to_id)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        if not dt_list:
            return {"map50": 0.0, "map50_95": 0.0, "ar100": 0.0}
        coco_dt = coco_gt.loadRes(dt_list)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"map50": float(ev.stats[1]), "map50_95": float(ev.stats[0]),
            "ar100": float(ev.stats[8])}


def evaluate(gt_records: list[dict], dt_records: list[dict],
             conf_thresh: float, iou_thresh: float = 0.5) -> dict:
    """Full metric bundle for one (model, dataset, condition) cell."""
    out = pr_at_threshold(gt_records, dt_records, conf_thresh, iou_thresh)
    out.update(coco_map(gt_records, dt_records))
    return out


def save_predictions(dt_records: list[dict], path: str | Path) -> None:
    serializable = [{"image_id": str(r["image_id"]),
                     "boxes": np.asarray(r["boxes"],
                                         np.float32).reshape(-1, 4).tolist(),
                     "scores": np.asarray(r["scores"],
                                          np.float32).reshape(-1).tolist()}
                    for r in dt_records]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_predictions(path: str | Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    return [{"image_id": r["image_id"],
             "boxes": np.asarray(r["boxes"], np.float32).reshape(-1, 4),
             "scores": np.asarray(r["scores"], np.float32)} for r in raw]
