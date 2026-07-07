"""Full-image inference driver shared by Phases 4-6.

Pipeline per image: (optional corruption at FULL resolution, before tiling,
so large-scale atmospheric structure is coherent across tiles) -> overlapping
tiles -> per-tile detection -> map back to full-image coordinates -> NMS
merge. Evaluation downstream is always against original full-image GT.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .evaluation.merge import merge_tile_detections
from .tiling import tile_image


def load_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def infer_records(records: list[dict], detector, *,
                  suite=None, corruption: str | None = None,
                  severity: int | None = None,
                  tile_size: int = 1024, overlap: int = 256,
                  imgsz: int = 1024, conf_floor: float = 0.001,
                  nms_iou: float = 0.5,
                  progress: bool = True) -> list[dict]:
    """Returns dt_records: [{'image_id', 'boxes', 'scores'}] in full-image
    coordinates. `corruption=None` runs the clean condition."""
    out = []
    for i, rec in enumerate(records):
        img = load_rgb(rec["image_path"])
        if corruption is not None:
            img = suite.apply(img, corruption, severity, rec["image_id"])

        tiles, _ = tile_image(img, np.zeros((0, 4)), rec["width"],
                              rec["height"], tile_size=tile_size,
                              overlap=overlap)
        per_tile = []
        for t in tiles:
            pred = detector.predict(t.image, conf=conf_floor, imgsz=imgsz)
            per_tile.append({"origin_x": t.origin_x, "origin_y": t.origin_y,
                             "boxes": pred["boxes"],
                             "scores": pred["scores"]})
        merged = merge_tile_detections(per_tile, iou_thresh=nms_iou)
        out.append({"image_id": rec["image_id"], **merged})
        if progress and (i + 1) % 10 == 0:
            print(f"  inferred {i + 1}/{len(records)} images", flush=True)
    return out
