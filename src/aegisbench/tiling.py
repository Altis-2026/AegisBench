"""Overlapping-tile pipeline for large aerial frames (e.g. 4000x3000 HERIDAL
images) with exact bounding-box remapping.

Box rule (documented for the paper):
  * A ground-truth box is assigned to a tile if its intersection with the
    tile covers >= `min_visibility` of the ORIGINAL box area (default 0.30).
    Kept boxes are clipped to tile bounds.
  * Clipped boxes narrower or shorter than `min_size_px` are dropped.
  * With `overlap >= max box side`, every box is FULLY contained in at least
    one tile, so no annotation is ever lost from the dataset as a whole;
    partially visible copies at tile edges are additionally kept (>= 30%
    visible) so detectors learn partial-person evidence. `tile_image`
    verifies the full-containment guarantee per box and reports violations
    instead of silently dropping annotations.

At inference time detections are produced per tile, mapped back to
full-image coordinates, and merged with class-agnostic NMS
(see evaluation/merge.py), so evaluation is always against the ORIGINAL
full-image ground truth — comparable to published full-image results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tile:
    """One tile crop. Boxes are [x1, y1, x2, y2] float in TILE coordinates;
    origin maps them back: full_xy = tile_xy + (ox, oy)."""
    origin_x: int
    origin_y: int
    width: int
    height: int
    boxes: np.ndarray                      # (N, 4) float32, tile coords
    fully_inside: np.ndarray               # (N,) bool — box uncut by edges
    source_indices: np.ndarray             # (N,) int — index into input boxes
    image: np.ndarray | None = None


@dataclass
class TilingReport:
    n_tiles: int = 0
    n_source_boxes: int = 0
    boxes_fully_contained_somewhere: int = 0
    violations: list[int] = field(default_factory=list)  # source box indices


def tile_starts(dim: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, dim) with a final tile flush to the edge."""
    if dim <= tile:
        return [0]
    starts = list(range(0, dim - tile + 1, stride))
    if starts[-1] + tile < dim:
        starts.append(dim - tile)
    return starts


def tile_image(img: np.ndarray | None,
               boxes: np.ndarray,
               img_w: int,
               img_h: int,
               tile_size: int = 1024,
               overlap: int = 256,
               min_visibility: float = 0.30,
               min_size_px: float = 4.0,
               keep_empty: bool = True) -> tuple[list[Tile], TilingReport]:
    """Split an image (or just its geometry, if img is None) into overlapping
    tiles and remap boxes per the rule documented above.

    boxes: (N, 4) [x1, y1, x2, y2] absolute full-image pixel coordinates.
    """
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    stride = tile_size - overlap
    report = TilingReport(n_source_boxes=len(boxes))

    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * \
        np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    contained_somewhere = np.zeros(len(boxes), bool)

    tiles: list[Tile] = []
    for oy in tile_starts(img_h, tile_size, stride):
        for ox in tile_starts(img_w, tile_size, stride):
            tw = min(tile_size, img_w - ox)
            th = min(tile_size, img_h - oy)

            kept, fully, src = [], [], []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                ix1, iy1 = max(x1, ox), max(y1, oy)
                ix2, iy2 = min(x2, ox + tw), min(y2, oy + th)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0 or areas[i] <= 0:
                    continue
                inter = iw * ih
                is_full = bool(np.isclose(inter, areas[i], rtol=1e-4))
                if is_full:
                    contained_somewhere[i] = True
                if inter / areas[i] < min_visibility:
                    continue
                if iw < min_size_px or ih < min_size_px:
                    continue
                kept.append([ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy])
                fully.append(is_full)
                src.append(i)

            if not kept and not keep_empty:
                continue
            crop = None
            if img is not None:
                crop = img[oy:oy + th, ox:ox + tw]
            tiles.append(Tile(
                origin_x=ox, origin_y=oy, width=tw, height=th,
                boxes=np.asarray(kept, np.float32).reshape(-1, 4),
                fully_inside=np.asarray(fully, bool),
                source_indices=np.asarray(src, int),
                image=crop))

    report.n_tiles = len(tiles)
    report.boxes_fully_contained_somewhere = int(contained_somewhere.sum())
    report.violations = [i for i in range(len(boxes))
                         if areas[i] > 0 and not contained_somewhere[i]]
    return tiles, report


def tile_to_full(boxes_tile: np.ndarray, origin_x: int,
                 origin_y: int) -> np.ndarray:
    """Map (N, 4) tile-coordinate boxes back to full-image coordinates."""
    out = np.asarray(boxes_tile, np.float32).reshape(-1, 4).copy()
    out[:, [0, 2]] += origin_x
    out[:, [1, 3]] += origin_y
    return out
