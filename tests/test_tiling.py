"""Tiling is the highest-risk silent bug in the pipeline: these tests pin
down the box-remapping rule exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.tiling import tile_image, tile_starts, tile_to_full


def test_tile_starts_cover_and_flush():
    starts = tile_starts(4000, 1024, 768)
    assert starts[0] == 0
    assert starts[-1] == 4000 - 1024          # flush to the edge
    covered = np.zeros(4000, bool)
    for s in starts:
        covered[s:s + 1024] = True
    assert covered.all()


def test_small_image_single_tile():
    starts = tile_starts(800, 1024, 768)
    assert starts == [0]


def test_full_containment_guarantee():
    """With overlap >= max box side, every box is fully inside >= 1 tile."""
    rng = np.random.default_rng(7)
    w, h, side = 4000, 3000, 60          # 60 px persons << 256 px overlap
    boxes = []
    for _ in range(200):
        x1 = rng.uniform(0, w - side)
        y1 = rng.uniform(0, h - side)
        boxes.append([x1, y1, x1 + rng.uniform(10, side),
                      y1 + rng.uniform(10, side)])
    _, report = tile_image(None, np.array(boxes), w, h,
                           tile_size=1024, overlap=256)
    assert report.violations == []
    assert report.boxes_fully_contained_somewhere == len(boxes)


def test_roundtrip_exact_for_fully_contained():
    boxes = np.array([[1500.0, 1100.0, 1550.0, 1160.0]])
    tiles, _ = tile_image(None, boxes, 4000, 3000,
                          tile_size=1024, overlap=256)
    found_exact = False
    for t in tiles:
        for b, full in zip(t.boxes, t.fully_inside):
            back = tile_to_full(b[None], t.origin_x, t.origin_y)[0]
            if full:
                assert np.allclose(back, boxes[0], atol=1e-3)
                found_exact = True
            # Clipped copies must lie inside the source box.
            assert back[0] >= boxes[0][0] - 1e-3
            assert back[1] >= boxes[0][1] - 1e-3
            assert back[2] <= boxes[0][2] + 1e-3
            assert back[3] <= boxes[0][3] + 1e-3
    assert found_exact


def test_visibility_rule_drops_slivers():
    """A box with <30% of its area in a tile must not appear in that tile."""
    # Tile grid for 2048x2048, tile 1024, overlap 256 -> x starts 0, 768,
    # 1024. Box straddles x=1024 with 20% inside the left [0,1024) tile.
    box = np.array([[1004.0, 100.0, 1104.0, 200.0]])  # 20 px of 100 inside
    tiles, report = tile_image(None, box, 2048, 2048,
                               tile_size=1024, overlap=256)
    for t in tiles:
        if t.origin_x == 0 and t.origin_y == 0:
            assert len(t.boxes) == 0        # 20% < 30% visibility
    assert report.violations == []          # fully inside the x=768 tile


def test_min_size_rule():
    box = np.array([[1021.0, 100.0, 1080.0, 103.0]])   # 3 px tall
    tiles, _ = tile_image(None, box, 2048, 2048,
                          tile_size=1024, overlap=256, min_size_px=4)
    for t in tiles:
        for b in t.boxes:
            assert b[3] - b[1] >= 4 or b[2] - b[0] >= 4


def test_image_crop_matches_geometry():
    img = np.zeros((3000, 4000, 3), np.uint8)
    img[1100:1160, 1500:1550] = 255       # white block = the "person"
    boxes = np.array([[1500.0, 1100.0, 1550.0, 1160.0]])
    tiles, _ = tile_image(img, boxes, 4000, 3000,
                          tile_size=1024, overlap=256)
    for t in tiles:
        for b, full in zip(t.boxes, t.fully_inside):
            if not full:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            patch = t.image[y1:y2, x1:x2]
            assert patch.mean() > 250      # remapped box lands on the block


def test_overlap_must_be_smaller_than_tile():
    with pytest.raises(ValueError):
        tile_image(None, np.zeros((0, 4)), 100, 100,
                   tile_size=512, overlap=512)
