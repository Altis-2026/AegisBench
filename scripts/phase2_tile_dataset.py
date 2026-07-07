#!/usr/bin/env python3
"""Phase 2: tile full-resolution records into a YOLO-format training set.

Writes images/ + labels/ per split, plus a tiling report (including any
boxes that violate the full-containment guarantee — should be none when
overlap exceeds the largest person box).

  python scripts/phase2_tile_dataset.py --records data/heridal/records \
      --out data/heridal/tiles --tile 1024 --overlap 256
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import load_records, write_dataset_yaml
from aegisbench.inference import load_rgb
from aegisbench.tiling import tile_image


def tile_split(records, out_dir: Path, tile: int, overlap: int,
               keep_empty_fraction: float) -> dict:
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"n_source_images": len(records), "n_tiles_written": 0,
             "n_tiles_with_boxes": 0, "n_boxes_written": 0,
             "containment_violations": []}
    rng = np.random.default_rng(0)
    for rec in records:
        img = load_rgb(rec["image_path"])
        tiles, report = tile_image(img, rec["boxes"], rec["width"],
                                   rec["height"], tile_size=tile,
                                   overlap=overlap)
        if report.violations:
            stats["containment_violations"].append(
                {"image_id": rec["image_id"], "boxes": report.violations})
        for t in tiles:
            has_boxes = len(t.boxes) > 0
            if not has_boxes and rng.random() > keep_empty_fraction:
                continue
            name = f"{rec['image_id']}__{t.origin_x}_{t.origin_y}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"),
                        cv2.cvtColor(t.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for x1, y1, x2, y2 in t.boxes:
                cx, cy = (x1 + x2) / 2 / t.width, (y1 + y2) / 2 / t.height
                bw, bh = (x2 - x1) / t.width, (y2 - y1) / t.height
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_dir / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            stats["n_tiles_written"] += 1
            stats["n_tiles_with_boxes"] += int(has_boxes)
            stats["n_boxes_written"] += len(t.boxes)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="directory containing train/val/test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--keep-empty-fraction", type=float, default=0.10,
                    help="fraction of person-free tiles kept as negatives "
                         "in TRAIN/VAL (test tiling happens at inference "
                         "time and never drops tiles)")
    args = ap.parse_args()

    rec_dir = Path(args.records)
    out = Path(args.out)
    full_report = {"tile": args.tile, "overlap": args.overlap}
    for split in ("train", "val"):
        records = load_records(rec_dir / f"{split}.json")
        stats = tile_split(records, out / split, args.tile, args.overlap,
                           args.keep_empty_fraction)
        full_report[split] = stats
        print(f"[{split}] {json.dumps(stats)[:400]}")
        if stats["containment_violations"]:
            print(f"  WARNING: {len(stats['containment_violations'])} images "
                  "contain a box larger than the tile overlap — those boxes "
                  "are never fully inside one tile. Increase --overlap or "
                  "accept clipped-only supervision for them.")

    write_dataset_yaml(out / "dataset.yaml", out,
                       "train/images", "val/images")
    (out / "phase2_report.json").write_text(json.dumps(full_report, indent=2))
    print(f"\nWrote {out}/dataset.yaml. Now run phase2_visual_check.py and "
          "INSPECT remapped boxes before training (Phase 2 checkpoint).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
