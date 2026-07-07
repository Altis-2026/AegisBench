#!/usr/bin/env python3
"""Phase 2 checkpoint: verify tiling box remaps visually AND numerically.

For N random source images this script:
  1. draws GT on the full image,
  2. draws remapped GT on every tile that contains boxes,
  3. re-projects tile boxes back to full-image coordinates and checks they
     land inside the matching source box (exact for fully contained boxes),
     printing a PASS/FAIL per image.

  python scripts/phase2_visual_check.py \
      --records data/heridal/records/train.json --out results/phase2 --n 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import load_records
from aegisbench.inference import load_rgb
from aegisbench.seeding import stable_seed
from aegisbench.tiling import tile_image, tile_to_full
from aegisbench.visualize import GREEN, YELLOW, draw_boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    args = ap.parse_args()

    records = [r for r in load_records(args.records) if len(r["boxes"])]
    rng = np.random.default_rng(stable_seed("phase2-visual", args.records))
    picks = rng.choice(len(records), size=min(args.n, len(records)),
                       replace=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_pass = True
    for i in picks:
        rec = records[int(i)]
        img = load_rgb(rec["image_path"])
        cv2.imwrite(str(out / f"{rec['image_id']}_full.jpg"),
                    draw_boxes(img, rec["boxes"]))

        tiles, report = tile_image(img, rec["boxes"], rec["width"],
                                   rec["height"], tile_size=args.tile,
                                   overlap=args.overlap)
        ok = not report.violations
        for t in tiles:
            if not len(t.boxes):
                continue
            colors_ok = t.fully_inside
            overlay = draw_boxes(t.image, t.boxes[colors_ok], GREEN)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay = draw_boxes(overlay, t.boxes[~colors_ok], YELLOW)
            cv2.imwrite(str(out / f"{rec['image_id']}__"
                            f"{t.origin_x}_{t.origin_y}.jpg"), overlay)
            # Numeric round-trip: fully-inside boxes must exactly equal
            # their source box after re-projection.
            back = tile_to_full(t.boxes, t.origin_x, t.origin_y)
            for b, src_idx, full in zip(back, t.source_indices,
                                        t.fully_inside):
                if full and not np.allclose(b, rec["boxes"][src_idx],
                                            atol=0.5):
                    ok = False
        n_full = report.boxes_fully_contained_somewhere
        status = "PASS" if ok and n_full == len(rec["boxes"]) else "FAIL"
        all_pass &= status == "PASS"
        print(f"{rec['image_id']}: {len(rec['boxes'])} boxes, "
              f"{n_full} fully contained in >=1 tile, round-trip {status}")

    print("\nGreen boxes = fully contained; yellow = clipped edge copies "
          "(the full copy lives in a neighboring tile).")
    print("PHASE 2 CHECK:", "PASS" if all_pass else
          "FAIL — do not train on these tiles")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
