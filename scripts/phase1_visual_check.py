#!/usr/bin/env python3
"""Phase 1 checkpoint: draw GT boxes on N sample images for eyeball
verification that labels line up with people.

  python scripts/phase1_visual_check.py --records data/heridal/records/train.json \
      --out results/phase1/heridal_train --n 5
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import load_records
from aegisbench.inference import load_rgb
from aegisbench.seeding import stable_seed
from aegisbench.visualize import draw_boxes

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    records = load_records(args.records)
    with_boxes = [r for r in records if len(r["boxes"])]
    rng = np.random.default_rng(stable_seed("phase1-visual", args.records))
    picks = rng.choice(len(with_boxes), size=min(args.n, len(with_boxes)),
                       replace=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in picks:
        rec = with_boxes[int(i)]
        img = load_rgb(rec["image_path"])
        overlay = draw_boxes(img, rec["boxes"])
        path = out / f"{rec['image_id']}_gt.jpg"
        cv2.imwrite(str(path), overlay)
        print(f"wrote {path}  ({len(rec['boxes'])} boxes)")
    print("\nCHECKPOINT: open these files and confirm every box sits on a "
          "person before moving to Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
