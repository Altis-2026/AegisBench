#!/usr/bin/env python3
"""Phase 1 (HERIDAL): parse the manually downloaded archive, report counts,
carve a validation split from official train, and save canonical records.

Usage:
  python scripts/phase1_prepare_heridal.py \
      --train-images data/heridal/trainImages \
      --test-images data/heridal/testImages \
      [--train-labels DIR --test-labels DIR] --out data/heridal/records

Then run scripts/phase1_visual_check.py and INSPECT the overlays before
continuing (Phase 1 checkpoint).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import save_records, summarize_records
from aegisbench.datasets.heridal import load_split, train_val_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--train-labels", default=None)
    ap.add_argument("--test-labels", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_all = load_split(args.train_images, args.train_labels)
    test = load_split(args.test_images, args.test_labels)
    train, val = train_val_split(train_all, args.val_fraction)

    report = {}
    for name, recs in (("train", train), ("val", val), ("test", test)):
        save_records(recs, out / f"{name}.json")
        report[name] = summarize_records(recs)

    print(json.dumps(report, indent=2))
    (out / "phase1_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nRecords written to {out}/. Sanity expectations: the official "
          "HERIDAL split has roughly 1500-1600 train and ~101 test images; "
          "if your counts are wildly different, the archive layout was not "
          "parsed correctly — fix that BEFORE the visual check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
