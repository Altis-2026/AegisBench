#!/usr/bin/env python3
"""Phase 1 (HERIDAL): parse the manually downloaded archive, report counts,
obtain a validation split, and save canonical records.

Usage (official 2-folder HERIDAL: carves 15% of train as validation):
  python scripts/phase1_prepare_heridal.py \
      --train-images data/heridal/trainImages \
      --test-images data/heridal/testImages \
      [--train-labels DIR --test-labels DIR] --out data/heridal/records

Usage (Roboflow-style export that already has its own train/valid/test):
  python scripts/phase1_prepare_heridal.py \
      --train-images data/heridal_raw/train \
      --val-images data/heridal_raw/valid \
      --test-images data/heridal_raw/test --out data/heridal/records

Roboflow's images and XML labels typically sit together in the same
directory (no separate labels/ subfolder) — load_split() already handles
that layout automatically, no --train-labels/--test-labels needed.

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
    ap.add_argument("--val-images", default=None,
                    help="if the source already provides its own "
                         "validation split (e.g. Roboflow's valid/ "
                         "folder), use it directly instead of carving "
                         "--val-fraction out of train")
    ap.add_argument("--train-labels", default=None)
    ap.add_argument("--test-labels", default=None)
    ap.add_argument("--val-labels", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_all = load_split(args.train_images, args.train_labels)
    test = load_split(args.test_images, args.test_labels)
    if args.val_images:
        train, val = train_all, load_split(args.val_images, args.val_labels)
    else:
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
