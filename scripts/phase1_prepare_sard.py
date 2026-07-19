#!/usr/bin/env python3
"""Phase 1 (SARD): parse, GROUP-AWARE split (frames of one video sequence
never straddle splits), report counts, save canonical records.

Two input modes:
  --images DIR [--labels DIR]   a flat VOC-XML SARD release (e.g. the
                                 original IEEE DataPort distribution).
  --roboflow-root DIR            a Roboflow-style YOLO export laid out as
                                 DIR/{train,valid,test}/{images,labels}/
                                 (common for Kaggle/Roboflow SARD mirrors).
                                 All splits are POOLED and re-split by our
                                 own group_split() — Roboflow's own split
                                 is a per-frame shuffle, not video-aware,
                                 and is discarded rather than trusted.

Usage:
  python scripts/phase1_prepare_sard.py --images data/sard/images \
      [--labels data/sard/labels] --out data/sard/records \
      [--group-regex '^(.*?)[-_]?\\d+$']
  python scripts/phase1_prepare_sard.py \
      --roboflow-root data/sard_raw/search-and-rescue --out data/sard/records

REVIEW THE PRINTED GROUP TABLE. If groups do not correspond to video
sequences (one group per file, or one giant group), the default regex does
not fit your SARD filenames — adjust --group-regex until it does. This is
the difference between a defensible split and silent train/test leakage.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import save_records, summarize_records
from aegisbench.datasets.sard import (DEFAULT_GROUP_REGEX, group_split,
                                      load_all, load_pooled_roboflow_yolo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--roboflow-root", default=None,
                    help="root of a train/valid/test Roboflow YOLO export; "
                         "mutually exclusive with --images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    args = ap.parse_args()

    if bool(args.images) == bool(args.roboflow_root):
        raise SystemExit("pass exactly one of --images or --roboflow-root")

    if args.roboflow_root:
        records = load_pooled_roboflow_yolo(args.roboflow_root)
    else:
        records = load_all(args.images, args.labels)
    print(f"loaded {len(records)} labeled images")
    train, val, test, info = group_split(records, regex=args.group_regex)

    print(f"groups discovered: {info['n_groups']}")
    for k, n in sorted(info["group_sizes"].items()):
        print(f"  {k:40s} {n:5d} frames")
    print(f"split sizes: {info['split_sizes']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"groups": info}
    for name, recs in (("train", train), ("val", val), ("test", test)):
        save_records(recs, out / f"{name}.json")
        report[name] = summarize_records(recs)
    print(json.dumps({k: v for k, v in report.items() if k != "groups"},
                     indent=2))
    (out / "phase1_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
