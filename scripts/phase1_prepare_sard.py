#!/usr/bin/env python3
"""Phase 1 (SARD): parse, GROUP-AWARE split (frames of one video sequence
never straddle splits), report counts, save canonical records.

Usage:
  python scripts/phase1_prepare_sard.py --images data/sard/images \
      [--labels data/sard/labels] --out data/sard/records \
      [--group-regex '^(.*?)[-_]?\\d+$']

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
from aegisbench.datasets.sard import DEFAULT_GROUP_REGEX, group_split, load_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    args = ap.parse_args()

    records = load_all(args.images, args.labels)
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
