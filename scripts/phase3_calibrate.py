#!/usr/bin/env python3
"""Phase 3 calibration audit: verify on REAL images that each corruption's
calibration statistic moves monotonically with severity, and log the
measured values (these numbers go into the paper's taxonomy table).

  python scripts/phase3_calibrate.py --records data/heridal/records/test.json \
      --out results/phase3/calibration.csv --n 20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import DEFAULT_CONFIG_PATH, CorruptionSuite
from aegisbench.corruptions.base import to_float
from aegisbench.corruptions.calibration import measure
from aegisbench.datasets.common import load_records
from aegisbench.inference import load_rgb
from aegisbench.seeding import stable_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args()

    suite = CorruptionSuite(args.config)
    records = load_records(args.records)
    rng = np.random.default_rng(stable_seed("phase3-calib", args.records))
    picks = rng.choice(len(records), size=min(args.n, len(records)),
                       replace=False)

    rows, failures = [], []
    for name in suite.names():
        calib = suite.calibration(name)
        stat, direction = calib["stat"], calib["direction"]
        means = {}
        for sev in (1, 2, 3):
            vals = []
            for i in picks:
                rec = records[int(i)]
                clean = to_float(load_rgb(rec["image_path"]))
                corrupted = suite.apply(clean, name, sev, rec["image_id"])
                vals.append(measure(stat, corrupted, clean))
            means[sev] = float(np.mean(vals))
            rows.append({"corruption": name, "severity": sev, "stat": stat,
                         "mean_value": means[sev],
                         "std_value": float(np.std(vals)), "n": len(vals)})
        seq = [means[1], means[2], means[3]]
        mono = all(a < b for a, b in zip(seq, seq[1:])) \
            if direction == "increasing" \
            else all(a > b for a, b in zip(seq, seq[1:]))
        print(f"{name:16s} {stat:20s} {direction:10s} "
              f"s1={seq[0]:.4f} s2={seq[1]:.4f} s3={seq[2]:.4f} "
              f"{'OK' if mono else 'NOT MONOTONIC'}")
        if not mono:
            failures.append(name)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    if failures:
        print(f"FAIL: non-monotonic severity ladders: {failures} — fix the "
              "parameter table before sweeping.")
        return 1
    print("PHASE 3 CALIBRATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
