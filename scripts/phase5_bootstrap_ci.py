#!/usr/bin/env python3
"""Phase 5c: bootstrap confidence intervals on the sweep's headline metrics.

The master sweep CSV is one number per (model, dataset, corruption,
severity) -- a single point estimate with no sense of how much it would
move on a different sample of the same test set. This resamples the test
set (with replacement, by image) many times per condition and reports a
95% CI on precision/recall/F1 at the model's frozen operating point, using
the exact predictions the sweep already produced.

Requires phase5_sweep.py to have been run with --pred-dir (predictions
must be saved to disk -- this script reruns no inference, it only
resamples already-computed detections).

  python scripts/phase5_bootstrap_ci.py \
      --pred-dir results/sweep/preds --records data/heridal/records/test.json \
      --dataset heridal --sweep-csv results/sweep/master.csv \
      --out results/sweep/ci_heridal.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import load_records
from aegisbench.evaluation import load_predictions, pr_at_threshold
from aegisbench.seeding import stable_seed

FIELDS = ["model", "dataset", "family", "corruption", "severity",
          "conf_thresh", "n_images", "n_boot",
          "recall_mean", "recall_ci_lo", "recall_ci_hi",
          "precision_mean", "precision_ci_lo", "precision_ci_hi",
          "f1_mean", "f1_ci_lo", "f1_ci_hi"]


def bootstrap_one(gt_records: list[dict], dt_records: list[dict],
                  conf_thresh: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(gt_records)
    recall = np.empty(n_boot)
    precision = np.empty(n_boot)
    f1 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_gt = [gt_records[i] for i in idx]
        m = pr_at_threshold(resampled_gt, dt_records, conf_thresh)
        recall[b], precision[b], f1[b] = m["recall"], m["precision"], m["f1"]

    def ci(arr):
        return (float(arr.mean()), float(np.percentile(arr, 2.5)),
               float(np.percentile(arr, 97.5)))

    r_mean, r_lo, r_hi = ci(recall)
    p_mean, p_lo, p_hi = ci(precision)
    f_mean, f_lo, f_hi = ci(f1)
    return {"recall_mean": r_mean, "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
           "precision_mean": p_mean, "precision_ci_lo": p_lo,
           "precision_ci_hi": p_hi,
           "f1_mean": f_mean, "f1_ci_lo": f_lo, "f1_ci_hi": f_hi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--records", required=True,
                    help="test.json for this dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sweep-csv", required=True,
                    help="master.csv, to read each model's frozen "
                         "conf_thresh and the condition list actually run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    gt = load_records(args.records)
    df = pd.read_csv(args.sweep_csv)
    df = df[df["dataset"] == args.dataset]
    pred_dir = Path(args.pred_dir)

    rows = []
    for model, mdf in df.groupby("model"):
        conf_thresh = float(mdf["conf_thresh"].iloc[0])
        for _, r in mdf.iterrows():
            corruption, severity, family = (r["corruption"], r["severity"],
                                            r["family"])
            pred_path = pred_dir / f"{model}_{args.dataset}_{corruption}_s{severity}.json"
            if not pred_path.exists():
                print(f"  skip {model} {corruption} s{severity} "
                     "(no saved predictions)")
                continue
            dt = load_predictions(pred_path)
            seed = stable_seed("bootstrap-ci", model, args.dataset,
                               corruption, severity)
            stats = bootstrap_one(gt, dt, conf_thresh, args.n_boot, seed)
            print(f"  {model} {corruption} s{severity}: "
                 f"recall={stats['recall_mean']:.3f} "
                 f"[{stats['recall_ci_lo']:.3f}, {stats['recall_ci_hi']:.3f}]")
            rows.append({"model": model, "dataset": args.dataset,
                        "family": family, "corruption": corruption,
                        "severity": severity, "conf_thresh": conf_thresh,
                        "n_images": len(gt), "n_boot": args.n_boot, **stats})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDS).to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(rows)} conditions)")
    if not rows:
        print("Nothing to do -- rerun phase5_sweep.py with --pred-dir set "
             "first so predictions are actually saved to disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
