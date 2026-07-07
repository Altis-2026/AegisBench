#!/usr/bin/env python3
"""Phase 5: the stress-test sweep. Every detector x corruption x severity
on the test split, at each model's FROZEN clean-val operating point.

Appends one row per cell to the master CSV (resumable: already-present
cells are skipped), with git SHA, seed, config hash, and timestamp.

  python scripts/phase5_sweep.py --models configs/sweep_models.yaml \
      --records data/heridal/records --dataset heridal \
      --out results/sweep/master.csv
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import (DEFAULT_CONFIG_PATH, SEVERITIES,
                                    CorruptionSuite)
from aegisbench.datasets.common import load_records
from aegisbench.evaluation import (evaluate, save_predictions,
                                   select_operating_point)
from aegisbench.inference import infer_records
from aegisbench.models import load_detector

FIELDS = ["timestamp", "git_sha", "config_hash", "model", "dataset",
          "family", "corruption", "severity", "conf_thresh", "seed",
          "precision", "recall", "f1", "map50", "map50_95",
          "recall_small", "recall_medium", "recall_large",
          "n_gt", "n_gt_small", "n_gt_medium", "n_gt_large", "tp", "fp"]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def existing_cells(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(r["model"], r["dataset"], r["corruption"], r["severity"])
                for r in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="yaml: [{kind, weights, imgsz}]")
    ap.add_argument("--records", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--pred-dir", default=None,
                    help="if set, raw predictions are archived here")
    args = ap.parse_args()

    suite = CorruptionSuite(args.config)
    with open(args.config, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    with open(args.models) as f:
        model_specs = yaml.safe_load(f)

    rec_dir = Path(args.records)
    val = load_records(rec_dir / "val.json")
    test = load_records(rec_dir / "test.json")
    csv_path = Path(args.out)
    done = existing_cells(csv_path)
    sha = git_sha()

    conditions = [("clean", 0)] + [(c, s) for c in suite.names()
                                   for s in SEVERITIES]
    for spec in model_specs:
        kind = spec["kind"]
        detector = load_detector(kind, spec["weights"])
        imgsz = spec.get("imgsz", 1024)

        print(f"\n=== {kind}: operating point on clean val ===")
        val_dt = infer_records(val, detector, tile_size=args.tile,
                               overlap=args.overlap, imgsz=imgsz)
        conf = select_operating_point(val, val_dt)
        print(f"    conf={conf:.2f} (frozen for all conditions)")

        for corruption, severity in conditions:
            key = (kind, args.dataset, corruption, str(severity))
            if key in done:
                print(f"  skip {corruption} s{severity} (already in CSV)")
                continue
            print(f"  -> {corruption} s{severity}")
            kwargs = {} if corruption == "clean" else {
                "suite": suite, "corruption": corruption,
                "severity": severity}
            dt = infer_records(test, detector, tile_size=args.tile,
                               overlap=args.overlap, imgsz=imgsz, **kwargs)
            if args.pred_dir:
                save_predictions(dt, Path(args.pred_dir) /
                                 f"{kind}_{args.dataset}_{corruption}"
                                 f"_s{severity}.json")
            metrics = evaluate(test, dt, conf)
            row = {**metrics,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "git_sha": sha, "config_hash": config_hash,
                   "model": kind, "dataset": args.dataset,
                   "family": ("clean" if corruption == "clean"
                              else suite.family(corruption)),
                   "corruption": corruption, "severity": severity,
                   "conf_thresh": conf, "seed": suite.global_seed}
            append_row(csv_path, row)
            print(f"     recall={metrics['recall']:.3f} "
                  f"map50={metrics['map50']:.3f}")

    print(f"\nmaster table: {csv_path}")
    print("Next: python scripts/phase5_heatmap.py --csv "
          f"{csv_path} --out results/sweep/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
