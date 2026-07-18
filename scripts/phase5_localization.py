#!/usr/bin/env python3
"""Phase 5b: localization-stability table from saved sweep predictions.

Requires phase5_sweep.py to have been run with --pred-dir (so per-condition
predictions were archived). For each model x corruption x severity, this
joins the corrupted-condition predictions against that model's CLEAN
predictions and reports, over survivors detected in BOTH conditions, how
much the box drifted (see evaluation/localization.py).

  python scripts/phase5_localization.py \
      --records data/heridal/records/test.json --dataset heridal \
      --pred-dir results/sweep/preds --sweep-csv results/sweep/master.csv \
      --out results/sweep/localization.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import DEFAULT_CONFIG_PATH, SEVERITIES, CorruptionSuite
from aegisbench.datasets.common import load_records
from aegisbench.evaluation import load_predictions, localization_stability

FIELDS = ["model", "dataset", "family", "corruption", "severity",
          "conf_thresh", "loc_stability_iou", "loc_center_shift",
          "loc_iou_clean", "loc_iou_corrupt", "loc_iou_drop", "n_common"]


def _conf_by_model(sweep_csv: Path, dataset: str) -> dict[str, float]:
    """Recover each model's frozen operating point from the sweep CSV so
    the same threshold is applied here."""
    conf = {}
    if sweep_csv.exists():
        with open(sweep_csv) as f:
            for r in csv.DictReader(f):
                if r["dataset"] == dataset and r.get("conf_thresh"):
                    conf[r["model"]] = float(r["conf_thresh"])
    return conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--sweep-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--default-conf", type=float, default=0.25,
                    help="fallback operating point if not found in the CSV")
    args = ap.parse_args()

    suite = CorruptionSuite(args.config)
    gt = load_records(args.records)
    pred_dir = Path(args.pred_dir)
    conf_by_model = _conf_by_model(Path(args.sweep_csv), args.dataset)

    models = sorted({p.name.split("_")[0] for p in pred_dir.glob("*_clean_*")}
                    or {p.name.split("_")[0] for p in pred_dir.glob("*.json")})
    rows = []
    for model in models:
        clean_path = pred_dir / f"{model}_{args.dataset}_clean_s0.json"
        if not clean_path.exists():
            print(f"[skip] no clean predictions for {model} "
                  f"({clean_path.name})")
            continue
        clean_dt = load_predictions(clean_path)
        conf = conf_by_model.get(model, args.default_conf)
        for corruption in suite.names():
            for sev in SEVERITIES:
                cpath = (pred_dir /
                         f"{model}_{args.dataset}_{corruption}_s{sev}.json")
                if not cpath.exists():
                    continue
                corrupt_dt = load_predictions(cpath)
                m = localization_stability(gt, clean_dt, corrupt_dt, conf)
                rows.append({"model": model, "dataset": args.dataset,
                             "family": suite.family(corruption),
                             "corruption": corruption, "severity": sev,
                             "conf_thresh": conf, **m})
                print(f"{model:12s} {corruption:16s} s{sev} "
                      f"stability_iou={m['loc_stability_iou']:.3f} "
                      f"center_shift={m['loc_center_shift']:.3f} "
                      f"iou_drop={m['loc_iou_drop']:.3f} "
                      f"n={m['n_common']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
