#!/usr/bin/env python3
"""Build the released per-condition result tables from existing CSVs.

Nothing here re-runs training or inference. It reformats numbers that are
already computed elsewhere into the shape a reader of the paper needs,
written to results/workshop/ so nothing here can be confused with the raw
sweep records in results/sweep/.

Two rules this script follows, because the paper depends on them:

  * Recall, precision and F1 are the POINT ESTIMATES from the master
    sweep record (the `recall` column), not the bootstrap means. The
    bootstrap files contribute only the interval endpoints. Mixing the
    two is how a table ends up half a thousandth off from its own source,
    which is exactly the kind of drift the verifier exists to catch.
  * If an input is missing, say so and emit nothing for that table,
    rather than emitting a short or empty one that looks complete.

  python scripts/workshop_tables.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/workshop"

BASELINES = ["fasterrcnn", "rtdetr", "yolo11"]
MITIGATED = {"fasterrcnn": "fasterrcnn_aug_lowlight",
             "rtdetr": "rtdetr_aug_lowlight",
             "yolo11": "yolo11_aug_lowlight"}
SEEDS = {"17": "yolo11_seed17", "42": "yolo11_seed42", "123": "yolo11_seed123"}
CORRUPTIONS = ["water_glare", "turbidity_cast", "inundation", "smoke_haze",
               "fire_warm_tint", "rain_streaks", "motion_blur", "low_light",
               "dust_haze"]
CONDITIONS = [("clean", "0")] + [(c, s) for c in CORRUPTIONS for s in "123"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def index(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def f(row, field, nd=4):
    return round(float(row[field]), nd)


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):4d} rows -> {path.relative_to(ROOT)}")


# ------------------------------------------------------------------ 1

BENCH_FIELDS = ["model", "dataset", "family", "corruption", "severity",
                "conf_thresh", "n_images", "n_gt", "tp", "fp",
                "recall", "recall_ci_lo", "recall_ci_hi",
                "precision", "f1", "map50", "map50_95"]


def benchmark_table(point, ci):
    """The 168 headline conditions: 3 models x 2 datasets x 28 conditions."""
    rows, missing = [], 0
    for dataset in ("heridal", "sard"):
        for model in BASELINES:
            for corr, sev in CONDITIONS:
                p = point.get((model, dataset, corr, sev))
                c = ci.get((model, dataset, corr, sev))
                if not (p and c):
                    missing += 1
                    continue
                rows.append({
                    "model": model, "dataset": dataset, "family": p["family"],
                    "corruption": corr, "severity": sev,
                    "conf_thresh": p["conf_thresh"], "n_images": c["n_images"],
                    "n_gt": p["n_gt"], "tp": p["tp"], "fp": p["fp"],
                    "recall": f(p, "recall"),
                    "recall_ci_lo": f(c, "recall_ci_lo"),
                    "recall_ci_hi": f(c, "recall_ci_hi"),
                    "precision": f(p, "precision"), "f1": f(p, "f1"),
                    "map50": f(p, "map50"), "map50_95": f(p, "map50_95"),
                })
    if missing:
        print(f"  NOTE: {missing} of 168 conditions missing from the source "
              "files; the table below is incomplete. Check "
              "results/sweep/master_ci.csv and ci_{heridal,sard}.csv.")
    return rows


# ------------------------------------------------------------------ 2

MIT_FIELDS = ["model", "dataset", "corruption", "severity",
              "recall_before", "recall_after", "delta",
              "recall_after_ci_lo", "recall_after_ci_hi",
              "conf_thresh_before", "conf_thresh_after", "trained_on"]


def mitigation_table(point, ci):
    """Before/after for the low-light augmentation arms, all 28 conditions.

    `trained_on` marks which rows the augmentation actually saw, so a
    reader can tell the generalisation result (low_light severity 3) from
    the in-distribution ones without consulting the paper.
    """
    if not any((m, "sard", "clean", "0") in point for m in MITIGATED.values()):
        print("mitigation: no *_aug_lowlight rows in "
              "results/sweep/master_ci.csv yet. Run "
              "configs/sweep_models_sard_mitigation.yaml first.")
        return []
    rows = []
    for base, mit in MITIGATED.items():
        thr_b = point[(base, "sard", "clean", "0")]["conf_thresh"]
        thr_a = point[(mit, "sard", "clean", "0")]["conf_thresh"]
        for corr, sev in CONDITIONS:
            b = point.get((base, "sard", corr, sev))
            a = point.get((mit, "sard", corr, sev))
            c = ci.get((mit, "sard", corr, sev))
            if not (b and a and c):
                continue
            rb, ra = float(b["recall"]), float(a["recall"])
            rows.append({
                "model": base, "dataset": "sard", "corruption": corr,
                "severity": sev, "recall_before": round(rb, 4),
                "recall_after": round(ra, 4), "delta": round(ra - rb, 4),
                "recall_after_ci_lo": f(c, "recall_ci_lo"),
                "recall_after_ci_hi": f(c, "recall_ci_hi"),
                "conf_thresh_before": thr_b, "conf_thresh_after": thr_a,
                "trained_on": "yes" if (corr == "low_light" and sev in "12") else "no",
            })
    return rows


# ------------------------------------------------------------------ 3

SEED_FIELDS = ["seed", "model", "dataset", "corruption", "severity",
               "conf_thresh", "recall", "recall_ci_lo", "recall_ci_hi",
               "precision", "f1"]


def multiseed_table():
    """All 28 conditions for each of the three SARD YOLOv11 seeds.

    The paper quotes only the low_light rows, but the full grid is what
    lets a reader check that the seeds are comparable everywhere else and
    not only where we looked.
    """
    point = index(read_csv(OUT / "master_ci_seeds.csv"),
                  "model", "corruption", "severity")
    ci = index(read_csv(OUT / "ci_seeds.csv"), "model", "corruption", "severity")
    if not point or not ci:
        print("multi-seed: results/workshop/{master_ci_seeds,ci_seeds}.csv "
              "not found. Train configs/train_yolo11_sard_seed{42,123}.yaml, "
              "then run configs/sweep_models_sard_seeds.yaml.")
        return []
    rows = []
    for seed, model in SEEDS.items():
        for corr, sev in CONDITIONS:
            p, c = point.get((model, corr, sev)), ci.get((model, corr, sev))
            if not (p and c):
                continue
            rows.append({
                "seed": seed, "model": "yolo11", "dataset": "sard",
                "corruption": corr, "severity": sev,
                "conf_thresh": p["conf_thresh"], "recall": f(p, "recall"),
                "recall_ci_lo": f(c, "recall_ci_lo"),
                "recall_ci_hi": f(c, "recall_ci_hi"),
                "precision": f(p, "precision"), "f1": f(p, "f1"),
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    point = index(read_csv(ROOT / "results/sweep/master_ci.csv"),
                  "model", "dataset", "corruption", "severity")
    ci = {}
    for name in ("ci_heridal.csv", "ci_sard.csv"):
        ci.update(index(read_csv(ROOT / "results/sweep" / name),
                        "model", "dataset", "corruption", "severity"))
    if not point:
        print("results/sweep/master_ci.csv not found; nothing to do.")
        return 1

    bench = benchmark_table(point, ci)
    write_csv(bench, OUT / "benchmark_168_conditions.csv", BENCH_FIELDS)
    if len(bench) != 168:
        print(f"  WARNING: {len(bench)} rows, not 168.")

    mit = mitigation_table(point, ci)
    if mit:
        write_csv(mit, OUT / "mitigation_sard_lowlight.csv", MIT_FIELDS)

    seeds = multiseed_table()
    if seeds:
        write_csv(seeds, OUT / "multiseed_sard_yolo11.csv", SEED_FIELDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
