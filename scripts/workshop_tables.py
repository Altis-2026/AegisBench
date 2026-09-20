#!/usr/bin/env python3
"""Build the three workshop-requested tables from existing CSVs.

Nothing here re-runs training or inference. It only reformats numbers
that are already computed elsewhere, into the column shapes the workshop
prompt asked for, written to results/workshop/ so nothing here can be
confused with the GRSL letter's own committed numbers.

Task 1 (mitigation before/after) and Task 3 (per-condition CI) read from
results/sweep/master_ci.csv and results/sweep/ci_{sard,heridal}.csv --
the same files the GRSL letter is verified against -- once the mitigation
sweep has actually been run and appended there.

Task 2 (multi-seed) reads from results/workshop/master_ci_seeds.csv and
results/workshop/ci_seeds.csv, which do not exist until
configs/sweep_models_sard_seeds.yaml has been run. This script says so
plainly rather than producing an empty or fabricated table.

  python scripts/workshop_tables.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/workshop"

BASELINE_NAME = {"yolo11": "yolo11", "rtdetr": "rtdetr", "fasterrcnn": "fasterrcnn"}
MITIGATED_NAME = {"yolo11": "yolo11_aug_lowlight", "rtdetr": "rtdetr_aug_lowlight",
                  "fasterrcnn": "fasterrcnn_aug_lowlight"}
CORRUPTIONS = ["water_glare", "turbidity_cast", "inundation", "smoke_haze",
              "fire_warm_tint", "rain_streaks", "motion_blur", "low_light",
              "dust_haze"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def task1_mitigation_table():
    """[Model, Dataset, Condition, Recall_before, Recall_after, Delta]."""
    ci_sard = {(r["model"], r["corruption"], r["severity"]): r
              for r in read_csv(ROOT / "results/sweep/ci_sard.csv")}
    if not any(k[0] in MITIGATED_NAME.values() for k in ci_sard):
        print("Task 1: no mitigated-model rows found in "
              "results/sweep/ci_sard.csv yet. Run the mitigation sweep "
              "(configs/sweep_models_sard_mitigation.yaml) first -- this "
              "is the same run GRSL needs, nothing extra to do for it.")
        return []

    rows = []
    for base, mit in zip(BASELINE_NAME.values(), MITIGATED_NAME.values()):
        conditions = [("clean", "0")] + [(c, s) for c in CORRUPTIONS for s in "123"]
        for corr, sev in conditions:
            before = ci_sard.get((base, corr, sev))
            after = ci_sard.get((mit, corr, sev))
            if not (before and after):
                continue
            b = float(before["recall_mean"])
            a = float(after["recall_mean"])
            rows.append({
                "Model": base, "Dataset": "sard",
                "Condition": "clean" if corr == "clean" else f"{corr}_s{sev}",
                "Recall_before": round(b, 4), "Recall_after": round(a, 4),
                "Delta": round(a - b, 4),
                "Recall_after_CI_lo": round(float(after["recall_ci_lo"]), 4),
                "Recall_after_CI_hi": round(float(after["recall_ci_hi"]), 4),
            })
    return rows


def task2_multiseed_table():
    """Recall at low_light severity 3, one row per seed."""
    ci_path = OUT / "ci_seeds.csv"
    ci = read_csv(ci_path)
    if not ci:
        print(f"Task 2: {ci_path} does not exist yet. Run "
              "configs/train_yolo11_sard_seed{42,123}.yaml, then the sweep "
              "at configs/sweep_models_sard_seeds.yaml with "
              "--out results/workshop/master_ci_seeds.csv and bootstrap CI "
              "into --out results/workshop/ci_seeds.csv, matching the "
              "column shape of results/sweep/ci_sard.csv.")
        return []
    rows = []
    for r in ci:
        if r["corruption"] == "low_light" and r["severity"] == "3":
            rows.append({
                "Seed": r["model"].replace("yolo11_seed", ""),
                "Recall": round(float(r["recall_mean"]), 4),
                "CI_lo": round(float(r["recall_ci_lo"]), 4),
                "CI_hi": round(float(r["recall_ci_hi"]), 4),
            })
    return rows


def task3_consolidated_ci():
    """[Model, Dataset, Corruption, Severity, Recall, CI_low, CI_high]
    for all 168 conditions. Already fully computed -- this just merges
    the two existing per-dataset files into one."""
    rows = []
    for dataset in ("heridal", "sard"):
        for r in read_csv(ROOT / f"results/sweep/ci_{dataset}.csv"):
            rows.append({
                "Model": r["model"], "Dataset": dataset,
                "Corruption": r["corruption"], "Severity": r["severity"],
                "Recall": round(float(r["recall_mean"]), 4),
                "CI_low": round(float(r["recall_ci_lo"]), 4),
                "CI_high": round(float(r["recall_ci_hi"]), 4),
            })
    return rows


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    t1 = task1_mitigation_table()
    if t1:
        write_csv(t1, OUT / "task1_mitigation.csv",
                  ["Model", "Dataset", "Condition", "Recall_before",
                   "Recall_after", "Delta", "Recall_after_CI_lo",
                   "Recall_after_CI_hi"])

    t2 = task2_multiseed_table()
    if t2:
        write_csv(t2, OUT / "task2_multiseed.csv", ["Seed", "Recall", "CI_lo", "CI_hi"])

    t3 = task3_consolidated_ci()
    write_csv(t3, OUT / "task3_all_conditions_ci.csv",
             ["Model", "Dataset", "Corruption", "Severity", "Recall",
              "CI_low", "CI_high"])
    if len(t3) != 168:
        print(f"NOTE: {len(t3)} rows, not 168 -- some conditions are "
              "missing CI. Check results/sweep/ci_heridal.csv and "
              "ci_sard.csv directly before trusting this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
