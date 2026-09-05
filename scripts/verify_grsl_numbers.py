#!/usr/bin/env python3
"""Check every number in the GRSL letter against the results CSVs.

The letter's central claim is that its numbers are traceable, so this
parses the tables straight out of the .tex and re-derives each value from
results/. A mismatch anywhere is a failure, not a warning.

  python scripts/verify_grsl_numbers.py
"""

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/grsl/aegisbench_grsl.tex"

MODEL_KEY = {"Faster R-CNN": "fasterrcnn", "RT-DETR": "rtdetr", "YOLOv11": "yolo11"}
failures = []
checks = 0


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check(label, got, want, tol=0.0006):
    global checks
    checks += 1
    if got is None or abs(got - want) > tol:
        failures.append(f"{label}: tex={got} source={want}")


def load():
    point = {(r["model"], r["dataset"], r["corruption"], r["severity"]): r
             for r in read_csv(ROOT / "results/sweep/master_ci.csv")}
    ci = {}
    for name in ("ci_heridal.csv", "ci_sard.csv"):
        for r in read_csv(ROOT / "results/sweep" / name):
            ci[(r["model"], r["dataset"], r["corruption"], r["severity"])] = r
    calib = {(r["corruption"], r["severity"]): r
             for r in read_csv(ROOT / "results/phase3/calibration.csv")}
    return point, ci, calib


def table_body(tex, label):
    """Rows of the tabular whose \\label is `label`."""
    start = tex.index(f"\\label{{{label}}}")
    body = tex[start:tex.index("\\end{tabular}", start)]
    body = body[body.index("\\toprule"):]
    return [ln.strip() for ln in body.split("\\\\") if ln.strip()]


def cells(row):
    row = re.sub(r"\\(midrule|toprule|bottomrule|cmidrule)(\(lr\))?(\{[^}]*\})?", "", row)
    return [c.strip() for c in row.split("&")]


def clean_name(cell):
    m = re.search(r"\\texttt\{([a-z\\_]+)\}", cell)
    if m:
        return m.group(1).replace("\\_", "_")
    return re.sub(r"\\(emph|textbf)\{|\}|\\", "", cell).strip()


def num(cell):
    cell = re.sub(r"\\textbf\{|\}", "", cell)
    m = re.match(r"^\s*(-?[\d.]+)", cell.replace("$-$", "-").replace("$", ""))
    return float(m.group(1)) if m else None


def interval(cell):
    m = re.search(r"\[([\d.]+),\s*([\d.]+)\]", cell)
    return (float(m.group(1)), float(m.group(2))) if m else None


def main():
    global checks
    tex = TEX.read_text()
    point, ci, calib = load()

    # ---- Table I: calibration statistics -------------------------------
    for row in table_body(tex, "tab:taxonomy"):
        c = cells(row)
        if len(c) < 7 or "Corruption" in c[0]:
            continue
        name = clean_name(c[0])
        if name == "rain_streaks":
            continue  # declared by generation parameters, not a measured stat
        for sev, cell in zip("123", c[4:7]):
            want = float(calib[(name, sev)]["mean_value"])
            check(f"Table I {name} s{sev}", num(cell), want)

    # ---- Table I: derived optical depth and exposure stops -------------
    cfg = (ROOT / "configs/corruptions.yaml").read_text()
    for name, shown in (("smoke_haze", [0.36, 0.80, 1.39]),
                        ("dust_haze", [0.43, 0.87, 1.43])):
        blk = cfg.split(f"  {name}:", 1)[1].split("\n    fixed:", 1)[0]
        ts = [float(m) for m in re.findall(r"t_mean:\s*([\d.]+)", blk)]
        for sev, (t, s) in enumerate(zip(ts, shown), 1):
            check(f"Table I {name} tau s{sev}", s, -math.log(t), tol=0.005)

    blk = cfg.split("  low_light:", 1)[1].split("\n    fixed:", 1)[0]
    gains = [float(m) for m in re.findall(r"linear_gain:\s*([\d.]+)", blk)]
    for sev, (g, s) in enumerate(zip(gains, [-1.6, -2.9, -4.5]), 1):
        check(f"Table I low_light stops s{sev}", s, math.log2(g), tol=0.05)

    # ---- Table II: severity-3 recall, all detectors and datasets -------
    for row in table_body(tex, "tab:severity3"):
        c = cells(row)
        if len(c) < 8 or "Corruption" in c[0] or "HERIDAL" in c[0]:
            continue
        name = clean_name(c[0])
        corr, sev = ("clean", "0") if name == "clear condition" else (name, "3")
        for i, (ds, model) in enumerate(
                [("heridal", "fasterrcnn"), ("heridal", "rtdetr"), ("heridal", "yolo11"),
                 ("sard", "fasterrcnn"), ("sard", "rtdetr"), ("sard", "yolo11")]):
            want = float(point[(model, ds, corr, sev)]["recall"])
            check(f"Table II {name} {ds}/{model}", num(c[2 + i]), want)

    # ---- Table III: low_light recall, intervals, and mAP50 -------------
    for row in table_body(tex, "tab:lowlight"):
        c = cells(row)
        if len(c) < 9 or c[0].startswith("Model") or "Recall" in c[0]:
            continue
        model = MODEL_KEY.get(clean_name(c[0]))
        if model is None:
            continue
        ds = c[1].strip().lower()
        check(f"Table III {model}/{ds} threshold",
              num(c[2]), float(point[(model, ds, "low_light", "1")]["conf_thresh"]))
        for sev, rcell, mcell in zip("123", c[3:6], c[6:9]):
            key = (model, ds, "low_light", sev)
            check(f"Table III recall {model}/{ds} s{sev}",
                  num(rcell), float(point[key]["recall"]))
            lo, hi = interval(rcell)
            check(f"Table III CI-lo {model}/{ds} s{sev}",
                  lo, float(ci[key]["recall_ci_lo"]))
            check(f"Table III CI-hi {model}/{ds} s{sev}",
                  hi, float(ci[key]["recall_ci_hi"]))
            check(f"Table III mAP {model}/{ds} s{sev}",
                  num(mcell), float(point[key]["map50"]))

    # ---- Prose claims --------------------------------------------------
    def rec(m, d, c, s):
        return float(point[(m, d, c, s)]["recall"])

    clean_y = rec("yolo11", "sard", "clean", "0")
    check("prose: rain cost on SARD (3.6 points)",
          3.6, 100 * (clean_y - rec("yolo11", "sard", "rain_streaks", "3")), tol=0.06)
    check("prose: smoke/dust ratio (6.7-fold)", 6.7,
          rec("yolo11", "sard", "smoke_haze", "3") / rec("yolo11", "sard", "dust_haze", "3"),
          tol=0.05)

    # every low_light severity-3 recall is exactly zero
    for d in ("heridal", "sard"):
        for m in MODEL_KEY.values():
            check(f"prose: {m}/{d} zero at s3", rec(m, d, "low_light", "3"), 0.0, tol=0)

    # mAP50 at severity 3 is at most 0.010 everywhere
    worst = max(float(point[(m, d, "low_light", "3")]["map50"])
                for d in ("heridal", "sard") for m in MODEL_KEY.values())
    check("prose: max mAP50 at s3 (0.010)", 0.010, worst, tol=0.0005)

    # localization stability band and n_common for low_light
    loc = [r for r in read_csv(ROOT / "results/sweep/localization_sard.csv")
           if r["model"] == "yolo11" and r["loc_stability_iou"] != "nan"]
    stab = [float(r["loc_stability_iou"]) for r in loc]
    check("prose: min stability 0.779", 0.779, min(stab))
    check("prose: max stability 0.984", 0.984, max(stab))
    ncommon = [r["n_common"] for r in read_csv(ROOT / "results/sweep/localization_sard.csv")
               if r["model"] == "yolo11" and r["corruption"] == "low_light"]
    if ncommon != ["202", "8", "0"]:
        failures.append(f"prose: n_common {ncommon} != ['202','8','0']")
    checks += 1

    # clear-condition precision and mAP ranges quoted in Section IV-A
    for ds, plo, phi, mlo, mhi in [("heridal", 0.855, 0.903, 0.785, 0.835),
                                   ("sard", 0.955, 0.972, 0.896, 0.929)]:
        ps = [float(point[(m, ds, "clean", "0")]["precision"]) for m in MODEL_KEY.values()]
        ms = [float(point[(m, ds, "clean", "0")]["map50"]) for m in MODEL_KEY.values()]
        check(f"prose: {ds} precision low", plo, min(ps))
        check(f"prose: {ds} precision high", phi, max(ps))
        check(f"prose: {ds} mAP low", mlo, min(ms))
        check(f"prose: {ds} mAP high", mhi, max(ms))

    # test-set sizes quoted in the table headers
    for ds, n in (("heridal", 101), ("sard", 862)):
        check(f"n_images {ds}", float(ci[("yolo11", ds, "clean", "0")]["n_images"]), n)

    print(f"{checks} checks run")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print("  ", f)
        return 1
    print("all numbers in the letter match their source files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
