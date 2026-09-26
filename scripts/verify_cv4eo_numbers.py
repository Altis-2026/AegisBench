#!/usr/bin/env python3
"""Check every number in the CV4EO paper against the results CSVs.

Same contract as scripts/verify_grsl_numbers.py: the paper claims its
numbers are traceable, so this parses the tables and the inline claims
straight out of the .tex sources and re-derives each value from
results/. A mismatch anywhere is a failure, not a warning.

Recall is always compared against the point estimate in the master
record (the `recall` column), never against the bootstrap mean, so the
paper reports what was measured and the interval separately. Interval
endpoints are compared against the bootstrap files.

  python scripts/verify_cv4eo_numbers.py
"""

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "paper/cv4eo/sec"

MODEL_KEY = {"Faster R-CNN": "fasterrcnn", "F.\\ R-CNN": "fasterrcnn",
             "RT-DETR": "rtdetr", "YOLOv11": "yolo11"}
MITIGATED = {"fasterrcnn": "fasterrcnn_aug_lowlight",
             "rtdetr": "rtdetr_aug_lowlight",
             "yolo11": "yolo11_aug_lowlight"}
FAMILY = {"flood": ["water_glare", "turbidity_cast", "inundation"],
          "wildfire": ["smoke_haze", "fire_warm_tint"],
          "storm": ["rain_streaks", "motion_blur", "low_light"],
          "earthquake": ["dust_haze"]}
OTHER8 = ["water_glare", "turbidity_cast", "inundation", "smoke_haze",
          "fire_warm_tint", "rain_streaks", "motion_blur", "dust_haze"]

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
    seed_point = {(r["model"], r["corruption"], r["severity"]): r
                  for r in read_csv(ROOT / "results/workshop/master_ci_seeds.csv")}
    seed_ci = {(r["model"], r["corruption"], r["severity"]): r
               for r in read_csv(ROOT / "results/workshop/ci_seeds.csv")}
    return point, ci, seed_point, seed_ci


def tex(name):
    return (SEC / name).read_text()


def table_body(source, label):
    """Rows of the tabular whose \\label is `label`."""
    start = source.index(f"\\label{{{label}}}")
    body = source[start:source.index("\\end{tabular}", start)]
    body = body[body.index("\\toprule"):]
    return [ln.strip() for ln in body.split("\\\\") if ln.strip()]


def cells(row):
    row = re.sub(r"\\(midrule|toprule|bottomrule|cmidrule)(\(lr\))?(\{[^}]*\})?",
                 "", row)
    return [c.strip() for c in row.split("&")]


def num(cell):
    """The single numeric value in a table cell, or None."""
    cell = cell.replace("\\textbf{", "").replace("}", "")
    m = re.fullmatch(r"\$?([+-]?\d*\.?\d+)\$?", cell.strip())
    return float(m.group(1)) if m else None


def interval(cell):
    """(lo, hi) from a \\ci{lo}{hi} cell, or None."""
    m = re.search(r"\\ci\{([\d.]+)\}\{([\d.]+)\}", cell)
    return (float(m.group(1)), float(m.group(2))) if m else None


# ---------------------------------------------------------------- tables

def check_spectrum(point):
    """tab:spectrum -- YOLOv11 on SARD, every corruption and severity."""
    for row in table_body(tex("4_results.tex"), "tab:spectrum"):
        c = cells(row)
        m = re.search(r"\\cfd\{([a-z\\_]+)\}", c[0])
        if not m:
            continue
        corr = m.group(1).replace("\\_", "_")
        for i, sev in enumerate("123", start=2):
            got = num(c[i])
            want = float(point[("yolo11", "sard", corr, sev)]["recall"])
            check(f"tab:spectrum {corr} s{sev}", got, want)


def check_lowlight(point):
    """tab:lowlight -- clean and low_light s1-s3, both datasets."""
    dataset = None
    for row in table_body(tex("4_results.tex"), "tab:lowlight"):
        if "HERIDAL" in row:
            dataset = "heridal"
            continue
        if "SARD" in row:
            dataset = "sard"
            continue
        c = cells(row)
        if len(c) < 6 or c[0] not in MODEL_KEY:
            continue
        model = MODEL_KEY[c[0]]
        check(f"tab:lowlight {model}/{dataset} thr", num(c[1]),
              float(point[(model, dataset, "clean", "0")]["conf_thresh"]),
              tol=0.005)
        check(f"tab:lowlight {model}/{dataset} clean", num(c[2]),
              float(point[(model, dataset, "clean", "0")]["recall"]))
        for i, sev in enumerate("123", start=3):
            check(f"tab:lowlight {model}/{dataset} s{sev}", num(c[i]),
                  float(point[(model, dataset, "low_light", sev)]["recall"]))


def check_family(point):
    """tab:family -- mean relative recall drop per disaster family."""
    dataset = None
    for row in table_body(tex("4_results.tex"), "tab:family"):
        if "HERIDAL" in row:
            dataset = "heridal"
            continue
        if "SARD" in row:
            dataset = "sard"
            continue
        c = cells(row)
        fam = c[0].strip().lower()
        if fam not in FAMILY:
            continue
        for col, model in enumerate(("fasterrcnn", "rtdetr", "yolo11"), start=1):
            clean = float(point[(model, dataset, "clean", "0")]["recall"])
            drops = [(clean - float(point[(model, dataset, corr, s)]["recall"])) / clean
                     for corr in FAMILY[fam] for s in "123"]
            check(f"tab:family {dataset}/{fam}/{model}", num(c[col]),
                  statistics.mean(drops))


def check_seeds(seed_point, seed_ci):
    """tab:seeds -- three seeds, YOLOv11 on SARD under low_light."""
    models = ["yolo11_seed17", "yolo11_seed42", "yolo11_seed123"]
    rows = table_body(tex("4_results.tex"), "tab:seeds")
    pending = None
    for row in rows:
        c = cells(row)
        head = c[0].strip()
        if head == "Threshold":
            for i, m in enumerate(models, start=1):
                check(f"tab:seeds {m} thr", num(c[i]),
                      float(seed_point[(m, "clean", "0")]["conf_thresh"]), tol=0.005)
        elif head == "Clean":
            for i, m in enumerate(models, start=1):
                check(f"tab:seeds {m} clean", num(c[i]),
                      float(seed_point[(m, "clean", "0")]["recall"]))
        elif head in ("s1", "s2", "s3"):
            pending = head[1]
            for i, m in enumerate(models, start=1):
                check(f"tab:seeds {m} s{pending}", num(c[i]),
                      float(seed_point[(m, "low_light", pending)]["recall"]))
        elif pending and interval(c[1]):
            for i, m in enumerate(models, start=1):
                lo, hi = interval(c[i])
                r = seed_ci[(m, "low_light", pending)]
                check(f"tab:seeds {m} s{pending} lo", lo, float(r["recall_ci_lo"]))
                check(f"tab:seeds {m} s{pending} hi", hi, float(r["recall_ci_hi"]))
            pending = None


def check_mitigation(point, ci):
    """tab:mitigation -- before/after and intervals, SARD low_light."""
    rows = table_body(tex("5_mitigation.tex"), "tab:mitigation")
    model = None
    for row in rows:
        c = cells(row)
        m = re.search(r"multirow\{3\}\{\*\}\{([^}]+)\}", c[0])
        if m:
            model = MODEL_KEY[m.group(1).strip()]
        if model is None or len(c) < 6:
            continue
        kind = c[1].strip()
        key = model if kind == "before" else MITIGATED[model]
        if kind in ("before", "after"):
            check(f"tab:mitigation {model} {kind} clean", num(c[2]),
                  float(point[(key, "sard", "clean", "0")]["recall"]))
            for i, sev in enumerate("123", start=3):
                check(f"tab:mitigation {model} {kind} s{sev}", num(c[i]),
                      float(point[(key, "sard", "low_light", sev)]["recall"]))
        elif kind.startswith("95"):
            for i, sev in enumerate("123", start=3):
                lo, hi = interval(c[i])
                r = ci[(MITIGATED[model], "sard", "low_light", sev)]
                check(f"tab:mitigation {model} s{sev} lo", lo, float(r["recall_ci_lo"]))
                check(f"tab:mitigation {model} s{sev} hi", hi, float(r["recall_ci_hi"]))


def check_tradeoff(point):
    """tab:tradeoff -- change on the 8 corruptions never trained on."""
    for row in table_body(tex("5_mitigation.tex"), "tab:tradeoff"):
        c = cells(row)
        if c[0].strip() not in MODEL_KEY or len(c) < 5:
            continue
        model = MODEL_KEY[c[0].strip()]
        mit = MITIGATED[model]
        deltas = [float(point[(mit, "sard", corr, s)]["recall"])
                  - float(point[(model, "sard", corr, s)]["recall"])
                  for corr in OTHER8 for s in "123"]
        d_clean = (float(point[(mit, "sard", "clean", "0")]["recall"])
                   - float(point[(model, "sard", "clean", "0")]["recall"]))
        check(f"tab:tradeoff {model} d_clean", num(c[1]), d_clean)
        check(f"tab:tradeoff {model} mean", num(c[2]), statistics.mean(deltas))
        check(f"tab:tradeoff {model} worst", num(c[3]), min(deltas))
        m = re.match(r"(\d+)/24", c[4].strip())
        check(f"tab:tradeoff {model} n_worse", float(m.group(1)) if m else None,
              float(sum(1 for d in deltas if d < -0.03)), tol=0.001)
        global checks
        if len(deltas) != 24:
            failures.append(f"tab:tradeoff {model}: {len(deltas)} conditions, not 24")


# ------------------------------------------------------- inline claims

def check_inline(point, ci, seed_point):
    res = tex("4_results.tex")
    mit = tex("5_mitigation.tex")
    abstract = tex("0_abstract.tex")

    # Clean-recall ranges quoted in prose.
    for dataset, blob in (("heridal", res), ("sard", res)):
        vals = [float(point[(m, dataset, "clean", "0")]["recall"])
                for m in ("fasterrcnn", "rtdetr", "yolo11")]
        m = re.search(rf"(\d\.\d\d\d)--(\d\.\d\d\d) on {dataset.upper()}", blob)
        if m:
            check(f"inline clean-min {dataset}", float(m.group(1)), min(vals))
            check(f"inline clean-max {dataset}", float(m.group(2)), max(vals))

    # The rain-streaks headline, quoted in both abstract and results.
    clean = float(point[("yolo11", "sard", "clean", "0")]["recall"])
    rain = float(point[("yolo11", "sard", "rain_streaks", "3")]["recall"])
    for blob, where in ((res, "results"), (abstract, "abstract")):
        m = re.search(r"(\d\.\d)-point drop|roughly (\d\.\d) points", blob)
        if m:
            got = float(m.group(1) or m.group(2))
            check(f"inline rain drop ({where})", got, (clean - rain) * 100, tol=0.06)

    # mAP50 clean -> low_light severity 1 on HERIDAL. The prose names the
    # three models in this order and elides the metric after the first.
    pairs = re.findall(r"(?:'s mAP\$_\{50\}\$ falls from|'s from)\s+"
                       r"(\d\.\d+)\s+(?:clean\s+)?to\s+(\d\.\d+)", res)
    for (a, b), model in zip(pairs, ("rtdetr", "yolo11", "fasterrcnn")):
        check(f"inline map50 {model} clean", float(a),
              float(point[(model, "heridal", "clean", "0")]["map50"]))
        check(f"inline map50 {model} s1", float(b),
              float(point[(model, "heridal", "low_light", "1")]["map50"]))

    # Severity-3 mitigation values quoted in prose and abstract.
    for model, want_name in (("fasterrcnn", "Faster R-CNN"),
                             ("rtdetr", "RT-DETR"), ("yolo11", "YOLOv11")):
        v = float(point[(MITIGATED[model], "sard", "low_light", "3")]["recall"])
        m = re.search(rf"(\d\.\d\d\d) for {re.escape(want_name)}", mit)
        if m:
            check(f"inline mitigation s3 {model}", float(m.group(1)), v)

    s3 = [float(point[(MITIGATED[m], "sard", "low_light", "3")]["recall"])
          for m in MITIGATED]
    m = re.search(r"(\d\.\d\d\d)--(\d\.\d\d\d) at a held-out severity", abstract)
    if m:
        check("abstract mitigation range lo", float(m.group(1)), min(s3))
        check("abstract mitigation range hi", float(m.group(2)), max(s3))

    # Ground-truth and image counts.
    for dataset, n_img in (("heridal", 101), ("sard", 862)):
        r = point[("yolo11", dataset, "clean", "0")]
        n_gt = int(r["n_gt"])
        if f"{n_img} images, {n_gt} persons" not in res:
            failures.append(f"inline counts {dataset}: "
                            f"expected '{n_img} images, {n_gt} persons'")
        global checks
        checks += 1
        c_img = int(ci[("yolo11", dataset, "clean", "0")]["n_images"])
        if c_img != n_img:
            failures.append(f"inline n_images {dataset}: tex={n_img} source={c_img}")

    # Every model is exactly zero at low_light severity 3, both datasets.
    for dataset in ("heridal", "sard"):
        for model in ("fasterrcnn", "rtdetr", "yolo11"):
            checks += 1
            if float(point[(model, dataset, "low_light", "3")]["recall"]) != 0.0:
                failures.append(f"zero-claim {model}/{dataset} is not 0.000")
    for m in ("yolo11_seed17", "yolo11_seed42", "yolo11_seed123"):
        checks += 1
        if float(seed_point[(m, "low_light", "3")]["recall"]) != 0.0:
            failures.append(f"zero-claim {m} is not 0.000")


def main():
    point, ci, seed_point, seed_ci = load()
    check_spectrum(point)
    check_lowlight(point)
    check_family(point)
    check_seeds(seed_point, seed_ci)
    check_mitigation(point, ci)
    check_tradeoff(point)
    check_inline(point, ci, seed_point)

    print(f"{checks} checks against results/")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all numbers agree with their source records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
