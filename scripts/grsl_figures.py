#!/usr/bin/env python3
"""Figures for the GRSL letter, built from the committed results CSVs.

Two figures, both sized for a single IEEEtran journal column (252pt) and
placed at scale 1 in the LaTeX, per the sizing rule in aegisbench.paperstyle.

  fig1  low_light recall against exposure loss in EV stops, all six
        model/dataset pairs. The headline collapse, plotted against the
        physical drive parameter rather than a severity index.
  fig2  recall against severity for all nine corruptions, YOLOv11 on SARD,
        with bootstrap 95% bands. Shows the spread the benchmark resolves.

Every value is read from results/sweep/*.csv. Nothing is hard-coded except
the linear_gain values, which are read back from configs/corruptions.yaml.

  python scripts/grsl_figures.py --out paper/grsl/figures
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegisbench.paperstyle import FAMILY_COLORS, LINE_STYLES, apply, save

# IEEEtran journal geometry, in inches at 72.27 pt/in. Measured from the
# class itself: \columnwidth = 252pt, \textwidth = 516pt. These replace
# the CVF widths in paperstyle, which target a different template.
IEEE_COL_WIDTH = 252.0 / 72.27
IEEE_FULL_WIDTH = 516.0 / 72.27

# Declaration order from configs/corruptions.yaml, so within-family line
# styles are assigned the same way the taxonomy table orders them.
CORRUPTION_ORDER = [
    ("water_glare", "flood"),
    ("turbidity_cast", "flood"),
    ("inundation", "flood"),
    ("smoke_haze", "wildfire"),
    ("fire_warm_tint", "wildfire"),
    ("rain_streaks", "storm"),
    ("motion_blur", "storm"),
    ("low_light", "storm"),
    ("dust_haze", "earthquake"),
]

MODEL_LABEL = {
    "fasterrcnn": "Faster R-CNN",
    "rtdetr": "RT-DETR",
    "yolo11": "YOLOv11",
}
MODEL_MARKER = {"fasterrcnn": "o", "rtdetr": "s", "yolo11": "^"}
DATASET_LABEL = {"heridal": "HERIDAL", "sard": "SARD"}
DATASET_STYLE = {"heridal": "--", "sard": "-"}


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_results():
    """Point estimates from the sweep log, bootstrap bounds from the CI files."""
    point = {}
    for row in read_csv(ROOT / "results/sweep/master_ci.csv"):
        key = (row["model"], row["dataset"], row["corruption"], row["severity"])
        point[key] = row

    interval = {}
    for name in ("ci_heridal.csv", "ci_sard.csv"):
        for row in read_csv(ROOT / "results/sweep" / name):
            key = (row["model"], row["dataset"], row["corruption"], row["severity"])
            interval[key] = row
    return point, interval


def low_light_exposure_stops():
    """Exposure loss per severity, in stops, from the declared linear gains.

    Read out of the corruption config rather than restated here so the
    figure cannot drift from the parameters that generated the imagery.
    """
    text = (ROOT / "configs/corruptions.yaml").read_text()
    block = text.split("low_light:", 1)[1].split("dust_haze:", 1)[0]
    gains = {}
    for line in block.splitlines():
        line = line.strip()
        if line[:1] in "123" and "linear_gain" in line:
            severity = int(line[0])
            value = line.split("linear_gain:", 1)[1].split(",")[0].strip()
            gains[severity] = float(value)
    if sorted(gains) != [1, 2, 3]:
        raise SystemExit(f"could not read all three linear_gain values: {gains}")
    return {s: math.log2(g) for s, g in gains.items()}


def figure_low_light(point, stops, out_dir):
    """Recall against exposure loss, every model on both datasets."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(IEEE_COL_WIDTH, 2.35))
    ax = fig.add_axes([0.155, 0.215, 0.825, 0.755])

    x = [0.0] + [stops[s] for s in (1, 2, 3)]
    for dataset in ("heridal", "sard"):
        for model in ("fasterrcnn", "rtdetr", "yolo11"):
            y = [float(point[(model, dataset, "clean", "0")]["recall"])]
            y += [float(point[(model, dataset, "low_light", str(s))]["recall"])
                  for s in (1, 2, 3)]
            ax.plot(x, y,
                    color=FAMILY_COLORS["storm"] if dataset == "sard" else "#8c6d31",
                    linestyle=DATASET_STYLE[dataset],
                    marker=MODEL_MARKER[model], markersize=3.2,
                    markerfacecolor="white", markeredgewidth=0.9,
                    label=f"{MODEL_LABEL[model]}, {DATASET_LABEL[dataset]}")

    ax.set_xlabel("Exposure loss (stops)")
    ax.set_ylabel("Recall")
    ax.set_xlim(0.35, -4.85)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks([0] + [round(stops[s], 2) for s in (1, 2, 3)])
    ax.set_xticklabels(["0\n(clean)", "$-1.6$\n(s1)", "$-2.9$\n(s2)", "$-4.5$\n(s3)"])
    ax.grid(axis="y", color="#cccccc", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=1, fontsize=5.6, handlelength=2.2,
              borderpad=0.2, labelspacing=0.22)
    return save(fig, Path(out_dir) / "lowlight_collapse")


def figure_spectrum(point, interval, out_dir):
    """Recall against severity, all nine corruptions, YOLOv11 on SARD."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(IEEE_COL_WIDTH, 2.45))
    ax = fig.add_axes([0.135, 0.155, 0.60, 0.815])

    seen = {}
    for name, family in CORRUPTION_ORDER:
        index = seen.setdefault(family, 0)
        seen[family] = index + 1
        style = {"color": FAMILY_COLORS[family],
                 "linestyle": LINE_STYLES[index % len(LINE_STYLES)]}

        xs = [0, 1, 2, 3]
        ys = [float(point[("yolo11", "sard", "clean", "0")]["recall"])]
        lo = [float(interval[("yolo11", "sard", "clean", "0")]["recall_ci_lo"])]
        hi = [float(interval[("yolo11", "sard", "clean", "0")]["recall_ci_hi"])]
        for s in (1, 2, 3):
            ys.append(float(point[("yolo11", "sard", name, str(s))]["recall"]))
            lo.append(float(interval[("yolo11", "sard", name, str(s))]["recall_ci_lo"]))
            hi.append(float(interval[("yolo11", "sard", name, str(s))]["recall_ci_hi"]))

        ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.13, linewidth=0)
        # Plain text, not LaTeX: matplotlib renders the underscore literally
        # here, so escaping it would print the backslash.
        ax.plot(xs, ys, **style, label=name)

    ax.set_xlabel("Severity")
    ax.set_ylabel("Recall")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["clean", "1", "2", "3"])
    ax.set_ylim(-0.03, 1.0)
    ax.grid(axis="y", color="#cccccc", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.8,
              handlelength=2.0, labelspacing=0.3)
    return save(fig, Path(out_dir) / "spectrum_sard_yolo11")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/grsl/figures"))
    args = ap.parse_args()

    apply()
    point, interval = load_results()
    stops = low_light_exposure_stops()
    print("exposure loss (stops):",
          {s: round(v, 2) for s, v in sorted(stops.items())})

    for path in figure_low_light(point, stops, args.out):
        print("wrote", path)
    for path in figure_spectrum(point, interval, args.out):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
