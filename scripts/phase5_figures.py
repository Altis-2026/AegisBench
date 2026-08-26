#!/usr/bin/env python3
"""Generate the paper figures at publication quality.

Writes PDF (for LaTeX) and PNG (for quick viewing) at exact CVF column or
text width, so `\\includegraphics[width=\\columnwidth]{...}` places them at
scale 1 and no label gets shrunk by LaTeX rescaling. See
src/aegisbench/paperstyle.py for the sizing and typography rules.

  # result figures (needs the bootstrap-CI and localization CSVs)
  python scripts/phase5_figures.py results \\
      --ci results/sweep/ci_heridal.csv \\
      --localization results/sweep/localization_heridal.csv \\
      --sweep-csv results/sweep/master_ci.csv \\
      --dataset heridal --out results/figures

  # corruption taxonomy panel (needs one clean image)
  python scripts/phase5_figures.py taxonomy \\
      --image data/heridal/testImages/some_frame.JPG \\
      --severity 3 --out results/figures
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import DEFAULT_CONFIG_PATH, CorruptionSuite
from aegisbench.corruptions.base import to_float
from aegisbench.inference import load_rgb
from aegisbench.visualize import (corruption_panel, localization_decoupling,
                                  robustness_heatmap_panels, severity_curves)


def cmd_results(args) -> int:
    out = Path(args.out)
    ci = pd.read_csv(args.ci)
    loc = pd.read_csv(args.localization)
    models = sorted(ci["model"].unique())

    # Severity curves. One panel per model keeps every corruption legible;
    # a single combined panel would overplot 27 lines.
    for model in models:
        p = severity_curves(ci, out / f"severity_{args.dataset}_{model}",
                            [(model, args.dataset)])
        print(f"  {p[0].name}")
    if len(models) > 1:
        p = severity_curves(ci, out / f"severity_{args.dataset}_all",
                            [(m, args.dataset) for m in models], height=2.6)
        print(f"  {p[0].name}  (all models, one panel each)")

    for model in models:
        p = localization_decoupling(loc, ci,
                                    out / f"decoupling_{args.dataset}_{model}",
                                    model=model, dataset=args.dataset)
        print(f"  {p[0].name}")

    if args.sweep_csv:
        df = pd.read_csv(args.sweep_csv)
        p = robustness_heatmap_panels(df, out / f"heatmap_{args.dataset}",
                                      dataset=args.dataset)
        print(f"  {p[0].name}")
    else:
        print("  (skipping heatmap: pass --sweep-csv master_ci.csv)")
    return 0


def cmd_taxonomy(args) -> int:
    suite = CorruptionSuite(args.config)
    img = to_float(load_rgb(args.image))
    crop = [int(v) for v in args.crop.split(",")] if args.crop else None
    stem = Path(args.image).stem
    p = corruption_panel(img, suite, stem,
                         Path(args.out) / f"taxonomy_s{args.severity}",
                         severity=args.severity, ncols=args.ncols, crop=crop)
    print(f"  {p[0].name}")
    print("Check every cell before using this figure: severe smoke must "
          "read as smoke rather than noise, dust must read browner and "
          "patchier than smoke, and inundation must occlude rather than "
          "merely tint.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("results", help="severity curves, decoupling, heatmap")
    r.add_argument("--ci", required=True)
    r.add_argument("--localization", required=True)
    r.add_argument("--sweep-csv", default=None,
                   help="master_ci.csv, needed for the heatmap")
    r.add_argument("--dataset", required=True)
    r.add_argument("--out", default="results/figures")
    r.set_defaults(func=cmd_results)

    t = sub.add_parser("taxonomy", help="the nine-corruption panel")
    t.add_argument("--image", required=True, help="one CLEAN source image")
    t.add_argument("--severity", type=int, default=3)
    t.add_argument("--ncols", type=int, default=5)
    t.add_argument("--crop", default=None,
                   help="x1,y1,x2,y2 in pixels; crop to an annotated region "
                        "so the corruption shows at detection scale")
    t.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    t.add_argument("--out", default="results/figures")
    t.set_defaults(func=cmd_taxonomy)

    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
