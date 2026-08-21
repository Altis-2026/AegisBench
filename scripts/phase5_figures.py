#!/usr/bin/env python3
"""Phase 5d: the two figures the paper plan calls for that phase5_heatmap.py
doesn't produce -- severity curves with bootstrap CI bands, and the
recall-vs-localization-stability decoupling scatter.

Pure plotting over already-computed CSVs (bootstrap CI + localization
stability); no inference, no GPU, seconds to run.

  python scripts/phase5_figures.py \
      --ci results/sweep/ci_heridal.csv \
      --localization results/sweep/localization_heridal.csv \
      --dataset heridal --out results/sweep
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.visualize import localization_decoupling, severity_curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", required=True, help="phase5_bootstrap_ci.py output")
    ap.add_argument("--localization", required=True,
                    help="phase5_localization.py output")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ci_df = pd.read_csv(args.ci)
    loc_df = pd.read_csv(args.localization)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for model in sorted(ci_df["model"].unique()):
        p1 = severity_curve(ci_df, out / f"severity_{args.dataset}_{model}.png",
                            model=model, dataset=args.dataset)
        print(f"wrote {p1}")
        p2 = localization_decoupling(loc_df, ci_df,
                                     out / f"decoupling_{args.dataset}_{model}.png",
                                     model=model, dataset=args.dataset)
        print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
