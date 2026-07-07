#!/usr/bin/env python3
"""Phase 5 checkpoint artifacts: robustness heatmap + per-family summary
from the master sweep CSV.

  python scripts/phase5_heatmap.py --csv results/sweep/master.csv \
      --out results/sweep
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.evaluation.robustness import summarize
from aegisbench.visualize import robustness_heatmap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="recall")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        p = robustness_heatmap(df, out / f"heatmap_{dataset}_"
                               f"{args.metric}.png",
                               metric=args.metric, dataset=dataset)
        print(f"wrote {p}")

    summary = summarize(df, args.metric)
    summary_path = out / f"summary_{args.metric}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
