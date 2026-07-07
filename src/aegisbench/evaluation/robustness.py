"""Robustness aggregation: relative performance drop (rPD), following the
reporting convention of corruption-robustness benchmarks (ImageNet-C
lineage): drop is measured relative to the same model's clean score, per
corruption and severity, then averaged."""

from __future__ import annotations

import pandas as pd

METRICS = ("recall", "precision", "f1", "map50", "map50_95",
           "recall_small", "recall_medium", "recall_large")


def relative_drop(df: pd.DataFrame, metric: str = "recall") -> pd.DataFrame:
    """df: one row per (model, dataset, corruption, severity) including
    'clean' rows with corruption == 'clean'. Adds rPD_<metric> =
    (clean - corrupted) / clean for each non-clean row."""
    clean = (df[df["corruption"] == "clean"]
             .set_index(["model", "dataset"])[metric])
    out = df[df["corruption"] != "clean"].copy()

    def _rpd(row):
        c = clean.loc[(row["model"], row["dataset"])]
        return (c - row[metric]) / c if c > 0 else float("nan")

    out[f"rpd_{metric}"] = out.apply(_rpd, axis=1)
    return out


def summarize(df: pd.DataFrame, metric: str = "recall") -> pd.DataFrame:
    """Mean rPD per (model, dataset, corruption) across severities, plus a
    per-family mean — the headline robustness table."""
    rpd = relative_drop(df, metric)
    per_corruption = (rpd.groupby(["model", "dataset", "family",
                                   "corruption"])[f"rpd_{metric}"]
                      .mean().reset_index())
    per_family = (rpd.groupby(["model", "dataset", "family"])
                  [f"rpd_{metric}"].mean().reset_index()
                  .rename(columns={f"rpd_{metric}":
                                   f"rpd_{metric}_family_mean"}))
    return per_corruption.merge(per_family, on=["model", "dataset", "family"])
