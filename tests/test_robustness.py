import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.evaluation.robustness import relative_drop, summarize


def _df():
    rows = [
        {"model": "yolo11", "dataset": "heridal", "family": "clean",
         "corruption": "clean", "severity": 0, "recall": 0.90},
        {"model": "yolo11", "dataset": "heridal", "family": "wildfire",
         "corruption": "smoke_haze", "severity": 1, "recall": 0.72},
        {"model": "yolo11", "dataset": "heridal", "family": "wildfire",
         "corruption": "smoke_haze", "severity": 3, "recall": 0.45},
        {"model": "yolo11", "dataset": "heridal", "family": "storm",
         "corruption": "motion_blur", "severity": 1, "recall": 0.81},
    ]
    return pd.DataFrame(rows)


def test_relative_drop():
    rpd = relative_drop(_df(), "recall")
    smoke1 = rpd[(rpd.corruption == "smoke_haze")
                 & (rpd.severity == 1)].iloc[0]
    assert abs(smoke1["rpd_recall"] - (0.90 - 0.72) / 0.90) < 1e-9


def test_summarize_family_means():
    s = summarize(_df(), "recall")
    wildfire = s[s.family == "wildfire"].iloc[0]
    expected = ((0.90 - 0.72) / 0.90 + (0.90 - 0.45) / 0.90) / 2
    assert abs(wildfire["rpd_recall"] - expected) < 1e-9
