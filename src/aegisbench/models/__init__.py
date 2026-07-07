"""Detector factory: every model exposes predict(image_rgb) ->
{'boxes' (N,4) xyxy, 'scores' (N,)} so the sweep code is model-agnostic."""

from __future__ import annotations

from pathlib import Path


def load_detector(kind: str, weights: str | Path):
    kind = kind.lower()
    if kind in ("yolo11", "rtdetr"):
        from .ultralytics_wrapper import UltralyticsDetector
        return UltralyticsDetector(weights)
    if kind == "fasterrcnn":
        from .fasterrcnn import FasterRCNNDetector
        return FasterRCNNDetector(weights)
    raise ValueError(f"unknown detector kind '{kind}' "
                     "(expected yolo11 | rtdetr | fasterrcnn)")
