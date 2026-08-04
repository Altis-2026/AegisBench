"""_is_fasterrcnn_ckpt must key off the filename convention alone -- it's
the only thing that lets `--resume` route to the right architecture's
resume function without having to load and inspect either checkpoint
file first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase4_train import _is_fasterrcnn_ckpt


def test_detects_fasterrcnn_checkpoint():
    path = "runs/fasterrcnn/fasterrcnn_sard_clean_last_ckpt.pt"
    assert _is_fasterrcnn_ckpt(path)


def test_ultralytics_checkpoint_not_misdetected():
    path = "runs/rtdetr/rtdetr_heridal_clean/weights/last.pt"
    assert not _is_fasterrcnn_ckpt(path)


def test_ultralytics_yolo_checkpoint_not_misdetected():
    path = "runs/yolo11/yolo11_heridal_clean/weights/last.pt"
    assert not _is_fasterrcnn_ckpt(path)
