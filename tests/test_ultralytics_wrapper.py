"""_is_rtdetr must key off the full path, not just the filename stem --
ultralytics always names checkpoints best.pt/last.pt regardless of
architecture, so only the run-directory name carries an 'rtdetr' marker
for a TRAINED checkpoint (as opposed to a base weights file like
rtdetr-l.pt, where the stem itself says it)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.models.ultralytics_wrapper import _is_rtdetr


def test_detects_rtdetr_base_weights():
    assert _is_rtdetr("rtdetr-l.pt")


def test_detects_yolo_base_weights():
    assert not _is_rtdetr("yolo11m.pt")


def test_detects_rtdetr_from_checkpoint_path_not_filename():
    # The bug this guards against: the checkpoint file itself is always
    # named best.pt/last.pt, so only the run directory says "rtdetr".
    path = "runs/rtdetr/rtdetr_heridal_clean/weights/best.pt"
    assert _is_rtdetr(path)


def test_yolo_checkpoint_path_not_misdetected():
    path = "runs/yolo11/yolo11_heridal_clean/weights/best.pt"
    assert not _is_rtdetr(path)


def test_case_insensitive():
    assert _is_rtdetr("runs/RTDETR/RTDETR_heridal_clean/weights/last.pt")
