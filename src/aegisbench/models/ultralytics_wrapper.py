"""YOLOv11 and RT-DETR via the ultralytics API, wrapped behind the common
detector interface: predict(image RGB uint8) -> {'boxes' xyxy, 'scores'}.

Training goes through ultralytics with fixed seeds and deterministic mode;
all hyperparameters live in configs/train_*.yaml and are logged verbatim
into the run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _is_rtdetr(weights: str | Path) -> bool:
    """Detect RT-DETR from the full path, not just the filename stem --
    trained checkpoints are always named best.pt/last.pt by ultralytics
    regardless of architecture, so only the run directory name (e.g.
    '.../rtdetr_heridal_clean/weights/best.pt') carries the 'rtdetr' marker.
    Checking just the stem would silently misdetect every trained RT-DETR
    checkpoint as YOLO."""
    return "rtdetr" in str(weights).lower()


class UltralyticsDetector:
    def __init__(self, weights: str | Path):
        from ultralytics import YOLO, RTDETR

        w = str(weights)
        self.model = RTDETR(w) if _is_rtdetr(w) else YOLO(w)

    def predict(self, image_rgb: np.ndarray, conf: float = 0.001,
                imgsz: int = 1024, device: str | int = 0) -> dict:
        # ultralytics treats a numpy source as BGR (cv2 convention) and flips
        # it to RGB internally. Our pipeline carries RGB, and training read
        # the on-disk JPEGs via cv2 (BGR) -- so we must hand predict() BGR
        # here for the inference channel order to match training. Passing RGB
        # would silently swap R/B and degrade accuracy without any error.
        image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
        res = self.model.predict(image_bgr, conf=conf, imgsz=imgsz,
                                 device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        return {"boxes": boxes, "scores": scores}


def train_from_config(config_path: str | Path, run_dir: str | Path) -> Path:
    """Fine-tune per a config yaml; returns path to best weights."""
    from ultralytics import YOLO, RTDETR

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Must be absolute: a relative `project` path lets ultralytics silently
    # prepend its own default "runs/<task>/" prefix, so the directory it
    # actually saves to no longer matches the path this function returns.
    # Observed live: project="runs/yolo11" saved to
    # ".../runs/detect/runs/yolo11/..." instead of ".../runs/yolo11/...".
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config_used.yaml").write_text(yaml.safe_dump(cfg))

    base = cfg["base_weights"]
    model = RTDETR(base) if _is_rtdetr(base) else YOLO(base)
    model.train(
        data=cfg["data_yaml"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        seed=cfg["seed"],
        deterministic=True,
        amp=cfg.get("amp", True),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        patience=cfg.get("patience", 20),
        workers=cfg.get("workers", 4),
        project=str(run_dir),
        name=cfg["run_name"],
        exist_ok=True,
        single_cls=True,
        verbose=True,
    )
    return run_dir / cfg["run_name"] / "weights" / "best.pt"


def resume_training(last_weights: str | Path) -> Path:
    """Resume an interrupted run (e.g. after Ctrl+C, or a killed process)
    from its last checkpoint. Ultralytics stores the original training
    args (data, epochs, project, name, ...) alongside the checkpoint, so
    resume=True picks training back up from the next epoch with no need
    to restate any config."""
    from ultralytics import YOLO, RTDETR

    w = str(last_weights)
    model = RTDETR(w) if _is_rtdetr(w) else YOLO(w)
    model.train(resume=True)
    # .../<run_name>/weights/last.pt -> .../<run_name>/weights/best.pt
    return Path(last_weights).parent / "best.pt"
