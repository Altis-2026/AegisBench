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


class UltralyticsDetector:
    def __init__(self, weights: str | Path):
        from ultralytics import YOLO, RTDETR

        w = str(weights)
        self.model = RTDETR(w) if "rtdetr" in Path(w).stem.lower() else YOLO(w)

    def predict(self, image_rgb: np.ndarray, conf: float = 0.001,
                imgsz: int = 1024, device: str | int = 0) -> dict:
        res = self.model.predict(image_rgb, conf=conf, imgsz=imgsz,
                                 device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        return {"boxes": boxes, "scores": scores}


def train_from_config(config_path: str | Path, run_dir: str | Path) -> Path:
    """Fine-tune per a config yaml; returns path to best weights."""
    from ultralytics import YOLO, RTDETR

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config_used.yaml").write_text(yaml.safe_dump(cfg))

    base = cfg["base_weights"]
    model = RTDETR(base) if "rtdetr" in base.lower() else YOLO(base)
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
