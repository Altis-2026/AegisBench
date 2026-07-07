"""Faster R-CNN (torchvision, ResNet-50 FPN v2) — the classic two-stage
contrast baseline, trained on the same YOLO-format tiles as the other
detectors and exposed behind the same predict() interface."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml


def _lazy_torch():
    import torch
    import torchvision
    return torch, torchvision


class YoloTileDataset:
    """Reads a YOLO-format tile directory (images/ + labels/) into
    torchvision (image, target) pairs."""

    def __init__(self, images_dir: str | Path, labels_dir: str | Path):
        from PIL import Image  # noqa: F401 (validated import)
        self.images = sorted(Path(images_dir).glob("*"))
        self.images = [p for p in self.images
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        self.labels_dir = Path(labels_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        torch, _ = _lazy_torch()
        from PIL import Image

        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = []
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes_t,
                  "labels": torch.ones((len(boxes),), dtype=torch.int64),
                  "image_id": torch.tensor([idx])}
        img_t = torch.as_tensor(np.asarray(img), dtype=torch.float32
                                ).permute(2, 0, 1) / 255.0
        return img_t, target


def build_model(num_classes: int = 2):
    _, torchvision = _lazy_torch()
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    return model


def train_from_config(config_path: str | Path, run_dir: str | Path) -> Path:
    torch, _ = _lazy_torch()
    from torch.utils.data import DataLoader

    from ..seeding import seed_everything

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config_used.yaml").write_text(yaml.safe_dump(cfg))

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = YoloTileDataset(cfg["train_images"], cfg["train_labels"])
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True,
                        num_workers=cfg.get("workers", 4),
                        collate_fn=lambda b: tuple(zip(*b)),
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.get("lr0", 0.005),
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))

    best_path = run_dir / "fasterrcnn_best.pt"
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            with torch.autocast("cuda", enabled=cfg.get("amp", True)):
                losses = model(images, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
        sched.step()
        print(f"[fasterrcnn] epoch {epoch + 1}/{cfg['epochs']} "
              f"loss={running / max(len(loader), 1):.4f} "
              f"({time.time() - t0:.0f}s)")
        torch.save(model.state_dict(), best_path)
    return best_path


class FasterRCNNDetector:
    def __init__(self, weights: str | Path):
        torch, _ = _lazy_torch()
        self.torch = torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model().to(self.device)
        self.model.load_state_dict(torch.load(str(weights),
                                              map_location=self.device))
        self.model.eval()

    def predict(self, image_rgb: np.ndarray, conf: float = 0.001,
                imgsz: int = 0, device=None) -> dict:
        torch = self.torch
        t = torch.as_tensor(image_rgb, dtype=torch.float32
                            ).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            out = self.model([t.to(self.device)])[0]
        keep = out["scores"] >= conf
        return {"boxes": out["boxes"][keep].cpu().numpy(),
                "scores": out["scores"][keep].cpu().numpy()}
