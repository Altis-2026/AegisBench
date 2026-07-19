"""Shared dataset plumbing.

Canonical record schema used everywhere downstream:
  {'image_id': str, 'image_path': str, 'width': int, 'height': int,
   'boxes': np.ndarray (N, 4) float32 xyxy absolute pixels}

Only one class exists in this benchmark: person.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PERSON_ALIASES = {"person", "human", "people", "pedestrian"}


def parse_voc_xml(xml_path: str | Path,
                  person_aliases: set[str] = PERSON_ALIASES) -> dict:
    """Parse one PASCAL-VOC annotation file into the canonical record.
    Objects whose class is not a person alias are ignored (and counted)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))
    boxes, ignored = [], 0
    for obj in root.findall("object"):
        cls = (obj.find("name").text or "").strip().lower()
        if cls not in person_aliases:
            ignored += 1
            continue
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        # Guard against inverted/degenerate boxes in the source annotations.
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            boxes.append([x1, y1, x2, y2])
    return {"image_id": Path(xml_path).stem,
            "width": width, "height": height,
            "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
            "n_ignored_objects": ignored}


def image_size(image_path: str | Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as im:
        return im.size          # (width, height)


def parse_yolo_label(image_path: str | Path, label_path: str | Path,
                     person_class_ids: set[int] | None = None) -> dict:
    """Parse one YOLO-format label file (normalized 'cls cx cy w h' per
    line) into the canonical record. Image dimensions are read from the
    image itself since YOLO coordinates carry no size metadata.

    person_class_ids: if given, only these class ids become boxes (objects
    of other classes are counted in n_ignored_objects). If None, EVERY box
    is treated as person — the correct default for single-class person
    detection exports where class 0 is the only class.
    """
    width, height = image_size(image_path)
    boxes, ignored = [], 0
    label_path = Path(label_path)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            if person_class_ids is not None and cls_id not in person_class_ids:
                ignored += 1
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
    return {"image_id": Path(image_path).stem,
            "width": width, "height": height,
            "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
            "n_ignored_objects": ignored}


def write_yolo_labels(records: list[dict], out_dir: str | Path) -> None:
    """Write one YOLO txt per record (class 0 = person, normalized cxcywh)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        w, h = rec["width"], rec["height"]
        lines = []
        for x1, y1, x2, y2 in rec["boxes"]:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / f"{rec['image_id']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))


def write_dataset_yaml(path: str | Path, root: str | Path,
                       train_dir: str, val_dir: str,
                       test_dir: str | None = None) -> None:
    """Ultralytics dataset yaml (single class)."""
    lines = [f"path: {Path(root).resolve()}",
             f"train: {train_dir}", f"val: {val_dir}"]
    if test_dir:
        lines.append(f"test: {test_dir}")
    lines += ["names:", "  0: person"]
    Path(path).write_text("\n".join(lines) + "\n")


def save_records(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = []
    for r in records:
        d = dict(r)
        d["boxes"] = np.asarray(r["boxes"],
                                np.float32).reshape(-1, 4).tolist()
        out.append(d)
    with open(path, "w") as f:
        json.dump(out, f)


def load_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    for r in raw:
        r["boxes"] = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
    return raw


def summarize_records(records: list[dict]) -> dict:
    n_boxes = sum(len(r["boxes"]) for r in records)
    empty = sum(1 for r in records if len(r["boxes"]) == 0)
    sides = np.concatenate(
        [np.stack([r["boxes"][:, 2] - r["boxes"][:, 0],
                   r["boxes"][:, 3] - r["boxes"][:, 1]], 1)
         for r in records if len(r["boxes"])]) if n_boxes else np.zeros((0, 2))
    return {
        "n_images": len(records),
        "n_boxes": n_boxes,
        "n_images_without_boxes": empty,
        "box_side_px_median": float(np.median(sides)) if n_boxes else 0.0,
        "box_side_px_p95": float(np.percentile(sides, 95)) if n_boxes else 0.0,
        "box_side_px_max": float(sides.max()) if n_boxes else 0.0,
    }
