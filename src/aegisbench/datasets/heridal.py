"""HERIDAL loader.

HERIDAL (Bozic-Stulic et al., IJCV 2019) ships full-size ~4000x3000 aerial
images with PASCAL-VOC XML annotations and an official folder-level
train/test split (trainImages / testImages). We keep the official split and
carve a validation set out of TRAIN only (never test) for operating-point
selection and early stopping.

Expected layout after manual download from the IPSAR research page
(see docs/DATA.md — the archive is free but must be fetched by a human):

    data/heridal/
      trainImages/          *.JPG
      trainImages/labels/   *.xml     (or trainLabels/ — both are handled)
      testImages/           *.JPG
      testImages/labels/    *.xml

If your archive unpacks differently, pass the image/label dirs explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..seeding import stable_seed
from .common import parse_voc_xml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _find_label_dir(image_dir: Path) -> Path:
    for cand in (image_dir / "labels", image_dir,
                 image_dir.parent / (image_dir.name.replace("Images",
                                                            "Labels"))):
        if cand.is_dir() and list(cand.glob("*.xml")):
            return cand
    raise FileNotFoundError(
        f"No XML labels found near {image_dir}. Pass label_dir explicitly.")


def load_split(image_dir: str | Path,
               label_dir: str | Path | None = None) -> list[dict]:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir) if label_dir else _find_label_dir(image_dir)
    records = []
    missing_labels = []
    for img_path in sorted(p for p in image_dir.iterdir()
                           if p.suffix in IMG_EXTS):
        xml_path = label_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            missing_labels.append(img_path.name)
            continue
        rec = parse_voc_xml(xml_path)
        rec["image_id"] = img_path.stem
        rec["image_path"] = str(img_path)
        records.append(rec)
    if missing_labels:
        print(f"[heridal] WARNING: {len(missing_labels)} images without "
              f"labels were skipped (first few: {missing_labels[:5]})")
    return records


def train_val_split(train_records: list[dict], val_fraction: float = 0.15,
                    seed_key: str = "heridal-val-split") -> tuple[list, list]:
    """Deterministic validation carve-out from the official train split."""
    rng = np.random.default_rng(stable_seed(seed_key))
    idx = rng.permutation(len(train_records))
    n_val = max(1, int(round(val_fraction * len(train_records))))
    val_ids = set(idx[:n_val].tolist())
    train = [r for i, r in enumerate(train_records) if i not in val_ids]
    val = [r for i, r in enumerate(train_records) if i in val_ids]
    return train, val
