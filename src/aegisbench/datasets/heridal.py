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
        rec = parse_voc_xml(xml_path, image_path=img_path)
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


def load_from_imagesets(voc_root: str | Path, split_name: str,
                        images_subdir: str = "JPEGImages",
                        annotations_subdir: str = "Annotations",
                        imagesets_subdir: str = "ImageSets/Main"
                        ) -> list[dict]:
    """Load a split from a standard PASCAL-VOC layout where images and
    annotations live in ONE shared pool (JPEGImages/, Annotations/) and
    ImageSets/Main/<split_name>.txt just lists which image ids belong to
    that split -- as opposed to load_split()'s per-split directory layout.

    This is what re-packaged VOC-format HERIDAL mirrors typically use
    (e.g. the keras-retinanet conversion), and it carries the ORIGINAL
    author-provided train/val/test membership rather than a re-split.
    """
    root = Path(voc_root)
    ids_file = root / imagesets_subdir / f"{split_name}.txt"
    ids = [line.strip() for line in ids_file.read_text().splitlines()
          if line.strip()]
    img_dir = root / images_subdir
    ann_dir = root / annotations_subdir
    records, missing = [], []
    for image_id in ids:
        img_path = next((img_dir / f"{image_id}{ext}" for ext in IMG_EXTS
                         if (img_dir / f"{image_id}{ext}").exists()), None)
        xml_path = ann_dir / f"{image_id}.xml"
        if img_path is None or not xml_path.exists():
            missing.append(image_id)
            continue
        rec = parse_voc_xml(xml_path, image_path=img_path)
        rec["image_id"] = image_id
        rec["image_path"] = str(img_path)
        records.append(rec)
    if missing:
        print(f"[heridal] WARNING: {len(missing)} ids listed in "
              f"'{split_name}.txt' had no image/annotation pair, skipped "
              f"(first few: {missing[:5]}) -- likely negative/no-person "
              "images excluded from this repackaging's annotation set.")
    return records
