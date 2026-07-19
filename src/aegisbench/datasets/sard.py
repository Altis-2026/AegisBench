"""SARD loader.

SARD (Sambolek & Ivasic-Kos, IEEE Access 2021) consists of 1920x1080 frames
extracted from video of actors simulating injured/lost persons. Depending on
where you obtained it, annotations are either the original PASCAL-VOC XML
(IEEE DataPort release) or YOLO-format .txt (common on Roboflow/Kaggle
re-exports, which typically also pre-split into train/valid/test folders).
See docs/DATA.md.

CRITICAL SPLIT RULE — frames extracted from the same video sequence are
near-duplicates. A random frame-level split leaks train content into test
and inflates every metric. We therefore split at the GROUP level, where a
group is derived from the filename prefix shared by frames of one sequence.
This means any train/valid/test split a third-party mirror already applied
must NOT be trusted as-is (it's typically a random per-frame shuffle) —
load_pooled_roboflow_yolo() deliberately pools all of a mirror's splits back
together so our own group_split() can rebuild a non-leaking one.

The default group regex strips a trailing frame counter:
    'video7_frame00123' / 'seq_03-0456' / 'DJI_0042_00123' -> prefix
VERIFY THIS AGAINST YOUR ACTUAL SARD FILENAMES before trusting the split:
scripts/phase1_prepare_sard.py prints the discovered groups and sizes for
manual review, and refuses to proceed if it finds only one group or as many
groups as files (both mean the regex failed).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..seeding import stable_seed
from .common import parse_voc_xml, parse_yolo_label

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
DEFAULT_GROUP_REGEX = r"^(.*?)[-_]?\d+$"


def load_all(image_dir: str | Path,
             label_dir: str | Path | None = None) -> list[dict]:
    """Load a VOC-XML-annotated SARD release (e.g. the original IEEE
    DataPort distribution) from a single images(+labels) directory."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir) if label_dir else image_dir
    records, missing = [], []
    for img_path in sorted(p for p in image_dir.rglob("*")
                           if p.suffix in IMG_EXTS):
        xml_path = Path(label_dir) / img_path.relative_to(
            image_dir).with_suffix(".xml")
        if not xml_path.exists():
            xml_path = Path(label_dir) / f"{img_path.stem}.xml"
        if not xml_path.exists():
            missing.append(img_path.name)
            continue
        rec = parse_voc_xml(xml_path)
        rec["image_id"] = img_path.stem
        rec["image_path"] = str(img_path)
        records.append(rec)
    if missing:
        print(f"[sard] WARNING: {len(missing)} images without labels "
              f"skipped (first few: {missing[:5]})")
    return records


def load_pooled_roboflow_yolo(root_dir: str | Path,
                              splits: tuple[str, ...] = ("train", "valid",
                                                         "test")
                              ) -> list[dict]:
    """Load a Roboflow-style YOLO export laid out as
    root/{train,valid,test}/{images,labels}/, POOLING every split back into
    one list. Roboflow's own split is a per-frame shuffle, not video-aware,
    so we deliberately discard it here and let group_split() rebuild a
    non-leaking one from the pooled set.
    """
    root = Path(root_dir)
    records, missing = [], []
    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(p for p in img_dir.iterdir()
                               if p.suffix in IMG_EXTS):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing.append(img_path.name)
                continue
            rec = parse_yolo_label(img_path, label_path)
            rec["image_path"] = str(img_path)
            records.append(rec)
    if missing:
        print(f"[sard] WARNING: {len(missing)} images without labels "
              f"skipped (first few: {missing[:5]})")
    seen = set()
    dupes = [r["image_id"] for r in records if r["image_id"] in seen
             or seen.add(r["image_id"])]
    if dupes:
        print(f"[sard] WARNING: {len(dupes)} duplicate image_id(s) across "
              f"pooled splits (first few: {dupes[:5]}) — Roboflow "
              "sometimes re-emits the same source frame with augmented "
              "copies in multiple splits; verify these aren't near-"
              "duplicate leaks before trusting the rebuilt split.")
    return records


def group_key(image_id: str, regex: str = DEFAULT_GROUP_REGEX) -> str:
    m = re.match(regex, image_id)
    return m.group(1) if m and m.group(1) else image_id


def group_split(records: list[dict], fractions=(0.70, 0.15, 0.15),
                regex: str = DEFAULT_GROUP_REGEX,
                seed_key: str = "sard-group-split"
                ) -> tuple[list, list, list, dict]:
    """Group-aware train/val/test split. Returns (train, val, test, info)."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(group_key(r["image_id"], regex), []).append(r)

    info = {"n_groups": len(groups),
            "group_sizes": {k: len(v) for k, v in groups.items()}}
    if len(groups) <= 1 or len(groups) == len(records):
        raise ValueError(
            f"Group regex produced {len(groups)} groups for "
            f"{len(records)} images — it is not capturing video sequences. "
            "Inspect filenames and pass a corrected regex.")

    keys = sorted(groups)
    rng = np.random.default_rng(stable_seed(seed_key))
    rng.shuffle(keys)
    n = len(records)
    train, val, test = [], [], []
    acc = 0
    for k in keys:
        frac = acc / n
        bucket = (train if frac < fractions[0]
                  else val if frac < fractions[0] + fractions[1]
                  else test)
        bucket.extend(groups[k])
        acc += len(groups[k])
    info["split_sizes"] = {"train": len(train), "val": len(val),
                           "test": len(test)}
    return train, val, test, info
