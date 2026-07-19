import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from aegisbench.datasets.common import parse_voc_xml, parse_yolo_label
from aegisbench.datasets.sard import (group_key, group_split,
                                      load_pooled_roboflow_yolo)

VOC = textwrap.dedent("""\
    <annotation>
      <filename>img001.jpg</filename>
      <size><width>4000</width><height>3000</height><depth>3</depth></size>
      <object>
        <name>human</name>
        <bndbox><xmin>120</xmin><ymin>200</ymin><xmax>160</xmax>
        <ymax>260</ymax></bndbox>
      </object>
      <object>
        <name>dog</name>
        <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax>
        <ymax>50</ymax></bndbox>
      </object>
      <object>
        <name>person</name>
        <bndbox><xmin>3990</xmin><ymin>2990</ymin><xmax>4100</xmax>
        <ymax>3100</ymax></bndbox>
      </object>
    </annotation>
""")


def test_parse_voc_xml(tmp_path):
    xml = tmp_path / "img001.xml"
    xml.write_text(VOC)
    rec = parse_voc_xml(xml)
    assert rec["width"] == 4000 and rec["height"] == 3000
    # dog ignored; out-of-bounds person clipped to image and kept
    assert rec["n_ignored_objects"] == 1
    assert len(rec["boxes"]) == 2
    assert np.allclose(rec["boxes"][0], [120, 200, 160, 260])
    assert rec["boxes"][1][2] <= 4000 and rec["boxes"][1][3] <= 3000


def test_group_key_strips_frame_counter():
    assert group_key("video7_frame00123") == "video7_frame"
    assert group_key("seq_03-0456") == "seq_03"
    assert group_key("DJI_0042_00123") == "DJI_0042"


def _fake_records(n_groups=8, frames_per_group=30):
    recs = []
    for g in range(n_groups):
        for f in range(frames_per_group):
            recs.append({"image_id": f"vid{g:02d}_{f:05d}",
                         "image_path": "x", "width": 1920, "height": 1080,
                         "boxes": np.zeros((0, 4), np.float32)})
    return recs


def test_group_split_no_leakage():
    recs = _fake_records()
    train, val, test, info = group_split(recs)
    assert len(train) + len(val) + len(test) == len(recs)
    for a, b in ((train, val), (train, test), (val, test)):
        ga = {group_key(r["image_id"]) for r in a}
        gb = {group_key(r["image_id"]) for r in b}
        assert not ga & gb, "video group straddles splits"


def test_group_split_rejects_degenerate_grouping():
    recs = _fake_records(n_groups=1)
    with pytest.raises(ValueError):
        group_split(recs)


def _make_image(path, width=200, height=100):
    Image.new("RGB", (width, height), (0, 128, 0)).save(path)


def test_parse_yolo_label_converts_normalized_to_absolute(tmp_path):
    img = tmp_path / "im001.jpg"
    _make_image(img, width=200, height=100)
    label = tmp_path / "im001.txt"
    # class cx cy w h, normalized; box centered at (0.5,0.5) size (0.2,0.4)
    label.write_text("0 0.5 0.5 0.2 0.4\n")
    rec = parse_yolo_label(img, label)
    assert rec["width"] == 200 and rec["height"] == 100
    assert len(rec["boxes"]) == 1
    assert np.allclose(rec["boxes"][0], [80, 30, 120, 70], atol=1e-3)


def test_parse_yolo_label_missing_file_returns_empty(tmp_path):
    img = tmp_path / "im002.jpg"
    _make_image(img)
    rec = parse_yolo_label(img, tmp_path / "does_not_exist.txt")
    assert len(rec["boxes"]) == 0


def test_parse_yolo_label_filters_by_class_id(tmp_path):
    img = tmp_path / "im003.jpg"
    _make_image(img, width=200, height=100)
    label = tmp_path / "im003.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1\n")
    rec_all = parse_yolo_label(img, label)
    assert len(rec_all["boxes"]) == 2
    rec_person_only = parse_yolo_label(img, label, person_class_ids={0})
    assert len(rec_person_only["boxes"]) == 1
    assert rec_person_only["n_ignored_objects"] == 1


def _make_roboflow_split(root, split, image_ids, width=200, height=100):
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        _make_image(img_dir / f"{image_id}.jpg", width, height)
        (lbl_dir / f"{image_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_load_pooled_roboflow_yolo_combines_all_splits(tmp_path):
    root = tmp_path / "sard_export"
    _make_roboflow_split(root, "train", ["vidA_001", "vidA_002"])
    _make_roboflow_split(root, "valid", ["vidB_001"])
    _make_roboflow_split(root, "test", ["vidC_001"])
    records = load_pooled_roboflow_yolo(root)
    assert len(records) == 4
    assert {r["image_id"] for r in records} == {
        "vidA_001", "vidA_002", "vidB_001", "vidC_001"}
    for r in records:
        assert len(r["boxes"]) == 1


def test_load_pooled_roboflow_yolo_then_group_split_no_leakage(tmp_path):
    root = tmp_path / "sard_export"
    ids_a = [f"vidA_{i:03d}" for i in range(10)]
    ids_b = [f"vidB_{i:03d}" for i in range(10)]
    # Deliberately split one video's frames across Roboflow's train/valid
    # to simulate the exact leakage risk this function exists to undo.
    _make_roboflow_split(root, "train", ids_a[:5] + ids_b[:5])
    _make_roboflow_split(root, "valid", ids_a[5:] + ids_b[5:])
    records = load_pooled_roboflow_yolo(root)
    train, val, test, info = group_split(records,
                                         fractions=(0.5, 0.25, 0.25))
    for a, b in ((train, val), (train, test), (val, test)):
        ga = {group_key(r["image_id"]) for r in a}
        gb = {group_key(r["image_id"]) for r in b}
        assert not ga & gb, "video group straddles splits after rebuild"
