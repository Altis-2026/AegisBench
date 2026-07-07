import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.datasets.common import parse_voc_xml
from aegisbench.datasets.sard import group_key, group_split

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
