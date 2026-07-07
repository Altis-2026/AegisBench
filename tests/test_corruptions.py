"""Corruption engine invariants: shape/range preservation, determinism,
severity distinctness, and monotonic calibration statistics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.corruptions import SEVERITIES, CorruptionSuite
from aegisbench.corruptions.calibration import measure
from aegisbench.corruptions.noise_fields import value_noise

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corruptions.yaml"


@pytest.fixture(scope="module")
def suite():
    return CorruptionSuite(CONFIG)


@pytest.fixture(scope="module")
def textured_image():
    """Deterministic textured RGB image — corruptions need real structure
    (contrast/edges) for calibration stats to be meaningful."""
    rng = np.random.default_rng(123)
    base = value_noise((360, 480), rng, octaves=5, base_res=4)
    detail = value_noise((360, 480), rng, octaves=3, base_res=80)
    img = np.stack([0.25 + 0.5 * base, 0.35 + 0.4 * base,
                    0.2 + 0.3 * base], -1) + 0.15 * (detail[..., None] - 0.5)
    return np.clip(img, 0, 1).astype(np.float32)


def test_expected_corruption_set(suite):
    assert set(suite.names()) == {
        "water_glare", "turbidity_cast", "inundation", "smoke_haze",
        "fire_warm_tint", "rain_streaks", "motion_blur", "low_light",
        "dust_haze"}
    families = {suite.family(n) for n in suite.names()}
    assert families == {"flood", "wildfire", "storm", "earthquake"}


def test_output_contract(suite, textured_image):
    for name in suite.names():
        for sev in SEVERITIES:
            out = suite.apply(textured_image, name, sev, "imgA")
            assert out.shape == textured_image.shape
            assert out.dtype == np.float32
            assert out.min() >= 0.0 and out.max() <= 1.0
            assert not np.allclose(out, textured_image), \
                f"{name} s{sev} is a no-op"


def test_uint8_roundtrip(suite, textured_image):
    u8 = (textured_image * 255).astype(np.uint8)
    out = suite.apply(u8, "smoke_haze", 2, "imgA")
    assert out.dtype == np.uint8 and out.shape == u8.shape


def test_determinism_and_image_id_sensitivity(suite, textured_image):
    for name in suite.names():
        a = suite.apply(textured_image, name, 2, "imgA")
        b = suite.apply(textured_image, name, 2, "imgA")
        assert np.array_equal(a, b), f"{name} not deterministic"
    # Stochastic corruptions must differ across image ids (fresh streams).
    a = suite.apply(textured_image, "rain_streaks", 2, "imgA")
    c = suite.apply(textured_image, "rain_streaks", 2, "imgB")
    assert not np.array_equal(a, c)


def test_severities_are_distinct(suite, textured_image):
    for name in suite.names():
        outs = [suite.apply(textured_image, name, s, "imgA")
                for s in SEVERITIES]
        assert not np.array_equal(outs[0], outs[1])
        assert not np.array_equal(outs[1], outs[2])


def test_calibration_monotonicity(suite, textured_image):
    """The measurable statistic behind each severity ladder must move the
    documented direction — averaged over a few image ids to absorb
    stochastic wiggle, exactly like phase3_calibrate.py does on real data."""
    ids = [f"img{i}" for i in range(6)]
    for name in suite.names():
        calib = suite.calibration(name)
        stat, direction = calib["stat"], calib["direction"]
        if stat == "streak_density":
            continue  # audited visually; no closed-form statistic
        means = []
        for sev in SEVERITIES:
            vals = [measure(stat, suite.apply(textured_image, name, sev, i),
                            textured_image) for i in ids]
            means.append(float(np.mean(vals)))
        pairs = list(zip(means, means[1:]))
        if direction == "increasing":
            assert all(a < b for a, b in pairs), f"{name}: {means}"
        else:
            assert all(a > b for a, b in pairs), f"{name}: {means}"


def test_smoke_and_dust_are_chromatically_distinct(suite, textured_image):
    """Dust must skew warmer (brown airlight) than smoke at equal-ish
    optical depth — the physical distinction the taxonomy claims."""
    smoke = suite.apply(textured_image, "smoke_haze", 3, "imgA")
    dust = suite.apply(textured_image, "dust_haze", 3, "imgA")
    rb_smoke = smoke[..., 0].mean() / smoke[..., 2].mean()
    rb_dust = dust[..., 0].mean() / dust[..., 2].mean()
    assert rb_dust > rb_smoke + 0.1
