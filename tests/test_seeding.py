import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegisbench.seeding import rng_for, stable_seed


def test_stable_seed_reproducible():
    assert stable_seed("img1", "smoke_haze", 2) == \
        stable_seed("img1", "smoke_haze", 2)


def test_stable_seed_distinct_across_keys():
    seeds = {stable_seed(img, c, s)
             for img in ("a", "b")
             for c in ("smoke_haze", "dust_haze")
             for s in (1, 2, 3)}
    assert len(seeds) == 12


def test_no_separator_collisions():
    # ("ab", "c") must not collide with ("a", "bc").
    assert stable_seed("ab", "c") != stable_seed("a", "bc")


def test_rng_streams_independent():
    a = rng_for("img1", "smoke_haze", 1).random(8)
    b = rng_for("img1", "smoke_haze", 2).random(8)
    assert not (a == b).all()
