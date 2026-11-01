import numpy as np
import pytest

from ipshakti.config import species_of
from ipshakti.traits import estimate_traits, linear_score


def synthetic_animal():
    """Cow-like side silhouette, head on the left, flat back, four legs."""
    m = np.zeros((300, 400), dtype=bool)
    m[60:141, 100:300] = True          # torso
    m[70:111, 60:100] = True           # head/neck
    for a, b in [(110, 125), (160, 175), (250, 265), (280, 295)]:
        m[140:210, a:b] = True         # legs
    return m


def test_measurements_reasonable():
    r = estimate_traits(synthetic_animal())
    assert 140 <= r.measurements["withers_height_px"] <= 160
    assert 0.45 <= r.measurements["body_depth_ratio"] <= 0.60
    assert abs(r.measurements["rump_angle_deg"]) < 3
    assert all(1 <= s <= 9 for s in r.scores.values())
    assert "stature" not in r.scores


def test_scale_adds_stature():
    r = estimate_traits(synthetic_animal(), pixels_per_cm=1.0)
    assert "stature" in r.scores
    assert r.measurements["withers_height_cm"] == r.measurements["withers_height_px"]


def test_mirrored_image_gives_same_result():
    a = estimate_traits(synthetic_animal())
    b = estimate_traits(np.fliplr(synthetic_animal()))
    assert a.measurements == b.measurements


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        estimate_traits(np.zeros((50, 50), dtype=bool))


def test_linear_score_clipping():
    assert linear_score(-100, 0, 10) == 1
    assert linear_score(100, 0, 10) == 9
    assert linear_score(5, 0, 10) == 5


def test_species_lookup():
    assert species_of("Murrah") == "buffalo"
    assert species_of("Red Sindhi") == "cattle"
    assert species_of("unknown-breed") == "unknown"
