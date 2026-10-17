import numpy as np
import pytest

from src.ml.drift import level, psi, psi_edges


def test_identical_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    y = rng.normal(size=20000)  # same distribution, different sample
    assert psi(x, y) < 0.01


def test_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    shifted = rng.normal(loc=1.5, size=20000)
    assert psi(ref, shifted) > 0.25


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20000)
    small = psi(ref, rng.normal(loc=0.2, size=20000))
    big = psi(ref, rng.normal(loc=1.0, size=20000))
    assert 0 < small < big


def test_heavily_tied_reference_does_not_crash():
    ref = np.r_[np.zeros(900), np.linspace(0.1, 1, 100)]
    assert len(psi_edges(ref)) < 9  # tied quantiles collapse
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, np.ones(500)) > 0.25


def test_levels():
    assert level(0.05) == "ok"
    assert level(0.1) == "moderate"
    assert level(0.25) == "major"
