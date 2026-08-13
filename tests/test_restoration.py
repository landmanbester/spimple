"""The three restored flux scales."""

import numpy as np
import pytest

from spimple.utils.restoration import restore_products


def _inputs(npix=64, ncorr=1):
    model = np.zeros((ncorr, npix, npix))
    model[:, npix // 2, npix // 2] = 1.0
    residual = np.zeros((ncorr, npix, npix))
    v = np.arange(npix) - npix // 2
    yy, xx = np.meshgrid(v, v, indexing="ij")
    beam = np.exp(-(xx**2 + yy**2) / (2 * (npix / 3.0) ** 2))[None]
    beam = np.repeat(beam, ncorr, axis=0)
    return model, residual, beam


def test_mixed_is_the_convolved_model_plus_the_residual():
    model, residual, beam = _inputs()
    residual[:] = 0.25
    gaussparf = np.tile([6.0, 4.0, 0.0], (1, 1))

    out = restore_products(model, residual, beam, gaussparf, products=("k",))

    # the model is a delta, so its convolution peaks at 1 for an unnormalised kernel
    assert out["k"].shape == model.shape
    assert out["k"][0, 32, 32] == pytest.approx(1.25, rel=1e-6)
    assert out["k"][0, 0, 0] == pytest.approx(0.25, abs=1e-8)


def test_intrinsic_divides_the_residual_by_the_beam_and_floors_it():
    model, residual, beam = _inputs()
    residual[:] = 0.25
    gaussparf = np.tile([6.0, 4.0, 0.0], (1, 1))

    out = restore_products(model, residual, beam, gaussparf, products=("i",), pb_min=0.5)

    centre = out["i"][0, 32, 32]
    assert centre == pytest.approx(1.0 + 0.25 / beam[0, 32, 32], rel=1e-6)
    assert np.all(out["i"][0][beam[0] <= 0.5] == 0.0)


def test_apparent_attenuates_before_convolving_not_after():
    """(B*m) convolved is not B*(m convolved) for a spatially varying beam."""
    npix = 64
    model = np.zeros((1, npix, npix))
    model[0, npix // 2 - 8, npix // 2] = 1.0
    model[0, npix // 2 + 8, npix // 2] = 1.0
    residual = np.zeros((1, npix, npix))
    v = np.arange(npix) - npix // 2
    yy, xx = np.meshgrid(v, v, indexing="ij")
    beam = np.exp(-((xx - 20.0) ** 2 + yy**2) / (2 * 15.0**2))[None]
    gaussparf = np.tile([6.0, 6.0, 0.0], (1, 1))

    out = restore_products(model, residual, beam, gaussparf, products=("a", "k"))

    naive = beam * out["k"]
    assert not np.allclose(out["a"], naive, atol=1e-6)


def test_residual_is_reconvolved_when_the_resolutions_differ():
    npix = 96
    model = np.zeros((1, npix, npix))
    residual = np.zeros((1, npix, npix))
    residual[0, npix // 2, npix // 2] = 1.0
    beam = np.ones((1, npix, npix))
    gausspari = np.tile([4.0, 4.0, 0.0], (1, 1))
    gaussparf = np.tile([8.0, 8.0, 0.0], (1, 1))

    same = restore_products(model, residual, beam, gausspari, gausspari=gausspari, products=("k",))
    wider = restore_products(model, residual, beam, gaussparf, gausspari=gausspari, products=("k",))

    # reconvolution to a coarser beam spreads the delta out
    np.testing.assert_allclose(same["k"], residual, atol=1e-8)
    assert wider["k"][0, npix // 2, npix // 2] < residual[0, npix // 2, npix // 2]
    assert wider["k"].sum() > residual.sum() * 0.9


def test_requesting_no_products_returns_an_empty_dict():
    model, residual, beam = _inputs()
    gaussparf = np.tile([6.0, 4.0, 0.0], (1, 1))

    assert restore_products(model, residual, beam, gaussparf, products=()) == {}
