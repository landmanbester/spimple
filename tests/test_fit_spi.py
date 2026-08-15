"""Per component weighting in the spectral index fit.

The load bearing property is that fitting the apparent image with the beam in
the model and fitting the intrinsic image with B ** 2 weights are the same
least squares problem:

    sum_v w_v (d_app - B_v M_v) ** 2 = sum_v (w_v B_v ** 2) (d_int - M_v) ** 2

That identity is what makes spifit's intrinsic path a consistency check on the
apparent path rather than a second, differently biased estimator.
"""

import numpy as np
import pytest

from spimple.utils.fit_spi import fit_spi_components_np

pmp = pytest.mark.parametrize


def _spectra(nband=8, ncomps=64, alpha=-0.7, seed=42, noise=0.0):
    """Apparent and intrinsic spectra on a beam that narrows with frequency."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.9e9, 1.7e9, nband)
    freq0 = freqs[nband // 2]

    i0 = rng.uniform(0.5, 5.0, ncomps)
    alphas = alpha + rng.normal(0.0, 0.1, ncomps)
    intrinsic = i0[:, None] * (freqs[None, :] / freq0) ** alphas[:, None]

    # a gaussian primary beam whose width scales as 1 / nu, sampled over a range
    # of radii so the components span the usable part of the field
    radius = np.linspace(0.0, 0.5, ncomps)
    width = 0.45 * freqs[0] / freqs
    beam = np.exp(-0.5 * (radius[:, None] / width[None, :]) ** 2)

    apparent = beam * intrinsic
    if noise:
        apparent = apparent + rng.normal(0.0, noise, apparent.shape)
    return freqs, freq0, beam, apparent, i0, alphas


@pmp("noise", [0.0, 1e-3])
def test_intrinsic_with_beam_squared_weights_matches_the_apparent_fit(noise):
    """The two parameterisations are one least squares problem, to machine precision."""
    freqs, freq0, beam, apparent, _, _ = _spectra(noise=noise)
    w = np.linspace(1.0, 0.4, freqs.size)  # unequal band weights, as WSUM gives

    app = fit_spi_components_np(apparent, w, freqs, freq0, beam=beam, tol=1e-10, maxiter=500)
    intr = fit_spi_components_np(apparent / beam, w[None, :] * beam**2, freqs, freq0, tol=1e-10, maxiter=500)

    np.testing.assert_allclose(intr[0], app[0], rtol=0, atol=1e-10)  # alpha
    np.testing.assert_allclose(intr[2], app[2], rtol=0, atol=1e-10)  # I0
    # the variances are chi squared rescaled, so noiseless they are pure
    # roundoff; atol keeps that from being compared relatively
    np.testing.assert_allclose(intr[1], app[1], rtol=1e-9, atol=1e-20)  # alpha variance
    np.testing.assert_allclose(intr[3], app[3], rtol=1e-9, atol=1e-20)  # I0 variance


def test_flat_weights_on_the_intrinsic_image_disagree_with_the_apparent_fit():
    """Guards the equivalence above against a vacuous pass.

    Unity weights are what spifit used before, and they give a measurably
    different answer wherever the beam varies.
    """
    freqs, freq0, beam, apparent, _, _ = _spectra(noise=1e-3)
    w = np.ones(freqs.size)

    app = fit_spi_components_np(apparent, w, freqs, freq0, beam=beam, tol=1e-10, maxiter=500)
    flat = fit_spi_components_np(apparent / beam, w, freqs, freq0, tol=1e-10, maxiter=500)

    assert np.abs(flat[0] - app[0]).max() > 1e-3


def test_beam_squared_weights_beat_flat_weights_on_noisy_edge_components():
    """The point of the change: lower alpha scatter where the beam varies most."""
    freqs, freq0, beam, apparent, _, alphas = _spectra(ncomps=400, noise=2e-3)
    w = np.ones(freqs.size)

    flat = fit_spi_components_np(apparent / beam, w, freqs, freq0, tol=1e-10, maxiter=500)
    wgt = fit_spi_components_np(apparent / beam, w[None, :] * beam**2, freqs, freq0, tol=1e-10, maxiter=500)

    edge = beam.min(axis=1) < 0.5
    assert np.std(wgt[0][edge] - alphas[edge]) < np.std(flat[0][edge] - alphas[edge])


def test_a_1d_weight_vector_is_broadcast_over_components():
    """Back compatibility with the africanus signature."""
    freqs, freq0, beam, apparent, _, _ = _spectra()
    w = np.linspace(1.0, 0.4, freqs.size)

    one_d = fit_spi_components_np(apparent, w, freqs, freq0, beam=beam, tol=1e-10, maxiter=500)
    two_d = fit_spi_components_np(
        apparent, np.broadcast_to(w, apparent.shape).copy(), freqs, freq0, beam=beam, tol=1e-10, maxiter=500
    )

    np.testing.assert_allclose(two_d, one_d, rtol=0, atol=0)


def test_mismatched_weights_are_rejected():
    freqs, freq0, beam, apparent, _, _ = _spectra()

    with pytest.raises(ValueError, match="does not match data"):
        fit_spi_components_np(apparent, np.ones((3, freqs.size)), freqs, freq0, beam=beam)


def test_the_dask_wrapper_accepts_per_component_weights():
    da = pytest.importorskip("dask.array")
    freqs, freq0, beam, apparent, _, _ = _spectra(ncomps=64)
    from spimple.utils.fit_spi import fit_spi_components

    w = np.linspace(1.0, 0.4, freqs.size)[None, :] * beam**2
    chunks = (16, freqs.size)

    out = fit_spi_components(
        da.from_array(apparent / beam, chunks=chunks),
        da.from_array(w, chunks=chunks),
        da.from_array(freqs, chunks=(freqs.size,)),
        np.float64(freq0),
        tol=1e-10,
        maxiter=500,
    ).compute()
    expected = fit_spi_components_np(apparent / beam, w, freqs, freq0, tol=1e-10, maxiter=500)

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
