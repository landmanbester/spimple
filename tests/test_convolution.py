import numpy as np
import pytest
from africanus.model.spi import fit_spi_components
from numpy.testing._private.utils import assert_allclose

from spimple.utils.convolution import Gaussian2D, convolve2gaussres

pmp = pytest.mark.parametrize


@pmp("nx", [128])
@pmp("ny", [80, 220])
@pmp("nband", [4, 8])
@pmp("alpha", [-0.5, 0.0, 0.5])
def test_convolve2gaussres(nx, ny, nband, alpha):
    freq = np.linspace(0.5e9, 1.5e9, nband)
    ref_freq = freq[0]

    Gausspari = ()
    es = np.linspace(15, 5, nband)
    for v in range(nband):
        Gausspari += ((es[v], es[v], 0.0),)

    x = np.arange(-nx / 2, nx / 2)
    y = np.arange(-ny / 2, ny / 2)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    restored = np.zeros((nband, nx, ny))
    for v in range(nband):
        restored[v] = Gaussian2D(xx, yy, Gausspari[v], normalise=False) * (freq[v] / ref_freq) ** alpha

    conv_model, gausskern = convolve2gaussres(restored, xx, yy, Gausspari[0], 8, gausspari=Gausspari)

    Ix, Iy = np.where(conv_model[-1] > 0.05)

    comps = conv_model[:, Ix, Iy]
    weights = np.ones(nband)

    out = fit_spi_components(comps.T, weights, freq, ref_freq, tol=1e-7, maxiter=250)

    # offset for relative difference
    assert_allclose(1 + alpha, 1 + out[0, :], atol=5e-4, rtol=5e-4)
    assert_allclose(out[2, :], restored[0, Ix, Iy], atol=5e-4, rtol=5e-4)


def test_gaussian2d_has_the_requested_fwhm():
    """Half maximum falls at emaj/2 along y and emin/2 along x when pa is zero.

    The quadratic form puts the major axis along y at pa = 0, matching the FITS
    convention where BPA is measured from north.
    """
    n = 129
    c = n // 2
    v = np.arange(n) - c
    xx, yy = np.meshgrid(v, v, indexing="ij")

    kern = Gaussian2D(xx, yy, (20.0, 10.0, 0.0), normalise=False, nsigma=10)

    assert kern[c, c] == pytest.approx(1.0)
    assert kern[c, c + 10] == pytest.approx(0.5, rel=1e-12)
    assert kern[c + 5, c] == pytest.approx(0.5, rel=1e-12)


def test_gaussian2d_support_is_nsigma_standard_deviations():
    """The kernel is truncated at nsigma sigma, not nsigma FWHM."""
    n = 257
    c = n // 2
    v = np.arange(n) - c
    xx, yy = np.meshgrid(v, v, indexing="ij")

    kern = Gaussian2D(xx, yy, (20.0, 20.0, 0.0), normalise=False, nsigma=5)

    sigma = 20.0 / (2 * np.sqrt(2 * np.log(2)))
    inside = int(4.0 * sigma)
    outside = int(6.0 * sigma)
    assert kern[c + inside, c] > 0.0
    assert kern[c + outside, c] == 0.0


def _grids(nx, ny):
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    return np.meshgrid(x, y, indexing="ij")


def test_convolve2gaussres_yx_order_matches_the_transposed_call():
    """Convolving (n, y, x) with yx_order equals convolving its transpose without."""
    nx, ny, nband = 48, 32, 3
    xx, yy = _grids(nx, ny)
    rng = np.random.default_rng(7)
    xmajor = rng.normal(size=(nband, nx, ny))
    yxmajor = xmajor.transpose(0, 2, 1).copy()
    target = (6.0, 4.0, 0.4)

    ref, ref_kern = convolve2gaussres(xmajor, xx, yy, target, 1)
    out, out_kern = convolve2gaussres(yxmajor, xx, yy, target, 1, yx_order=True)

    np.testing.assert_allclose(out, ref.transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(out_kern, ref_kern.transpose(0, 2, 1), atol=1e-12)


def test_convolve2gaussres_accepts_a_target_per_plane():
    """A (nplane, 3) gaussparf convolves each plane to its own resolution."""
    nx, ny = 64, 64
    xx, yy = _grids(nx, ny)
    delta = np.zeros((2, nx, ny))
    delta[:, nx // 2, ny // 2] = 1.0
    targets = np.array([[8.0, 8.0, 0.0], [4.0, 4.0, 0.0]])

    out, _ = convolve2gaussres(delta, xx, yy, targets, 1, norm_kernel=False)

    for plane in range(targets.shape[0]):
        expected = Gaussian2D(xx, yy, tuple(targets[plane]), normalise=False)
        np.testing.assert_allclose(out[plane], expected, atol=1e-8)


def test_convolve2gaussres_rejects_a_mismatched_per_plane_target():
    nx, ny = 32, 32
    xx, yy = _grids(nx, ny)
    image = np.zeros((3, nx, ny))
    targets = np.array([[8.0, 8.0, 0.0], [4.0, 4.0, 0.0]])

    with pytest.raises(ValueError, match="gaussparf"):
        convolve2gaussres(image, xx, yy, targets, 1)


def test_convolve2gaussres_accepts_a_numpy_gausspari():
    """gausspari as an array must not trip an elementwise truth test."""
    nx, ny = 48, 48
    xx, yy = _grids(nx, ny)
    image = np.zeros((2, nx, ny))
    image[:, nx // 2, ny // 2] = 1.0
    gausspari = np.array([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0]])
    gaussparf = np.array([[8.0, 8.0, 0.0], [8.0, 8.0, 0.0]])

    out, _ = convolve2gaussres(image, xx, yy, gaussparf, 1, gausspari=gausspari)

    assert np.isfinite(out).all()
    assert out[0, nx // 2, ny // 2] > 0.0
