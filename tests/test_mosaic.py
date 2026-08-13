"""Combining image-space partitions into a band image."""

import numpy as np

from spimple.utils.mosaic import conjugate_gradient, stitch_band

NY, NX = 16, 24


def _beam(shift, sigma=8.0):
    v_y = np.arange(NY) - NY // 2
    v_x = np.arange(NX) - NX // 2 - shift
    yy, xx = np.meshgrid(v_y, v_x, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def _scene(shifts, sky=None):
    sky = np.ones((NY, NX)) if sky is None else sky
    beams = np.stack([_beam(s)[None] for s in shifts])
    apparent = beams * sky[None, None]
    masks = np.ones((len(shifts), NY, NX), dtype=bool)
    return sky, apparent, beams, masks


def test_stitch_recovers_the_intrinsic_sky():
    rng = np.random.default_rng(4)
    sky = 1.0 + rng.uniform(size=(NY, NX))
    _, apparent, beams, masks = _scene([-4, 4], sky=sky)

    out = stitch_band(apparent, beams, masks, eta=1e-8)

    np.testing.assert_allclose(out["IMAGE"][0], sky, rtol=1e-5)


def test_stitch_beam_reduces_to_the_common_beam():
    """Identical partition beams must give BEAM == B."""
    _, apparent, beams, masks = _scene([0, 0])

    out = stitch_band(apparent, beams, masks, eta=0.0)

    np.testing.assert_allclose(out["BEAM"][0], beams[0, 0], rtol=1e-9)


def test_stitch_beam_reduces_to_the_only_covering_partition():
    """Where one partition alone covers a pixel, BEAM is that partition's beam."""
    _, apparent, beams, masks = _scene([-6, 6])
    masks[1, :, : NX // 2] = False

    out = stitch_band(apparent, beams, masks, eta=0.0)

    np.testing.assert_allclose(out["BEAM"][0, :, 0], beams[0, 0, :, 0], rtol=1e-9)


def test_bimage_is_the_beam_weighted_mean_apparent_image():
    _, apparent, beams, masks = _scene([-4, 4])

    out = stitch_band(apparent, beams, masks, eta=0.0)

    expected = (beams * apparent).sum(axis=0) / beams.sum(axis=0)
    np.testing.assert_allclose(out["BIMAGE"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["BIMAGE"], out["BEAM"] * out["IMAGE"], rtol=1e-6)


def test_spatial_weight_is_the_sum_of_squared_beams():
    _, apparent, beams, masks = _scene([-4, 4])
    eta = 1e-3

    out = stitch_band(apparent, beams, masks, eta=eta)

    np.testing.assert_allclose(out["SPATIALWGT"], (beams**2).sum(axis=0) + eta, rtol=1e-9)


def test_masked_partitions_do_not_contribute():
    _, apparent, beams, masks = _scene([-4, 4])
    masks[1] = False
    apparent[1] = 1e6  # would dominate if it leaked in

    out = stitch_band(apparent, beams, masks, eta=1e-8)

    np.testing.assert_allclose(out["IMAGE"][0], 1.0, rtol=1e-5)


def test_uncovered_pixels_are_zero_not_nan():
    _, apparent, beams, masks = _scene([-4, 4])
    masks[:, :, 0] = False

    out = stitch_band(apparent, beams, masks, eta=0.0)

    assert np.isfinite(out["IMAGE"]).all()
    np.testing.assert_allclose(out["IMAGE"][:, :, 0], 0.0)


def test_mixed_scale_is_averaged_not_resolved():
    _, apparent, beams, masks = _scene([-4, 4])
    mixed = apparent + 0.5

    out = stitch_band(apparent, beams, masks, mixed=mixed, eta=0.0)

    expected = (beams * mixed).sum(axis=0) / beams.sum(axis=0)
    np.testing.assert_allclose(out["KIMAGE"], expected, rtol=1e-9)


def test_wsum_and_rms_are_propagated():
    _, apparent, beams, masks = _scene([-4, 4])
    wsums = np.array([[1.0], [3.0]])
    rms = np.array([[0.1], [0.2]])

    out = stitch_band(apparent, beams, masks, wsums=wsums, rms=rms, eta=0.0)

    np.testing.assert_allclose(out["WSUM"], [4.0])
    assert out["RMS"][0] > 0.0


def test_the_direct_solve_agrees_with_conjugate_gradient():
    """The normal equations are diagonal, so CG must reach the same fixed point."""
    rng = np.random.default_rng(9)
    sky = 1.0 + rng.uniform(size=(NY, NX))
    _, apparent, beams, masks = _scene([-4, 4], sky=sky)
    eta = 1e-4

    direct = stitch_band(apparent, beams, masks, eta=eta)["IMAGE"][0]

    rhs = (beams[:, 0] * apparent[:, 0] * masks).sum(axis=0)
    diag = (beams[:, 0] ** 2 * masks).sum(axis=0) + eta

    def hess(x):
        return diag * x

    cg, _ = conjugate_gradient(hess, rhs, max_iter=50, tol=1e-14)

    np.testing.assert_allclose(direct, cg, rtol=1e-8)
