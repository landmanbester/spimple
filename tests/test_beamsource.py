"""The primary-beam backends init can attach to a partition."""

import numpy as np
import pytest
from astropy.wcs import WCS

from spimple.utils.beamsource import beam_for_grid, lm_grid

CELL_DEG = 10.0 / 3600.0
NY, NX = 20, 28


def _wcs(cell_deg=CELL_DEG):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cdelt = [-cell_deg, cell_deg]
    w.wcs.crpix = [1 + NX // 2, 1 + NY // 2]
    w.wcs.crval = [30.0, -30.0]
    w.array_shape = (NY, NX)
    return w


def _wide_wcs():
    """A grid spanning about a degree, comparable to MeerKAT's L-band beam.

    On the 10-arcsecond grid the analytic beam varies by only 0.3 percent across
    the whole image, so its argmax is numerical noise rather than the pointing.
    """
    return _wcs(cell_deg=200.0 / 3600.0)


def test_lm_grid_has_the_yx_raster_and_the_right_signs():
    ll, mm = lm_grid(_wcs(), (NY, NX))

    assert ll.shape == (NY, NX)
    assert mm.shape == (NY, NX)
    # l runs along x and decreases, m runs along y and increases
    assert ll[0, 0] > ll[0, -1]
    assert mm[0, 0] < mm[-1, 0]
    # the reference pixel sits at the origin
    assert ll[NY // 2, NX // 2] == pytest.approx(0.0)
    assert mm[NY // 2, NX // 2] == pytest.approx(0.0)


def test_no_beam_model_returns_ones():
    freqs = np.array([1.0e9, 1.1e9])

    beam = beam_for_grid(None, "L", freqs, _wcs(), (NY, NX), ncorr=1)

    assert beam.shape == (2, 1, NY, NX)
    np.testing.assert_allclose(beam, 1.0)


def test_jimbeam_peaks_at_the_pointing_centre_and_falls_off():
    pytest.importorskip("katbeam")
    freqs = np.array([1.0e9])

    beam = beam_for_grid("JimBeam", "L", freqs, _wide_wcs(), (NY, NX), ncorr=1)

    assert beam.shape == (1, 1, NY, NX)
    peak = np.unravel_index(np.argmax(beam[0, 0]), (NY, NX))
    assert peak == (NY // 2, NX // 2)
    assert beam[0, 0, 0, 0] < beam[0, 0, NY // 2, NX // 2]


def test_jimbeam_broadcasts_over_correlations():
    pytest.importorskip("katbeam")
    freqs = np.array([1.0e9])

    beam = beam_for_grid("JimBeam", "L", freqs, _wide_wcs(), (NY, NX), ncorr=2)

    assert beam.shape == (1, 2, NY, NX)
    np.testing.assert_allclose(beam[0, 0], beam[0, 1])


def test_jimbeam_rejects_an_unknown_band():
    pytest.importorskip("katbeam")

    with pytest.raises(ValueError, match="band"):
        beam_for_grid("JimBeam", "P", np.array([1.0e9]), _wide_wcs(), (NY, NX), ncorr=1)


def test_a_fits_beam_is_reprojected_onto_the_target_grid(fits_beam_cube):
    """A beam cube on a coarser grid must land on the image grid, peak in the centre."""
    freqs = np.array([1.0e9, 1.1e9])

    beam = beam_for_grid(fits_beam_cube, "L", freqs, _wcs(), (NY, NX), ncorr=1)

    assert beam.shape == (2, 1, NY, NX)
    peak = np.unravel_index(np.argmax(beam[0, 0]), (NY, NX))
    assert peak == (NY // 2, NX // 2)
    assert beam[0, 0].max() == pytest.approx(1.0, abs=2e-2)


def test_a_fits_beam_is_not_transposed(fits_beam_cube_elliptical):
    """An elliptical beam catches a transpose that a circular one hides."""
    freqs = np.array([1.0e9])

    beam = beam_for_grid(fits_beam_cube_elliptical, "L", freqs, _wcs(), (NY, NX), ncorr=1)

    plane = beam[0, 0]
    # the fixture is elongated along m (the y axis), so the half-power extent
    # measured along y must exceed the extent along x
    above = plane > 0.5 * plane.max()
    extent_y = above[:, NX // 2].sum()
    extent_x = above[NY // 2, :].sum()
    assert extent_y > extent_x


def test_an_unknown_beam_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown beam model"):
        beam_for_grid("/no/such/thing.txt", "L", np.array([1.0e9]), _wcs(), (NY, NX), ncorr=1)
