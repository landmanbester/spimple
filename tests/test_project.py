"""The union grid and the reprojection onto it."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spimple.utils.project import reproject_cube, same_frame, union_wcs

CELL_DEG = 10.0 / 3600.0
NPIX = 32


def _hdr(ra, dec, npix=NPIX):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = npix
    hdr["NAXIS2"] = npix
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = ra
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CDELT1"] = -CELL_DEG
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = dec
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CDELT2"] = CELL_DEG
    hdr["CUNIT2"] = "deg"
    return hdr


def test_union_of_one_header_is_that_header_s_frame():
    hdr = _hdr(30.0, -30.0)

    wcs, shape = union_wcs([hdr])

    assert shape == (NPIX, NPIX)
    assert same_frame(wcs, shape, WCS(hdr).celestial, (NPIX, NPIX))


def test_union_of_two_offset_pointings_covers_both():
    a = _hdr(30.0, -30.0)
    b = _hdr(30.0 + 20 * CELL_DEG, -30.0)

    wcs, shape = union_wcs([a, b])

    assert shape[1] > NPIX  # wider in x to hold both pointings
    for hdr in (a, b):
        centre = WCS(hdr).celestial.pixel_to_world(NPIX // 2, NPIX // 2)
        x, y = wcs.world_to_pixel(centre)
        assert 0 <= x < shape[1]
        assert 0 <= y < shape[0]


def test_same_frame_rejects_a_shifted_reference():
    a = _hdr(30.0, -30.0)
    b = _hdr(30.1, -30.0)

    assert not same_frame(WCS(a).celestial, (NPIX, NPIX), WCS(b).celestial, (NPIX, NPIX))


def test_same_frame_rejects_a_different_shape():
    hdr = _hdr(30.0, -30.0)

    assert not same_frame(WCS(hdr).celestial, (NPIX, NPIX), WCS(hdr).celestial, (NPIX, NPIX + 2))


def test_reproject_onto_the_same_frame_is_a_near_identity():
    hdr = _hdr(30.0, -30.0)
    wcs = WCS(hdr).celestial
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(1, NPIX, NPIX))

    out, mask = reproject_cube(cube, wcs, wcs, (NPIX, NPIX))

    assert mask.all()
    np.testing.assert_allclose(out, cube, atol=1e-9)


def test_reproject_moves_a_source_to_the_predicted_pixel():
    """A source at a known sky position must land where the target WCS says."""
    src_hdr = _hdr(30.0, -30.0)
    src_wcs = WCS(src_hdr).celestial
    cube = np.zeros((1, NPIX, NPIX))
    y0, x0 = 8, 21
    cube[0, y0, x0] = 1.0

    tgt_wcs, shape = union_wcs([src_hdr, _hdr(30.0 + 20 * CELL_DEG, -30.0)])
    out, mask = reproject_cube(cube, src_wcs, tgt_wcs, shape)

    sky = src_wcs.pixel_to_world(x0, y0)
    xt, yt = tgt_wcs.world_to_pixel(sky)
    peak = np.unravel_index(np.argmax(out[0]), out[0].shape)
    assert peak[0] == pytest.approx(yt, abs=1.0)
    assert peak[1] == pytest.approx(xt, abs=1.0)


def test_reproject_masks_the_region_outside_the_footprint():
    src_hdr = _hdr(30.0, -30.0)
    tgt_wcs, shape = union_wcs([src_hdr, _hdr(30.0 + 40 * CELL_DEG, -30.0)])
    cube = np.ones((1, NPIX, NPIX))

    out, mask = reproject_cube(cube, WCS(src_hdr).celestial, tgt_wcs, shape)

    assert not mask.all()
    np.testing.assert_allclose(out[0][~mask], 0.0)
