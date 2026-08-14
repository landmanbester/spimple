"""Rendering band-node variables back to FITS.

The orientation test here is the one that catches a (Y, X) mistake; every other
test in the suite passes while the image is transposed.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spimple.utils.datatree import band_node_name, create_store, write_node
from spimple.utils.render import dt2fits

CELL_DEG = 1.0 / 3600.0
BPAR = ["BMAJ", "BMIN", "BPA"]
NY, NX = 24, 32
Y0, X0 = 5, 27  # deliberately asymmetric, and inside a non-square raster


@pytest.fixture
def spike_tree(tmp_path):
    """Two bands, each a single bright pixel at (Y0, X0) on a non-square grid."""
    url = str(tmp_path / "spike_I.dt")
    create_store(
        url,
        {"product": "I", "nband": 2, "ntime": 1, "nx": NX, "ny": NY, "cell_rad": np.deg2rad(CELL_DEG)},
        overwrite=True,
    )
    for band in range(2):
        data = np.zeros((1, NY, NX), dtype=np.float32)
        data[0, Y0, X0] = 1.0 + band
        write_node(
            url,
            band_node_name(band, 0),
            {
                "IMAGE": (("corr", "y", "x"), data),
                "WSUM": (("corr",), np.array([1.0 + band], dtype=np.float32)),
                "PSFPARSF": (("corr", "bpar"), np.array([[8.0, 6.0, 0.0]], dtype=np.float32)),
            },
            {
                "bandid": band,
                "timeid": 0,
                "freq_out": 1.0e9 + band * 1.0e8,
                "time_out": 1.0e9,
                "ra": np.deg2rad(30.0),
                "dec": np.deg2rad(-30.0),
                "cell_rad": np.deg2rad(CELL_DEG),
                "l0": 0.0,
                "m0": 0.0,
            },
            {"corr": ["I"], "bpar": BPAR},
        )
    return url


def test_render_preserves_the_raster(spike_tree, tmp_path):
    """The bright pixel must land at [.., Y0, X0] in the FITS array, not transposed."""
    written = dt2fits(spike_tree, "IMAGE", str(tmp_path / "out"), do_mfs=False)
    assert len(written) == 1

    data = fits.getdata(written[0])
    assert data.shape[-2:] == (NY, NX)
    for band in range(2):
        peak = np.unravel_index(np.argmax(data[0, band]), data[0, band].shape)
        assert peak == (Y0, X0)


def test_render_wcs_agrees_with_the_raster(spike_tree, tmp_path):
    """The header must describe the pixel the data actually occupies."""
    written = dt2fits(spike_tree, "IMAGE", str(tmp_path / "out"), do_mfs=False)
    hdr = fits.getheader(written[0])
    wcs = WCS(hdr).celestial

    sky = wcs.pixel_to_world(X0, Y0)
    ref = wcs.pixel_to_world(hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1)

    # CDELT1 is negative, so RA decreases with increasing x
    d_ra = (sky.ra.deg - ref.ra.deg) * np.cos(np.deg2rad(ref.dec.deg))
    d_dec = sky.dec.deg - ref.dec.deg
    assert d_ra == pytest.approx(-(X0 - (hdr["CRPIX1"] - 1)) * CELL_DEG, rel=1e-3)
    assert d_dec == pytest.approx((Y0 - (hdr["CRPIX2"] - 1)) * CELL_DEG, rel=1e-3)


def test_render_mfs_is_the_wsum_weighted_mean(spike_tree, tmp_path):
    written = dt2fits(spike_tree, "IMAGE", str(tmp_path / "out"), do_cube=False)
    assert len(written) == 1

    data = fits.getdata(written[0])
    # band values 1 and 2 with wsums 1 and 2 -> (1*1 + 2*2)/3
    assert data[0, 0, Y0, X0] == pytest.approx((1.0 * 1.0 + 2.0 * 2.0) / 3.0, rel=1e-5)


def test_render_writes_a_beams_table_from_psfparsf(spike_tree, tmp_path):
    written = dt2fits(spike_tree, "IMAGE", str(tmp_path / "out"), do_mfs=False)

    with fits.open(written[0]) as hdul:
        assert "BEAMS" in hdul
        beams = hdul["BEAMS"].data
        np.testing.assert_allclose(beams["BMAJ"], [8.0 * CELL_DEG] * 2, rtol=1e-5)
        assert hdul[0].header["BMAJ"] == pytest.approx(8.0 * CELL_DEG, rel=1e-5)


def test_render_skips_a_column_no_band_carries(spike_tree, tmp_path):
    assert dt2fits(spike_tree, "BIMAGE", str(tmp_path / "out")) == []


def test_render_handles_the_two_band_minimum(pfb_tree, tmp_path):
    """A cube of any size must render; set_wcs used to read out of bounds at nband 2."""
    written = dt2fits(pfb_tree, "IMAGE", str(tmp_path / "pfb"), do_mfs=True, do_cube=True)

    assert len(written) == 2
    for path in written:
        assert fits.getdata(path).ndim == 4
