"""Unit tests for the shared FITS helpers."""

import numpy as np
import pytest
from astropy.io import fits as afits

from spimple.utils.fits import (
    add_beampars,
    create_beams_table,
    data_from_header,
    expand_image_patterns,
    freq_axis_of,
    load_cube,
    load_fits,
    save_fits,
    set_wcs,
    to4d,
)


def test_expand_passes_through_literal_paths():
    assert expand_image_patterns(["a.fits", "b.fits"]) == ["a.fits", "b.fits"]


def test_expand_resolves_globs(tmp_path, monkeypatch):
    for name in ("img_02.fits", "img_01.fits"):
        (tmp_path / name).touch()
    monkeypatch.chdir(tmp_path)

    assert expand_image_patterns(["img_*.fits"]) == ["img_01.fits", "img_02.fits"]


def test_expand_deduplicates(tmp_path, monkeypatch):
    (tmp_path / "img_01.fits").touch()
    monkeypatch.chdir(tmp_path)

    assert expand_image_patterns(["img_*.fits", "img_01.fits"]) == ["img_01.fits"]


def test_expand_raises_on_no_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="No files match pattern"):
        expand_image_patterns(["nothing_*.fits"])


def test_expand_resolves_absolute_globs(tmp_path):
    for name in ("img_02.fits", "img_01.fits"):
        (tmp_path / name).touch()

    pattern = str(tmp_path / "img_*.fits")

    assert expand_image_patterns([pattern]) == [
        str(tmp_path / "img_01.fits"),
        str(tmp_path / "img_02.fits"),
    ]


def test_expand_raises_on_no_match_absolute(tmp_path):
    pattern = str(tmp_path / "nothing_*.fits")

    with pytest.raises(FileNotFoundError, match="No files match pattern"):
        expand_image_patterns([pattern])


EPOCH_VECTORS = [
    (4453401600.0, "2000-01-01 00:00:00", "2000-01-01T00:00:00.000"),
    (5049129600.0, "2018-11-17 00:00:00", "2018-11-17T00:00:00.000"),
    (5.0e9, "2017-04-27 08:53:20", "2017-04-27T08:53:20.000"),
    (4886784123.0, "2013-09-25 00:02:03", "2013-09-25T00:02:03.000"),
]


@pytest.mark.parametrize(("ms_time", "expected_utc", "expected_dateobs"), EPOCH_VECTORS)
def test_epoch_conversion_matches_known_vectors(ms_time, expected_utc, expected_dateobs):
    """MJD seconds -> UTC_TIME / DATE-OBS, pinned against the casacore implementation.

    Vectors were captured from the pre-swap code. astropy agreed on all four to
    better than 2.4e-7 s, far below the whole-second truncation applied here.
    """
    header = set_wcs(
        1 / 3600,
        1 / 3600,
        64,
        64,
        [np.deg2rad(30.0), np.deg2rad(-30.0)],
        np.array([1.0e9]),
        ms_time=ms_time,
    )

    assert header["UTC_TIME"] == expected_utc
    assert header["DATE-OBS"] == expected_dateobs


def test_origin_names_this_package():
    """set_wcs stamped ORIGIN = "pfb-imaging" -- a copy-paste leftover."""
    header = set_wcs(1 / 3600, 1 / 3600, 64, 64, [0.0, 0.0], np.array([1.0e9]))

    assert header["ORIGIN"] == "spimple"


def test_data_from_header_reads_ctype4_frequency(image_cube):
    hdr = afits.getheader(image_cube)

    freqs, ref_freq = data_from_header(hdr, axis=4)

    assert freqs.size == 4
    # ref_freq is CRVAL, i.e. the reference pixel's value, not the band centre.
    assert ref_freq == pytest.approx(hdr["CRVAL4"])


def test_data_from_header_reads_ctype3_frequency(image_cube_ctype3):
    hdr = afits.getheader(image_cube_ctype3)

    freqs, ref_freq = data_from_header(hdr, axis=3)

    assert freqs.size == 4
    assert ref_freq == pytest.approx(hdr["CRVAL3"])


def test_data_from_header_agrees_across_conventions(image_cube, image_cube_ctype3):
    """The same frequencies, whichever FITS axis carries them."""
    freqs4, _ = data_from_header(afits.getheader(image_cube), axis=4)
    freqs3, _ = data_from_header(afits.getheader(image_cube_ctype3), axis=3)

    np.testing.assert_allclose(freqs4, freqs3)


def test_load_save_roundtrip(image_cube, tmp_path):
    hdr = afits.getheader(image_cube)
    data = load_fits(image_cube)
    out = tmp_path / "roundtrip.fits"

    save_fits(str(out), data, hdr)

    np.testing.assert_allclose(load_fits(str(out)), data)


def test_to4d_promotes_lower_rank():
    assert to4d(np.zeros((8, 8))).ndim == 4
    assert to4d(np.zeros((2, 8, 8))).ndim == 4
    assert to4d(np.zeros((1, 2, 8, 8))).shape == (1, 2, 8, 8)


def test_to4d_rejects_5d():
    with pytest.raises(ValueError, match="ndim <= 4"):
        to4d(np.zeros((1, 1, 1, 8, 8)))


def test_add_beampars_writes_keywords(image_cube, beam_params):
    hdr = add_beampars(afits.getheader(image_cube), beam_params)

    assert hdr["BMAJ"] == pytest.approx(beam_params[0])
    assert hdr["BMIN"] == pytest.approx(beam_params[1])


def test_load_cube_is_identical_across_frequency_axis_conventions(image_cube, image_cube_ctype3):
    """The two fixtures hold the same data under CTYPE4 and CTYPE3; load_cube must agree."""
    cube4, freq4 = load_cube(image_cube)
    cube3, freq3 = load_cube(image_cube_ctype3)

    assert cube4.shape == cube3.shape
    np.testing.assert_array_equal(cube4, cube3)
    np.testing.assert_allclose(freq4, freq3)


def test_load_cube_puts_band_first_and_keeps_the_raster(image_cube):
    """Shape is (nband, ncorr, ny, nx) with the FITS rows and columns untouched."""
    cube, freqs = load_cube(image_cube)
    hdr = afits.getheader(image_cube)

    assert cube.ndim == 4
    assert cube.shape[0] == freqs.size
    assert cube.shape[2] == hdr["NAXIS2"]
    assert cube.shape[3] == hdr["NAXIS1"]

    raw = afits.getdata(image_cube)
    np.testing.assert_array_equal(cube[0, 0], raw[0, 0])


def test_save_fits_yx_order_round_trips_through_load_cube(tmp_path):
    """save_fits(yx_order=True) then load_cube is the identity on (nband, ncorr, ny, nx)."""
    nband, ncorr, ny, nx = 3, 1, 5, 7
    rng = np.random.default_rng(0)
    data = rng.normal(size=(nband, ncorr, ny, nx)).astype(np.float32)
    freqs = np.array([1.0e9, 1.1e9, 1.2e9])

    hdr = set_wcs(1 / 3600, 1 / 3600, nx, ny, (0.5, -0.5), freqs)
    name = str(tmp_path / "cube.fits")
    save_fits(name, data, hdr, yx_order=True)

    out, out_freqs = load_cube(name)
    np.testing.assert_allclose(out, data, rtol=1e-6)
    np.testing.assert_allclose(out_freqs, freqs, rtol=1e-9)


def test_freq_axis_of_detects_both_conventions(image_cube, image_cube_ctype3):
    assert freq_axis_of(afits.getheader(image_cube)) == 4
    assert freq_axis_of(afits.getheader(image_cube_ctype3)) == 3


def test_set_wcs_reference_frequency_matches_crpix3():
    """CRVAL3 must be the frequency of the channel CRPIX3 names, one-based."""
    freqs = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])
    hdr = set_wcs(1 / 3600, 1 / 3600, 64, 64, (0.5, -0.5), freqs)

    crpix3 = int(hdr["CRPIX3"])
    assert hdr["CRVAL3"] == pytest.approx(freqs[crpix3 - 1])


def test_set_wcs_handles_a_two_channel_cube():
    """Two bands is spifit's minimum; it must not read out of bounds."""
    freqs = np.array([1.0e9, 1.1e9])
    hdr = set_wcs(1 / 3600, 1 / 3600, 64, 64, (0.5, -0.5), freqs)

    assert hdr["CRVAL3"] == pytest.approx(freqs[int(hdr["CRPIX3"]) - 1])


def test_set_wcs_unix_time_agrees_with_mjd_time():
    """time_is_unix shifts the epoch; the same instant must render identically."""
    mjd_seconds = 5.0e9
    unix_seconds = mjd_seconds - 3506716800.0

    from_mjd = set_wcs(1 / 3600, 1 / 3600, 8, 8, (0.5, -0.5), 1.0e9, ms_time=mjd_seconds)
    from_unix = set_wcs(1 / 3600, 1 / 3600, 8, 8, (0.5, -0.5), 1.0e9, ms_time=unix_seconds, time_is_unix=True)

    assert from_mjd["DATE-OBS"] == from_unix["DATE-OBS"]
    assert from_mjd["UTC_TIME"] == from_unix["UTC_TIME"]


def test_set_wcs_target_offset_shifts_crpix_not_crval():
    """l0/m0 move the centre pixel; the tangent point stays put."""
    cell = 1 / 3600
    base = set_wcs(cell, cell, 64, 64, (0.5, -0.5), 1.0e9)
    offset = set_wcs(cell, cell, 64, 64, (0.5, -0.5), 1.0e9, l0=np.deg2rad(cell), m0=0.0)

    assert offset["CRVAL1"] == pytest.approx(base["CRVAL1"])
    assert offset["CRPIX1"] == pytest.approx(base["CRPIX1"] + 1.0)
    assert offset["CRPIX2"] == pytest.approx(base["CRPIX2"])


def test_create_beams_table_converts_pixels_to_degrees():
    import xarray as xr

    cell_deg = 1 / 3600
    pars = np.array([[[8.0, 6.0, 0.5]], [[4.0, 3.0, 0.5]]])  # (nband, ncorr, 3), pixels/radians
    da = xr.DataArray(
        pars,
        dims=("band", "corr", "bpar"),
        coords={"band": [0, 1], "corr": ["I"], "bpar": ["BMAJ", "BMIN", "BPA"]},
    )

    hdu = create_beams_table(da, cell2deg=cell_deg)

    assert hdu.name == "BEAMS"
    np.testing.assert_allclose(hdu.data["BMAJ"], [8.0 * cell_deg, 4.0 * cell_deg], rtol=1e-6)
    np.testing.assert_allclose(hdu.data["BPA"], [np.rad2deg(0.5)] * 2, rtol=1e-6)
    np.testing.assert_array_equal(hdu.data["CHAN"], [0, 1])
