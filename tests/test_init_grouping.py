"""How init maps a pile of FITS files onto the tree's band and partition axes."""

import numpy as np
import pytest
from astropy.io import fits

from spimple.core.init import assign_bands, field_name_for, group_partitions, partition_key

CELL_DEG = 1.0 / 3600.0


def _write(path, ra, dec, freqs, npix=8):
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = ra
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CDELT1"] = -CELL_DEG
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = dec
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CDELT2"] = CELL_DEG
    hdr["CTYPE4"] = "FREQ"
    hdr["CRVAL4"] = float(freqs[0])
    hdr["CRPIX4"] = 1
    hdr["CDELT4"] = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0e8
    hdr["CTYPE3"] = "STOKES"
    hdr["CRVAL3"] = 1.0
    hdr["CRPIX3"] = 1
    hdr["CDELT3"] = 1.0
    data = np.zeros((len(freqs), 1, npix, npix), dtype=np.float32)
    fits.writeto(path, data, hdr, overwrite=True)
    return str(path)


def test_files_sharing_a_phase_centre_and_grid_form_one_partition(tmp_path):
    a0 = _write(tmp_path / "fieldA-0000.fits", 30.0, -30.0, [1.0e9])
    a1 = _write(tmp_path / "fieldA-0001.fits", 30.0, -30.0, [1.1e9])
    b0 = _write(tmp_path / "fieldB-0000.fits", 31.0, -30.0, [1.0e9])

    groups = group_partitions([a0, a1, b0])

    assert len(groups) == 2
    assert sorted(groups[0][1]) == sorted([a0, a1])
    assert groups[1][1] == [b0]


def test_partitions_are_ordered_by_phase_centre(tmp_path):
    east = _write(tmp_path / "e.fits", 31.0, -30.0, [1.0e9])
    west = _write(tmp_path / "w.fits", 30.0, -30.0, [1.0e9])

    groups = group_partitions([east, west])

    assert groups[0][1] == [west]
    assert groups[1][1] == [east]


def test_a_different_grid_size_splits_the_partition(tmp_path):
    small = _write(tmp_path / "small.fits", 30.0, -30.0, [1.0e9], npix=8)
    large = _write(tmp_path / "large.fits", 30.0, -30.0, [1.0e9], npix=16)

    assert len(group_partitions([small, large])) == 2


def test_partition_key_is_insensitive_to_floating_point_noise(tmp_path):
    a = _write(tmp_path / "a.fits", 30.0, -30.0, [1.0e9])
    b = _write(tmp_path / "b.fits", 30.0 + 1e-12, -30.0, [1.0e9])

    assert partition_key(fits.getheader(a)) == partition_key(fits.getheader(b))


def test_matching_frequencies_across_partitions_share_a_band():
    freqs = [np.array([1.0e9, 1.1e9]), np.array([1.0e9, 1.1e9])]

    nominal, mapping = assign_bands(freqs, freq_tol=None)

    np.testing.assert_allclose(nominal, [1.0e9, 1.1e9])
    assert mapping == [{0: 0, 1: 1}, {0: 0, 1: 1}]


def test_a_partition_missing_a_band_simply_omits_it():
    freqs = [np.array([1.0e9, 1.1e9]), np.array([1.1e9])]

    nominal, mapping = assign_bands(freqs, freq_tol=None)

    np.testing.assert_allclose(nominal, [1.0e9, 1.1e9])
    assert mapping == [{0: 0, 1: 1}, {1: 0}]


def test_frequencies_within_the_tolerance_are_one_band():
    freqs = [np.array([1.0e9]), np.array([1.0e9 + 1.0e6])]

    nominal, mapping = assign_bands(freqs, freq_tol=1.0e7)

    assert nominal.size == 1
    assert mapping == [{0: 0}, {0: 0}]


def test_two_channels_of_one_partition_in_one_band_is_an_error():
    freqs = [np.array([1.0e9, 1.0e9 + 1.0e6])]

    with pytest.raises(ValueError, match="two channels"):
        assign_bands(freqs, freq_tol=1.0e7)


def test_field_name_is_the_common_prefix_of_the_group(tmp_path):
    paths = [str(tmp_path / "deep2-0000-image.fits"), str(tmp_path / "deep2-0001-image.fits")]

    assert field_name_for(paths, 0) == "deep2"


def test_field_name_falls_back_to_the_partition_id(tmp_path):
    paths = [str(tmp_path / "alpha.fits"), str(tmp_path / "beta.fits")]

    assert field_name_for(paths, 3) == "part0003"


def test_field_name_keeps_a_digit_that_is_part_of_the_name(tmp_path):
    """deep2 must not become deep just because the name ends in a digit."""
    paths = [str(tmp_path / "deep2.fits"), str(tmp_path / "deep2.fits")]

    assert field_name_for(paths, 0) == "deep2"


def test_field_name_strips_a_wsclean_channel_counter(tmp_path):
    paths = [str(tmp_path / "img_01-image.fits"), str(tmp_path / "img_02-image.fits")]

    assert field_name_for(paths, 0) == "img"
