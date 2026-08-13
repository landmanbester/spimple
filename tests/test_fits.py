"""Unit tests for the shared FITS helpers."""

import numpy as np
import pytest

from spimple.utils.fits import expand_image_patterns, set_wcs


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
