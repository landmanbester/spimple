"""Unit tests for the shared FITS helpers."""

import pytest

from spimple.core.fits import expand_image_patterns


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
