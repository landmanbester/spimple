"""spimple mosaic, the partition combiner."""

import numpy as np
import pytest

from spimple.core.init import init, store_name
from spimple.core.mosaic import mosaic
from spimple.utils.datatree import band_nodes, open_store


@pytest.fixture
def two_pointing_store(two_pointing_fits, tmp_path):
    models, residuals = two_pointing_fits
    out = str(tmp_path / "mos")
    init(models, out, residual=residuals, overwrite=True)
    return store_name(out, "I")


def test_mosaic_populates_every_band_node(two_pointing_store):
    mosaic(two_pointing_store, fits_outputs="")

    dt = open_store(two_pointing_store)
    for node in band_nodes(dt):
        ds = dt[node].ds
        for name in ("IMAGE", "BIMAGE", "KIMAGE", "BEAM", "WSUM", "SPATIALWGT"):
            assert name in ds, f"{name} missing from {node}"
        assert ds.IMAGE.dims == ("corr", "y", "x")


def test_mosaic_preserves_the_band_attributes(two_pointing_store):
    dt = open_store(two_pointing_store)
    before = {n: dict(dt[n].ds.attrs) for n in band_nodes(dt)}

    mosaic(two_pointing_store, fits_outputs="")

    dt = open_store(two_pointing_store)
    for node, attrs in before.items():
        for key in ("bandid", "timeid", "freq_out", "cell_rad", "ra", "dec"):
            assert dt[node].ds.attrs[key] == attrs[key]


def test_mosaic_output_is_finite_everywhere(two_pointing_store):
    mosaic(two_pointing_store, fits_outputs="")

    dt = open_store(two_pointing_store)
    for node in band_nodes(dt):
        assert np.isfinite(dt[node].ds.IMAGE.values).all()


def test_mosaic_is_idempotent(two_pointing_store):
    mosaic(two_pointing_store, fits_outputs="")
    dt = open_store(two_pointing_store)
    first = dt[band_nodes(dt)[0]].ds.IMAGE.values.copy()

    mosaic(two_pointing_store, fits_outputs="")
    dt = open_store(two_pointing_store)
    second = dt[band_nodes(dt)[0]].ds.IMAGE.values

    np.testing.assert_allclose(first, second, rtol=1e-9)


def test_mosaic_renders_the_requested_fits(two_pointing_store, tmp_path):
    folder = tmp_path / "rendered"
    mosaic(two_pointing_store, fits_outputs="I", fits_output_folder=str(folder))

    assert sorted(folder.glob("*.fits")), "no FITS rendered"


def test_mosaic_refuses_a_store_with_no_image_partitions(pfb_tree):
    """A pfb tree mosaics in visibility space; there is nothing here to combine."""
    with pytest.raises(ValueError, match="already populated"):
        mosaic(pfb_tree, fits_outputs="")
