"""spimple init, the FITS-to-DataTree ingest."""

import numpy as np
import pytest
from astropy.io import fits

from spimple.core.init import init, resolve_target, store_name
from spimple.utils.datatree import band_nodes, open_store

CELL_DEG = 1.0 / 3600.0


def test_store_name_appends_the_product():
    assert store_name("/data/out", "I") == "/data/out_I.dt"


def test_resolve_target_takes_the_largest_axes_times_dilate():
    psfparsn = np.array([[[8.0, 6.0, 0.0]], [[4.0, 5.0, 0.0]]])  # (nband, ncorr, 3), pixels

    target = resolve_target(psfparsn, None, False, 1.05, CELL_DEG)

    np.testing.assert_allclose(target[0, 0], 8.0 * 1.05)
    np.testing.assert_allclose(target[0, 1], 6.0 * 1.05)


def test_resolve_target_converts_psf_pars_from_degrees_to_pixels():
    psfparsn = np.array([[[8.0, 6.0, 0.0]]])

    target = resolve_target(psfparsn, (10.0 * CELL_DEG, 9.0 * CELL_DEG, 30.0), False, 1.05, CELL_DEG)

    np.testing.assert_allclose(target[0], [10.0, 9.0, np.deg2rad(30.0)])


def test_resolve_target_rejects_a_target_finer_than_the_input():
    psfparsn = np.array([[[8.0, 6.0, 0.0]]])

    with pytest.raises(ValueError, match="finer"):
        resolve_target(psfparsn, (4.0 * CELL_DEG, 3.0 * CELL_DEG, 0.0), False, 1.05, CELL_DEG)


def test_resolve_target_circularises_when_asked():
    psfparsn = np.array([[[8.0, 6.0, 0.3]]])

    target = resolve_target(psfparsn, None, True, 1.0, CELL_DEG)

    assert target[0, 0] == target[0, 1]
    assert target[0, 2] == 0.0


def test_init_writes_a_store_with_one_band_node_per_channel(image_cube, residual_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], overwrite=True)

    dt = open_store(store_name(out, "I"))
    nodes = band_nodes(dt)

    assert len(nodes) == 4
    assert [int(dt[n].ds.attrs["bandid"]) for n in nodes] == [0, 1, 2, 3]
    assert dt.attrs["product"] == "I"


def test_init_populates_the_band_node_for_a_single_partition(image_cube, residual_cube, tmp_path):
    """One pointing needs no mosaic step, so init writes the band products itself."""
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], overwrite=True)

    dt = open_store(store_name(out, "I"))
    ds = dt[band_nodes(dt)[0]].ds

    for name in ("IMAGE", "BIMAGE", "KIMAGE", "BEAM", "WSUM", "RMS", "PSFPARSF"):
        assert name in ds, f"{name} missing from the band node"
    assert ds.IMAGE.dims == ("corr", "y", "x")


def test_init_homogenises_every_band_to_one_resolution(image_cube, residual_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], overwrite=True)

    dt = open_store(store_name(out, "I"))
    pars = np.stack([dt[n].ds.PSFPARSF.values for n in band_nodes(dt)])

    assert pars.shape[0] == 4
    for band in pars:
        np.testing.assert_allclose(band, pars[0], rtol=1e-9)


def test_init_takes_wsum_from_the_channel_weights_keyword(image_cube, residual_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], overwrite=True)

    dt = open_store(store_name(out, "I"))
    wsum = dt[band_nodes(dt)[0]].ds.WSUM.values

    # the residual_cube fixture carries WSCVWSUM = 1.0
    np.testing.assert_allclose(wsum, 1.0)


def test_init_without_a_beam_leaves_the_flux_scales_equal(image_cube, residual_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], overwrite=True)

    dt = open_store(store_name(out, "I"))
    ds = dt[band_nodes(dt)[0]].ds

    np.testing.assert_allclose(ds.BEAM.values, 1.0)
    np.testing.assert_allclose(ds.IMAGE.values, ds.BIMAGE.values, atol=1e-6)


def test_init_refuses_to_clobber_without_overwrite(image_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, overwrite=True)

    with pytest.raises(FileExistsError):
        init([image_cube], out, overwrite=False)


def test_init_renders_requested_fits(image_cube, residual_cube, tmp_path):
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], fits_outputs="I", overwrite=True)

    written = sorted((tmp_path / "fits").glob("*.fits"))
    assert written, "no FITS rendered"
    assert fits.getdata(str(written[0])).ndim == 4


def test_init_rejects_a_beam_model_until_task_12(image_cube, tmp_path):
    with pytest.raises(NotImplementedError, match="beam"):
        init([image_cube], str(tmp_path / "out"), beam_model="JimBeam", overwrite=True)
