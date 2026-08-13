"""The DataTree schema layer.

These tests pin the cross-repo contract described in the design spec section 3.
"""

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from spimple.utils.datatree import (
    PRODUCT_VARS,
    band_node_name,
    band_nodes,
    check_homogeneous,
    create_store,
    open_store,
    partition_node_name,
    partition_nodes,
    psfpars_from_header,
    require_vars,
    timeids,
    write_node,
)

BPAR = ["BMAJ", "BMIN", "BPA"]


def test_node_names_are_zero_padded_to_four_digits():
    assert band_node_name(0, 0) == "band0000_time0000"
    assert band_node_name(12, 3) == "band0012_time0003"
    assert partition_node_name(7) == "part0007"


def test_band_nodes_are_ordered_by_bandid_not_by_frequency(tmp_path):
    """Effective frequencies are data-dependent and can invert; bandid cannot."""
    url = str(tmp_path / "out.dt")
    create_store(url, {"product": "I"}, overwrite=True)
    # band 0 deliberately carries the HIGHER freq_out
    write_node(url, band_node_name(0, 0), {}, {"bandid": 0, "timeid": 0, "freq_out": 2.0e9}, {})
    write_node(url, band_node_name(1, 0), {}, {"bandid": 1, "timeid": 0, "freq_out": 1.0e9}, {})

    dt = open_store(url)
    assert band_nodes(dt) == ["band0000_time0000", "band0001_time0000"]


def test_band_nodes_filter_by_timeid(tmp_path):
    url = str(tmp_path / "out.dt")
    create_store(url, {"product": "I"}, overwrite=True)
    write_node(url, band_node_name(0, 0), {}, {"bandid": 0, "timeid": 0}, {})
    write_node(url, band_node_name(0, 1), {}, {"bandid": 0, "timeid": 1}, {})

    dt = open_store(url)
    assert timeids(dt) == [0, 1]
    assert band_nodes(dt, timeid=1) == ["band0000_time0001"]


def test_partition_nodes_are_ordered_and_scoped_to_their_band(tmp_path):
    url = str(tmp_path / "out.dt")
    create_store(url, {"product": "I"}, overwrite=True)
    node = band_node_name(0, 0)
    write_node(url, node, {}, {"bandid": 0, "timeid": 0}, {})
    for pid in (1, 0):
        write_node(url, f"{node}/{partition_node_name(pid)}", {}, {"field_name": f"f{pid}"}, {})

    dt = open_store(url)
    assert partition_nodes(dt, node) == ["part0000", "part0001"]


def test_write_node_preserves_existing_attrs(tmp_path):
    """to_zarr(mode='a') replaces attrs wholesale; write_node must merge."""
    url = str(tmp_path / "out.dt")
    create_store(url, {"product": "I"}, overwrite=True)
    node = band_node_name(0, 0)
    write_node(url, node, {"WSUM": (("corr",), np.ones(1))}, {"bandid": 0, "timeid": 0}, {"corr": ["I"]})
    write_node(url, node, {"IMAGE": (("corr", "y", "x"), np.zeros((1, 4, 4)))}, {"pb_min": 0.15}, {"corr": ["I"]})

    ds = open_store(url)[node].ds
    assert ds.attrs["bandid"] == 0
    assert ds.attrs["pb_min"] == 0.15
    assert "WSUM" in ds and "IMAGE" in ds


def test_require_vars_names_the_fix_in_its_error():
    ds = xr.Dataset({"IMAGE": (("y", "x"), np.zeros((2, 2)))})

    with pytest.raises(ValueError, match="spimple mosaic"):
        require_vars(ds, ["IMAGE", "BEAM"], "band0000_time0000", "run spimple mosaic")


def test_check_homogeneous_accepts_equal_psfparsf_and_returns_it():
    pars = np.array([[8.0, 6.0, 0.3]])
    nodes = [xr.Dataset({"PSFPARSF": (("corr", "bpar"), pars.copy())}) for _ in range(3)]

    out = check_homogeneous(nodes)

    np.testing.assert_allclose(out, pars)


def test_check_homogeneous_rejects_differing_psfparsf():
    a = xr.Dataset({"PSFPARSF": (("corr", "bpar"), np.array([[8.0, 6.0, 0.3]]))})
    b = xr.Dataset({"PSFPARSF": (("corr", "bpar"), np.array([[7.0, 6.0, 0.3]]))})

    with pytest.raises(ValueError, match="pfb restore"):
        check_homogeneous([a, b])


def test_check_homogeneous_rejects_a_missing_psfparsf():
    with pytest.raises(ValueError, match="PSFPARSF"):
        check_homogeneous([xr.Dataset({"IMAGE": (("y", "x"), np.zeros((2, 2)))})])


def test_psfpars_from_header_converts_degrees_to_pixels_and_radians():
    cell_deg = 1 / 3600
    hdr = fits.Header()
    for i in (1, 2):
        hdr[f"BMAJ{i}"] = 8.0 * cell_deg * i
        hdr[f"BMIN{i}"] = 6.0 * cell_deg * i
        hdr[f"BPA{i}"] = 30.0

    pars = psfpars_from_header(hdr, nband=2, ncorr=1, cell_deg=cell_deg)

    assert pars.shape == (2, 1, 3)
    np.testing.assert_allclose(pars[0, 0], [8.0, 6.0, np.deg2rad(30.0)])
    np.testing.assert_allclose(pars[1, 0], [16.0, 12.0, np.deg2rad(30.0)])


def test_psfpars_from_header_falls_back_to_the_scalar_cards():
    cell_deg = 1 / 3600
    hdr = fits.Header()
    hdr["BMAJ"] = 8.0 * cell_deg
    hdr["BMIN"] = 6.0 * cell_deg
    hdr["BPA"] = 0.0

    pars = psfpars_from_header(hdr, nband=3, ncorr=2, cell_deg=cell_deg)

    assert pars.shape == (3, 2, 3)
    np.testing.assert_allclose(pars[:, :, 0], 8.0)


def test_psfpars_from_header_raises_when_no_beam_is_present():
    with pytest.raises(ValueError, match="BMAJ"):
        psfpars_from_header(fits.Header(), nband=1, ncorr=1, cell_deg=1 / 3600)


def test_product_vars_match_the_pfb_restore_contract():
    assert PRODUCT_VARS == {"a": "BIMAGE", "i": "IMAGE", "k": "KIMAGE"}
