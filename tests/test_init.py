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


def test_init_writes_one_partition_per_pointing(two_pointing_fits, tmp_path):
    from spimple.utils.datatree import partition_nodes

    models, residuals = two_pointing_fits
    out = str(tmp_path / "out")
    init(models, out, residual=residuals, overwrite=True)

    dt = open_store(store_name(out, "I"))
    node = band_nodes(dt)[0]

    assert partition_nodes(dt, node) == ["part0000", "part0001"]


def test_init_leaves_the_band_node_bare_for_multiple_partitions(two_pointing_fits, tmp_path):
    """Combining is mosaic's job; init must not guess a band-level image."""
    models, residuals = two_pointing_fits
    out = str(tmp_path / "out")
    init(models, out, residual=residuals, overwrite=True)

    dt = open_store(store_name(out, "I"))
    ds = dt[band_nodes(dt)[0]].ds

    assert "IMAGE" not in ds
    assert "PSFPARSF" in ds
    assert "cell_rad" in ds.attrs


def test_init_puts_every_partition_on_the_union_grid(two_pointing_fits, tmp_path):
    from spimple.utils.datatree import partition_nodes

    models, residuals = two_pointing_fits
    out = str(tmp_path / "out")
    init(models, out, residual=residuals, overwrite=True)

    dt = open_store(store_name(out, "I"))
    node = band_nodes(dt)[0]
    shapes = {dt[f"{node}/{p}"].ds.IMAGE.shape for p in partition_nodes(dt, node)}

    assert len(shapes) == 1
    assert dt.attrs["nx"] > 64  # wider than one pointing


def test_init_masks_each_partition_to_its_own_footprint(two_pointing_fits, tmp_path):
    from spimple.utils.datatree import partition_nodes

    models, residuals = two_pointing_fits
    out = str(tmp_path / "out")
    init(models, out, residual=residuals, overwrite=True)

    dt = open_store(store_name(out, "I"))
    node = band_nodes(dt)[0]
    masks = [dt[f"{node}/{p}"].ds.MASK.values[0] for p in partition_nodes(dt, node)]

    assert not masks[0].all()
    assert (masks[0] & masks[1]).any(), "the pointings should overlap"
    assert not (masks[0] & masks[1]).all(), "the pointings should not coincide"


def test_init_applies_a_jimbeam_to_each_partition(image_cube, residual_cube, tmp_path):
    pytest.importorskip("katbeam")
    out = str(tmp_path / "out")
    init([image_cube], out, residual=[residual_cube], beam_model="JimBeam", overwrite=True)

    dt = open_store(store_name(out, "I"))
    ds = dt[band_nodes(dt)[0]].ds

    assert ds.BEAM.values.max() <= 1.0 + 1e-6
    assert ds.BEAM.values.min() < 1.0


def test_init_uses_each_partition_s_own_pointing_for_the_beam(two_pointing_fits, tmp_path):
    """Beams are evaluated around each partition's phase centre, not a shared one."""
    pytest.importorskip("katbeam")
    from spimple.utils.datatree import partition_nodes

    models, residuals = two_pointing_fits
    out = str(tmp_path / "out")
    init(models, out, residual=residuals, beam_model="JimBeam", overwrite=True)

    dt = open_store(store_name(out, "I"))
    node = band_nodes(dt)[0]
    peaks = []
    for part in partition_nodes(dt, node):
        beam = dt[f"{node}/{part}"].ds.BEAM.values[0]
        peaks.append(np.unravel_index(np.argmax(beam), beam.shape))

    assert peaks[0] != peaks[1], "both beams peak at the same pixel; the pointing was ignored"


def _single_band_fits(path, freq, bmaj_deg, npix=32, ra=30.0):
    """One channel, one file, with its own scalar beam cards."""
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = ra
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CDELT1"] = -CELL_DEG
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = -30.0
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CDELT2"] = CELL_DEG
    hdr["CUNIT2"] = "deg"
    hdr["CTYPE4"] = "FREQ"
    hdr["CRVAL4"] = float(freq)
    hdr["CRPIX4"] = 1
    hdr["CDELT4"] = 1.0e8
    hdr["CTYPE3"] = "STOKES"
    hdr["CRVAL3"] = 1.0
    hdr["CRPIX3"] = 1
    hdr["CDELT3"] = 1.0
    hdr["BMAJ"] = float(bmaj_deg)
    hdr["BMIN"] = float(bmaj_deg)
    hdr["BPA"] = 0.0
    data = np.zeros((1, 1, npix, npix), dtype=np.float32)
    data[0, 0, 5, 7] = 1.0
    fits.writeto(path, data, hdr, overwrite=True)
    return str(path)


def test_init_labels_planes_correctly_on_a_descending_frequency_axis(tmp_path):
    """A cube whose CDELT4 is negative must not have its spectrum reversed."""
    npix = 32
    freqs = [1.3e9, 1.2e9, 1.1e9, 1.0e9]
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = 30.0
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CDELT1"] = -CELL_DEG
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = -30.0
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CDELT2"] = CELL_DEG
    hdr["CUNIT2"] = "deg"
    hdr["CTYPE4"] = "FREQ"
    hdr["CRVAL4"] = freqs[0]
    hdr["CRPIX4"] = 1
    hdr["CDELT4"] = freqs[1] - freqs[0]  # negative
    hdr["CTYPE3"] = "STOKES"
    hdr["CRVAL3"] = 1.0
    hdr["CRPIX3"] = 1
    hdr["CDELT3"] = 1.0
    hdr["BMAJ"] = 6.0 * CELL_DEG
    hdr["BMIN"] = 6.0 * CELL_DEG
    hdr["BPA"] = 0.0
    data = np.zeros((4, 1, npix, npix), dtype=np.float32)
    for i in range(4):
        data[i, 0, 5, 7] = float(i + 1)  # plane i, at freqs[i], carries value i+1
    path = str(tmp_path / "desc.fits")
    fits.writeto(path, data, hdr, overwrite=True)

    out = str(tmp_path / "out")
    init([path], out, overwrite=True)

    dt = open_store(store_name(out, "I"))
    for node in band_nodes(dt):
        ds = dt[node].ds
        freq = float(ds.attrs["freq_out"])
        expected = freqs.index(pytest.approx(freq)) + 1 if False else None
        # the plane written at this frequency carried value (index in freqs) + 1
        idx = min(range(4), key=lambda i: abs(freqs[i] - freq))
        expected = idx + 1
        assert ds.IMAGE.values[0, 5, 7] == pytest.approx(expected, rel=0.1), (
            f"{node} at {freq:.3e} Hz carries the wrong plane"
        )


def test_init_reads_the_beam_from_each_source_file(tmp_path):
    """Split single-channel files each carry their own scalar BMAJ."""
    beams = [6.0, 5.0, 4.0]
    files = [
        _single_band_fits(tmp_path / f"s-{i:04d}.fits", 1.0e9 + i * 1.0e8, b * CELL_DEG) for i, b in enumerate(beams)
    ]

    out = str(tmp_path / "out")
    init(files, out, overwrite=True)

    dt = open_store(store_name(out, "I"))
    recorded = [np.asarray(dt[f"{n}/part0000"].ds.attrs["psfparsn"])[0][0] for n in band_nodes(dt)]

    np.testing.assert_allclose(recorded, beams, rtol=1e-6)
