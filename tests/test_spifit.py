"""spimple spifit, the spectral index fit over a datatree."""

import os

import numpy as np
import pytest
from astropy.io import fits

from spimple.core.init import init, store_name
from spimple.core.spifit import spifit
from spimple.utils.datatree import band_nodes, open_store, write_node

PB_MIN = 0.15


@pytest.fixture
def narrowing_beam_tree(pfb_tree, tmp_path):
    """pfb_tree with a beam that narrows as 1 / freq, and BIMAGE kept consistent.

    The shipped pfb_tree writes one band independent beam, which makes every
    band's beam mask identical and hides any difference between an all bands
    cut and a per band one. A real primary beam narrows with frequency, so the
    two disagree over an annulus.
    """
    import shutil

    from spimple.utils.datatree import partition_node_name

    tree = str(tmp_path / "narrowing_I.dt")
    shutil.copytree(pfb_tree, tree)
    dt = open_store(tree)
    nodes = band_nodes(dt)
    freqs = np.array([float(dt[n].ds.attrs["freq_out"]) for n in nodes])

    npix = dt[nodes[0]].ds.IMAGE.shape[-1]
    v = np.arange(npix) - npix // 2
    yy, xx = np.meshgrid(v, v, indexing="ij")
    rsq = (xx**2 + yy**2).astype(np.float64)

    for node, freq in zip(nodes, freqs):
        sigma = 6.0 * freqs[0] / freq
        beam = np.exp(-rsq / (2 * sigma**2))[None].astype(np.float32)
        intrinsic = dt[node].ds.IMAGE.values
        write_node(
            tree,
            node,
            {
                "BEAM": (("corr", "y", "x"), beam),
                "BIMAGE": (("corr", "y", "x"), (intrinsic * beam).astype(np.float32)),
            },
            {},
            {"corr": ["I"]},
        )
        write_node(
            tree,
            f"{node}/{partition_node_name(0)}",
            {"BEAM": (("corr", "y", "x"), beam)},
            {},
            {"corr": ["I"]},
        )
    return tree


def _band_beams(tree):
    """(nband, ny, nx) stack of the tree's band beams, Stokes I only."""
    dt = open_store(tree)
    return np.stack([dt[n].ds.BEAM.values[0] for n in band_nodes(dt)])


def test_beam_cut_excludes_pixels_failing_in_any_band(narrowing_beam_tree, tmp_path):
    """pb_min is an all bands cut, set by the band with the smallest beam.

    A pixel whose beam clears pb_min at the bottom of the band but not at the
    top must not be fitted at all. Fitting it would either weight the top band
    by a near zero beam or divide by one, which is what biases alpha towards
    the field edge.
    """
    out = str(tmp_path / "spi_cut")
    spifit(narrowing_beam_tree, out, flux_scale="intrinsic", products="a", pb_min=PB_MIN)
    alpha = fits.getdata(f"{out}_time0.alpha.fits").squeeze()

    beams = _band_beams(narrowing_beam_tree)
    fitted = np.isfinite(alpha)

    assert fitted.any()
    assert (beams.min(axis=0)[fitted] > PB_MIN).all(), "a fitted pixel fails pb_min in at least one band"

    # Non-vacuity: the straddle annulus must exist and must hold flux that
    # would otherwise have been fitted, or the assertion above proves nothing.
    straddle = (beams.min(axis=0) <= PB_MIN) & (beams.max(axis=0) > PB_MIN)
    assert straddle.any()
    lo = str(tmp_path / "spi_nocut")
    spifit(narrowing_beam_tree, lo, flux_scale="intrinsic", products="a", pb_min=1e-3)
    alpha_lo = fits.getdata(f"{lo}_time0.alpha.fits").squeeze()
    assert np.isfinite(alpha_lo[straddle]).any(), "no fittable flux in the straddle annulus"


def test_beam_cut_is_identical_across_flux_scales(narrowing_beam_tree, tmp_path):
    """The mask comes from BEAM, so both scales must drop the same pixels.

    The apparent image is attenuated, so its flux threshold bites harder; the
    beam cut itself must not differ, hence comparing supersets rather than
    equality.
    """
    app = str(tmp_path / "spi_app")
    intr = str(tmp_path / "spi_int")
    spifit(narrowing_beam_tree, app, flux_scale="apparent", products="a", pb_min=PB_MIN)
    spifit(narrowing_beam_tree, intr, flux_scale="intrinsic", products="a", pb_min=PB_MIN)

    beams = _band_beams(narrowing_beam_tree)
    for path in (app, intr):
        alpha = fits.getdata(f"{path}_time0.alpha.fits").squeeze()
        fitted = np.isfinite(alpha)
        assert fitted.any()
        assert (beams.min(axis=0)[fitted] > PB_MIN).all()


def test_recovers_the_injected_spectral_index(pfb_tree, true_alpha, tmp_path):
    out = str(tmp_path / "spi")
    spifit(pfb_tree, out, flux_scale="intrinsic", products="a")

    alpha = fits.getdata(f"{out}_time0.alpha.fits").squeeze()
    fitted = alpha[np.isfinite(alpha)]

    assert fitted.size > 0
    assert np.median(fitted) == pytest.approx(true_alpha, abs=0.05)


def test_apparent_scale_recovers_the_same_index(pfb_tree, true_alpha, tmp_path):
    """Fitting the apparent image with its beam must give the intrinsic index."""
    out = str(tmp_path / "spi_app")
    spifit(pfb_tree, out, flux_scale="apparent", products="a")

    alpha = fits.getdata(f"{out}_time0.alpha.fits").squeeze()
    fitted = alpha[np.isfinite(alpha)]

    assert np.median(fitted) == pytest.approx(true_alpha, abs=0.05)


def test_intrinsic_and_apparent_scales_give_the_same_alpha(pfb_tree, tmp_path):
    """The two scales are one weighted least squares problem, not two estimators.

    The intrinsic fit drops the beam from the model and carries it as a B ** 2
    weight instead, which is algebraically the same normal equations as fitting
    the apparent image with the beam in the model. Any disagreement therefore
    means BIMAGE, IMAGE and BEAM disagree in the tree rather than that the
    estimators differ. Only the shared pixels are compared, because each scale
    thresholds on its own image and the apparent one is attenuated.
    """
    app = str(tmp_path / "spi_app")
    intr = str(tmp_path / "spi_int")
    spifit(pfb_tree, app, flux_scale="apparent", products="a")
    spifit(pfb_tree, intr, flux_scale="intrinsic", products="a")

    a = fits.getdata(f"{app}_time0.alpha.fits").squeeze()
    i = fits.getdata(f"{intr}_time0.alpha.fits").squeeze()
    both = np.isfinite(a) & np.isfinite(i)

    assert both.sum() > 0
    np.testing.assert_allclose(i[both], a[both], rtol=0, atol=1e-5)


def test_unfitted_pixels_are_nan_not_zero(pfb_tree, tmp_path):
    out = str(tmp_path / "spi")
    spifit(pfb_tree, out, flux_scale="intrinsic", products="a")

    alpha = fits.getdata(f"{out}_time0.alpha.fits").squeeze()

    assert np.isnan(alpha).any()


def test_products_letters_select_outputs(pfb_tree, tmp_path):
    out = str(tmp_path / "spi")
    spifit(pfb_tree, out, flux_scale="intrinsic", products="ai")

    assert os.path.exists(f"{out}_time0.alpha.fits")
    assert os.path.exists(f"{out}_time0.I0.fits")
    assert not os.path.exists(f"{out}_time0.alpha_err.fits")
    assert not os.path.exists(f"{out}_time0.Irec_cube.fits")


def test_deselect_bands_reduces_the_bands_used(pfb_tree, tmp_path):
    out = str(tmp_path / "spi")
    spifit(pfb_tree, out, flux_scale="intrinsic", products="I", deselect_bands=[3])

    cube = fits.getdata(f"{out}_time0.Irec_cube.fits")

    assert cube.shape[1] == 3


def test_output_carries_the_tree_wcs(pfb_tree, tmp_path):
    out = str(tmp_path / "spi")
    spifit(pfb_tree, out, flux_scale="intrinsic", products="a")

    hdr = fits.getheader(f"{out}_time0.alpha.fits")
    dt = open_store(pfb_tree)
    ds = dt[band_nodes(dt)[0]].ds

    assert hdr["CRVAL1"] == pytest.approx(np.rad2deg(ds.attrs["ra"]))
    assert hdr["CRVAL2"] == pytest.approx(np.rad2deg(ds.attrs["dec"]))
    assert hdr["NAXIS1"] == ds.IMAGE.shape[-1]
    assert hdr["NAXIS2"] == ds.IMAGE.shape[-2]


def test_refuses_a_tree_whose_bands_disagree_in_resolution(pfb_tree, tmp_path):
    """The message must name the command that fixes it."""
    # pfb_tree is session-scoped, so corrupt a copy -- writing PSFPARSF into the
    # shared fixture would break every test that runs after this one
    import shutil

    corrupt = str(tmp_path / "corrupt_I.dt")
    shutil.copytree(pfb_tree, corrupt)
    dt = open_store(corrupt)
    node = band_nodes(dt)[0]
    write_node(
        corrupt,
        node,
        {"PSFPARSF": (("corr", "bpar"), np.array([[99.0, 99.0, 0.0]], dtype=np.float32))},
        {},
        {"corr": ["I"], "bpar": ["BMAJ", "BMIN", "BPA"]},
    )

    with pytest.raises(ValueError, match="pfb restore"):
        spifit(corrupt, str(tmp_path / "spi"), flux_scale="intrinsic", products="a")


def test_refuses_an_uncombined_multi_partition_tree(two_pointing_fits, tmp_path):
    models, residuals = two_pointing_fits
    out = str(tmp_path / "raw")
    init(models, out, residual=residuals, overwrite=True)

    with pytest.raises(ValueError, match="spimple mosaic"):
        spifit(store_name(out, "I"), str(tmp_path / "spi"), flux_scale="apparent", products="a")


def test_accepts_a_mosaicked_tree(two_pointing_fits, true_alpha, tmp_path):
    from spimple.core.mosaic import mosaic

    models, residuals = two_pointing_fits
    out = str(tmp_path / "mos")
    init(models, out, residual=residuals, overwrite=True)
    mosaic(store_name(out, "I"), fits_outputs="")

    spi = str(tmp_path / "spi")
    spifit(store_name(out, "I"), spi, flux_scale="intrinsic", products="a")

    alpha = fits.getdata(f"{spi}_time0.alpha.fits").squeeze()
    fitted = alpha[np.isfinite(alpha)]
    assert fitted.size > 0
    assert np.median(fitted) == pytest.approx(true_alpha, abs=0.15)


def test_warns_when_the_beam_includes_the_n_term(pfb_tree, tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="spimple.SPIFIT"):
        spifit(pfb_tree, str(tmp_path / "spi"), flux_scale="intrinsic", products="a")

    assert any("beam_includes_n" in record.message or "B/n" in record.message for record in caplog.records)


def test_skips_fully_flagged_bands(pfb_tree, true_alpha, tmp_path):
    """pfb leaves WSUM == 0 band nodes in the tree; its restore skips them.

    Such a node may lack the restored product entirely, so keeping it would
    either abort the fit or give a dead band a weight.
    """
    import shutil

    from spimple.utils.datatree import write_node

    flagged = str(tmp_path / "flagged_I.dt")
    shutil.copytree(pfb_tree, flagged)
    dt = open_store(flagged)
    node = band_nodes(dt)[-1]
    ds = dt[node].ds
    write_node(
        flagged,
        node,
        {
            "WSUM": (("corr",), np.zeros(1, dtype=np.float32)),
            # a dead band that restore never wrote a product for
            "IMAGE": (("corr", "y", "x"), np.full_like(ds.IMAGE.values, np.nan)),
        },
        {},
        {"corr": ["I"]},
    )

    out = str(tmp_path / "spi")
    spifit(flagged, out, flux_scale="intrinsic", products="aI")

    cube = fits.getdata(f"{out}_time0.Irec_cube.fits")
    assert cube.shape[1] == 3, "the fully flagged band should have been dropped"
    alpha = fits.getdata(f"{out}_time0.alpha.fits").squeeze()
    fitted = alpha[np.isfinite(alpha)]
    assert np.median(fitted) == pytest.approx(true_alpha, abs=0.05)


def test_missing_product_error_names_what_is_available(pfb_tree, tmp_path):
    """pfb restore defaults to outputs='kK', so only KIMAGE exists."""
    import shutil

    konly = str(tmp_path / "konly_I.dt")
    shutil.copytree(pfb_tree, konly)
    for node in band_nodes(open_store(konly)):
        for var in ("BIMAGE", "IMAGE"):
            shutil.rmtree(f"{konly}/{node}/{var}", ignore_errors=True)

    with pytest.raises(ValueError, match="KIMAGE") as excinfo:
        spifit(konly, str(tmp_path / "spi"), flux_scale="apparent", products="a")

    # a KIMAGE-only tree has no fittable product at all, so the message has to
    # send the user back to pfb restore rather than name another --flux-scale
    assert "pfb restore" in str(excinfo.value)
    assert "--outputs a" in str(excinfo.value)


def test_mixed_flux_scale_is_rejected(pfb_tree, tmp_path):
    """The mixed scale was removed; the error must say what to run instead."""
    with pytest.raises(ValueError, match="mixed was removed") as excinfo:
        spifit(pfb_tree, str(tmp_path / "spi"), flux_scale="mixed", products="a")

    assert "pfb restore" in str(excinfo.value)
