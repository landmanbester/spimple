"""End-to-end coverage for `spimple imconv` via the core implementation.

These call the core function directly -- the same entry point the generated
Stimela cab's `command:` targets -- so they exercise what a recipe actually
runs, not just the Typer wrapper.
"""

import numpy as np
import pytest
from astropy.io import fits

from spimple.core.imconv import imconv


@pytest.mark.slow
def test_writes_convolved_image(image_cube, beam_params, tmp_path):
    outname = tmp_path / "conv"

    imconv(images=[image_cube], output_filename=outname, products="i", psf_pars=beam_params, nthreads=1)

    written = tmp_path / "conv.convolved.fits"
    assert written.exists(), f"expected {written}, found {sorted(p.name for p in tmp_path.iterdir())}"
    assert np.all(np.isfinite(fits.getdata(str(written))))


@pytest.mark.slow
def test_products_letters_select_outputs(image_cube, beam_params, tmp_path):
    """'c' adds the restoring beam alongside 'i's convolved image."""
    outname = tmp_path / "multi"

    imconv(images=[image_cube], output_filename=outname, products="ic", psf_pars=beam_params, nthreads=1)

    assert (tmp_path / "multi.convolved.fits").exists()
    assert (tmp_path / "multi.clean_psf.fits").exists()


@pytest.mark.slow
def test_power_beam_product_requires_a_beam_model(image_cube, beam_params, tmp_path):
    """'b' without a beam model is a clear error, not a crash deep in the beam code."""
    with pytest.raises(ValueError, match="no beam model provided"):
        imconv(
            images=[image_cube],
            output_filename=tmp_path / "nobeam",
            products="b",
            psf_pars=beam_params,
            nthreads=1,
        )


@pytest.mark.slow
def test_out_dtype_is_honoured(image_cube, beam_params, tmp_path):
    outname = tmp_path / "f8"

    imconv(
        images=[image_cube],
        output_filename=outname,
        products="i",
        psf_pars=beam_params,
        nthreads=1,
        out_dtype="f8",
    )

    # FITS stores big-endian, so compare the itemsize rather than the dtype object.
    written_dtype = fits.getdata(str(tmp_path / "f8.convolved.fits")).dtype
    assert written_dtype.kind == "f"
    assert written_dtype.itemsize == 8


@pytest.mark.slow
def test_circ_psf_changes_the_convolution(image_cube, beam_params, tmp_path):
    """--circ-psf convolves with a circularised beam, so the pixels must differ.

    Note it does NOT alter the reported BMAJ/BMIN in the output header; the
    observable effect is in the data.
    """
    for circ in (False, True):
        imconv(
            images=[image_cube],
            output_filename=tmp_path / f"circ{circ}",
            products="i",
            psf_pars=beam_params,
            nthreads=1,
            circ_psf=circ,
        )

    elliptical = fits.getdata(str(tmp_path / "circFalse.convolved.fits"))
    circular = fits.getdata(str(tmp_path / "circTrue.convolved.fits"))

    assert not np.allclose(elliptical, circular)


@pytest.mark.slow
def test_convolution_scales_flux_by_the_beam_area_ratio(image_cube, beam_params, tmp_path):
    """Jy/beam summed over the image scales with the restoring-beam area.

    The data are in Jy/beam, so convolving from a native beam to a coarser
    common beam multiplies the summed pixel values by the ratio of beam areas.
    The fixture's sharpest channel is 3.0 x 2.4 arcsec and beam_params is
    8.0 x 6.4, giving a factor of ~7.1. This is the physically meaningful
    check: a normalisation regression in convolve2gaussres moves this number.
    """
    outname = tmp_path / "flux"

    imconv(images=[image_cube], output_filename=outname, products="i", psf_pars=beam_params, nthreads=1)

    original = np.squeeze(fits.getdata(image_cube))
    convolved = np.squeeze(fits.getdata(str(tmp_path / "flux.convolved.fits")))

    hdr = fits.getheader(image_cube)
    native_area = hdr["BMAJ4"] * hdr["BMIN4"]  # sharpest channel
    target_area = beam_params[0] * beam_params[1]
    expected_ratio = target_area / native_area

    assert convolved[-1].sum() / original[-1].sum() == pytest.approx(expected_ratio, rel=0.05)


def test_imconv_warns_that_it_is_deprecated(image_cube, beam_params, tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="spimple.IMCONV"):
        imconv([image_cube], tmp_path / "conv", psf_pars=beam_params)

    assert any("spimple init" in record.message for record in caplog.records)
