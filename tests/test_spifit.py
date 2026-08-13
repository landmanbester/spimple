"""End-to-end coverage for `spimple spifit` via the core implementation.

These call the core function directly -- the same entry point the generated
Stimela cab's `command:` targets.
"""

import numpy as np
import pytest
from astropy.io import fits

from spimple.core.spifit import spifit


@pytest.mark.slow
def test_recovers_the_injected_spectral_index(image_cube, residual_cube, true_alpha, tmp_path):
    """The fixture cube is a pure power law; the fit must find its index."""
    outname = tmp_path / "spi"

    spifit(
        images=[image_cube],
        output_filename=str(outname),
        residual=[residual_cube],
        products="ai",
        nthreads=1,
        dont_convolve=True,
    )

    alpha_map = np.squeeze(fits.getdata(str(tmp_path / "spi.alpha.fits")))
    fitted = alpha_map[np.isfinite(alpha_map) & (alpha_map != 0)]

    assert fitted.size > 0, "no components were fit"
    assert np.median(fitted) == pytest.approx(true_alpha, abs=0.05)


@pytest.mark.slow
def test_products_letters_select_outputs(image_cube, residual_cube, tmp_path):
    outname = tmp_path / "prod"

    spifit(
        images=[image_cube],
        output_filename=str(outname),
        residual=[residual_cube],
        products="ai",
        nthreads=1,
        dont_convolve=True,
    )

    assert (tmp_path / "prod.alpha.fits").exists()
    assert (tmp_path / "prod.I0.fits").exists()


@pytest.mark.slow
def test_i0_map_is_positive_where_fit(image_cube, residual_cube, tmp_path):
    """I0 is a reference-frequency intensity, so every fitted pixel is positive.

    Pixels below the fitting threshold come back NaN rather than zero, so the
    mask has to be isfinite -- `!= 0` lets NaN through, which is how an earlier
    version of this test silently measured nothing.
    """
    outname = tmp_path / "i0"

    spifit(
        images=[image_cube],
        output_filename=str(outname),
        residual=[residual_cube],
        products="ai",
        nthreads=1,
        dont_convolve=True,
    )

    i0 = np.squeeze(fits.getdata(str(tmp_path / "i0.I0.fits")))
    fitted = i0[np.isfinite(i0) & (i0 != 0)]

    assert fitted.size > 0, "no components were fit"
    assert np.all(fitted > 0)


@pytest.mark.slow
def test_max_dr_threshold_used_without_residual(image_cube, tmp_path):
    """With no residual the threshold comes from maxDR instead of the rms.

    Task 9 renames this parameter to max_dr; this call site is deliberately
    written against the current name so the rename fails loudly if applied
    inconsistently.
    """
    outname = tmp_path / "nodr"

    spifit(
        images=[image_cube],
        output_filename=str(outname),
        products="a",
        nthreads=1,
        dont_convolve=True,
        max_dr=100,
    )

    assert (tmp_path / "nodr.alpha.fits").exists()
