"""Coverage for `spimple binterp` via the core implementation.

The end-to-end FITS-beam path is not exercised here: it is broken independently
of this port (see test_beam_path_is_known_broken below for the specifics), and
repairing it is out of scope. What is covered is the argument plumbing and the
error contract, which is what the port itself touches.
"""

import pytest

from spimple.core.binterp import binterp


def test_requires_a_beam_model(image_cube, tmp_path):
    """No beam model means nothing to interpolate; that must not pass silently."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        binterp(image=[image_cube], output_filename=str(tmp_path / "pb.fits"), nthreads=1)


def test_rejects_unknown_corr_type(image_cube, tmp_path):
    """corr_type is validated before any beam file is touched."""
    with pytest.raises(KeyError, match="Unknown corr_type"):
        binterp(
            image=[image_cube],
            output_filename=str(tmp_path / "pb.fits"),
            beam_model=str(tmp_path / "nonexistent_beam"),
            corr_type="elliptical",
            nthreads=1,
        )


@pytest.mark.skip(
    reason=(
        "The FITS primary-beam path is broken independently of the hip-cargo port. "
        "make_power_beam is internally inconsistent: load_fits transposes (1,0,3,2) "
        "and the code then drops axis 0 as the correlation axis, which requires the "
        "frequency axis on NAXIS4; but the frequency metadata is read from "
        "CTYPE3/NAXIS3/CRVAL3 a few lines later. No beam cube satisfies both, so the "
        "path raises for every input. Fixing it needs a decision on the intended "
        "on-disk beam layout, which needs real MeerKAT beam files to settle."
    )
)
def test_produces_a_power_beam_cube(image_cube, tmp_path):
    """Placeholder for the real end-to-end beam test once the layout is settled."""
