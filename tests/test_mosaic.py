"""Coverage for `spimple mosaic`.

The end-to-end path is not exercised here: `mosaic` requires a primary beam in
the meerkat-beams `.bds.zarr` format and has no code path for running without
one, so a hermetic end-to-end test would have to synthesise that dataset. See
test_end_to_end_needs_a_bds_beam below.

What is covered is `mosaic_info`, which does the coordinate-frame and output-naming
work and runs on plain FITS.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from spimple.utils.mosaic import mosaic_info


def test_mosaic_info_derives_a_common_frame(image_cube_ctype3, tmp_path):
    """A single input's optimal frame must cover it and carry its frequencies."""
    ref_wcs, ufreqs, out_names = mosaic_info([image_cube_ctype3], str(tmp_path / "mos.fits"))

    assert isinstance(ref_wcs, WCS)
    assert ref_wcs.array_shape is not None
    assert ufreqs.size == 4
    assert np.all(np.diff(ufreqs) > 0), "frequencies must come back sorted and unique"


def test_mosaic_info_names_one_output_per_corr_and_channel(image_cube_ctype3, tmp_path):
    """out_names drives the per-slice zarr scratch products."""
    _, _, out_names = mosaic_info([image_cube_ctype3], str(tmp_path / "mos.fits"))

    # fixture is 1 correlation x 4 channels
    assert len(out_names) == 4
    assert all(name.endswith(".zarr") for name in out_names)
    assert all("_im0_" in name for name in out_names)


def test_mosaic_info_deduplicates_shared_frequencies(image_cube_ctype3, tmp_path):
    """The same cube twice spans the same band, so ufreqs must not double up."""
    _, ufreqs, out_names = mosaic_info([image_cube_ctype3, image_cube_ctype3], str(tmp_path / "mos.fits"))

    assert ufreqs.size == 4, "identical inputs must not multiply the frequency axis"
    assert len(out_names) == 8, "but each input still gets its own scratch products"


def test_mosaic_info_rejects_a_reference_image(image_cube_ctype3, tmp_path):
    """ref_image is declared but not implemented; it must say so rather than ignore it."""
    with pytest.raises(NotImplementedError, match="Reference image"):
        mosaic_info([image_cube_ctype3], str(tmp_path / "mos.fits"), ref_image=image_cube_ctype3)


@pytest.mark.skip(
    reason=(
        "`mosaic` cannot run without a primary beam: utils/mosaic.project calls "
        "xr.open_zarr(beam) unconditionally, so beam=None dies with an opaque "
        "zarr GroupNotFoundError. Covering this end to end needs a synthetic "
        "meerkat-beams .bds.zarr fixture. Separately, project indexes the "
        "frequency axis with the correlation index (`beamo((freq[c], ll, mm))`), "
        "which looks like it should be freq[f]."
    )
)
def test_end_to_end_needs_a_bds_beam(image_cube_ctype3, tmp_path):
    """Placeholder for the real end-to-end mosaic test."""
