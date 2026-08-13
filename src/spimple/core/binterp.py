#!/usr/bin/env python

import multiprocessing

from astropy.io import fits
import numpy as np

from spimple.utils.beam import interpolate_beam
from spimple.utils.fits import data_from_header, expand_image_patterns, save_fits, set_header_info
from spimple.utils.logging import get_logger, log_options

log = get_logger("BINTERP")


def binterp(
    image: list[str],
    output_filename: str,
    ms: list[str] | None = None,
    field: int = 0,
    beam_model: str | None = None,
    sparsify_time: int = 10,
    nthreads: int | None = None,
    corr_type: str = "linear",
):
    """
    Interpolates a primary beam model onto the coordinate grid of a
    specified FITS image and saves the result as a new FITS file.

    This function extracts spatial and frequency coordinates from an input
    FITS image, interpolates the primary beam pattern using optional
    measurement set and beam model information, updates the FITS header
    with relevant frequency metadata, and writes the resulting beam cube
    to the specified output file.
    """
    log_options(log, **locals())

    image = expand_image_patterns(image)

    if not nthreads:
        nthreads = multiprocessing.cpu_count()

    # get coord info
    hdr = fits.getheader(image[0])
    l_coord, ref_l = data_from_header(hdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(hdr, axis=2)
    m_coord -= ref_m
    if hdr["CTYPE4"].lower() == "freq":
        freq_axis = 4
    elif hdr["CTYPE3"].lower() == "freq":
        freq_axis = 3
    else:
        raise ValueError("Freq axis must be 3rd or 4th")
    freqs, ref_freq = data_from_header(hdr, axis=freq_axis)

    xx, yy = np.meshgrid(l_coord, m_coord, indexing="ij")

    # interpolate primary beam to fits header and optionally average over time
    # Note: interpolate_beam expects an opts object - need to create compatible structure
    class BeamOpts:
        pass

    beam_opts = BeamOpts()
    beam_opts.beam_model = beam_model
    beam_opts.ms = ms
    beam_opts.field = field
    beam_opts.sparsify_time = sparsify_time
    beam_opts.corr_type = corr_type
    beam_image = interpolate_beam(xx, yy, freqs, beam_opts)

    # new header for cubes if ref_freqs or freq_axis differs
    new_hdr = set_header_info(hdr, ref_freq, freq_axis)

    # save power beam
    save_fits(output_filename, beam_image, new_hdr)
    log.info("Wrote interpolated beam cube to %s", output_filename)
