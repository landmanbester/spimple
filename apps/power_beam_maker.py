#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import pyscilog
pyscilog.init('spimple')
log = pyscilog.get_logger('BINTERP')
import argparse
from omegaconf import OmegaConf
import numpy as np
from astropy.io import fits
import warnings
from spimple.utils import (load_fits, save_fits, data_from_header,
                           interpolate_beam)
from daskms import xds_from_ms, xds_from_table


def power_beam_maker():
    parser = argparse.ArgumentParser(description='Beam intrepolation tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-image', "--image", type=str, required=True)
    parser.add_argument('-ms', "--ms", nargs="+", type=str,
                        help="Mesurement sets used to make the image. \n"
                        "Used to get paralactic angles if doing primary beam correction")
    parser.add_argument('-f', "--field", type=int, default=0,
                        help="Field ID")
    parser.add_argument('-o', '--output-filename', type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument('-bm', '--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "It is assumed that the pattern is path_to_beam/"
                        "name_corr_re/im.fits. \n"
                        "Provide only the path up to name "
                        "e.g. /home/user/beams/meerkat_lband. \n"
                        "Patterns mathing corr are determined "
                        "automatically. \n"
                        "Only real and imaginary beam models currently "
                        "supported.")
    parser.add_argument('-st', "--sparsify-time", type=int, default=10,
                   help="Used to select a subset of time ")
    parser.add_argument('-nthreads', '--nthreads', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    parser.add_argument('-ct', '--corr-type', type=str, default='linear',
                   help="Correlation typ i.e. linear or circular. ")

    opts = parser.parse_args()
    opts = OmegaConf.create(vars(opts))
    pyscilog.log_to_file(f'binterp.log')

    if not opts.nthreads:
        import multiprocessing
        opts.nthreads = multiprocessing.cpu_count()

    OmegaConf.set_struct(opts, True)

    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    # get coord info
    hdr = fits.getheader(opts.image)
    l_coord, ref_l = data_from_header(hdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(hdr, axis=2)
    m_coord -= ref_m
    if hdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
    elif hdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
    else:
        raise ValueError("Freq axis must be 3rd or 4th")
    freqs, ref_freq = data_from_header(hdr, axis=freq_axis)

    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

    # interpolate primary beam to fits header and optionally average over time
    beam_image = interpolate_beam(xx, yy, freqs, opts)

    # save power beam
    save_fits(opts.output_filename, beam_image, hdr)
    print(f"Wrote interpolated beam cube to {opts.output_filename}", file=log)


    return
