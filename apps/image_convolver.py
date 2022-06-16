#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
import pyscilog
pyscilog.init('spimple')
log = pyscilog.get_logger('IMCONV')
import argparse
from omegaconf import OmegaConf
import numpy as np
from astropy.io import fits
from spimple.utils import load_fits, save_fits, convolve2gaussres, data_from_header


def image_convolver():
    parser = argparse.ArgumentParser(description='Convolve images to a common resolution.',
                                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-image', "--image", type=str, required=True)
    parser.add_argument('-o', '--output-filename', type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument('-pp', '--psf-pars', default=None, nargs='+', type=float,
                        help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the image.")
    parser.add_argument('-nthreads', '--nthreads', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    parser.add_argument('-cp', "--circ-psf", action="store_true",
                        help="Passing this flag will convolve with a circularised "
                        "beam instead of an elliptical one")
    parser.add_argument('-bm', '--beam-model', default=None, type=str,
                        help="Fits beam model to use. \n"
                        "Use power_beam_maker to make power beam "
                        "corresponding to image. ")
    parser.add_argument('-band', "--band", type=str, default='l',
                        help="Band to use with JimBeam. L or UHF")
    parser.add_argument('-pb-min', '--pb-min', type=float, default=0.05,
                        help="Set image to zero where pb falls below this value")
    parser.add_argument('-pf', '--padding-frac', type=float, default=0.5,
                        help="Padding fraction for FFTs (half on either side)")
    parser.add_argument('-otype', '--out_dtype', default='f4', type=str,
                        help="Data type of output. Default is single precision")
    opts = parser.parse_args()
    opts = OmegaConf.create(vars(opts))
    pyscilog.log_to_file(f'image_convolver.log')

    if not opts.nthreads:
        import multiprocessing
        opts.nthreads = multiprocessing.cpu_count()

    OmegaConf.set_struct(opts, True)

    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    # read coords from fits file
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

    nchan = freqs.size
    gausspari = ()
    if freqs.size > 1:
        for i in range(1,nchan+1):
            key = 'BMAJ' + str(i)
            if key in hdr.keys():
                emaj = hdr[key]
                emin = hdr['BMIN' + str(i)]
                pa = hdr['BPA' + str(i)]
                gausspari += ((emaj, emin, pa),)
    else:
        if 'BMAJ' in hdr.keys():
            emaj = hdr['BMAJ']
            emin = hdr['BMIN']
            pa = hdr['BPA']
            # using key of 1 for consistency with fits standard
            gausspari = ((emaj, emin, pa),)

    if len(gausspari) == 0 and opts.psf_pars is None:
        raise ValueError("No psf parameters in fits file and none passed in.", file=log)

    if len(gausspari) == 0:
        print("No psf parameters in fits file. "
              "Convolving model to resolution specified by psf-pars.", file=log)
        gaussparf = tuple(opts.psf_pars)
    else:
        if opts.psf_pars is None:
            gaussparf = gausspari[0]
        else:
            gaussparf = tuple(opts.psf_pars)

    if opts.circ_psf:
        e = (gaussparf[0] + gaussparf[1])/2.0
        gaussparf[0] = e
        gaussparf[1] = e

    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % gaussparf, file=log)

    # update header
    if freqs.size > 1:
        for i in range(1, nchan+1):
            hdr['BMAJ' + str(i)] = gaussparf[0]
            hdr['BMIN' + str(i)] = gaussparf[1]
            hdr['BPA' + str(i)] = gaussparf[2]
    else:
        hdr['BMAJ'] = gaussparf[0]
        hdr['BMIN'] = gaussparf[1]
        hdr['BPA'] = gaussparf[2]

    # coodinate grid
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

    # convolve image
    imagei = load_fits(opts.image, dtype=np.float32).squeeze()
    if imagei.ndim==2:
        imagei = imagei[None, :, :]
    if imagei.ndim != 3:
        raise ValueError("Unsupported number of image dimensions")
    print('Convolving image', file=log)
    image, gausskern = convolve2gaussres(imagei, xx, yy, gaussparf,
                                         opts.nthreads, gausspari,
                                         opts.padding_frac)

    # load beam and correct
    if opts.beam_model is not None:
        if opts.beam_model.endswith('.fits'):
            bhdr = fits.getheader(opts.beam_model)
            l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
            l_coord_beam -= ref_lb
            if not np.array_equal(l_coord_beam, l_coord):
                raise ValueError("l coordinates of beam model do not match those of image. "
                                "Use power_beam_maker to interpolate to fits header.")

            m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
            m_coord_beam -= ref_mb
            if not np.array_equal(m_coord_beam, m_coord):
                raise ValueError("m coordinates of beam model do not match those of image. "
                                "Use power_beam_maker to interpolate to fits header.")

            freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
            if not np.array_equal(freqs, freqs_beam):
                raise ValueError("Freqs of beam model do not match those of image. "
                                "Use power_beam_maker to interpolate to fits header.")

            beam_image = load_fits(opts.beam_model, dtype=np.float32).squeeze()
        elif opts.beam_model == "JimBeam":
            from katbeam import JimBeam
            if opts.band.lower() == 'l':
                beam = JimBeam('MKAT-AA-L-JIM-2020')
            else:
                beam = JimBeam('MKAT-AA-UHF-JIM-2020')
            beam_image = np.zeros(image.shape, dtype=opts.out_dtype)
            for v in range(freqs.size):
                beam_image[v] = beam.I(xx, yy, freqs[v]/1e6)  # freqs in MHz
        else:
            raise ValueError(f"Unknown beam model {opts.beam_model}")

        image = np.where(beam_image >= opts.pb_min, image/beam_image, 0.0)


    outfile = opts.output_filename

    # save images
    name = outfile + '.clean_psf.fits'
    save_fits(name, gausskern, hdr, dtype=opts.out_dtype)
    print(f"Wrote clean psf to {name}", file=log)

    name = outfile + '.convolved.fits'
    save_fits(name, image, hdr, dtype=opts.out_dtype)
    print(f"Wrote convolved image to {name}", file=log)

    print("All done here", file=log)
