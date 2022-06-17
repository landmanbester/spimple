#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import pyscilog
pyscilog.init('spimple')
log = pyscilog.get_logger('SPIFIT')
import argparse
from omegaconf import OmegaConf
import numpy as np
from astropy.io import fits
from spimple.utils import (load_fits, save_fits, convolve2gaussres, data_from_header,
                           set_header_info, str2bool, interpolate_beam)
import dask
import dask.array as da
from africanus.model.spi.dask import fit_spi_components

def spi_fitter():
    parser = argparse.ArgumentParser(description='Simple spectral index fitting tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-model', "--model", type=str)
    parser.add_argument('-residual', "--residual", type=str)
    parser.add_argument('-o', '--output-filename', type=str, required=True,
                        help="Path to output directory + prefix.")
    parser.add_argument('-pp', '--psf-pars', default=None, nargs='+', type=float,
                        help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    parser.add_argument('-cp', "--circ-psf", type=str2bool, nargs='?', const=True, default=False,
                        help="Passing this flag will convolve with a circularised "
                        "beam instead of an elliptical one")
    parser.add_argument('-th', '--threshold', default=10, type=float,
                        help="Multiple of the rms in the residual to threshold "
                        "on. \n"
                        "Only components above threshold*rms will be fit.")
    parser.add_argument('-maxDR', '--maxDR', default=100, type=float,
                        help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit. \n"
                        "Only used if residual is not passed in.")
    parser.add_argument('-nthreads', '--nthreads', default=0, type=int,
                        help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    parser.add_argument('-pb-min', '--pb-min', type=float, default=0.15,
                        help="Set image to zero where pb falls below this value")
    parser.add_argument('-products', '--products', default='aeikIcmrb', type=str,
                        help="Outputs to write. Letter correspond to: \n"
                        "a - alpha map \n"
                        "e - alpha error map \n"
                        "i - I0 map \n"
                        "k - I0 error map \n"
                        "I - reconstructed cube form alpha and I0 \n"
                        "c - restoring beam used for convolution \n"
                        "m - convolved model \n"
                        "r - convolved residual \n"
                        "b - average power beam \n"
                        "Default is to write all of them")
    parser.add_argument('-pf', "--padding-frac", default=0.5, type=float,
                        help="Padding factor for FFT's.")
    parser.add_argument('-dc', "--dont-convolve", type=str2bool, nargs='?', const=True, default=False,
                        help="Passing this flag bypasses the convolution "
                        "by the clean beam")
    parser.add_argument('-cw', "--channel_weights", default=None, nargs='+', type=float,
                        help="Per-channel weights to use during fit to frequency axis. \n "
                        "Only has an effect if no residual is passed in (for now).")
    parser.add_argument('-rf', '--ref-freq', default=None, type=np.float64,
                        help='Reference frequency where the I0 map is sought. \n'
                        "Will overwrite in fits headers of output.")
    parser.add_argument('-otype', '--out_dtype', default='f4', type=str,
                        help="Data type of output. Default is single precision")
    parser.add_argument('-acr', '--add-convolved-residuals', type=str2bool, nargs='?', const=True, default=True,
                        help='Flag to add in the convolved residuals before fitting components')
    parser.add_argument('-ms', "--ms", nargs="+", type=str,
                        help="Mesurement sets used to make the image. \n"
                        "Used to get paralactic angles if doing primary beam correction")
    parser.add_argument('-f', "--field", type=int, default=0,
                        help="Field ID")
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
    parser.add_argument('-ct', '--corr-type', type=str, default='linear',
                        help="Correlation typ i.e. linear or circular. ")
    parser.add_argument('-band', "--band", type=str, default='l',
                        help="Band to use with JimBeam. L or UHF")

    opts = parser.parse_args()
    opts = OmegaConf.create(vars(opts))
    pyscilog.log_to_file(f'spifit.log')

    if not opts.nthreads:
        import multiprocessing
        opts.nthreads = multiprocessing.cpu_count()

    OmegaConf.set_struct(opts, True)

    print('Input Options:', file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    if opts.psf_pars is None:
        print("Attempting to take psf_pars from residual/image fits header", file=log)
        try:
            rhdr = fits.getheader(opts.residual)
        except Exception as e:
            try:
                rhdr = fits.getheader(opts.model)
            except Exception as e:
                raise e

        if 'BMAJ1' in rhdr.keys():
            emaj = rhdr['BMAJ1']
            emin = rhdr['BMIN1']
            pa = rhdr['BPA1']
            gaussparf = (emaj, emin, pa)
        elif 'BMAJ' in rhdr.keys():
            emaj = rhdr['BMAJ']
            emin = rhdr['BMIN']
            pa = rhdr['BPA']
            gaussparf = (emaj, emin, pa)
        else:
            raise ValueError("No beam parameters found in residual."
                             "You will have to provide them manually.")

    else:
        gaussparf = tuple(opts.psf_pars)

    if opts.circ_psf:
        e = np.maximum(gaussparf[0], gaussparf[1])
        gaussparf = list(gaussparf)
        gaussparf[0] = e
        gaussparf[1] = e
        gaussparf[2] = 0.0
        gaussparf = tuple(gaussparf)

    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % gaussparf, file=log)

    # load model image
    model = load_fits(opts.model, dtype=opts.out_dtype).squeeze()
    orig_shape = model.shape
    mhdr = fits.getheader(opts.model)

    l_coord, ref_l = data_from_header(mhdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(mhdr, axis=2)
    m_coord -= ref_m
    if mhdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
        stokes_axis = 3
    elif mhdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
        stokes_axis = 4
    else:
        raise ValueError("Freq axis must be 3rd or 4th")

    mfs_shape = list(orig_shape)
    mfs_shape[0] = 1
    mfs_shape = tuple(mfs_shape)
    freqs, ref_freq = data_from_header(mhdr, axis=freq_axis)

    nband = freqs.size
    if nband < 2:
        raise ValueError("Can't produce alpha map from a single band image")
    npix_l = l_coord.size
    npix_m = m_coord.size

    # update cube psf-pars
    for i in range(1, nband+1):
        mhdr['BMAJ' + str(i)] = gaussparf[0]
        mhdr['BMIN' + str(i)] = gaussparf[1]
        mhdr['BPA' + str(i)] = gaussparf[2]

    if opts.ref_freq is not None and opts.ref_freq != ref_freq:
        ref_freq = opts.ref_freq
        print("Provided reference frequency does not match that of fits file. "
              "Will overwrite.", file=log)

    print("Cube frequencies:", file=log)
    with np.printoptions(precision=2):
        print(freqs, file=log)
    print("Reference frequency is %3.2e Hz" % ref_freq, file=log)

    # LB - new header for cubes if ref_freqs differ
    new_hdr = set_header_info(mhdr, ref_freq, freq_axis, opts, gaussparf)

    # save next to model if no outfile is provided
    outfile = opts.output_filename

    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

    # load beam
    if opts.beam_model is not None:
        # we can pass in either a fits file with the already interpolated beam or we can interpolate from scratch
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

            beam_image = load_fits(opts.beam_model, dtype=opts.out_dtype).squeeze()
        elif opts.beam_model == "JimBeam":
            from katbeam import JimBeam
            if opts.band.lower() == 'l':
                beam = JimBeam('MKAT-AA-L-JIM-2020')
            else:
                beam = JimBeam('MKAT-AA-UHF-JIM-2020')
            beam_image = np.zeros(model.shape, dtype=opts.out_dtype)
            for v in range(freqs.size):
                beam_image[v] = beam.I(xx, yy, freqs[v]/1e6)  # freqs in MHz

        else:
            beam_image = interpolate_beam(xx, yy, freqs, opts)

        if 'b' in opts.products:
            name = outfile + '.power_beam.fits'
            save_fits(name, np.expand_dims(beam_image, axis=4 - stokes_axis), mhdr, dtype=opts.out_dtype)
            print(f"Wrote average power beam to {name}", file=log)

    else:
        beam_image = np.ones(model.shape, dtype=opts.out_dtype)

    # beam cut off
    model = np.where(beam_image > opts.pb_min, model, 0.0)

    if not opts.dont_convolve:
        print("Convolving model", file=log)
        # convolve model to desired resolution
        model, gausskern = convolve2gaussres(model, xx, yy, gaussparf, opts.nthreads, None, opts.padding_frac)

        # save clean beam
        if 'c' in opts.products:
            name = outfile + '.clean_psf.fits'
            save_fits(name, gausskern, new_hdr, dtype=opts.out_dtype)
            print(f"Wrote clean psf to {name}", file=log)

        # save convolved model
        if 'm' in opts.products:
            name = outfile + '.convolved_model.fits'
            save_fits(name, model, new_hdr, dtype=opts.out_dtype)
            print(f"Wrote convolved model to {name}", file=log)


    # add in residuals and set threshold
    if opts.residual is not None:
        resid = load_fits(opts.residual, dtype=opts.out_dtype).squeeze()
        rhdr = fits.getheader(opts.residual)
        l_res, ref_lb = data_from_header(rhdr, axis=1)
        l_res -= ref_lb
        if not np.array_equal(l_res, l_coord):
            raise ValueError("l coordinates of residual do not match those of model")

        m_res, ref_mb = data_from_header(rhdr, axis=2)
        m_res -= ref_mb
        if not np.array_equal(m_res, m_coord):
            raise ValueError("m coordinates of residual do not match those of model")

        freqs_res, _ = data_from_header(rhdr, axis=freq_axis)
        if not np.array_equal(freqs, freqs_res):
            raise ValueError("Freqs of residual do not match those of model")

        # convolve residual to same resolution as model
        gausspari = ()
        for i in range(1,nband+1):
            key = 'BMAJ' + str(i)
            if key in rhdr.keys():
                emaj = rhdr[key]
                emin = rhdr['BMIN' + str(i)]
                pa = rhdr['BPA' + str(i)]
                gausspari += ((emaj, emin, pa),)
            else:
                print("Can't find Gausspars in residual header, unable to add residuals back in", file=log)
                gausspari = None
                break

        if gausspari is not None and opts.add_convolved_residuals:
            print("Convolving residuals", file=log)
            resid, _ = convolve2gaussres(resid, xx, yy, gaussparf, opts.nthreads, gausspari, opts.padding_frac, norm_kernel=False)
            model += resid
            print("Convolved residuals added to convolved model", file=log)

            if 'r' in opts.products:
                name = outfile + '.convolved_residual.fits'
                save_fits(name, resid, rhdr)
                print(f"Wrote convolved residuals to {name}", file=log)

        counts = np.sum(resid != 0)
        rms = np.sqrt(np.sum(resid**2)/counts)
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
        threshold = opts.threshold * rms
        print(f"Setting cutoff threshold as {opts.threshold} times the rms "
              "of the residual ", file=log)
        del resid
    else:
        print("No residual provided. Setting  threshold i.t.o dynamic range. "
              f"Max dynamic range is {opts.maxDR}", file=log)
        threshold = model.max()/opts.maxDR
        rms_cube = None

    print(f"Threshold set to {threshold} Jy.", file=log)

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    nanindices = np.argwhere(minimage <= threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                        "Try lowering your threshold."
                        "Max of convolved model is %3.2e" % model.max())
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T
    beam_comps = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T

    # set weights for fit
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights.", file=log)
        weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        if opts.channel_weights is not None:
            weights = np.array(opts.channel_weights)
            print("Using provided channel weights.", file=log)
        else:
            print("No residual or channel weights provided. Using equal weights.", file=log)
            weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//opts.nthreads, nband))
    beam_comps = da.from_array(beam_comps.astype(np.float64),
                               chunks=(ncomps//opts.nthreads, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting {ncomps} components", file=log)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                        np.float64(ref_freq), beam=beam_comps).compute()
    print("Done. Writing output.", file=log)

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[...] = np.nan
    alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    alpha_err_map[...] = np.nan
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    i0map[...] = np.nan
    i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    i0_err_map[...] = np.nan
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
    i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

    if 'I' in opts.products:
        # get the reconstructed cube
        Irec_cube = i0map[None, :, :] * \
            (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
        name = outfile + '.Irec_cube.fits'
        save_fits(name, np.expand_dims(Irec_cube, axis=4 - stokes_axis), mhdr, dtype=opts.out_dtype)
        print(f"Wrote reconstructed cube to {name}", file=log)

    # save alpha map
    if 'a' in opts.products:
        name = outfile + '.alpha.fits'
        save_fits(name, alphamap, mhdr, dtype=opts.out_dtype)
        print(f"Wrote alpha map to {name}", file=log)

    # save alpha error map
    if 'e' in opts.products:
        name = outfile + '.alpha_err.fits'
        save_fits(name, alpha_err_map, mhdr, dtype=opts.out_dtype)
        print(f"Wrote alpha error map to {name}", file=log)

    # save I0 map
    if 'i' in opts.products:
        name = outfile + '.I0.fits'
        save_fits(name, i0map, mhdr, dtype=opts.out_dtype)
        print(f"Wrote I0 map to {name}", file=log)

    # save I0 error map
    if 'k' in opts.products:
        name = outfile + '.I0_err.fits'
        save_fits(name, i0_err_map, mhdr, dtype=opts.out_dtype)
        print(f"Wrote I0 error map to {name}", file=log)

    print("All done here", file=log)
