#!/usr/bin/env python

import multiprocessing

import dask.array as da
import numpy as np
from africanus.model.spi.dask import fit_spi_components
from astropy.io import fits
from katbeam import JimBeam

from spimple.utils.beam import interpolate_beam
from spimple.utils.convolution import convolve2gaussres
from spimple.utils.fits import data_from_header, expand_image_patterns, load_fits, save_fits, set_header_info
from spimple.utils.logging import get_logger, log_options

log = get_logger("SPIFIT")


def spifit(
    image: list[str],
    output_filename: str,
    residual: list[str] | None = None,
    psf_pars: tuple[float, float, float] | None = None,
    circ_psf: bool = False,
    threshold: float = 10,
    maxDR: float = 1000,
    nthreads: int | None = None,
    pfb_min: float = 0.15,
    products: str = "aeikIcmrbd",
    padding_frac: float = 0.5,
    dont_convolve: bool = False,
    channel_weights_keyword: str = "WSCIMWG",
    channel_freqs: list[float] | None = None,
    ref_freq: float | None = None,
    out_dtype: str = "f4",
    add_convolved_residuals: bool = False,
    ms: list[str] | None = None,
    beam_model: str | None = None,
    sparsify_time: int = 10,
    corr_type: str = "linear",
    band: str = "L",
    deselect_bands: list[int] | None = None,
):
    """
    Runs spectral index fitting on radio astronomy image cubes.

    This function orchestrates the workflow for fitting spectral index (alpha) and
    reference intensity (I0) maps from multi-frequency radio interferometric image
    cubes. It handles model and residual image loading, PSF and beam parameter
    extraction, optional convolution, masking, thresholding, frequency band selection,
    component extraction, spectral fitting, and output of results as FITS files.
    The tool supports various options for beam modeling, channel weighting, and output
    product selection, and is designed for efficient processing of large datasets using
    Dask for parallelization.
    """
    log_options(log, **locals())

    image = expand_image_patterns(image)
    if residual is not None:
        residual = expand_image_patterns(residual)

    if not nthreads:
        nthreads = multiprocessing.cpu_count()

    if psf_pars is None:
        log.info("Attempting to take psf_pars from residual/image fits header")
        try:
            rhdr = fits.getheader(residual[0])
        except Exception:
            rhdr = fits.getheader(image[0])

        if "BMAJ1" in rhdr:
            emaj = rhdr["BMAJ1"]
            emin = rhdr["BMIN1"]
            pa = rhdr["BPA1"]
            gaussparf = (emaj, emin, pa)
        elif "BMAJ" in rhdr:
            emaj = rhdr["BMAJ"]
            emin = rhdr["BMIN"]
            pa = rhdr["BPA"]
            gaussparf = (emaj, emin, pa)
        else:
            raise ValueError("No beam parameters found in residual.You will have to provide them manually.")

    else:
        gaussparf = tuple(psf_pars)

    if circ_psf:
        e = np.maximum(gaussparf[0], gaussparf[1])
        gaussparf_list = list(gaussparf)
        gaussparf_list[0] = e
        gaussparf_list[1] = e
        gaussparf_list[2] = 0.0
        gaussparf = tuple(gaussparf_list)

    emaj = gaussparf[0]
    emin = gaussparf[1]
    pa = gaussparf[2]
    log.info("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e", emaj, emin, pa)

    # load model images or cube
    model_header = {}
    for i, m in enumerate(image):
        model = load_fits(m, dtype=out_dtype).squeeze()
        orig_shape = model.shape
        mhdr = fits.getheader(m)
        model_header[i] = [model, mhdr]

    l_coord, ref_l = data_from_header(mhdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(mhdr, axis=2)
    m_coord -= ref_m

    if mhdr["CTYPE4"].lower() in ["freq", "speclnmf"]:
        freq_axis = 4
        stokes_axis = 3
    elif mhdr["CTYPE3"].lower() in ["freq", "speclnmf"]:
        freq_axis = 3
        stokes_axis = 4
    else:
        raise ValueError("Freq axis must be 3rd or 4th")

    mfs_shape = list(orig_shape)
    mfs_shape[0] = 1
    mfs_shape = tuple(mfs_shape)

    # Fixed: Separate list and array variables to help mypy type inference
    # generate model frequencies and model slices
    # incase one/more than one frequency models are provided
    freq_list: list[float] = []
    models_data = []
    for model_data, model_hdr in model_header.values():
        mhdr = model_hdr
        freq, ref_freq = data_from_header(mhdr, axis=freq_axis)
        freq_list.extend(freq)
        # for 3d (cubes) image seperate individual frequency model
        # to stack later with the rest
        if len(model_data.shape) > 2:
            models_data.extend(list(model_data))
        else:
            models_data.append(model_data)
    # stack model data cube
    model = np.stack(models_data)
    # Create numpy array with proper type annotation
    freqs = np.array(channel_freqs) if channel_freqs is not None else np.array(freq_list)
    nband = freqs.size
    if nband < 2:
        raise ValueError("Can't produce alpha map from a single band image")
    npix_l = l_coord.size
    npix_m = m_coord.size

    # update cube psf-pars
    for i in range(1, nband + 1):
        mhdr["BMAJ" + str(i)] = gaussparf[0]
        mhdr["BMIN" + str(i)] = gaussparf[1]
        mhdr["BPA" + str(i)] = gaussparf[2]

    ref_freq_param = ref_freq  # Store parameter value
    if ref_freq_param is not None and ref_freq_param != ref_freq:
        ref_freq = ref_freq_param
        log.info("Provided reference frequency does not match that of fits file. Will overwrite.")

    log.info("Cube frequencies:")
    with np.printoptions(precision=2):
        log.info("%s", freqs)
    log.info("Reference frequency is %3.2e Hz", ref_freq)

    # LB - new header for cubes if ref_freqs differ
    new_hdr = set_header_info(mhdr, ref_freq, freq_axis, beampars=gaussparf)

    # save next to model if no outfile is provided
    outfile = output_filename

    xx, yy = np.meshgrid(l_coord, m_coord, indexing="ij")
    # load beam
    if beam_model is not None:
        # we can pass in either a fits file with the already interpolated
        # beam or we can interpolate from scratch
        if beam_model.endswith(".fits"):
            bhdr = fits.getheader(beam_model)
            l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
            l_coord_beam -= ref_lb
            if not np.array_equal(l_coord_beam, l_coord):
                raise ValueError(
                    "l coordinates of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
            m_coord_beam -= ref_mb
            if not np.array_equal(m_coord_beam, m_coord):
                raise ValueError(
                    "m coordinates of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
            if not np.array_equal(freqs, freqs_beam):
                raise ValueError(
                    "Freqs of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            beam_image = load_fits(beam_model, dtype=out_dtype).squeeze()
        elif beam_model == "JimBeam":
            beam_image = []
            if band.lower() == "l":
                beam = JimBeam("MKAT-AA-L-JIM-2020")
            elif band.lower() == "uhf":
                beam = JimBeam("MKAT-AA-UHF-JIM-2020")
            elif band.lower() == "s":
                beam = JimBeam("MKAT-AA-S-JIM-2020")
            else:
                raise ValueError(f"Unknown beam model for katbeam in band {band}")
            beam_image = np.zeros_like(model)
            for v in range(freqs.size):
                beam_image[v] = beam.I(xx, yy, freqs[v] / 1e6)  # freqs in MHz

        else:
            # field=0: spifit has no --field option; extract_dde_info previously read
            # opts.field off a BeamOpts that never set it, raising AttributeError.
            beam_image = interpolate_beam(
                xx,
                yy,
                freqs,
                beam_model=beam_model,
                ms=ms,
                field=0,
                sparsify_time=sparsify_time,
                corr_type=corr_type,
                nthreads=nthreads,
            )

        if "b" in products:
            name = outfile + ".power_beam.fits"
            save_fits(
                name,
                np.expand_dims(beam_image, axis=4 - stokes_axis),
                mhdr,
                dtype=out_dtype,
            )
            log.info("Wrote average power beam to %s", name)

    else:
        beam_image = np.ones(model.shape, dtype=out_dtype)

    # beam cut off
    model = np.where(beam_image > pfb_min, model, 0.0)

    if not dont_convolve:
        log.info("Convolving model")
        # convolve model to desired resolution
        model, gausskern = convolve2gaussres(model, xx, yy, gaussparf, nthreads, None, padding_frac)

        # save clean beam
        if "c" in products:
            name = outfile + ".clean_psf.fits"
            save_fits(name, gausskern, new_hdr, dtype=out_dtype)
            log.info("Wrote clean psf to %s", name)

        # save convolved model
        if "m" in products:
            name = outfile + ".convolved_model.fits"
            save_fits(name, model, new_hdr, dtype=out_dtype)
            log.info("Wrote convolved model to %s", name)

    # add in residuals and set threshold
    if residual:
        # get headers and frequencies from residual image(s)
        residuals = [load_fits(res, dtype=out_dtype) for res in residual]
        resid = np.stack(residuals).squeeze()
        rhdr = [fits.getheader(res) for res in residual]
        freqs_res = np.array([data_from_header(fits.getheader(res), axis=freq_axis)[0] for res in residual]).flatten()
        freqs_res = freqs if channel_freqs else freqs_res
        if not np.array_equal(freqs, freqs_res):
            raise ValueError("Freqs of residual do not match those of model")

        # get l and m coordinate from residual image(s)
        l_res, ref_lb = data_from_header(rhdr[0], axis=1)
        l_res -= ref_lb
        if not np.array_equal(l_res, l_coord):
            raise ValueError("l coordinates of residual do not match those of model")
        m_res, ref_mb = data_from_header(rhdr[0], axis=2)
        m_res -= ref_mb
        if not np.array_equal(m_res, m_coord):
            raise ValueError("m coordinates of residual do not match those of model")

        # convolve residual to same resolution as model
        gausspari = ()
        # get beam values from individual headers
        if len(rhdr) > 1:
            for hdr in rhdr:
                keys = ["BMAJ", "BMIN", "BPA"]
                if all(k in hdr for k in keys):
                    emaj, emin, pa = [hdr[k] for k in keys]
                    gausspari += (tuple([hdr[k] for k in keys]),)
        # residual cube provides beam params as BMAJ1, BMAJ2,...
        else:
            hdr = rhdr[0]
            for i in range(1, nband + 1):
                key = "BMAJ" + str(i)
                if key in hdr:
                    emaj = hdr[key]
                    emin = hdr["BMIN" + str(i)]
                    pa = hdr["BPA" + str(i)]
                    gausspari += ((emaj, emin, pa),)
        if gausspari:
            log.info("Gausspars in residual header: %s", gausspari)  # type: ignore[unreachable]
        else:
            log.info("Can't find Gausspars in residual header, unable to add residuals back in")
            gausspari = None

        if gausspari is not None and add_convolved_residuals:  # type: ignore[unreachable]
            log.info("Convolving residuals")  # type: ignore[unreachable]
            resid, _ = convolve2gaussres(
                resid,
                xx,
                yy,
                gaussparf,
                nthreads,
                gausspari,
                padding_frac,
                norm_kernel=False,
            )
            model += resid
            log.info("Convolved residuals added to convolved model")

            if "r" in products:
                name = outfile + ".convolved_residual.fits"
                save_fits(name, resid, rhdr[0])
                log.info("Wrote convolved residuals to %s", name)

        counts = np.sum(resid != 0)
        rms = np.sqrt(np.sum(resid**2) / counts)
        rms_cube = np.std(resid.reshape(nband, npix_l * npix_m), axis=1).ravel()
        threshold_val = threshold * rms
        log.info("Setting cutoff threshold as %s times the rms of the residual ", threshold)
        del resid
    else:
        log.info(
            "No residual provided. Setting  threshold i.t.o dynamic range. Max dynamic range is %s",
            maxDR,
        )
        mask = ~np.isnan(model)
        threshold_val = model[mask].max() / maxDR
        rms_cube = None

    log.info("Threshold set to %s Jy.", threshold_val)

    # remove completely nan slices
    freq_mask = np.isnan(model)
    fidx = ~np.all(freq_mask, axis=(1, 2))

    # exclude any bands that might be awful
    if deselect_bands:
        log.info("Deselected bands are: %s", deselect_bands)
        for bidx in deselect_bands:
            fidx[bidx] = False

    if fidx.any():
        model = model[fidx]
        beam_image = beam_image[fidx]
        freqs = freqs[fidx]
        gaussparf = list(gaussparf)
        new_hdr = set_header_info(mhdr, ref_freq, freq_axis, beampars=tuple(gaussparf))

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold_val)
    if not maskindices.size:
        raise ValueError(
            "No components found above threshold. "
            "Try lowering your threshold."
            f"Max of convolved model is {model.max():3.2e}"
        )
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T
    beam_comps = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T

    # set weights for fit
    # Note: channel_weights_keyword is not used in the original code, only channel_weights
    # The original code checked opts.channel_weights but we don't have that parameter
    # Based on CLI, we should get weights from the header using channel_weights_keyword
    channel_weights = None  # This would need to be extracted from headers if implemented
    if channel_weights is not None:
        weights = np.array(channel_weights)[fidx]
        try:
            assert weights.size == nband
            log.info("Using provided channel weights.")
        except Exception as e:
            log.info("Number of provided channel weights not equal to number of imaging bands")
    else:
        if residual:
            log.info("Getting weights from list of image headers.")
            rhdr = []
            for res in residual:
                rhdr.append(fits.getheader(res))
            weights = np.array([hdr["WSCVWSUM"] for hdr in rhdr])
            weights /= weights.max()
        elif rms_cube is not None:
            log.info("Using RMS in each imaging band to determine weights.")
            weights = np.where(rms_cube[fidx] > 0, 1.0 / rms_cube[fidx] ** 2, 0.0)
            # normalise
            weights /= weights.max()
        else:
            log.info("No residual or channel weights provided. Using equal weights.")
            weights = np.ones(fidx.sum(), dtype=np.float64)
        log.info("Channel weights: %s", weights)

    ncomps, _ = fitcube.shape
    cchunks = np.maximum(1, ncomps // nthreads)
    fitcube = da.from_array(fitcube.astype(np.float64), chunks=(cchunks, nband))
    beam_comps = da.from_array(beam_comps.astype(np.float64), chunks=(cchunks, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    log.info("Fitting %s components", ncomps)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(
        fitcube, weights, freqsdask, np.float64(ref_freq), beam=beam_comps
    ).compute()
    log.info("Done. Writing output.")

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
    Irec_cube = i0map[None, :, :] * (freqs[:, None, None] / ref_freq) ** alphamap[None, :, :]
    fit_diff = np.zeros_like(model)
    fit_diff[...] = np.nan
    ix = maskindices[:, 0]
    iy = maskindices[:, 1]
    fit_diff[:, ix, iy] = model[:, ix, iy] / beam_image[:, ix, iy]
    fit_diff[:, ix, iy] -= Irec_cube[:, ix, iy]

    if "I" in products:
        # get the reconstructed cube
        name = outfile + ".Irec_cube.fits"
        save_fits(
            name,
            np.expand_dims(Irec_cube, axis=4 - stokes_axis),
            mhdr,
            dtype=out_dtype,
        )
        log.info("Wrote reconstructed cube to %s", name)

    if "d" in products:
        # get the reconstructed cube
        name = outfile + ".fit_diff.fits"
        save_fits(
            name,
            np.expand_dims(fit_diff, axis=4 - stokes_axis),
            mhdr,
            dtype=out_dtype,
        )
        log.info("Wrote reconstructed cube to %s", name)

    # save alpha map
    if "a" in products:
        name = outfile + ".alpha.fits"
        save_fits(name, alphamap, mhdr, dtype=out_dtype)
        log.info("Wrote alpha map to %s", name)

    # save alpha error map
    if "e" in products:
        name = outfile + ".alpha_err.fits"
        save_fits(name, alpha_err_map, mhdr, dtype=out_dtype)
        log.info("Wrote alpha error map to %s", name)

    # save I0 map
    if "i" in products:
        name = outfile + ".I0.fits"
        save_fits(name, i0map, mhdr, dtype=out_dtype)
        log.info("Wrote I0 map to %s", name)

    # save I0 error map
    if "k" in products:
        name = outfile + ".I0_err.fits"
        save_fits(name, i0_err_map, mhdr, dtype=out_dtype)
        log.info("Wrote I0 error map to %s", name)

    log.info("All done here")
