import numpy as np
from ducc0.fft import good_size, r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from astropy.io import fits
from glob import glob
from africanus.rime.dask import beam_cube_dde
from africanus.rime.fast_beam_cubes import beam_cube_dde
from africanus.rime import parallactic_angles
from africanus.util.numba import jit

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')

def to4d(data):
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def data_from_header(hdr, axis=3, zero_ref=False):
    npix = hdr['NAXIS' + str(axis)]
    refpix = hdr['CRPIX' + str(axis)]
    delta = hdr['CDELT' + str(axis)]
    ref_val = hdr['CRVAL' + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta, ref_val


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data)[:, :, ::-1], axes=(0, 1, 3, 2))
    return np.require(data, dtype=dtype, requirements='C')


def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))[:, :, ::-1]
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)


def set_header_info(mhdr, ref_freq, freq_axis, args, beampars):
    hdr_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'NAXIS4', 'CTYPE1', 'CTYPE2', 'CTYPE3', 'CTYPE4', 'CRPIX1',
                'CRPIX2', 'CRPIX3', 'CRPIX4', 'CRVAL1', 'CRVAL2', 'CRVAL3',
                'CRVAL4', 'CDELT1', 'CDELT2', 'CDELT3', 'CDELT4']

    new_hdr = {}
    for key in hdr_keys:
        new_hdr[key] = mhdr[key]

    if freq_axis == 3:
        new_hdr["NAXIS3"] = 1
        new_hdr["CRVAL3"] = ref_freq
    elif freq_axis == 4:
        new_hdr["NAXIS4"] = 1
        new_hdr["CRVAL4"] = ref_freq

    new_hdr['BMAJ'] = beampars[0]
    new_hdr['BMIN'] = beampars[1]
    new_hdr['BPA'] = beampars[2]

    new_hdr = fits.Header(new_hdr)

    return new_hdr


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True):
    S0, S1, PA = GaussPar
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (5 * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    tmp = np.exp(-fwhm_conv*R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


def get_padding_info(nx, ny, pfrac):
    npad_x = int(pfrac * nx)
    nfft = good_size(nx + npad_x, True)
    npad_xl = (nfft - nx)//2
    npad_xr = nfft - nx - npad_xl

    npad_y = int(pfrac * ny)
    nfft = good_size(ny + npad_y, True)
    npad_yl = (nfft - ny)//2
    npad_yr = nfft - ny - npad_yl
    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y


def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None, pfrac=0.5, norm_kernel=False):
    """
    Convolves the image to a specified resolution.

    Parameters
    ----------
    Image - (nband, nx, ny) array to convolve
    xx/yy - coordinates on the grid in the same units as gaussparf.
    gaussparf - tuple containing Gaussian parameters of desired resolution (emaj, emin, pa).
    gausspari - initial resolution . By default it is assumed that the image is a clean component image with no associated resolution.
                If beampari is specified, it must be a tuple containing gausspars for each imaging band in the same format.
    nthreads - number of threads to use for the FFT's.
    pfrac - padding used for the FFT based convolution. Will pad by pfrac/2 on both sides of image
    """
    nband, nx, ny = image.shape
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
    gausskern = np.pad(gausskern[None], padding, mode='constant')
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    # convolve to desired resolution
    if gausspari is None:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel)
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

            convkernhat = np.where(np.abs(thiskernhat)>0.0, gausskernhat/thiskernhat, 0.0)

            imhat[i] *= convkernhat[0]

    image = Fs(c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=nthreads), axes=ax)[:, unpad_x, unpad_y]

    return image, gausskern[:, unpad_x, unpad_y]


@jit(nopython=True, nogil=True, cache=True)
def _unflagged_counts(flags, time_idx, out):
    for i in range(time_idx.size):
            ilow = time_idx[i]
            ihigh = time_idx[i+1]
            out[i] = np.sum(~flags[ilow:ihigh])
    return out

def extract_dde_info(opts, freqs):
    """
    Computes paralactic angles, antenna scaling and pointing information
    required for beam interpolation.
    """
    # get ms info required to compute paralactic angles and weighted sum
    nband = freqs.size
    if opts.ms is not None:
        utimes = []
        unflag_counts = []
        ant_pos = None
        phase_dir = None
        for ms_name in opts.ms:
            # get antenna positions
            ant = xds_from_table(ms_name + '::ANTENNA')[0].compute()
            if ant_pos is None:
                ant_pos = ant['POSITION'].data
            else: # check all are the same
                tmp = ant['POSITION']
                if not np.array_equal(ant_pos, tmp):
                    raise ValueError("Antenna positions not the same across measurement sets")

            # get phase center for field
            field = xds_from_table(ms_name + '::FIELD')[0].compute()
            if phase_dir is None:
                phase_dir = field['PHASE_DIR'][opts.field].data.squeeze()
            else:
                tmp = field['PHASE_DIR'][opts.field].data.squeeze()
                if not np.array_equal(phase_dir, tmp):
                    raise ValueError('Phase direction not the same across measurement sets')

            # get unique times and count flags
            xds = xds_from_ms(ms_name, columns=["TIME", "FLAG_ROW"], group_cols=["FIELD_ID"])[opts.field]
            utime, time_idx = np.unique(xds.TIME.data.compute(), return_index=True)
            ntime = utime.size
            # extract subset of times
            if opts.sparsify_time > 1:
                I = np.arange(0, ntime, opts.sparsify_time)
                utime = utime[I]
                time_idx = time_idx[I]
                ntime = utime.size

            utimes.append(utime)

            flags = xds.FLAG_ROW.data.compute()
            unflag_count = _unflagged_counts(flags.astype(np.int32), time_idx, np.zeros(ntime, dtype=np.int32))
            unflag_counts.append(unflag_count)

        utimes = np.concatenate(utimes)
        unflag_counts = np.concatenate(unflag_counts)
        ntimes = utimes.size

        # compute paralactic angles
        parangles = parallactic_angles(utimes, ant_pos, phase_dir)

        # mean over antanna nant -> 1
        parangles = np.mean(parangles, axis=1, keepdims=True)
        nant = 1

        # beam_cube_dde requirements
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)

        return (parangles,
                da.from_array(ant_scale, chunks=ant_scale.shape),
                point_errs,
                unflag_counts,
                True)
    else:
        ntimes = 1
        nant = 1
        parangles = np.zeros((ntimes, nant,), dtype=np.float64)
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
        unflag_counts = np.array([1])

        return (parangles, ant_scale, point_errs, unflag_counts, False)


def make_power_beam(opts, lm_source, freqs, use_dask):
    # print(f"Loading fits beam patterns from {opts.beam_model}", file=log)
    paths = glob(opts.beam_model + '**_**.fits')
    beam_hdr = None
    if opts.corr_type == 'linear':
        corr1 = 'XX'
        corr2 = 'YY'
    elif opts.corr_type == 'circular':
        corr1 = 'LL'
        corr2 = 'RR'
    else:
        raise KeyError("Unknown corr_type supplied. Only 'linear' or 'circular' supported")

    for path in paths:
        if corr1.lower() in path[-10::]:
            if 're' in path[-7::]:
                corr1_re = load_fits(path)
                if beam_hdr is None:
                    beam_hdr = fits.getheader(path)
            elif 'im' in path[-7::]:
                corr1_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
        elif corr2.lower() in path[-10::]:
            if 're' in path[-7::]:
                corr2_re = load_fits(path)
            elif 'im' in path[-7::]:
                corr2_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")

    # get power beam
    beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2)/2.0

    # get cube in correct shape for interpolation code
    beam_amp = np.ascontiguousarray(np.transpose(beam_amp, (1, 2, 0))
                                    [:, :, :, None, None])
    # get cube info
    if beam_hdr['CUNIT1'].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_l = beam_hdr['NAXIS1']
    refpix_l = beam_hdr['CRPIX1']
    delta_l = beam_hdr['CDELT1']
    l_min = (1 - refpix_l)*delta_l
    l_max = (1 + npix_l - refpix_l)*delta_l

    if beam_hdr['CUNIT2'].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_m = beam_hdr['NAXIS2']
    refpix_m = beam_hdr['CRPIX2']
    delta_m = beam_hdr['CDELT2']
    m_min = (1 - refpix_m)*delta_m
    m_max = (1 + npix_m - refpix_m)*delta_m

    if (l_min > lm_source[:, 0].min() or m_min > lm_source[:, 1].min() or
            l_max < lm_source[:, 0].max() or m_max < lm_source[:, 1].max()):
        raise ValueError("The supplied beam is not large enough")

    beam_extents = np.array([[l_min, l_max], [m_min, m_max]])

    # get frequencies
    if beam_hdr["CTYPE3"].lower() != 'freq':
        raise ValueError(
            "Cubes are assumed to be in format [nchan, nx, ny]")
    nchan = beam_hdr['NAXIS3']
    refpix = beam_hdr['CRPIX3']
    delta = beam_hdr['CDELT3']  # assumes units are Hz
    freq0 = beam_hdr['CRVAL3']
    bfreqs = freq0 + np.arange(1 - refpix, 1 + nchan - refpix) * delta
    if bfreqs[0] > freqs[0] or bfreqs[-1] < freqs[-1]:
        warnings.warn("The supplied beam does not have sufficient "
                       "bandwidth. Beam frequencies:", file=log)
        # with np.printoptions(precision=2):
        #     print(bfreqs, file=log)

    if use_dask:
        return (da.from_array(beam_amp, chunks=beam_amp.shape),
                da.from_array(beam_extents, chunks=beam_extents.shape),
                da.from_array(bfreqs, bfreqs.shape))
    else:
        return beam_amp, beam_extents, bfreqs

def interpolate_beam(ll, mm, freqs, opts):
    """
    Interpolate beam to image coordinates and optionally compute average
    over time if MS is provoded
    """
    nband = freqs.size
    # print("Interpolating beam", file=log)
    parangles, ant_scale, point_errs, unflag_counts, use_dask = extract_dde_info(opts, freqs)

    lm_source = np.vstack((ll.ravel(), mm.ravel())).T
    beam_amp, beam_extents, bfreqs = make_power_beam(opts, lm_source, freqs, use_dask)

    # interpolate beam
    if use_dask:
        lm_source = da.from_array(lm_source, chunks=lm_source.shape)
        freqs = da.from_array(freqs, chunks=freqs.shape)
        # compute nthreads images at a time to avoid memory errors
        ntimes = parangles.shape[0]
        I = np.arange(0, ntimes, opts.nthreads)
        nchunks = I.size
        I = np.append(I, ntimes)
        beam_image = np.zeros((ll.size, 1, nband), dtype=beam_amp.dtype)
        for i in range(nchunks):
            ilow = I[i]
            ihigh = I[i+1]
            part_parangles = da.from_array(parangles[ilow:ihigh], chunks=(1, 1))
            part_point_errs = da.from_array(point_errs[ilow:ihigh], chunks=(1, 1, freqs.size, 2))
            # interpolate and remove redundant axes
            part_beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                        lm_source, part_parangles, part_point_errs,
                                        ant_scale, freqs).compute()[:, :, 0, :, 0 , 0]
            # weighted sum over time
            beam_image += np.sum(part_beam_image * unflag_counts[None, ilow:ihigh, None], axis=1, keepdims=True)
        # normalise by sum of weights
        beam_image /= np.sum(unflag_counts)
        # remove time axis
        beam_image = beam_image[:, 0, :]
    else:
        beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).squeeze()



    # swap source and freq axes and reshape to image shape
    beam_source = np.transpose(beam_image, axes=(1, 0))
    return beam_source.squeeze().reshape((freqs.size, *ll.shape))
