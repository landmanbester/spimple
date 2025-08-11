import numpy as np
from ducc0.fft import good_size, r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
from africanus.rime import parallactic_angles
from africanus.util.numba import jit
from daskms import xds_from_ms, xds_from_table
import dask.array as da
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import ray
from time import time
from spimple.fits import save_fits, load_fits, set_wcs, data_from_header

def str2bool(v):
    """
    Converts a string or boolean input to a boolean value.
    
    Accepts common string representations of true and false. Raises an
    ArgumentTypeError if the input cannot be interpreted as a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True, nsigma=5):
    ''' 
    xin         - grid of x coordinates
    yin         - grid of y coordinates
    GaussPar    - (emaj, emin, pa) with emaj/emin in units x and pa in radians.
    normalise   - normalise kernel to have volume 1
    nsigma      - compute kernel out to this many sigmas
    '''
    Smaj, Smin, PA = GaussPar
    A = np.array([[1. / Smaj ** 2, 0],
                  [0, 1. / Smin ** 2]])
    # R = np.array([[np.cos(PA), -np.sin(PA)],
    #               [np.sin(PA), np.cos(PA)]])
    # this parametrisation is equivalent to the above with
    # t = np.pi/2 - pa
    # use this for compatibility with fits 
    R = np.array([[np.sin(PA), -np.cos(PA)],
                  [np.cos(PA), np.sin(PA)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (nsigma * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-fwhm_conv * R)
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

    image = np.pad(image, padding, mode='constant').astype(np.float64)
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    # convolve to desired resolution
    if gausspari in [None, ()]:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel).astype(np.float64)
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

            if not np.all(np.isnan(thiskernhat)):
                convkernhat = np.where(np.abs(thiskernhat)>1e-10, gausskernhat/thiskernhat, 0.0)
            else:
                print("Nan values have been encountered. Subverting RuntimeWarning")
                convkernhat = np.zeros_like(thiskernhat).astype("complex")

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
    Extracts parallactic angles, antenna scaling, pointing errors, and unflagged data counts for beam interpolation.
    
    If measurement set files are provided in `opts.ms`, computes these quantities from the data, ensuring consistency of antenna positions and phase centers across sets. Otherwise, returns default arrays suitable for beam interpolation.
    
    Returns:
        A tuple containing:
            - parangles: Array of parallactic angles averaged over antennas.
            - ant_scale: Array of antenna scaling factors.
            - point_errs: Array of antenna pointing errors.
            - unflag_counts: Array of unflagged data counts per time.
            - A boolean flag (always False).
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

        # mean over antenna nant -> 1
        parangles = np.mean(parangles, axis=1, keepdims=True)
        nant = 1

        # beam_cube_dde requirements
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
        return (parangles,
                # da.from_array(ant_scale, chunks=ant_scale.shape),
                ant_scale,
                point_errs,
                unflag_counts,
                False)
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
    """
    Loads and constructs a power beam cube from FITS beam model files for interpolation.
    
    Searches for FITS files matching the specified beam model pattern and loads the real and imaginary components for two correlations (linear or circular). Computes the power beam as the average squared magnitude of both correlations, verifies spatial and frequency coverage, and extracts spatial extents and frequency axis information. Returns the beam amplitude cube, spatial extents, and beam frequencies as either Dask arrays or NumPy arrays depending on the `use_dask` flag.
    
    Args:
        opts: Options object containing beam model pattern and correlation type.
        lm_source: Array of source direction cosines for spatial coverage validation.
        freqs: Array of frequencies to check against beam model coverage.
        use_dask: If True, returns Dask arrays; otherwise, returns NumPy arrays.
    
    Returns:
        Tuple containing the beam amplitude cube, spatial extents, and beam frequencies.
    """
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
            print(f'Loading beam from {path}')
            if 're' in path[-7::]:
                corr1_re = load_fits(path)
                if beam_hdr is None:
                    beam_hdr = fits.getheader(path)
            elif 'im' in path[-7::]:
                corr1_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
        elif corr2.lower() in path[-10::]:
            print(f'Loading beam from {path}')
            if 're' in path[-7::]:
                corr2_re = load_fits(path)
            elif 'im' in path[-7::]:
                corr2_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")

    # get power beam
    beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2)/2.0

    # get cube in correct shape for interpolation code
    beam_amp = beam_amp[0]  # drop corr axis
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
        raise ValueError("The supplied beam does not have sufficient "
                         f"bandwidth. min={bfreqs.min()}, max={bfreqs.max()}")
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
    Interpolates the beam model to specified image coordinates and frequencies.
    
    If measurement set (MS) data is provided in the options, computes a time-averaged beam using direction-dependent effects (DDE) such as parallactic angle, antenna scaling, and pointing errors. Supports both Dask-based and NumPy-based interpolation depending on the workflow. Returns the interpolated beam cube reshaped to match the frequency and image coordinate dimensions.
    
    Args:
        ll: 2D array of l (direction cosine) coordinates for the image grid.
        mm: 2D array of m (direction cosine) coordinates for the image grid.
        freqs: 1D array of frequencies at which to interpolate the beam.
        opts: Options object containing beam model paths, MS information, and processing parameters.
    
    Returns:
        A NumPy array of the interpolated beam, with shape (nfreq, *ll.shape).
    """
    nband = freqs.size
    # print("Interpolating beam", file=log)
    parangles, ant_scale, point_errs, unflag_counts, use_dask = extract_dde_info(opts, freqs)

    lm_source = np.vstack((ll.ravel(), mm.ravel())).T
    beam_amp, beam_extents, bfreqs = make_power_beam(opts, lm_source, freqs, use_dask)

    # interpolate beam
    if use_dask:
        # chunking is over time and antenna
        from africanus.rime.dask import beam_cube_dde
        lm_source = da.from_array(lm_source, chunks=lm_source.shape)
        freqs = da.from_array(freqs, chunks=freqs.shape)
        # compute nthreads images at a time to avoid memory errors
        ntimes = parangles.shape[0]
        I = np.arange(0, ntimes, opts.nthreads)
        nchunks = I.size
        I = np.append(I, ntimes)
        beam_image = np.zeros((ll.size, 1, nband), dtype=beam_amp.dtype)
        ant_scale = da.from_array(ant_scale, chunks=(1, freqs.size, 2))
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
        from africanus.rime.fast_beam_cubes import beam_cube_dde
        beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs) #.squeeze()
        beam_image = beam_image[:, :, 0, :, 0, 0]
        beam_image = np.mean(beam_image, axis=1)



    # swap source and freq axes and reshape to image shape
    beam_source = np.transpose(beam_image, axes=(1, 0))
    return beam_source.squeeze().reshape((freqs.size, *ll.shape))


def mosaic_info(im_list, oname, ref_image=None):
    if ref_image is not None:
        raise NotImplementedError("Reference image not supported yet")
    wcss = []
    out_names = []
    freqs = []
    flatfreqs = []
    shapes = []
    basename = oname.removesuffix('.fits')
    for imnum, im in enumerate(im_list):
        hdr = fits.getheader(im)
        nu = data_from_header(hdr, axis=3)[0]
        freqs.append(nu)
        flatfreqs.extend(nu)
        wcsi = WCS(hdr).dropaxis(-1).dropaxis(-1)
        wcss.append(wcsi)
        nchan = hdr['NAXIS3']
        ncorr = hdr['NAXIS4']
        shapes.append((hdr['NAXIS1'],hdr['NAXIS2']))
        for c in range(ncorr):
            for f in range(nchan):
                out_names.append(f'{basename}_im{imnum}_pol{c}_ch{f}.zarr')
    nu = np.array(flatfreqs)
    ufreqs = np.unique(nu)
    # get domain of intrinsic image
    ref_wcs, shape_out = find_optimal_celestial_wcs(
                                    list(zip(shapes, wcss)),
                                    projection='SIN')
    ref_wcs.array_shape = (shape_out[0], shape_out[1])

    return ref_wcs, ufreqs, out_names


@ray.remote
def project(im, imnum, ref_wcs, beam, oname, band='L', method='interp'):
    if method != 'interp':
        raise NotImplementedError("Only 'interp' method supported for now")
    
    # output shape
    nxo, nyo = ref_wcs.array_shape
    
    # interpolate beam
    bds = xr.open_zarr(beam, chunks=None)
    beam = bds.BEAM.values
    l_beam = bds.l_beam.values
    m_beam = bds.m_beam.values
    bfreq = bds.chan.values

    # make the power beam
    pbeam = ((beam[0]*beam[0].conj()).real + (beam[-1]*beam[-1].conj()).real)/2.0
    
    # this is cheap, evaluation is more expensive
    beamo = RegularGridInterpolator(
                            (bfreq, l_beam, m_beam),
                            pbeam,
                            bounds_error=False,
                            fill_value=None,
                            method='linear')
    
    cell_x, cell_y = ref_wcs.wcs.cdelt
    l_im = (-(nxo)//2 + np.arange(nxo)) * cell_x
    m_im = (-(nyo)//2 + np.arange(nyo)) * cell_y
    im_coords = {
        'l': ('l', l_im),
        'm': ('m', m_im),
    }

    # image = load_fits(im)
    image = fits.getdata(im)
    ncorr, nchan, nx, ny = image.shape
    hdr = fits.getheader(im)
    freq, _ = data_from_header(hdr, axis=3)
    wcs = WCS(hdr).dropaxis(-1).dropaxis(-1)
    cxi, cyi = wcs.wcs.cdelt
    nx, ny = wcs.array_shape
    l = (-(nx)//2 + np.arange(nx)) * cxi
    m = (-(ny)//2 + np.arange(ny)) * cyi
    ll, mm = np.meshgrid(l, m, indexing='ij')
    
    basename = oname.removesuffix('.fits')
    # TODO - make these in parallel
    for c in range(ncorr):
        for f in range(nchan):
            bdata = np.zeros((nx, ny), dtype=np.float64)
            # ti = time()
            beami = beamo((freq[c], ll, mm))
            # print('Interp ', time() - ti)
            # ti = time()
            step = 25
            angles = np.linspace(0, 359, step)
            for angle in angles:
                bdata += ndimage.rotate(beami, angle,
                                        reshape=False,
                                        order=1,
                                        mode='nearest')
            bdata /= angles.size
            # print('azimuthal average ', time() - ti)
            # project beam
            # ti = time()
            pbeam, footprint = reproject_interp((bdata, wcs),
                                                ref_wcs,
                                                shape_out=(nxo, nyo),
                                                block_size='auto',
                                                parallel=4)
            mask = footprint > 0
            pbeam[~mask] = 0
            # print('beam project ', time() - ti)
            
            # project image
            # ti = time()
            pdata, _ = reproject_interp((image[c, f], wcs),
                                        ref_wcs,
                                        shape_out=(nxo, nyo),
                                        block_size='auto',
                                        parallel=4)
            # print('Image project ', time() - ti)

            im_attrs = {
            'freq': freq[c],
            }

            data_vars = {
                'IMAGE': (('l','m'), pdata),
                'BEAM': (('l', 'm'), pbeam),
                'MASK': (('l', 'm'), mask)
            }

            ds = xr.Dataset(data_vars,
                            coords=im_coords,
                            attrs=im_attrs).chunk({'l': 512, 'm': 512})
            
            ds_name = f'{basename}_im{imnum}_pol{c}_ch{f}.zarr'
            ds.to_zarr(ds_name,
                       compute=True, mode='w',
                       consolidated=False)
            
            
            # hdri = wcs.to_header()
            # hdu = fits.PrimaryHDU(header=hdri)
            # hdu.data = image[c, f]
            # fname = ds_name.removesuffix('.zarr') + '_image.fits'
            # hdu.writeto(fname, overwrite=True)
            # hdu.data = bdata
            # fname = ds_name.removesuffix('.zarr') + '_beam.fits'
            # hdu.writeto(fname, overwrite=True)

            # ref_hdr = ref_wcs.to_header()
            # hdu = fits.PrimaryHDU(header=ref_hdr)
            # hdu.data = pdata
            # fname = ds_name.removesuffix('.zarr') + '_pimage.fits'
            # hdu.writeto(fname, overwrite=True)
            # hdu.data = pbeam
            # fname = ds_name.removesuffix('.zarr') + '_pbeam.fits'
            # hdu.writeto(fname, overwrite=True)

    return imnum
            

@ray.remote
def stitch_images(freq, im_list, eta=1e-3):
    # get all datasets in current plane
    xds = []
    for im in im_list:
        ds = xr.open_zarr(im, chunks=None,
                          consolidated=False)
        if ds.freq == freq:
            xds.append(ds)
    
    # accumulate
    nx = xds[0].l.size
    ny = xds[0].m.size
    y = np.zeros((nx, ny))
    for ds in xds:
        mask = ds.MASK.values
        beam = ds.BEAM.values
        image = ds.IMAGE.values
        y[mask] += beam[mask]*image[mask]
    ds = ds.drop_vars(('IMAGE'))

    def hess(x):
        out = np.zeros((nx, ny))
        for ds in xds:
            mask = ds.MASK.values
            beam = ds.BEAM.values
            res = beam**2 * x
            out[mask] += res[mask]
        return out + eta * x
    
    image, info = conjugate_gradient(hess, y, max_iter=10)

    weight = np.zeros((nx, ny))
    for ds in xds:
        mask = ds.MASK.values
        beam = ds.BEAM.values
        weight[mask] += beam[mask]**2
    weight += eta
    return image, weight, info, freq
    

# from astropy.wcs import WCS
# from scipy.interpolate import RegularGridInterpolator
# from katpoint.projection import plane_to_sphere_sin
# def beam2sin(beam, lbeam, mbeam, hdr):
#     beamf = np.flipud(beam)
#     ll, mm = np.meshgrid(lbeam, mbeam, indexing='ij')
#     beomo = RegularGridInterpolator((lbeam, mbeam), beamf,
#                                     bounds_error=True, method='linear')

#     wcs = WCS(hdr)
#     # RA--SIN and DEC-SIN reference
#     ra0, dec0 = wcs.wcs.crval
#     nx, ny = wcs.array_shape
#     x_pix, y_pix = np.mgrid[0:nx, 0:ny]
#     ra, dec = wcs.pixel_to_world(x_pix, y_pix)




def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=100, report=20):
   n = b.shape
   if x0 is None:
       x = np.zeros(n)
   else:
       x = x0.copy()
   
   r = A(x) - b
   p = -r
   rsold = np.vdot(r, r)
   rs0 = rsold
   if rs0 < tol:
       import ipdb; ipdb.set_trace()
       return x, 0  # already at minimum
   for i in range(max_iter):
       Ap = A(p)
       alpha = rsold / np.vdot(p, Ap)
       x = x + alpha * p
       r = r + alpha * Ap
       rsnew = np.vdot(r, r)
       
       if np.sqrt(rsnew) < tol:
           break
           
       beta = rsnew / rsold
       p = beta * p - r
       rsold = rsnew

       if i%report:
           print(f"At {i} norm frac = {rsnew/rs0}")
   
   return x, i