from pathlib import Path

from africanus.rime import parallactic_angles
from africanus.rime.dask import beam_cube_dde as beam_cube_dde_dask
from africanus.rime.fast_beam_cubes import beam_cube_dde
from africanus.util.numba import jit
from astropy.io import fits
import dask.array as da
from daskms import xds_from_ms, xds_from_table
import numpy as np

from spimple.utils.fits import load_fits


@jit(nopython=True, nogil=True, cache=True)
def _unflagged_counts(flags, time_idx, out):
    for i in range(time_idx.size):
        ilow = time_idx[i]
        ihigh = time_idx[i + 1]
        out[i] = np.sum(~flags[ilow:ihigh])
    return out


def extract_dde_info(opts, freqs):
    """
    Extracts parallactic angles, antenna scaling, pointing errors,
    and unflagged data counts for beam interpolation.

    If measurement set files are provided in `opts.ms`, computes these
    quantities from the data, ensuring consistency of antenna positions
    and phase centers across sets. Otherwise, returns default arrays
    suitable for beam interpolation.

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
        # Fixed: Eliminate None initialization pattern to help mypy type inference
        utimes_list = []
        unflag_counts_list = []
        ms_list = list(opts.ms)

        # Get reference values from first MS
        first_ants = xds_from_table(ms_list[0] + "::ANTENNA").compute()
        ant_pos = first_ants[0]["POSITION"].data

        first_field = xds_from_table(ms_list[0] + "::FIELD")[0].compute()
        phase_dir = first_field["PHASE_DIR"][opts.field].data.squeeze()

        # Process all MS files (including the first one for data extraction)
        for ms_name in ms_list:
            # get antenna positions and check consistency (skip check for first MS)
            if ms_name != ms_list[0]:
                ants = xds_from_table(ms_name + "::ANTENNA").compute()
                ant = ants[0]
                tmp = ant["POSITION"].data
                if not np.array_equal(ant_pos, tmp):
                    msg = "Antenna positions not the same across measurement sets"
                    raise ValueError(msg)

                # get phase center for field and check consistency
                field = xds_from_table(ms_name + "::FIELD")[0].compute()
                tmp = field["PHASE_DIR"][opts.field].data.squeeze()
                if not np.array_equal(phase_dir, tmp):
                    raise ValueError("Phase direction not the same across measurement sets")

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

            utimes_list.append(utime)

            flags = xds.FLAG_ROW.data.compute()
            unflag_count = _unflagged_counts(flags.astype(np.int32), time_idx, np.zeros(ntime, dtype=np.int32))
            unflag_counts_list.append(unflag_count)

        # Convert lists to numpy arrays
        utimes = np.concatenate(utimes_list)
        unflag_counts = np.concatenate(unflag_counts_list)
        ntimes = utimes.size

        # compute paralactic angles
        parangles = parallactic_angles(utimes, ant_pos, phase_dir)

        # mean over antenna nant -> 1
        parangles = np.mean(parangles, axis=1, keepdims=True)
        nant = 1

        # beam_cube_dde requirements
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
        return (
            parangles,
            ant_scale,
            point_errs,
            unflag_counts,
            False,
        )
    ntimes = 1
    nant = 1
    parangles = np.zeros(
        (
            ntimes,
            nant,
        ),
        dtype=np.float64,
    )
    ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
    point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
    unflag_counts = np.array([1])

    return (parangles, ant_scale, point_errs, unflag_counts, False)


def make_power_beam(opts, lm_source, freqs, use_dask):
    """
    Loads and constructs a power beam cube from FITS beam model files for interpolation.

    Searches for FITS files matching the specified beam model pattern and loads the real
    and imaginary components for two correlations (linear or circular). Computes the
    power beam as the average squared magnitude of both correlations, verifies spatial
    and frequency coverage, and extracts spatial extents and frequency axis information.
    Returns the beam amplitude cube, spatial extents, and beam frequencies as either
    Dask arrays or NumPy arrays depending on the `use_dask` flag.

    Args:
        opts: Options object containing beam model pattern and correlation type.
        lm_source: Array of source direction cosines for spatial coverage validation.
        freqs: Array of frequencies to check against beam model coverage.
        use_dask: If True, returns Dask arrays; otherwise, returns NumPy arrays.

    Returns:
        Tuple containing the beam amplitude cube, spatial extents, and beam frequencies.
    """
    paths = list(Path(opts.beam_model).parent.glob(Path(opts.beam_model).name + "**_**.fits"))
    beam_hdr = None
    if opts.corr_type == "linear":
        corr1 = "XX"
        corr2 = "YY"
    elif opts.corr_type == "circular":
        corr1 = "LL"
        corr2 = "RR"
    else:
        raise KeyError("Unknown corr_type supplied. Only 'linear' or 'circular' supported")

    for path in paths:
        path_str = str(path)
        if corr1.lower() in path_str[-10::]:
            print(f"Loading beam from {path}")
            if "re" in path_str[-7::]:
                corr1_re = load_fits(path)
                if beam_hdr is None:
                    beam_hdr = fits.getheader(path)
            elif "im" in path_str[-7::]:
                corr1_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
        elif corr2.lower() in path_str[-10::]:
            print(f"Loading beam from {path}")
            if "re" in path_str[-7::]:
                corr2_re = load_fits(path)
            elif "im" in path_str[-7::]:
                corr2_im = load_fits(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")

    # get power beam
    beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2) / 2.0

    # get cube in correct shape for interpolation code
    beam_amp = beam_amp[0]  # drop corr axis
    beam_amp = np.ascontiguousarray(np.transpose(beam_amp, (1, 2, 0))[:, :, :, None, None])
    # get cube info
    if beam_hdr["CUNIT1"].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_l = beam_hdr["NAXIS1"]
    refpix_l = beam_hdr["CRPIX1"]
    delta_l = beam_hdr["CDELT1"]
    l_min = (1 - refpix_l) * delta_l
    l_max = (1 + npix_l - refpix_l) * delta_l

    if beam_hdr["CUNIT2"].lower() != "deg":
        raise ValueError("Beam image units must be in degrees")
    npix_m = beam_hdr["NAXIS2"]
    refpix_m = beam_hdr["CRPIX2"]
    delta_m = beam_hdr["CDELT2"]
    m_min = (1 - refpix_m) * delta_m
    m_max = (1 + npix_m - refpix_m) * delta_m

    if (
        l_min > lm_source[:, 0].min()
        or m_min > lm_source[:, 1].min()
        or l_max < lm_source[:, 0].max()
        or m_max < lm_source[:, 1].max()
    ):
        raise ValueError("The supplied beam is not large enough")

    beam_extents = np.array([[l_min, l_max], [m_min, m_max]])

    # get frequencies
    if beam_hdr["CTYPE3"].lower() != "freq":
        raise ValueError("Cubes are assumed to be in format [nchan, nx, ny]")
    nchan = beam_hdr["NAXIS3"]
    refpix = beam_hdr["CRPIX3"]
    delta = beam_hdr["CDELT3"]  # assumes units are Hz
    freq0 = beam_hdr["CRVAL3"]
    bfreqs = freq0 + np.arange(1 - refpix, 1 + nchan - refpix) * delta
    if bfreqs[0] > freqs[0] or bfreqs[-1] < freqs[-1]:
        raise ValueError(f"The supplied beam does not have sufficient bandwidth. min={bfreqs.min()},max={bfreqs.max()}")

    if use_dask:
        return (
            da.from_array(beam_amp, chunks=beam_amp.shape),
            da.from_array(beam_extents, chunks=beam_extents.shape),
            da.from_array(bfreqs, bfreqs.shape),
        )
    return beam_amp, beam_extents, bfreqs


def interpolate_beam(ll, mm, freqs, opts):
    """
    Interpolates the beam model to specified image coordinates and frequencies.

    If measurement set (MS) data is provided in the options, computes a time-averaged
    beam using direction-dependent effects (DDE) such as parallactic angle, antenna
    scaling, and pointing errors. Supports both Dask-based and NumPy-based
    interpolation depending on the workflow. Returns the interpolated beam cube
    reshaped to match the frequency and image coordinate dimensions.

    Args:
        ll: 2D array of l (direction cosine) coordinates for the image grid.
        mm: 2D array of m (direction cosine) coordinates for the image grid.
        freqs: 1D array of frequencies at which to interpolate the beam.
        opts: Options object containing beam model paths, MS information, and
        processing parameters.

    Returns:
        A NumPy array of the interpolated beam, with shape (nfreq, *ll.shape).
    """
    nband = freqs.size
    parangles, ant_scale, point_errs, unflag_counts, use_dask = extract_dde_info(opts, freqs)

    lm_source = np.vstack((ll.ravel(), mm.ravel())).T
    beam_amp, beam_extents, bfreqs = make_power_beam(opts, lm_source, freqs, use_dask)

    # interpolate beam
    if use_dask:
        # chunking is over time and antenna

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
            ihigh = I[i + 1]
            part_parangles = da.from_array(parangles[ilow:ihigh], chunks=(1, 1))
            part_point_errs = da.from_array(point_errs[ilow:ihigh], chunks=(1, 1, freqs.size, 2))
            # interpolate and remove redundant axes
            part_beam_image = beam_cube_dde_dask(
                beam_amp,
                beam_extents,
                bfreqs,
                lm_source,
                part_parangles,
                part_point_errs,
                ant_scale,
                freqs,
            ).compute()[:, :, 0, :, 0, 0]
            # weighted sum over time
            beam_image += np.sum(
                part_beam_image * unflag_counts[None, ilow:ihigh, None],
                axis=1,
                keepdims=True,
            )
        # normalise by sum of weights
        beam_image /= np.sum(unflag_counts)
        # remove time axis
        beam_image = beam_image[:, 0, :]
    else:
        beam_image = beam_cube_dde(
            beam_amp,
            beam_extents,
            bfreqs,
            lm_source,
            parangles,
            point_errs,
            ant_scale,
            freqs,
        )  # .squeeze()
        beam_image = beam_image[:, :, 0, :, 0, 0]
        beam_image = np.mean(beam_image, axis=1)

    # swap source and freq axes and reshape to image shape
    beam_source = np.transpose(beam_image, axes=(1, 0))
    return beam_source.squeeze().reshape((freqs.size, *ll.shape))
