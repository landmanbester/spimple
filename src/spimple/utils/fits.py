import glob

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


def to4d(data):
    if data.ndim == 4:
        return data
    if data.ndim == 2:
        return data[None, None]
    if data.ndim == 3:
        return data[None]
    if data.ndim == 1:
        return data[None, None, None]
    raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def data_from_header(hdr, axis=3):
    npix = hdr["NAXIS" + str(axis)]
    refpix = hdr["CRPIX" + str(axis)]
    delta = hdr["CDELT" + str(axis)]
    ref_val = hdr["CRVAL" + str(axis)]
    return ref_val + np.arange(1 - refpix, 1 + npix - refpix) * delta, ref_val


def freq_axis_of(hdr) -> int:
    """Return the FITS axis number carrying frequency, 3 or 4.

    Args:
        hdr: FITS header.

    Returns:
        3 or 4.

    Raises:
        ValueError: If neither CTYPE3 nor CTYPE4 is a frequency axis.
    """
    for axis in (4, 3):
        ctype = hdr.get(f"CTYPE{axis}", "")
        if str(ctype).lower() in ("freq", "speclnmf"):
            return axis
    raise ValueError("Freq axis must be 3rd or 4th")


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data), axes=(1, 0, 3, 2))  # fits and beams table
    return np.require(data, dtype=dtype, requirements="C")


def load_cube(name, dtype=np.float32):
    """Load a FITS cube as (nband, ncorr, ny, nx) with the raster preserved.

    astropy maps numpy axes to FITS axes in reverse, so the data arrives as
    (NAXIS4, NAXIS3, NAXIS2, NAXIS1). This detects which of NAXIS3/NAXIS4 is
    frequency and moves it to axis 0, leaving the (ny, nx) raster alone. Use
    this on the DataTree path; ``load_fits`` keeps its legacy (X, Y) transpose
    for the FITS-to-FITS commands.

    Args:
        name: Path to the FITS file.
        dtype: Output dtype.

    Returns:
        A tuple of the (nband, ncorr, ny, nx) cube and the (nband,) frequencies.
    """
    hdr = fits.getheader(name)
    data = to4d(fits.getdata(name))
    axis = freq_axis_of(hdr)
    # numpy axis 0 is NAXIS4, numpy axis 1 is NAXIS3
    band_axis = 0 if axis == 4 else 1
    corr_axis = 1 - band_axis
    cube = np.transpose(data, axes=(band_axis, corr_axis, 2, 3))
    freqs, _ = data_from_header(hdr, axis=axis)
    return np.require(cube, dtype=dtype, requirements="C"), freqs


def save_fits(name, data, hdr, overwrite=True, dtype=np.float32, beams_hdu=None, yx_order=False):
    """Write an array to FITS.

    Args:
        name: Output path.
        data: (nband, ncorr, ny, nx) when yx_order is True, otherwise a legacy
            x-major array that is transposed as it always was.
        hdr: FITS header.
        overwrite: Overwrite an existing file.
        dtype: Output dtype.
        beams_hdu: Optional BEAMS BinTableHDU appended as an extension.
        yx_order: True for DataTree-path arrays, which already carry the FITS
            (ny, nx) raster and need only the band and corr axes swapped onto
            the FREQ and STOKES axes.
    """
    hdu = fits.PrimaryHDU(header=hdr)
    if yx_order:
        data = np.transpose(to4d(data), axes=(1, 0, 2, 3))
    else:
        data = np.transpose(to4d(data), axes=(1, 0, 3, 2))
    hdu.data = np.require(data, dtype=dtype, requirements="F")
    if beams_hdu is not None:
        hdul = fits.HDUList([hdu, beams_hdu])
        hdul.writeto(name, overwrite=overwrite)
    else:
        hdu.writeto(name, overwrite=overwrite)


def set_wcs(
    cell_x,
    cell_y,
    nx,
    ny,
    radec,
    freq,
    unit="Jy/beam",
    gausspar=None,
    gausspars=None,
    ms_time=None,
    time_is_unix=False,
    header=True,
    casambm=True,
    l0=0.0,
    m0=0.0,
):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in radians
    freq - frequencies in Hz
    unit - Jy/beam or Jy/pixel
    gausspar - MFS beam parameters in degrees
    gausspars - per-plane beam parameters in degrees, for a cube
    ms_time - measurement set time
    time_is_unix - if True, ms_time is already in unix seconds (the DataTree
        convention); otherwise it is MSv2 MJD seconds
    header - if True, return a header, otherwise return a WCS object
    casambm - if True, add the CASAMBM keyword to the header
    l0/m0 - image-centre offset from the tangent point in radians. CRVAL stays
        the tangent point; CRPIX shifts so the centre pixel lands on the target
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = "deg"
    w.wcs.cunit[1] = "deg"
    w.wcs.cunit[2] = "Hz"
    w.wcs.cunit[3] = ""
    if np.size(freq) > 1:
        nchan = freq.size
        crpix3 = nchan // 2 + 1
        # CRPIX is one-based; the matching numpy index is crpix3 - 1
        ref_freq = freq[crpix3 - 1]
        df = freq[1] - freq[0]
        w.wcs.cdelt[2] = df
    else:
        ref_freq = freq[0] if isinstance(freq, np.ndarray) and freq.size == 1 else freq
        crpix3 = 1
    w.wcs.crval = [radec[0] * 180.0 / np.pi, radec[1] * 180.0 / np.pi, ref_freq, 1]
    # CRVAL stays the tangent point; an l0/m0 offset shifts CRPIX so the centre
    # pixel lands on the target. The RA axis has cdelt = -cell_x, so its sign is
    # opposite the Dec axis's.
    crpix_x = 1 + nx // 2 + np.rad2deg(l0) / cell_x
    crpix_y = 1 + ny // 2 - np.rad2deg(m0) / cell_y
    w.wcs.crpix = [crpix_x, crpix_y, crpix3, 1]
    w.wcs.equinox = 2000.0

    if header:
        header = w.to_header()
        header["RESTFRQ"] = ref_freq
        header["ORIGIN"] = "spimple"
        header["BTYPE"] = "Intensity"
        header["BUNIT"] = unit
        header["SPECSYS"] = "TOPOCENT"
        if ms_time is not None:
            # MSv2 carries MJD seconds; the DataTree carries unix seconds. Truncate
            # to whole seconds before rendering DATE-OBS so the two header keys agree.
            mjd_seconds = ms_time + 3506716800.0 if time_is_unix else ms_time
            utc_iso = Time(mjd_seconds / 86400.0, format="mjd", scale="utc").strftime("%Y-%m-%d %H:%M:%S")
            header["UTC_TIME"] = utc_iso
            t = Time(utc_iso, scale="utc")
            t.format = "fits"
            header["DATE-OBS"] = t.value

        if "LONPOLE" in header:
            header.pop("LONPOLE")
        if "LATPOLE" in header:
            header.pop("LATPOLE")
        if "RADESYS" in header:
            header.pop("RADESYS")
        if "MJDREF" in header:
            header.pop("MJDREF")

        header["EQUINOX"] = 2000.0
        header["BSCALE"] = 1.0
        header["BZERO"] = 0.0
        if casambm:
            header["CASAMBM"] = casambm  # we need this to pick up the beams table

        if gausspar is not None or gausspars is not None:
            # gausspar/gausspars arrive in pixels (the PSFPARSF convention) with
            # the angle in radians; unit2deg converts the axes to degrees.
            header = add_beampars(header, gausspar, GaussPars=gausspars, unit2deg=cell_x)

        return header
    return w


def add_beampars(hdr, GaussPar, GaussPars=None, unit2deg=1.0):
    """
    Add beam keywords to header.
    GaussPar - MFS beam pars
    GaussPars - beam pars for cube
    unit2deg - conversion factor to convert BMAJ/BMIN to degrees

    PA is passed in radians and follows the parametrisation in

    pfb/utils/misc/Gaussian2D

    """
    if GaussPar is not None:
        if len(GaussPar) == 1:
            GaussPar = GaussPar[0]
        elif len(GaussPar) != 3:
            raise ValueError("Invalid value for GaussPar")

    if GaussPar is not None and not np.isnan(GaussPar).any():
        hdr["BMAJ"] = GaussPar[0] * unit2deg
        hdr["BMIN"] = GaussPar[1] * unit2deg
        hdr["BPA"] = GaussPar[2] * 180 / np.pi

    if GaussPars is not None:
        for i in range(len(GaussPars)):
            if not np.isnan(GaussPars[i]).any():
                hdr["BMAJ" + str(i + 1)] = GaussPars[i][0] * unit2deg
                hdr["BMIN" + str(i + 1)] = GaussPars[i][1] * unit2deg
                hdr["BPA" + str(i + 1)] = GaussPars[i][2] * 180 / np.pi

    return hdr


def create_beams_table(beams_data, cell2deg):
    """Build a CASA-style BEAMS BinTableHDU from per-band, per-corr resolutions.

    Args:
        beams_data: DataArray with dims (band, corr, bpar) and the bpar
            coordinate ["BMAJ", "BMIN", "BPA"], in pixels, pixels and radians.
        cell2deg: Cell size in degrees, converting the axes to degrees.

    Returns:
        An astropy BinTableHDU named BEAMS.
    """
    nband = beams_data.band.size
    npol = beams_data.corr.size
    band_id = []
    pol_id = []
    for b in range(nband):
        for p in range(npol):
            band_id.append(b)
            pol_id.append(p)

    bmaj = beams_data.sel({"bpar": "BMAJ"}).values.ravel() * cell2deg
    bmin = beams_data.sel({"bpar": "BMIN"}).values.ravel() * cell2deg
    bpa = beams_data.sel({"bpar": "BPA"}).values.ravel() * 180 / np.pi
    cols = fits.ColDefs(
        [
            fits.Column(name="BMAJ", format="1E", array=bmaj, unit="deg"),
            fits.Column(name="BMIN", format="1E", array=bmin, unit="deg"),
            fits.Column(name="BPA", format="1E", array=bpa, unit="deg"),
            fits.Column(name="CHAN", format="1J", array=np.array(band_id)),
            fits.Column(name="POL", format="1J", array=np.array(pol_id)),
        ]
    )
    beams_hdu = fits.BinTableHDU.from_columns(cols)
    beams_hdu.name = "BEAMS"
    beams_hdu.header["EXTNAME"] = "BEAMS"
    beams_hdu.header["EXTVER"] = 1
    beams_hdu.header["NCHAN"] = nband
    beams_hdu.header["NPOL"] = npol
    return beams_hdu


def set_header_info(mhdr, ref_freq, freq_axis, beampars=None):
    """
    Creates a new FITS header with updated frequency axis and
    optional beam parameters.

    Copies selected header keys from the input header, sets the
    specified frequency axis to length 1 with the given reference
    frequency, and optionally adds beam parameters (`BMAJ`, `BMIN`, `BPA`)
    if provided.

    Args:
        mhdr: Input FITS header to copy keys from.
        ref_freq: Reference frequency value to set on the specified axis.
        freq_axis: Axis index (3 or 4) to update with the reference frequency.
        beampars: Optional tuple of (major axis, minor axis, position angle)
                  for beam parameters.

    Returns:
        A new astropy.io.fits.Header object with updated frequency and
        optional beam information.
    """
    hdr_keys = [
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "NAXIS3",
        "NAXIS4",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CTYPE4",
        "CRPIX1",
        "CRPIX2",
        "CRPIX3",
        "CRPIX4",
        "CRVAL1",
        "CRVAL2",
        "CRVAL3",
        "CRVAL4",
        "CDELT1",
        "CDELT2",
        "CDELT3",
        "CDELT4",
    ]

    new_hdr = {}
    for key in hdr_keys:
        new_hdr[key] = mhdr[key]

    if freq_axis == 3:
        new_hdr["NAXIS3"] = 1
        new_hdr["CRVAL3"] = ref_freq
    elif freq_axis == 4:
        new_hdr["NAXIS4"] = 1
        new_hdr["CRVAL4"] = ref_freq

    if beampars is not None:
        new_hdr["BMAJ"] = beampars[0]
        new_hdr["BMIN"] = beampars[1]
        new_hdr["BPA"] = beampars[2]

    return fits.Header(new_hdr)


def expand_image_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns into a sorted list of existing file paths.

    Replaces the hip-cargo ``expand_patterns`` Typer callback, which was removed
    upstream. This lives in the implementation layer rather than the CLI wrapper
    because the generated cab's ``command:`` targets ``spimple.core.*``, so Stimela
    never executes the wrapper.

    Args:
        patterns: Glob patterns and/or literal paths.

    Returns:
        Sorted, de-duplicated list of matching paths as strings.

    Raises:
        FileNotFoundError: If any pattern matches no existing file.
    """
    expanded: list[str] = []
    for pattern in patterns:
        # Only treat entries containing glob metacharacters as patterns; a literal
        # path is passed straight through so a missing file fails later, where the
        # error message can say what was being read.
        is_glob = any(char in pattern for char in "*?[")
        # Path.glob() raises NotImplementedError for absolute patterns (e.g. the
        # Stimela cab's usual "/data/images/*-image.fits"), so the stdlib glob
        # module is used deliberately here instead of ruff's suggested Path.glob.
        matches = sorted(glob.glob(pattern)) if is_glob else [pattern]
        if not matches:
            msg = f"No files match pattern: {pattern}"
            raise FileNotFoundError(msg)
        expanded.extend(matches)
    return sorted(dict.fromkeys(expanded))
