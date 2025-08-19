from datetime import datetime, timezone

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from casacore.quanta import quantity
import numpy as np


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


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data), axes=(1, 0, 3, 2))  # fits and beams table
    return np.require(data, dtype=dtype, requirements="C")


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32, beams_hdu=None):
    hdu = fits.PrimaryHDU(header=hdr)
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
    GuassPar=None,
    ms_time=None,
    header=True,
    casambm=True,
):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in radians
    freq - frequencies in Hz
    unit - Jy/beam or Jy/pixel
    GuassPar - MFS beam parameters in degrees
    ms_time - measurement set time
    header - if True, return a header, otherwise return a WCS object
    casambm - if True, add the CASAMBM keyword to the header
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
        ref_freq = freq[crpix3]
        df = freq[1] - freq[0]
        w.wcs.cdelt[2] = df
    else:
        ref_freq = freq[0] if isinstance(freq, np.ndarray) else freq
        crpix3 = 1
    w.wcs.crval = [radec[0] * 180.0 / np.pi, radec[1] * 180.0 / np.pi, ref_freq, 1]
    w.wcs.crpix = [1 + nx // 2, 1 + ny // 2, crpix3, 1]
    w.wcs.equinox = 2000.0

    if header:
        header = w.to_header()
        header["RESTFRQ"] = ref_freq
        header["ORIGIN"] = "pfb-imaging"
        header["BTYPE"] = "Intensity"
        header["BUNIT"] = unit
        header["SPECSYS"] = "TOPOCENT"
        if ms_time is not None:
            # TODO - probably a round about way of doing this
            unix_time = quantity(f"{ms_time}s").to_unix_time()
            utc_iso = datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            header["UTC_TIME"] = utc_iso
            t = Time(utc_iso)
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

        if GuassPar is not None:
            header = add_beampars(header, GuassPar)

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
    if len(GaussPar) == 1:
        GaussPar = GaussPar[0]
    elif len(GaussPar) != 3:
        raise ValueError("Invalid value for GaussPar")

    if not np.isnan(GaussPar).any():
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
