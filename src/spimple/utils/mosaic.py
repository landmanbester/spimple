import numpy as np
import ray
import xarray as xr
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from spimple.utils.fits import data_from_header


def mosaic_info(im_list, oname, ref_image=None):
    if ref_image is not None:
        raise NotImplementedError("Reference image not supported yet")
    wcss = []
    out_names = []
    freqs = []
    flatfreqs = []
    shapes = []
    basename = oname.removesuffix(".fits")
    for imnum, im in enumerate(im_list):
        hdr = fits.getheader(im)
        nu = data_from_header(hdr, axis=3)[0]
        freqs.append(nu)
        flatfreqs.extend(nu)
        wcsi = WCS(hdr).dropaxis(-1).dropaxis(-1)
        wcss.append(wcsi)
        nchan = hdr["NAXIS3"]
        ncorr = hdr["NAXIS4"]
        shapes.append((hdr["NAXIS1"], hdr["NAXIS2"]))
        out_names.extend([f"{basename}_im{imnum}_pol{c}_ch{f}.zarr" for c in range(ncorr) for f in range(nchan)])
    nu = np.array(flatfreqs)
    ufreqs = np.unique(nu)
    # get domain of intrinsic image
    ref_wcs, shape_out = find_optimal_celestial_wcs(list(zip(shapes, wcss, strict=False)), projection="SIN")
    ref_wcs.array_shape = (shape_out[0], shape_out[1])

    return ref_wcs, ufreqs, out_names


@ray.remote
def project(im, imnum, ref_wcs, beam, oname, method="interp"):
    if method != "interp":
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
    pbeam = ((beam[0] * beam[0].conj()).real + (beam[-1] * beam[-1].conj()).real) / 2.0

    # this is cheap, evaluation is more expensive
    beamo = RegularGridInterpolator(
        (bfreq, l_beam, m_beam),
        pbeam,
        bounds_error=False,
        fill_value=None,
        method="linear",
    )

    cell_x, cell_y = ref_wcs.wcs.cdelt
    l_im = (-(nxo) // 2 + np.arange(nxo)) * cell_x
    m_im = (-(nyo) // 2 + np.arange(nyo)) * cell_y
    im_coords = {
        "l": ("l", l_im),
        "m": ("m", m_im),
    }

    image = fits.getdata(im)
    ncorr, nchan, nx, ny = image.shape
    hdr = fits.getheader(im)
    freq, _ = data_from_header(hdr, axis=3)
    wcs = WCS(hdr).dropaxis(-1).dropaxis(-1)
    cxi, cyi = wcs.wcs.cdelt
    nx, ny = wcs.array_shape
    l = (-(nx) // 2 + np.arange(nx)) * cxi
    m = (-(ny) // 2 + np.arange(ny)) * cyi
    ll, mm = np.meshgrid(l, m, indexing="ij")

    basename = oname.removesuffix(".fits")
    # TODO - make these in parallel
    for c in range(ncorr):
        for f in range(nchan):
            bdata = np.zeros((nx, ny), dtype=np.float64)
            beami = beamo((freq[c], ll, mm))
            step = 25
            angles = np.linspace(0, 359, step)
            for angle in angles:
                bdata += ndimage.rotate(beami, angle, reshape=False, order=1, mode="nearest")
            bdata /= angles.size
            pbeam, footprint = reproject_interp(
                (bdata, wcs),
                ref_wcs,
                shape_out=(nxo, nyo),
                block_size="auto",
                parallel=4,
            )
            mask = footprint > 0
            pbeam[~mask] = 0
            pdata, _ = reproject_interp(
                (image[c, f], wcs),
                ref_wcs,
                shape_out=(nxo, nyo),
                block_size="auto",
                parallel=4,
            )

            im_attrs = {
                "freq": freq[c],
            }

            data_vars = {
                "IMAGE": (("l", "m"), pdata),
                "BEAM": (("l", "m"), pbeam),
                "MASK": (("l", "m"), mask),
            }

            ds = xr.Dataset(data_vars, coords=im_coords, attrs=im_attrs).chunk(
                {
                    "l": 512,
                    "m": 512,
                }
            )

            ds_name = f"{basename}_im{imnum}_pol{c}_ch{f}.zarr"
            ds.to_zarr(ds_name, compute=True, mode="w", consolidated=False)

    return imnum


@ray.remote
def stitch_images(freq, im_list, eta=1e-3):
    # get all datasets in current plane
    xds = []
    for im in im_list:
        ds = xr.open_zarr(im, chunks=None, consolidated=False)
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
        y[mask] += beam[mask] * image[mask]
    ds = ds.drop_vars("IMAGE")

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
        weight[mask] += beam[mask] ** 2
    weight += eta
    return image, weight, info, freq


def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=100, report=20):
    n = b.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    r = A(x) - b
    p = -r
    rsold = np.vdot(r, r)
    rs0 = rsold
    if rs0 < tol:
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

        if i % report == 0:
            print(f"At {i} norm frac = {rsnew / rs0}")

    return x, i
