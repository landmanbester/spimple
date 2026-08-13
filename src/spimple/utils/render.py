"""Render band-node variables from a DataTree store to FITS.

The store's analogue of a FITS writer: it stacks a band variable across bands
into a cube, forms the WSUM-weighted MFS reduction, and writes both with the
store's own WCS. Everything is (corr, y, x) and goes out through
``save_fits(yx_order=True)`` -- no transposes.
"""

import numpy as np
import xarray as xr

from spimple.utils.datatree import band_nodes, open_store, timeids
from spimple.utils.fits import create_beams_table, save_fits, set_wcs
from spimple.utils.logging import get_logger

log = get_logger("RENDER")

BPAR = ["BMAJ", "BMIN", "BPA"]


def dt2fits(
    store_url: str,
    column: str,
    outname: str,
    unit: str = "Jy/beam",
    otype=np.float32,
    do_mfs: bool = True,
    do_cube: bool = True,
    timeid: int | None = None,
    extra_hdr: dict | None = None,
) -> list[str]:
    """Write a band-node variable to FITS, as a cube and an MFS reduction.

    Args:
        store_url: Path to the store.
        column: Band-node variable to render, e.g. "IMAGE".
        outname: Output basename. Files are named
            ``<outname>_<column>_time{t}.fits`` and ``..._mfs.fits``.
        unit: BUNIT value.
        otype: Output dtype.
        do_mfs: Write the WSUM-weighted mean over bands.
        do_cube: Write the per-band cube.
        timeid: Restrict to one timeid; None renders every one.
        extra_hdr: Extra header cards stamped into every file written.

    Returns:
        The paths written, in the order they were written.
    """
    basename = f"{outname}_{column.lower()}"
    written: list[str] = []

    dt = open_store(store_url)
    wanted = timeids(dt) if timeid is None else [timeid]

    for tid in wanted:
        nodes = [n for n in band_nodes(dt, timeid=tid) if column in dt[n].ds]
        if not nodes:
            continue
        datasets = [dt[n].ds for n in nodes]
        ref = datasets[0]

        cube = np.stack([ds[column].values for ds in datasets], axis=0)  # (band, corr, ny, nx)
        wsums = np.stack([np.atleast_1d(ds.WSUM.values) for ds in datasets], axis=0)  # (band, corr)
        wsum = wsums.sum(axis=0)
        freqs = np.array([float(ds.attrs["freq_out"]) for ds in datasets])
        nband, ncorr, ny, nx = cube.shape
        cell_deg = np.rad2deg(float(ref.attrs["cell_rad"]))
        radec = (float(ref.attrs["ra"]), float(ref.attrs["dec"]))
        l0 = float(ref.attrs.get("l0", 0.0))
        m0 = float(ref.attrs.get("m0", 0.0))
        time_out = ref.attrs.get("time_out")

        psfpars = None
        beams_hdu = None
        if all("PSFPARSF" in ds for ds in datasets):
            pars = np.stack([ds.PSFPARSF.values for ds in datasets], axis=0)  # (band, corr, 3)
            psfpars = pars[:, 0]  # FITS carries one beam per plane; use Stokes I
            beams_hdu = create_beams_table(
                xr.DataArray(
                    pars,
                    dims=("band", "corr", "bpar"),
                    coords={"band": np.arange(nband), "corr": list(ref.corr.values), "bpar": BPAR},
                ),
                cell2deg=cell_deg,
            )

        def _header(freq, gausspars, psfpars=psfpars):
            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freq,
                unit=unit,
                gausspar=None if psfpars is None else psfpars[0],
                gausspars=gausspars,
                ms_time=time_out,
                time_is_unix=True,
                l0=l0,
                m0=m0,
            )
            for key, value in (extra_hdr or {}).items():
                hdr[key] = value
            return hdr

        if do_mfs:
            with np.errstate(invalid="ignore", divide="ignore"):
                mfs = np.sum(cube * wsums[:, :, None, None], axis=0) / wsum[:, None, None]
            freq_mfs = float(np.sum(freqs[:, None] * wsums) / wsum.sum())
            hdr = _header(freq_mfs, None)
            hdr["WSUM"] = float(wsum[0])
            name = f"{basename}_time{tid}_mfs.fits"
            save_fits(name, mfs[None], hdr, dtype=otype, yx_order=True)
            written.append(name)
            log.info("Wrote %s", name)

        if do_cube:
            hdr = _header(freqs, psfpars)
            for band in range(nband):
                hdr[f"WSUM{band + 1}"] = float(wsums[band, 0])
            name = f"{basename}_time{tid}.fits"
            save_fits(name, cube, hdr, dtype=otype, beams_hdu=beams_hdu, yx_order=True)
            written.append(name)
            log.info("Wrote %s", name)

    return written
