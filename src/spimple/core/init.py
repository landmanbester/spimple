"""Ingest FITS images into a pfb-imaging-style DataTree.

Input files are grouped into partitions by phase centre and grid, and into bands
by frequency. Each partition is homogenised to a common resolution and, when
there is more than one, reprojected onto a union grid. See
``docs/wiki/datatree-contract.md``.
"""

import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from spimple.utils.datatree import (
    PRODUCT_VARS,
    band_node_name,
    create_store,
    partition_node_name,
    psfpars_from_header,
    write_node,
)
from spimple.utils.fits import data_from_header, expand_image_patterns, freq_axis_of, load_cube
from spimple.utils.logging import get_logger, log_options
from spimple.utils.render import dt2fits
from spimple.utils.restoration import restore_products

log = get_logger("INIT")

PartitionKey = tuple[float, float, float, float, int, int]

BPAR = ["BMAJ", "BMIN", "BPA"]
_STOKES = {1: "I", 2: "Q", 3: "U", 4: "V", -1: "RR", -2: "LL", -5: "XX", -6: "YY"}

# a thousandth of a pixel, so floating-point noise in a header never splits a partition
_CELL_FRACTION = 1e-3


def partition_key(hdr) -> PartitionKey:
    """Return the identity of the partition a FITS header belongs to.

    Files sharing a phase centre and an image grid are one pointing and become
    one partition. The celestial values are rounded to a thousandth of a pixel
    so header round-tripping cannot split a partition.

    Args:
        hdr: FITS header.

    Returns:
        A hashable key of (crval1, crval2, cdelt1, cdelt2, naxis1, naxis2).
    """
    cell = abs(float(hdr["CDELT1"]))
    quantum = cell * _CELL_FRACTION
    return (
        round(float(hdr["CRVAL1"]) / quantum) * quantum,
        round(float(hdr["CRVAL2"]) / quantum) * quantum,
        round(float(hdr["CDELT1"]) / quantum) * quantum,
        round(float(hdr["CDELT2"]) / quantum) * quantum,
        int(hdr["NAXIS1"]),
        int(hdr["NAXIS2"]),
    )


def group_partitions(paths: list[str]) -> list[tuple[PartitionKey, list[str]]]:
    """Group input paths into partitions, ordered by phase centre.

    Args:
        paths: Resolved FITS paths.

    Returns:
        A list of (key, paths) pairs sorted by (ra0, dec0), so partition ids are
        deterministic across runs.
    """
    groups: dict[PartitionKey, list[str]] = {}
    for path in paths:
        groups.setdefault(partition_key(fits.getheader(path)), []).append(path)
    return [(key, sorted(groups[key])) for key in sorted(groups, key=lambda k: (k[0], k[1]))]


def frequencies_of(paths: list[str]) -> np.ndarray:
    """Return the sorted distinct channel frequencies across a partition's files."""
    freqs: list[float] = []
    for path in paths:
        hdr = fits.getheader(path)
        values, _ = data_from_header(hdr, axis=freq_axis_of(hdr))
        freqs.extend(np.atleast_1d(values).tolist())
    return np.array(sorted(freqs), dtype=np.float64)


def assign_bands(
    freqs_per_partition: list[np.ndarray], freq_tol: float | None
) -> tuple[np.ndarray, list[dict[int, int]]]:
    """Cluster every partition's channels into a common set of bands.

    Args:
        freqs_per_partition: One ascending frequency array per partition.
        freq_tol: Frequencies within this many Hz are one band. Defaults to half
            the narrowest channel width present, or 1 Hz for single-channel input.

    Returns:
        The (nband,) nominal band frequencies (each cluster's midpoint), and one
        dict per partition mapping bandid to that partition's channel index.

    Raises:
        ValueError: If a partition would contribute two channels to one band.
    """
    every = np.sort(np.concatenate([np.atleast_1d(f) for f in freqs_per_partition]))
    if freq_tol is None:
        widths = [np.diff(np.atleast_1d(f)) for f in freqs_per_partition]
        widths = np.concatenate([w for w in widths if w.size]) if any(w.size for w in widths) else np.array([2.0])
        freq_tol = float(np.min(np.abs(widths))) / 2.0

    # single linkage: start a new cluster wherever the gap exceeds the tolerance
    edges = np.flatnonzero(np.diff(every) > freq_tol) + 1
    clusters = np.split(every, edges)
    nominal = np.array([0.5 * (c[0] + c[-1]) for c in clusters], dtype=np.float64)

    mapping: list[dict[int, int]] = []
    for pid, freqs in enumerate(freqs_per_partition):
        per_partition: dict[int, int] = {}
        for chan, freq in enumerate(np.atleast_1d(freqs)):
            bandid = int(np.argmin(np.abs(nominal - freq)))
            if bandid in per_partition:
                raise ValueError(
                    f"partition {pid} contributes two channels to band {bandid} "
                    f"({freq:.6g} Hz and {np.atleast_1d(freqs)[per_partition[bandid]]:.6g} Hz); "
                    "lower freq-tol to separate them"
                )
            per_partition[bandid] = chan
        mapping.append(per_partition)
    return nominal, mapping


def field_name_for(paths: list[str], pid: int) -> str:
    """Name a partition after the common prefix of its filenames.

    Args:
        paths: The partition's file paths.
        pid: Partition id, used for the fallback.

    Returns:
        The stripped common prefix of the basenames, or the partition node name
        when there is no useful prefix.
    """
    # drop the extension first, or a single-file partition keeps its ".fits"
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    prefix = os.path.commonprefix(names)
    # commonprefix is character-wise, so "deep2-0000-image" and "deep2-0001-image"
    # give "deep2-000". Strip a separator-prefixed digit run -- never a bare one,
    # or a field genuinely named "deep2" would lose its 2.
    prefix = re.sub(r"[-_.]\d*$", "", prefix).rstrip("-_. ")
    return prefix if prefix else partition_node_name(pid)


def stokes_labels(hdr) -> list[str]:
    """Return the correlation labels named by the FITS STOKES axis.

    Args:
        hdr: FITS header.

    Returns:
        One label per plane of the non-frequency, non-celestial axis.
    """
    axis = 3 if freq_axis_of(hdr) == 4 else 4
    ncorr = int(hdr.get(f"NAXIS{axis}", 1))
    crval = float(hdr.get(f"CRVAL{axis}", 1.0))
    cdelt = float(hdr.get(f"CDELT{axis}", 1.0))
    return [_STOKES.get(int(round(crval + i * cdelt)), f"C{i}") for i in range(ncorr)]


def store_name(output_filename: str, product: str) -> str:
    """Return the store path, mirroring pfb's <basename>_<PRODUCT>.dt naming."""
    return f"{output_filename}_{product.upper()}.dt"


def resolve_target(psfparsn, psf_pars, circ_psf: bool, dilate: float, cell_deg: float) -> np.ndarray:
    """Resolve the common target resolution for every band and partition.

    Args:
        psfparsn: (nband, ncorr, 3) native resolutions in pixels and radians.
        psf_pars: Requested (emaj, emin, pa) in degrees, or None to derive it.
        circ_psf: Force a circular beam.
        dilate: Safety factor applied when the target is derived.
        cell_deg: Cell size in degrees.

    Returns:
        (ncorr, 3) target resolution in pixels, pixels, radians.

    Raises:
        ValueError: If the requested target is finer than any input resolution.
    """
    psfparsn = np.asarray(psfparsn, dtype=float)
    if psf_pars is None:
        target = np.empty(psfparsn.shape[1:], dtype=float)
        target[:, 0] = np.nanmax(psfparsn[:, :, 0], axis=0) * dilate
        target[:, 1] = np.nanmax(psfparsn[:, :, 1], axis=0) * dilate
        target[:, 2] = np.nanmean(psfparsn[:, :, 2], axis=0)
    else:
        one = np.array([psf_pars[0] / cell_deg, psf_pars[1] / cell_deg, np.deg2rad(psf_pars[2])])
        target = np.tile(one, (psfparsn.shape[1], 1))
        if np.any(psfparsn[:, :, 0] > target[None, :, 0]) or np.any(psfparsn[:, :, 1] > target[None, :, 1]):
            raise ValueError(
                "the requested resolution is finer than an input resolution; convolution can only degrade resolution"
            )
    if circ_psf:
        axis = np.maximum(target[:, 0], target[:, 1])
        target[:, 0] = axis
        target[:, 1] = axis
        target[:, 2] = 0.0
    return target


def _time_out(hdr) -> float:
    """Return DATE-OBS as unix seconds, or 0.0 when the header has no epoch."""
    if "DATE-OBS" not in hdr:
        return 0.0
    return float(Time(hdr["DATE-OBS"]).unix)


def _read_band(paths: list[str], bandid: int, chan: int) -> np.ndarray:
    """Return one band's (ncorr, ny, nx) plane from a partition's files."""
    remaining = chan
    for path in paths:
        cube, _ = load_cube(path, dtype=np.float64)
        if remaining < cube.shape[0]:
            return cube[remaining]
        remaining -= cube.shape[0]
    raise ValueError(f"band {bandid} channel {chan} not found in {paths}")


def init(
    images: list[str],
    output_filename: str,
    residual: list[str] | None = None,
    psf_pars: tuple[float, float, float] | None = None,
    circ_psf: bool = False,
    dilate: float = 1.05,
    beam_model: str | None = None,
    band: str = "L",
    pb_min: float = 0.15,
    padding_frac: float = 0.5,
    products: str = "aik",
    channel_weights_keyword: str = "WSCVWSUM",
    freq_tol: float | None = None,
    fits_outputs: str = "",
    fits_output_folder: str | None = None,
    overwrite: bool = False,
    out_dtype: str = "f4",
    nthreads: int | None = None,
    nworkers: int = 1,
):
    """
    Ingest FITS images into a pfb-imaging style datatree.

    Input files are grouped into partitions by phase centre and grid and into
    bands by frequency, homogenised to a common resolution, and written to a
    zarr store that spifit and mosaic consume. A single pointing needs no
    mosaic step because init populates the band nodes itself.
    """
    log_options(log, **locals())

    images = expand_image_patterns(images)
    residual = expand_image_patterns(residual) if residual else None
    if not nthreads:
        nthreads = multiprocessing.cpu_count()

    groups = group_partitions(images)
    if len(groups) > 1:
        raise NotImplementedError(f"{len(groups)} partitions found; multi-pointing ingest is not implemented yet")
    if beam_model is not None:
        raise NotImplementedError("beam models are not implemented yet; run without beam-model")

    part_paths = groups[0][1]
    hdr = fits.getheader(part_paths[0])
    corr = stokes_labels(hdr)
    ncorr = len(corr)
    cell_deg = abs(float(hdr["CDELT1"]))
    cell_rad = np.deg2rad(cell_deg)
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
    product = "".join(corr)

    nominal, mapping = assign_bands([frequencies_of(part_paths)], freq_tol)
    nband = nominal.size

    psfparsn = psfpars_from_header(hdr, nband, ncorr, cell_deg)
    target = resolve_target(psfparsn, psf_pars, circ_psf, dilate, cell_deg)
    log.info(
        "Target resolution %.3e deg by %.3e deg at %.3e deg",
        target[0, 0] * cell_deg,
        target[0, 1] * cell_deg,
        np.rad2deg(target[0, 2]),
    )

    url = store_name(output_filename, product)
    Path(url).parent.mkdir(parents=True, exist_ok=True)
    create_store(
        url,
        {
            "product": product,
            "nband": int(nband),
            "ntime": 1,
            "nx": nx,
            "ny": ny,
            "cell_rad": cell_rad,
            "origin": "spimple-init",
        },
        overwrite=overwrite,
    )

    beam = np.ones((ncorr, ny, nx), dtype=np.float64)
    rhdr = fits.getheader(residual[0]) if residual else hdr
    time_out = _time_out(hdr)
    ra, dec = np.deg2rad(float(hdr["CRVAL1"])), np.deg2rad(float(hdr["CRVAL2"]))
    letters = tuple(k for k in ("a", "i", "k") if k in products)

    for bandid in range(nband):
        chan = mapping[0].get(bandid)
        if chan is None:
            continue
        model = _read_band(part_paths, bandid, chan)
        if residual:
            resid = _read_band(residual, bandid, chan)
        else:
            # no residual: the input is a restored image, so it carries the
            # resolution and there is no separate model to convolve
            resid = model
            model = np.zeros_like(resid)

        out = restore_products(
            model,
            resid,
            beam,
            target,
            gausspari=psfparsn[bandid],
            products=letters,
            pb_min=pb_min,
            nthreads=nthreads,
            padding_frac=padding_frac,
        )

        rms = np.array([float(np.std(resid[c])) for c in range(ncorr)])
        if channel_weights_keyword in rhdr:
            wsum = np.full(ncorr, float(rhdr[channel_weights_keyword]))
        else:
            wsum = np.where(rms > 0, 1.0 / np.maximum(rms, 1e-30) ** 2, 1.0)

        data_vars = {PRODUCT_VARS[k]: (("corr", "y", "x"), out[k].astype(out_dtype)) for k in letters}
        data_vars["BEAM"] = (("corr", "y", "x"), beam.astype(out_dtype))
        data_vars["WSUM"] = (("corr",), wsum.astype(np.float64))
        data_vars["RMS"] = (("corr",), rms.astype(np.float64))
        data_vars["PSFPARSF"] = (("corr", "bpar"), target.astype(np.float64))
        attrs = {
            "bandid": int(bandid),
            "timeid": 0,
            "freq_out": float(nominal[bandid]),
            "freq_nominal": float(nominal[bandid]),
            "time_out": time_out,
            "ra": ra,
            "dec": dec,
            "cell_rad": cell_rad,
            "l0": 0.0,
            "m0": 0.0,
            "pb_min": float(pb_min),
        }
        coords = {"corr": corr, "bpar": BPAR}
        node = band_node_name(bandid, 0)
        # one partition: the band product IS the partition product, so write both
        write_node(
            url,
            f"{node}/{partition_node_name(0)}",
            {**data_vars, "MASK": (("corr", "y", "x"), np.ones((ncorr, ny, nx), dtype=bool))},
            {
                "field_name": field_name_for(part_paths, 0),
                "ra0": ra,
                "dec0": dec,
                "freq_out": float(nominal[bandid]),
                "psfparsn": psfparsn[bandid].tolist(),
                "beam_includes_n": False,
            },
            coords,
        )
        write_node(url, node, data_vars, attrs, coords)
        log.info("Wrote band %d at %.4e Hz", bandid, nominal[bandid])

    if fits_outputs:
        folder = fits_output_folder or str(Path(url).parent / "fits")
        Path(folder).mkdir(parents=True, exist_ok=True)
        oname = f"{folder}/{Path(output_filename).name}_{product}"
        for letter in ("a", "i", "k"):
            if letter.upper() in fits_outputs or letter in fits_outputs:
                dt2fits(
                    url,
                    PRODUCT_VARS[letter],
                    oname,
                    otype=out_dtype,
                    do_mfs=letter in fits_outputs,
                    do_cube=letter.upper() in fits_outputs,
                )

    log.info("All done here")
