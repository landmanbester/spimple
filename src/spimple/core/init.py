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
from astropy.wcs import WCS

from spimple.utils.beamsource import beam_for_grid
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
from spimple.utils.project import reproject_cube, union_wcs
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
    """Return a partition's channel frequencies in traversal order.

    Deliberately NOT sorted. The returned index of each frequency is what
    ``assign_bands`` records in its mapping and what ``_locate_band`` then uses
    to find the plane, so it must be the order those two walk: files in the
    given order, planes in native cube-axis order. Sorting here silently
    mislabels every plane of a descending frequency axis.
    """
    freqs: list[float] = []
    for path in paths:
        hdr = fits.getheader(path)
        values, _ = data_from_header(hdr, axis=freq_axis_of(hdr))
        freqs.extend(np.atleast_1d(values).tolist())
    return np.array(freqs, dtype=np.float64)


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


def resolve_target(psfparsn, psf_pars, circ_psf: bool, dilate: float, cell_deg: float = 1.0) -> np.ndarray:
    """Resolve the common target resolution for every band and partition.

    Angular units are canonical: partitions may sit on different pixel scales,
    so a resolution in pixels is only meaningful once paired with a grid. The
    caller divides by whichever cell size it is about to work on.

    Args:
        psfparsn: (nband, ncorr, 3) native resolutions, axes in the same units
            as cell_deg implies and the angle in radians.
        psf_pars: Requested (emaj, emin, pa) in degrees, or None to derive it.
        circ_psf: Force a circular beam.
        dilate: Safety factor applied when the target is derived.
        cell_deg: Cell size that psfparsn's axes are expressed in, so 1.0 when
            they are already in degrees.

    Returns:
        (ncorr, 3) target resolution in the same axis units as psfparsn, with
        the angle in radians.

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


def _locate_band(paths: list[str], chan: int) -> tuple[str, int]:
    """Map a partition-wide channel index onto (file, plane within that file).

    ``chan`` counts in the same traversal order ``frequencies_of`` uses.

    Args:
        paths: The partition's file paths, in order.
        chan: Channel index within the partition.

    Returns:
        The owning path and the plane's index inside that file.

    Raises:
        ValueError: If the index runs past the end of the partition.
    """
    remaining = chan
    for path in paths:
        hdr = fits.getheader(path)
        nplane = int(hdr.get(f"NAXIS{freq_axis_of(hdr)}", 1))
        if remaining < nplane:
            return path, remaining
        remaining -= nplane
    raise ValueError(f"channel {chan} not found in {paths}")


def _read_band(paths: list[str], bandid: int, chan: int) -> np.ndarray:
    """Return one band's (ncorr, ny, nx) plane from a partition's files."""
    path, plane = _locate_band(paths, chan)
    cube, _ = load_cube(path, dtype=np.float64)
    return cube[plane]


def psfparsn_deg(paths: list[str], mapping: dict[int, int], ncorr: int) -> dict[int, np.ndarray]:
    """Read each band's native resolution from the file that supplies it.

    Reading the whole partition's beams from its first header is wrong for the
    common split-per-channel layout, where every file carries a single scalar
    BMAJ describing only its own plane, and for any partition whose files differ in
    resolution.

    Args:
        paths: The partition's file paths, in order.
        mapping: bandid to channel index, for this partition.
        ncorr: Number of correlations.

    Returns:
        bandid to an (ncorr, 3) array of (emaj, emin, pa) in degrees, degrees
        and radians.
    """
    out: dict[int, np.ndarray] = {}
    for bandid, chan in mapping.items():
        path, plane = _locate_band(paths, chan)
        hdr = fits.getheader(path)
        nplane = int(hdr.get(f"NAXIS{freq_axis_of(hdr)}", 1))
        # cell_deg=1.0 keeps the axes in degrees; the caller converts to the
        # pixel scale of whichever grid it is about to work on
        pars = psfpars_from_header(hdr, nplane, ncorr, cell_deg=1.0)
        out[bandid] = pars[plane]
    return out


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

    headers = [fits.getheader(paths[0]) for _, paths in groups]
    ref_hdr = headers[0]
    corr = stokes_labels(ref_hdr)
    ncorr = len(corr)
    cell_deg = abs(float(ref_hdr["CDELT1"]))
    cell_rad = np.deg2rad(cell_deg)
    product = "".join(corr)

    target_wcs, (ny, nx) = union_wcs(headers)
    ra = np.deg2rad(float(target_wcs.wcs.crval[0]))
    dec = np.deg2rad(float(target_wcs.wcs.crval[1]))
    log.info("Union grid is %d by %d pixels over %d partitions", ny, nx, len(groups))

    # match residuals to images by partition identity, never by list position:
    # two equal-sized sets with a swapped pointing would otherwise pair silently
    residual_by_key = None
    if residual:
        residual_by_key = dict(group_partitions(residual))
        missing = [key for key, _ in groups if key not in residual_by_key]
        if missing:
            raise ValueError(
                f"{len(missing)} image partition(s) have no residual at the same phase centre and grid; "
                "every --images pointing needs a matching --residual pointing"
            )

    freqs_per_partition = [frequencies_of(paths) for _, paths in groups]
    nominal, mapping = assign_bands(freqs_per_partition, freq_tol)
    nband = nominal.size

    # native resolutions in degrees, read from the file that supplies each band
    psfparsn = [psfparsn_deg(paths, mapping[pid], ncorr) for pid, (_, paths) in enumerate(groups)]
    all_native = np.stack([pars for per_part in psfparsn for pars in per_part.values()])
    target_deg = resolve_target(all_native, psf_pars, circ_psf, dilate)
    log.info(
        "Target resolution %.3e deg by %.3e deg at %.3e deg",
        target_deg[0, 0],
        target_deg[0, 1],
        np.rad2deg(target_deg[0, 2]),
    )
    # stored on the union grid, so PSFPARSF is in union-grid pixels
    target = target_deg / np.array([cell_deg, cell_deg, 1.0])

    url = store_name(output_filename, product)
    Path(url).parent.mkdir(parents=True, exist_ok=True)
    create_store(
        url,
        {
            "product": product,
            "nband": int(nband),
            "ntime": 1,
            "nx": int(nx),
            "ny": int(ny),
            "cell_rad": cell_rad,
            "origin": "spimple-init",
        },
        overwrite=overwrite,
    )

    letters = tuple(k for k in ("a", "i", "k") if k in products)
    single = len(groups) == 1

    for pid, (_, paths) in enumerate(groups):
        hdr = headers[pid]
        # the convolution happens on this partition's native grid, so the
        # resolutions must be in ITS pixels, not the union grid's
        part_cell_deg = abs(float(hdr["CDELT1"]))
        target_pix = target_deg / np.array([part_cell_deg, part_cell_deg, 1.0])
        part_wcs = WCS(hdr).celestial
        part_shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
        res_paths = residual_by_key[groups[pid][0]] if residual_by_key else None
        rhdr = fits.getheader(res_paths[0]) if res_paths else hdr
        band_ids = sorted(mapping[pid])
        band_freqs = np.array([freqs_per_partition[pid][mapping[pid][b]] for b in band_ids], dtype=float)
        beams = beam_for_grid(beam_model, band, band_freqs, part_wcs, part_shape, ncorr, nthreads=nthreads)
        ra0 = np.deg2rad(float(hdr["CRVAL1"]))
        dec0 = np.deg2rad(float(hdr["CRVAL2"]))

        for slot, bandid in enumerate(band_ids):
            chan = mapping[pid][bandid]
            model = _read_band(paths, bandid, chan)
            if res_paths:
                resid = _read_band(res_paths, bandid, chan)
            else:
                # no residual: the input is a restored image, so it carries the
                # resolution and there is no separate model to convolve
                resid = model
                model = np.zeros_like(resid)
            beam = beams[slot]

            out = restore_products(
                model,
                resid,
                beam,
                target_pix,
                gausspari=psfparsn[pid][bandid] / np.array([part_cell_deg, part_cell_deg, 1.0]),
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

            # Convolve first, reproject second: the products are Jy/beam, which
            # reproject_interp conserves. A Jy/pixel model would not survive it.
            # The beam goes first so the footprint mask is defined even when
            # products selects nothing.
            pbeam, mask = reproject_cube(beam, part_wcs, target_wcs, (ny, nx))
            pbeam[:, ~mask] = 0.0
            projected = {k: reproject_cube(arr, part_wcs, target_wcs, (ny, nx))[0] for k, arr in out.items()}

            data_vars = {PRODUCT_VARS[k]: (("corr", "y", "x"), projected[k].astype(out_dtype)) for k in letters}
            data_vars["BEAM"] = (("corr", "y", "x"), pbeam.astype(out_dtype))
            data_vars["WSUM"] = (("corr",), wsum.astype(np.float64))
            data_vars["RMS"] = (("corr",), rms.astype(np.float64))
            data_vars["PSFPARSF"] = (("corr", "bpar"), target.astype(np.float64))
            coords = {"corr": corr, "bpar": BPAR}
            node = band_node_name(bandid, 0)

            write_node(
                url,
                f"{node}/{partition_node_name(pid)}",
                {**data_vars, "MASK": (("corr", "y", "x"), np.repeat(mask[None], ncorr, axis=0))},
                {
                    "field_name": field_name_for(paths, pid),
                    "ra0": ra0,
                    "dec0": dec0,
                    "freq_out": float(nominal[bandid]),
                    "psfparsn": (psfparsn[pid][bandid] / np.array([cell_deg, cell_deg, 1.0])).tolist(),
                    "beam_includes_n": False,
                },
                coords,
            )
            attrs = {
                "bandid": int(bandid),
                "timeid": 0,
                "freq_out": float(nominal[bandid]),
                "freq_nominal": float(nominal[bandid]),
                "time_out": _time_out(hdr),
                "ra": ra,
                "dec": dec,
                "cell_rad": cell_rad,
                "l0": 0.0,
                "m0": 0.0,
                "pb_min": float(pb_min),
            }
            if single:
                # one pointing needs no mosaic step: the band product IS the
                # partition product, so write it straight through
                write_node(url, node, data_vars, attrs, coords)
            else:
                write_node(url, node, {"PSFPARSF": data_vars["PSFPARSF"]}, attrs, coords)
        log.info("Wrote partition %d with %d bands", pid, len(band_ids))

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
