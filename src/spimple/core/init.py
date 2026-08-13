"""Ingest FITS images into a pfb-imaging-style DataTree.

Input files are grouped into partitions by phase centre and grid, and into bands
by frequency. Each partition is homogenised to a common resolution and, when
there is more than one, reprojected onto a union grid. See
``docs/wiki/datatree-contract.md``.
"""

import os
import re

import numpy as np
from astropy.io import fits

from spimple.utils.datatree import partition_node_name
from spimple.utils.fits import data_from_header, freq_axis_of

PartitionKey = tuple[float, float, float, float, int, int]

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
