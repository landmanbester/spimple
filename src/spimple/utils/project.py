"""The common output frame and the reprojection onto it.

Every partition in a band must share one grid, as pfb's tree requires. This
module resolves that grid from the input headers and moves each partition's
(Y, X) arrays onto it.

Arrays are (corr, y, x) throughout, which is also what reproject wants: for a
2-axis celestial WCS, astropy binds arr[row, col] with col to WCS axis 1
(RA-like) and row to WCS axis 2 (Dec-like).
"""

import numpy as np
from astropy.wcs import WCS

from spimple.utils.logging import get_logger

log = get_logger("PROJECT")


def union_wcs(headers: list) -> tuple[WCS, tuple[int, int]]:
    """Resolve the smallest common frame covering every input.

    Args:
        headers: FITS headers, one per partition.

    Returns:
        The common WCS with array_shape set, and (ny, nx).
    """
    from reproject.mosaicking import find_optimal_celestial_wcs

    inputs = []
    for hdr in headers:
        wcs = WCS(hdr).celestial
        shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
        inputs.append((shape, wcs))
    wcs, shape_out = find_optimal_celestial_wcs(inputs, projection="SIN")
    # find_optimal_celestial_wcs returns numpy (ny, nx)
    shape = (int(shape_out[0]), int(shape_out[1]))
    wcs.array_shape = shape
    return wcs, shape


def same_frame(wcs_a, shape_a: tuple[int, int], wcs_b, shape_b: tuple[int, int]) -> bool:
    """Return True when reprojecting from one frame to the other is the identity.

    Used to skip a near-identity reprojection for the single-pointing case,
    which would otherwise resample the image for nothing.
    """
    if tuple(shape_a) != tuple(shape_b):
        return False
    return (
        np.allclose(wcs_a.wcs.crval, wcs_b.wcs.crval, rtol=0.0, atol=1e-10)
        and np.allclose(wcs_a.wcs.cdelt, wcs_b.wcs.cdelt, rtol=1e-12, atol=0.0)
        and np.allclose(wcs_a.wcs.crpix, wcs_b.wcs.crpix, rtol=0.0, atol=1e-6)
        and list(wcs_a.wcs.ctype) == list(wcs_b.wcs.ctype)
    )


def reproject_cube(cube: np.ndarray, ref_wcs, target_wcs, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a (ncorr, ny, nx) cube onto the target frame.

    Args:
        cube: (ncorr, ny, nx) on the reference frame.
        ref_wcs: 2-axis celestial WCS describing cube.
        target_wcs: 2-axis celestial WCS of the output.
        shape: (ny, nx) of the output.

    Returns:
        The reprojected (ncorr, ny, nx) cube, zeroed outside the footprint, and
        the (ny, nx) boolean footprint mask.
    """
    from reproject import reproject_interp

    if same_frame(ref_wcs, cube.shape[-2:], target_wcs, shape):
        return np.ascontiguousarray(cube), np.ones(shape, dtype=bool)

    out = np.zeros((cube.shape[0], shape[0], shape[1]), dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for c in range(cube.shape[0]):
        plane, footprint = reproject_interp((cube[c], ref_wcs), target_wcs, shape_out=shape)
        covered = footprint > 0
        plane = np.nan_to_num(plane, nan=0.0)
        plane[~covered] = 0.0
        out[c] = plane
        mask |= covered
    return out, mask
