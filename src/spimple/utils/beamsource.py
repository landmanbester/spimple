"""Primary-beam backends for the ingest path.

Every backend returns a (nband, ncorr, ny, nx) power beam on the grid described
by a celestial WCS, in (Y, X) order like everything else on the tree path.

Reprojection follows the construction validated in pfb-imaging's
docs/wiki/image-and-beam-orientation.md section 6: describe the beam's own grid
honestly with signed CDELT and a coordinate-derived CRPIX, and make the target
WCS equal the real output header. The pre-refactor utils/mosaic.project violated
two of those three -- a one-pixel CRPIX shift and a flipped CDELT1 sign.
"""

import numpy as np
from astropy.wcs import WCS

from spimple.utils.logging import get_logger

log = get_logger("BEAM")

_JIMBEAM = {"l": "MKAT-AA-L-JIM-2020", "uhf": "MKAT-AA-UHF-JIM-2020", "s": "MKAT-AA-S-JIM-2020"}


def lm_grid(wcs, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the (ny, nx) direction-cosine grids in degrees.

    For a SIN projection the intermediate world coordinate
    ``(pixel - crpix) * cdelt`` is the direction cosine itself, so no projection
    maths is needed. l runs along x and decreases (CDELT1 is negative); m runs
    along y and increases.

    Args:
        wcs: A 2-axis celestial WCS.
        shape: (ny, nx).

    Returns:
        (ll, mm), both (ny, nx), in degrees, zero at the reference pixel.
    """
    ny, nx = shape
    cdelt = wcs.wcs.cdelt
    crpix = wcs.wcs.crpix
    l_axis = (np.arange(nx) - (crpix[0] - 1)) * cdelt[0]
    m_axis = (np.arange(ny) - (crpix[1] - 1)) * cdelt[1]
    # meshgrid's default indexing gives (len(m), len(l)) == (ny, nx)
    return np.meshgrid(l_axis, m_axis)


def _reproject_plane(plane: np.ndarray, ref_wcs, target_wcs, shape: tuple[int, int]) -> np.ndarray:
    """Reproject one (ny, nx) plane onto the target WCS, zeroing outside the footprint."""
    from reproject import reproject_interp

    out, footprint = reproject_interp((plane, ref_wcs), target_wcs, shape_out=shape)
    out = np.nan_to_num(out, nan=0.0)
    out[footprint <= 0] = 0.0
    return out


def _wcs_from_coords(l_deg: np.ndarray, m_deg: np.ndarray, radec_deg: tuple[float, float]) -> WCS:
    """Build the reference WCS of a beam grid from its own coordinates.

    Signed CDELT from the coordinate spacing, CRPIX from where zero falls -- the
    two things the old project() got wrong.
    """
    dl = float(l_deg[1] - l_deg[0])
    dm = float(m_deg[1] - m_deg[0])
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cdelt = [dl, dm]
    w.wcs.crval = list(radec_deg)
    w.wcs.crpix = [1 + (0.0 - float(l_deg[0])) / dl, 1 + (0.0 - float(m_deg[0])) / dm]
    w.array_shape = (m_deg.size, l_deg.size)
    return w


def _jimbeam(band: str, freqs: np.ndarray, wcs, shape: tuple[int, int]) -> np.ndarray:
    from katbeam import JimBeam

    key = band.lower()
    if key not in _JIMBEAM:
        raise ValueError(f"Unknown band {band} for katbeam, expected one of {sorted(_JIMBEAM)}")
    jim = JimBeam(_JIMBEAM[key])
    ll, mm = lm_grid(wcs, shape)
    return np.stack([jim.I(ll, mm, float(freq) / 1e6) for freq in freqs])


def _fits_beam(path: str, freqs: np.ndarray, wcs, shape: tuple[int, int]) -> np.ndarray:
    from astropy.io import fits

    from spimple.utils.fits import load_cube

    cube, bfreqs = load_cube(path, dtype=np.float64)  # (nband, ncorr, ny, nx)
    ref_wcs = WCS(fits.getheader(path)).celestial
    out = np.empty((freqs.size, shape[0], shape[1]), dtype=np.float64)
    for i, freq in enumerate(freqs):
        nearest = int(np.argmin(np.abs(bfreqs - freq)))
        out[i] = _reproject_plane(cube[nearest, 0], ref_wcs, wcs, shape)
    return out


def _bds_beam(path: str, freqs: np.ndarray, wcs, shape: tuple[int, int], radec_deg) -> np.ndarray:
    """Rotation-averaged power beam from a meerkat-beams bds zarr store."""
    import xarray as xr
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator

    bds = xr.open_zarr(path, chunks=None)
    # meerkat-beams has used both spellings for the beam grid coordinates
    l_name = "l_beam" if "l_beam" in bds.coords or "l_beam" in bds else "X"
    m_name = "m_beam" if "m_beam" in bds.coords or "m_beam" in bds else "Y"
    l_beam = np.asarray(bds[l_name].values, dtype=float)
    m_beam = np.asarray(bds[m_name].values, dtype=float)
    bfreq = np.asarray(bds.chan.values, dtype=float)
    jones = bds.BEAM.values  # (ncorr, nchan, ny, nx) on the beam's own grid
    power = ((jones[0] * jones[0].conj()).real + (jones[-1] * jones[-1].conj()).real) / 2.0

    interp = RegularGridInterpolator(
        (bfreq, m_beam, l_beam), power, bounds_error=False, fill_value=None, method="linear"
    )
    ref_wcs = _wcs_from_coords(l_beam, m_beam, radec_deg)
    ll, mm = np.meshgrid(l_beam, m_beam)

    out = np.empty((freqs.size, shape[0], shape[1]), dtype=np.float64)
    angles = np.linspace(0, 359, 25)
    for i, freq in enumerate(freqs):
        plane = interp((float(freq), mm, ll))
        rotated = np.zeros_like(plane)
        for angle in angles:
            rotated += ndimage.rotate(plane, angle, reshape=False, order=1, mode="nearest")
        rotated /= angles.size
        out[i] = _reproject_plane(rotated, ref_wcs, wcs, shape)
    return out


def beam_for_grid(
    beam_model: str | None,
    band: str,
    freqs: np.ndarray,
    wcs,
    shape: tuple[int, int],
    ncorr: int,
    nthreads: int = 1,
) -> np.ndarray:
    """Evaluate a power beam on an image grid.

    Args:
        beam_model: None for a uniform response, the literal "JimBeam", a path
            ending in .fits for a beam cube, or a path to a bds zarr store.
        band: JimBeam band, one of L, UHF or S.
        freqs: (nband,) frequencies in Hz.
        wcs: The 2-axis celestial WCS of the target grid.
        shape: (ny, nx) of the target grid.
        ncorr: Number of correlations. The Stokes I power beam is broadcast.
        nthreads: Unused today; kept so callers need not special-case backends.

    Returns:
        (nband, ncorr, ny, nx) in [0, 1].

    Raises:
        ValueError: If beam_model is not one of the supported forms.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if beam_model is None:
        planes = np.ones((freqs.size, shape[0], shape[1]), dtype=np.float64)
    else:
        name = str(beam_model)
        if name == "JimBeam":
            planes = _jimbeam(band, freqs, wcs, shape)
        elif name.endswith(".fits"):
            planes = _fits_beam(name, freqs, wcs, shape)
        elif name.rstrip("/").endswith(".zarr"):
            radec = (float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]))
            planes = _bds_beam(name, freqs, wcs, shape, radec)
        else:
            raise ValueError(f"Unknown beam model {beam_model}; expected JimBeam, a .fits cube or a .zarr store")
    return np.repeat(planes[:, None], ncorr, axis=1)
