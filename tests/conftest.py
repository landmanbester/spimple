"""Synthetic FITS fixtures.

Everything is generated under tmp_path -- no downloads, no tests/data, nothing
written into the repo tree. The cubes are deliberately tiny so the functional
tests stay cheap.

Axis-order note: astropy maps numpy axes to FITS axes in reverse, so a numpy
array of shape (a, b, c, d) becomes NAXIS1=d, NAXIS2=c, NAXIS3=b, NAXIS4=a.
Putting the frequency axis on CTYPE4 therefore means numpy axis 0, and on
CTYPE3 means numpy axis 1. Both conventions occur in real WSClean/CASA output,
which is why both are fixtured.
"""

import numpy as np
import pytest
from astropy.io import fits

NCHAN = 4
NPIX = 64
REF_FREQ = 1.0e9
BAND_WIDTH = 4.0e8
CELL_DEG = 1.0 / 3600.0
TRUE_ALPHA = -0.7


def _base_header(freq_axis: int) -> fits.Header:
    """Celestial WCS with the frequency axis on CTYPE3 or CTYPE4."""
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = 30.0
    hdr["CRPIX1"] = NPIX // 2 + 1
    hdr["CDELT1"] = -CELL_DEG
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = -30.0
    hdr["CRPIX2"] = NPIX // 2 + 1
    hdr["CDELT2"] = CELL_DEG
    hdr["CUNIT2"] = "deg"

    stokes_axis = 3 if freq_axis == 4 else 4
    hdr[f"CTYPE{freq_axis}"] = "FREQ"
    hdr[f"CRVAL{freq_axis}"] = REF_FREQ
    hdr[f"CRPIX{freq_axis}"] = 1
    hdr[f"CDELT{freq_axis}"] = BAND_WIDTH / NCHAN
    hdr[f"CUNIT{freq_axis}"] = "Hz"
    hdr[f"CTYPE{stokes_axis}"] = "STOKES"
    hdr[f"CRVAL{stokes_axis}"] = 1.0
    hdr[f"CRPIX{stokes_axis}"] = 1
    hdr[f"CDELT{stokes_axis}"] = 1.0

    hdr["BUNIT"] = "Jy/beam"
    hdr["EQUINOX"] = 2000.0
    return hdr


def _add_per_channel_beams(hdr: fits.Header) -> fits.Header:
    """BMAJ1..BMAJn keywords, resolution sharpening with frequency."""
    emaj = np.linspace(6.0, 3.0, NCHAN) * CELL_DEG
    for i in range(1, NCHAN + 1):
        hdr[f"BMAJ{i}"] = float(emaj[i - 1])
        hdr[f"BMIN{i}"] = float(emaj[i - 1] * 0.8)
        hdr[f"BPA{i}"] = 0.0
    hdr["BMAJ"] = float(emaj[0])
    hdr["BMIN"] = float(emaj[0] * 0.8)
    hdr["BPA"] = 0.0
    return hdr


def frequencies() -> np.ndarray:
    """The channel frequencies the fixture cubes are built on."""
    return REF_FREQ + np.arange(NCHAN) * BAND_WIDTH / NCHAN


def _power_law_cube(alpha: float, seed: int) -> np.ndarray:
    """(nchan, npix, npix) of Gaussian blobs with a known spectral index."""
    rng = np.random.default_rng(seed)
    freqs = frequencies()
    x = np.arange(NPIX) - NPIX // 2
    xx, yy = np.meshgrid(x, x, indexing="ij")

    cube = np.zeros((NCHAN, NPIX, NPIX), dtype=np.float32)
    for _ in range(5):
        x0, y0 = rng.integers(-NPIX // 4, NPIX // 4, size=2)
        amp = rng.uniform(0.5, 2.0)
        blob = amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 3.0**2))
        for v, freq in enumerate(freqs):
            cube[v] += blob * (freq / REF_FREQ) ** alpha
    return cube


def _shape_for(cube: np.ndarray, freq_axis: int) -> np.ndarray:
    """Insert the degenerate Stokes axis so freq lands on the requested FITS axis."""
    if freq_axis == 4:
        return cube[:, None]  # (nchan, 1, ny, nx) -> NAXIS4 = nchan
    return cube[None]  # (1, nchan, ny, nx) -> NAXIS3 = nchan


def _write(path, data, hdr) -> str:
    fits.writeto(path, data, hdr, overwrite=True)
    return str(path)


@pytest.fixture(scope="session")
def true_alpha():
    """The spectral index injected into the fixture cubes."""
    return TRUE_ALPHA


@pytest.fixture(scope="session")
def beam_params():
    """(emaj, emin, pa) in degrees, coarser than every channel's native beam."""
    return (8.0 * CELL_DEG, 6.4 * CELL_DEG, 0.0)


@pytest.fixture(scope="session")
def image_cube(tmp_path_factory):
    """Model cube with spectral index TRUE_ALPHA, frequency on CTYPE4."""
    path = tmp_path_factory.mktemp("fits") / "model.fits"
    hdr = _add_per_channel_beams(_base_header(freq_axis=4))
    return _write(path, _shape_for(_power_law_cube(TRUE_ALPHA, seed=42), 4), hdr)


@pytest.fixture(scope="session")
def image_cube_ctype3(tmp_path_factory):
    """Same data with the frequency axis on CTYPE3."""
    path = tmp_path_factory.mktemp("fits") / "model_ctype3.fits"
    hdr = _add_per_channel_beams(_base_header(freq_axis=3))
    return _write(path, _shape_for(_power_law_cube(TRUE_ALPHA, seed=42), 3), hdr)


@pytest.fixture(scope="session")
def residual_cube(tmp_path_factory):
    """Noise-only residual on image_cube's grid, for threshold determination.

    Carries WSCVWSUM because spifit derives its per-band weights from that
    header keyword when a residual is supplied (core/spifit.py).
    """
    path = tmp_path_factory.mktemp("fits") / "residual.fits"
    hdr = _add_per_channel_beams(_base_header(freq_axis=4))
    hdr["WSCVWSUM"] = 1.0
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=1e-3, size=(NCHAN, NPIX, NPIX)).astype(np.float32)
    return _write(path, _shape_for(noise, 4), hdr)
