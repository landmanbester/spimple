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


NBAND_TREE = 4
NCORR_TREE = 1
CELL_RAD = np.deg2rad(CELL_DEG)
BPAR = ["BMAJ", "BMIN", "BPA"]


def _tree_beam(npix: int) -> np.ndarray:
    """A smooth, circularly symmetric response peaking at 1.0 in the centre."""
    v = np.arange(npix) - npix // 2
    yy, xx = np.meshgrid(v, v, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2 * (npix / 2.5) ** 2))


@pytest.fixture(scope="session")
def pfb_tree(tmp_path_factory):
    """A synthetic pfb-shaped DataTree, homogenised, with spectral index TRUE_ALPHA.

    Built to the layout documented in pfb-imaging's docs/wiki/imager-pipeline.md
    (verified against pfb-imaging commit b84407b, branch dev-0.1.0). If that page
    changes, this fixture must be re-checked -- it is the only thing standing
    between spimple's consumers and the real contract.

    Everything is (corr, y, x). PSFPARSN and PSFPARSF are in pixels and radians.
    """
    from spimple.utils.datatree import band_node_name, create_store, partition_node_name, write_node

    url = str(tmp_path_factory.mktemp("tree") / "synth_I.dt")
    freqs = frequencies()
    beam = _tree_beam(NPIX)[None]  # (ncorr, ny, nx)
    cube = _power_law_cube(TRUE_ALPHA, seed=11)  # (nband, ny, nx), intrinsic
    rng = np.random.default_rng(3)
    # homogenised: every band already sits at the same resolution
    psfparsf = np.tile([8.0, 6.4, 0.0], (NCORR_TREE, 1))

    create_store(
        url,
        {
            "spimple_version": "test",
            "product": "I",
            "nband": NBAND_TREE,
            "ntime": 1,
            "nx": NPIX,
            "ny": NPIX,
            "cell_rad": CELL_RAD,
        },
        overwrite=True,
    )

    for band in range(NBAND_TREE):
        intrinsic = cube[band][None].astype(np.float32)
        apparent = (intrinsic * beam).astype(np.float32)
        residual = rng.normal(scale=1e-3, size=intrinsic.shape).astype(np.float32)
        wsum = np.array([1.0 + band], dtype=np.float32)
        node = band_node_name(band, 0)
        write_node(
            url,
            node,
            {
                "MODEL": (("corr", "y", "x"), intrinsic),
                "RESIDUAL": (("corr", "y", "x"), (residual * wsum[:, None, None]).astype(np.float32)),
                "IMAGE": (("corr", "y", "x"), intrinsic),
                "BIMAGE": (("corr", "y", "x"), apparent),
                "KIMAGE": (("corr", "y", "x"), (apparent + residual).astype(np.float32)),
                "BEAM": (("corr", "y", "x"), beam.astype(np.float32)),
                "WSUM": (("corr",), wsum),
                "PSFPARSN": (("corr", "bpar"), psfparsf.astype(np.float32)),
                "PSFPARSF": (("corr", "bpar"), psfparsf.astype(np.float32)),
            },
            {
                "bandid": band,
                "timeid": 0,
                "freq_out": float(freqs[band]),
                "freq_nominal": float(freqs[band]),
                "time_out": 1.0e9,
                "ra": np.deg2rad(30.0),
                "dec": np.deg2rad(-30.0),
                "cell_rad": CELL_RAD,
                "l0": 0.0,
                "m0": 0.0,
                "pb_min": 0.15,
            },
            {"corr": ["I"], "bpar": BPAR},
        )
        write_node(
            url,
            f"{node}/{partition_node_name(0)}",
            {"BEAM": (("corr", "y", "x"), beam.astype(np.float32))},
            {
                "field_name": "synth",
                "ra0": np.deg2rad(30.0),
                "dec0": np.deg2rad(-30.0),
                "freq_out": float(freqs[band]),
                "beam_includes_n": True,
            },
            {"corr": ["I"]},
        )
    return url


def _beam_fits(path, freqs, sigma_l_deg, sigma_m_deg, npix=48, cell_deg=None):
    """A Gaussian power beam on its own coarse grid, centred on the pointing."""
    cell_deg = cell_deg if cell_deg is not None else 20.0 / 3600.0
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRVAL1"] = 30.0
    hdr["CRPIX1"] = npix // 2 + 1
    hdr["CDELT1"] = -cell_deg
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL2"] = -30.0
    hdr["CRPIX2"] = npix // 2 + 1
    hdr["CDELT2"] = cell_deg
    hdr["CUNIT2"] = "deg"
    hdr["CTYPE4"] = "FREQ"
    hdr["CRVAL4"] = float(freqs[0])
    hdr["CRPIX4"] = 1
    hdr["CDELT4"] = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0e8
    hdr["CUNIT4"] = "Hz"
    hdr["CTYPE3"] = "STOKES"
    hdr["CRVAL3"] = 1.0
    hdr["CRPIX3"] = 1
    hdr["CDELT3"] = 1.0

    v = (np.arange(npix) - npix // 2) * cell_deg
    ll, mm = np.meshgrid(-v, v)  # (ny, nx); l descends with x, m ascends with y
    plane = np.exp(-0.5 * (ll**2 / sigma_l_deg**2 + mm**2 / sigma_m_deg**2))
    data = np.repeat(plane[None, None], len(freqs), axis=0).astype(np.float32)
    fits.writeto(path, data, hdr, overwrite=True)
    return str(path)


@pytest.fixture(scope="session")
def fits_beam_cube(tmp_path_factory):
    """A circular Gaussian power beam on a coarser grid than the image."""
    path = tmp_path_factory.mktemp("beam") / "beam.fits"
    return _beam_fits(path, [1.0e9, 1.1e9], 0.05, 0.05)


@pytest.fixture(scope="session")
def fits_beam_cube_elliptical(tmp_path_factory):
    """Elongated along m, so a transposed read is detectable."""
    path = tmp_path_factory.mktemp("beam") / "beam_ell.fits"
    return _beam_fits(path, [1.0e9], 0.02, 0.06)


@pytest.fixture(scope="session")
def two_pointing_fits(tmp_path_factory):
    """Two overlapping pointings sharing a channelisation.

    Returns (model_paths, residual_paths). The pointings are offset in RA by a
    quarter of the image, so they overlap over most of their area.
    """
    folder = tmp_path_factory.mktemp("pointings")
    offsets = [0.0, NPIX // 4 * CELL_DEG]
    models, residuals = [], []
    rng = np.random.default_rng(21)
    for pid, offset in enumerate(offsets):
        for name, cube, out in (
            ("model", _power_law_cube(TRUE_ALPHA, seed=42), models),
            ("residual", rng.normal(scale=1e-3, size=(NCHAN, NPIX, NPIX)), residuals),
        ):
            hdr = _add_per_channel_beams(_base_header(freq_axis=4))
            hdr["CRVAL1"] = 30.0 + offset
            hdr["WSCVWSUM"] = 1.0 + pid
            path = folder / f"field{pid}-{name}.fits"
            out.append(_write(path, _shape_for(np.asarray(cube, dtype=np.float32), 4), hdr))
    return models, residuals
