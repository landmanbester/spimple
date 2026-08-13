"""Restoration arithmetic for one image grid.

Pure array functions -- no tree I/O, no FITS, no Ray. ``core/init.py`` owns
reading the FITS and writing the store.

Flux scales (see the pfb wiki's D22/D23): the model is intrinsic flux while the
residual is apparent, so a restored image must say which scale it is on. Three
are offered, keyed by their CLI letter and named as pfb's restore names them.
"""

import numpy as np

from spimple.utils.convolution import convolve2gaussres


def restore_products(
    model: np.ndarray,
    residual: np.ndarray,
    beam: np.ndarray,
    gaussparf: np.ndarray,
    gausspari: np.ndarray | None = None,
    products: tuple[str, ...] = ("k",),
    pb_min: float = 0.1,
    nthreads: int = 1,
    padding_frac: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build the restored images for one image grid.

    Args:
        model: (ncorr, ny, nx) intrinsic model in Jy/pixel. Pass zeros when the
            input is a restored image rather than a model.
        residual: (ncorr, ny, nx) apparent residual already in Jy/beam.
        beam: (ncorr, ny, nx) primary beam response.
        gaussparf: (ncorr, 3) target resolution in pixels, pixels, radians.
        gausspari: (ncorr, 3) intrinsic resolution of the residual, or None to
            skip the reconvolution. None is correct whenever gaussparf already
            is the residual's own resolution.
        products: Any of "a" (apparent), "i" (intrinsic) and "k" (intrinsic
            model plus apparent residual).
        pb_min: Beam floor below which the intrinsic image is zeroed.
        nthreads: Threads for the convolution FFTs.
        padding_frac: FFT padding fraction.

    Returns:
        A dict mapping each requested key to an (ncorr, ny, nx) array.
    """
    model = np.asarray(model)
    residual = np.asarray(residual)
    beam = np.asarray(beam)
    gaussparf = np.asarray(gaussparf, dtype=float)
    if not products:
        return {}

    _, ny, nx = model.shape
    # grids stay x-major; yx_order transposes the (corr, y, x) images onto them
    x = -(nx // 2) + np.arange(nx)
    y = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    kw = {"nthreads": nthreads, "pfrac": padding_frac, "norm_kernel": False, "yx_order": True}

    if gausspari is None or np.allclose(gaussparf, np.asarray(gausspari, dtype=float)):
        rconv = residual
    else:
        rconv, _ = convolve2gaussres(residual, xx, yy, gaussparf, gausspari=np.asarray(gausspari, dtype=float), **kw)

    out: dict[str, np.ndarray] = {}
    if "i" in products or "k" in products:
        # the model carries no intrinsic resolution -- the clean-component assumption
        mconv, _ = convolve2gaussres(model, xx, yy, gaussparf, **kw)
    if "k" in products:
        out["k"] = np.ascontiguousarray(mconv + rconv)
    if "i" in products:
        with np.errstate(invalid="ignore", divide="ignore"):
            rint = np.where(beam > pb_min, rconv / beam, 0.0)
        out["i"] = np.ascontiguousarray(np.where(beam > pb_min, mconv + rint, 0.0))
    if "a" in products:
        # (B*m) convolved, NOT B*(m convolved): convolution does not commute
        # with multiplication by a spatially varying beam, so mconv cannot be reused
        aconv, _ = convolve2gaussres(beam * model, xx, yy, gaussparf, **kw)
        out["a"] = np.ascontiguousarray(aconv + rconv)
    return out
