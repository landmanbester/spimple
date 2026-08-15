"""Weighted least-squares fit of a spectral index model.

Vendored from ``africanus.model.spi`` so the weights can be per component as
well as per channel. The model is

    I(nu) = A(nu) I0 (nu / nu0) ** alpha

and the kernel minimises ``sum_v w[comp, v] (data[comp, v] - model[v]) ** 2``
by Gauss-Newton.

Why the weights need a component axis: image plane noise is flat in the
*apparent* image, so the intrinsic image ``IMAGE = BIMAGE / BEAM`` has variance
``sigma_v ** 2 / B(v, p) ** 2``. The inverse variance weight is therefore
``B(v, p) ** 2 / sigma_v ** 2``, which depends on both position and frequency
because the beam narrows with frequency. It does not factorise, so a
``(chan,)`` weight vector cannot express it. With those weights, fitting the
intrinsic image is algebraically identical to fitting the apparent image with
the beam in the model:

    sum_v w_v (d_app - B_v M_v) ** 2 = sum_v (w_v B_v ** 2) (d_int - M_v) ** 2

which ``tests/test_fit_spi.py`` pins to machine precision.
"""

import numpy as np
from africanus.util.numba import jit
from dask.array.core import blockwise


@jit(nopython=True, nogil=True, cache=True)
def _fit_spi_components_impl(data, weights, freqs, freq0, out, jac, beam, ncomps, nfreqs, tol, maxiter, mindet):
    w = freqs / freq0
    dof = np.maximum(w.size - 2, 1)
    for comp in range(ncomps):
        eps = 1.0
        k = 0
        alphak = out[0, comp]
        i0k = out[2, comp]
        b = beam[comp]
        wgt = weights[comp]
        while eps > tol and k < maxiter:
            alphap = alphak
            i0p = i0k
            jac[1, :] = b * w**alphak
            model = i0k * jac[1, :]
            jac[0, :] = model * np.log(w)
            residual = data[comp] - model
            lik = 0.0
            hess00 = 0.0
            hess01 = 0.0
            hess11 = 0.0
            jr0 = 0.0
            jr1 = 0.0
            for v in range(nfreqs):
                lik += residual[v] * wgt[v] * residual[v]
                jr0 += jac[0, v] * wgt[v] * residual[v]
                jr1 += jac[1, v] * wgt[v] * residual[v]
                hess00 += jac[0, v] * wgt[v] * jac[0, v]
                hess01 += jac[0, v] * wgt[v] * jac[1, v]
                hess11 += jac[1, v] * wgt[v] * jac[1, v]
            det = np.maximum(hess00 * hess11 - hess01**2, mindet)
            alphak = alphap + (hess11 * jr0 - hess01 * jr1) / det
            i0k = i0p + (-hess01 * jr0 + hess00 * jr1) / det
            eps = np.maximum(np.abs(alphak - alphap), np.abs(i0k - i0p))
            k += 1
        out[0, comp] = alphak
        out[1, comp] = hess11 / det * lik / dof
        out[2, comp] = i0k
        out[3, comp] = hess00 / det * lik / dof
    return out


def fit_spi_components_np(
    data: np.ndarray,
    weights: np.ndarray,
    freqs: np.ndarray,
    freq0: float,
    alphai: np.ndarray | None = None,
    I0i: np.ndarray | None = None,
    beam: np.ndarray | None = None,
    tol: float = 1e-4,
    maxiter: int = 100,
) -> np.ndarray:
    """Fit alpha and I0 per component.

    Args:
        data: (comps, chan) spectrum per component.
        weights: (chan,) or (comps, chan) inverse variance. A 1d array is
            broadcast over components.
        freqs: (chan,) frequencies.
        freq0: reference frequency.
        alphai: (comps,) initial alphas, defaulting to -0.7.
        I0i: (comps,) initial intensities, defaulting to the band nearest freq0.
        beam: (comps, chan) attenuation applied to the model, defaulting to one.
        tol: absolute convergence tolerance on alpha and I0.
        maxiter: Gauss-Newton iteration cap.

    Returns:
        (4, comps) array of [alphas, alphavars, I0s, I0vars]. The variances are
        rescaled by the reduced chi squared, so only the relative scale of
        ``weights`` matters.

    Raises:
        ValueError: if data is not float32 or float64, or weights does not
            broadcast onto data.
    """
    ncomps, nfreqs = data.shape
    if beam is None:
        beam = np.ones(data.shape, data.dtype)
    weights = np.asarray(weights, dtype=data.dtype)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, data.shape)
    if weights.shape != data.shape:
        raise ValueError(f"weights of shape {weights.shape} does not match data of shape {data.shape}")
    # numba indexes this row by row; a broadcast view has a zero stride
    weights = np.ascontiguousarray(weights)
    jac = np.zeros((2, nfreqs), dtype=data.dtype)
    out = np.zeros((4, ncomps), dtype=data.dtype)
    if alphai is not None:
        out[0, :] = alphai
    else:
        out[0, :] = -0.7 * np.ones(ncomps, dtype=data.dtype)
    if I0i is not None:
        out[2, :] = I0i
    else:
        tmp = np.abs(freqs - freq0)
        ref_freq_idx = np.argwhere(tmp == tmp.min()).squeeze()
        if np.size(ref_freq_idx) > 1:
            ref_freq_idx = ref_freq_idx.min()
        out[2, :] = data[:, ref_freq_idx] / beam[:, ref_freq_idx]
    if data.dtype == np.float64:
        mindet = 1e-12
    elif data.dtype == np.float32:
        mindet = 1e-5
    else:
        raise ValueError("Unsupported data type. Must be float32 of float64.")

    return _fit_spi_components_impl(
        data,
        weights,
        freqs,
        freq0,
        out,
        jac,
        beam,
        ncomps,
        nfreqs,
        tol,
        maxiter,
        mindet,
    )


def _fit_spi_components_wrapper(data, weights, freqs, freq0, alphai, I0i, beam, tol, maxiter):
    return fit_spi_components_np(
        data[0],
        weights[0],
        freqs[0],
        freq0,
        alphai,
        I0i,
        beam[0] if beam is not None else beam,
        tol=tol,
        maxiter=maxiter,
    )


def fit_spi_components(data, weights, freqs, freq0, alphai=None, I0i=None, beam=None, tol=1e-5, maxiter=100):
    """Dask wrapper for :func:`fit_spi_components_np`.

    ``weights`` may be chunked as ``(chan,)`` or, to weight each component
    separately, as ``(comps, chan)`` matching ``data``.
    """
    return blockwise(
        _fit_spi_components_wrapper,
        ("vars", "comps"),
        data,
        ("comps", "chan"),
        weights,
        ("comps", "chan") if weights.ndim == 2 else ("chan",),
        freqs,
        ("chan",),
        freq0,
        None,
        alphai,
        ("comps",) if alphai is not None else None,
        I0i,
        ("comps",) if I0i is not None else None,
        beam,
        ("comps", "chan") if beam is not None else None,
        tol,
        None,
        maxiter,
        None,
        new_axes={"vars": 4},
        dtype=data.dtype,
    )
