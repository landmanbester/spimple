"""Combine image-space partitions into one band image.

Each partition p carries an apparent image A_p = B_p * S on the union grid,
masked by its own footprint. The optimal linear combination minimises
sum_p ||A_p - B_p S||^2, whose normal equations are

    (sum_p B_p^2 + eta) S = sum_p B_p A_p

The operator is diagonal -- purely elementwise -- so the exact solution is one
division. ``conjugate_gradient`` is kept for the day a non-diagonal term is
added, and ``tests/test_mosaic.py`` pins the two against each other.

Arrays are (npart, ncorr, ny, nx) with (npart, ny, nx) masks.
"""

import numpy as np

from spimple.utils.logging import get_logger

log = get_logger("MOSAIC")


def stitch_band(
    apparent: np.ndarray,
    beams: np.ndarray,
    masks: np.ndarray,
    mixed: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    wsums: np.ndarray | None = None,
    eta: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Combine one band's partitions into the band-level products.

    Args:
        apparent: (npart, ncorr, ny, nx) apparent images, the partitions' BIMAGE.
        beams: (npart, ncorr, ny, nx) primary beams.
        masks: (npart, ny, nx) boolean footprints.
        mixed: (npart, ncorr, ny, nx) partitions' KIMAGE, or None to skip it.
        rms: (npart, ncorr) per-partition residual rms, or None to skip it.
        wsums: (npart, ncorr) per-partition weight sums, or None to skip it.
        eta: Tikhonov floor, keeping the solve finite where no partition covers.

    Returns:
        A dict with IMAGE, BEAM, BIMAGE and SPATIALWGT, plus KIMAGE, RMS and
        WSUM when their inputs were supplied.
    """
    mask = masks[:, None, :, :]  # broadcast the footprint over correlations
    weighted_beam = beams * mask
    beam_sum = weighted_beam.sum(axis=0)
    beam_sq = (weighted_beam**2).sum(axis=0)
    rhs = (weighted_beam * apparent).sum(axis=0)

    spatial = beam_sq + eta
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(spatial > 0, rhs / spatial, 0.0)
        # beam-weighted mean response: reduces to B when the partitions agree,
        # and to B_p where only partition p covers
        beam = np.where(beam_sum > 0, beam_sq / beam_sum, 0.0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    beam = np.nan_to_num(beam, nan=0.0, posinf=0.0, neginf=0.0)

    out = {
        "IMAGE": image,
        "BEAM": beam,
        "BIMAGE": beam * image,
        "SPATIALWGT": spatial,
    }
    if mixed is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            km = np.where(beam_sum > 0, (weighted_beam * mixed).sum(axis=0) / beam_sum, 0.0)
        out["KIMAGE"] = np.nan_to_num(km, nan=0.0, posinf=0.0, neginf=0.0)
    if wsums is not None:
        out["WSUM"] = np.asarray(wsums, dtype=np.float64).sum(axis=0)
    if rms is not None:
        # propagate through the same beam weighting the images went through, so
        # spifit's threshold survives the mosaic. Weighted by each partition's
        # peak response, which is the scalar analogue of the per-pixel weighting.
        rms = np.asarray(rms, dtype=np.float64)
        peak = np.array([np.nanmax(beams[p], axis=(-2, -1)) for p in range(beams.shape[0])])
        denom = peak.sum(axis=0)
        num = np.sqrt(((peak * rms) ** 2).sum(axis=0))
        out["RMS"] = np.where(denom > 0, num / denom, 0.0)
    return out


def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=100, report=20):
    """Solve A x = b for a symmetric positive-definite operator A.

    Retained for a future non-diagonal formulation; ``stitch_band`` solves the
    current diagonal system directly.
    """
    n = b.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    r = A(x) - b
    p = -r
    rsold = np.vdot(r, r)
    rs0 = rsold
    if rs0 < tol:
        return x, 0  # already at minimum
    i = 0
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / np.vdot(p, Ap)
        x = x + alpha * p
        r = r + alpha * Ap
        rsnew = np.vdot(r, r)

        if np.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = beta * p - r
        rsold = rsnew

        if i % report == 0:
            log.debug("At %d norm frac = %s", i, rsnew / rs0)

    return x, i
