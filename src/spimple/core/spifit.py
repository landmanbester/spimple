#!/usr/bin/env python

import multiprocessing

import numpy as np

from spimple.utils.datatree import (
    PRODUCT_VARS,
    band_nodes,
    check_homogeneous,
    open_store,
    partition_nodes,
    timeids,
)
from spimple.utils.fits import save_fits, set_wcs
from spimple.utils.logging import get_logger, log_options

log = get_logger("SPIFIT")

_SCALES = {"apparent": "a", "intrinsic": "i", "mixed": "k"}


def spifit(
    store: str,
    output_filename: str,
    flux_scale: str,
    products: str = "aeikId",
    threshold: float = 10.0,
    max_dr: float = 1000.0,
    pb_min: float = 0.15,
    deselect_bands: list[int] | None = None,
    ref_freq: float | None = None,
    timeid: int | None = None,
    out_dtype: str = "f4",
    nthreads: int | None = None,
):
    """
    Fit a spectral index model to the band images of a datatree.

    The tree must already be homogenised to a single resolution, by spimple init
    or by pfb restore with a target resolution. Multi partition trees written by
    spimple init must be combined with spimple mosaic first.

    flux_scale has no default deliberately: the three scales carry different
    physical meanings and which products a tree even holds depends on how it was
    made, so the caller states which one they are fitting rather than having one
    chosen for them.
    """
    log_options(log, **locals())

    import dask.array as da
    from africanus.model.spi.dask import fit_spi_components

    if not nthreads:
        nthreads = multiprocessing.cpu_count()
    if flux_scale not in _SCALES:
        raise ValueError(f"Unknown flux-scale {flux_scale}, expected one of {sorted(_SCALES)}")
    column = PRODUCT_VARS[_SCALES[flux_scale]]

    store = str(store)
    output_filename = str(output_filename)
    dt = open_store(store)
    wanted = timeids(dt) if timeid is None else [timeid]
    if not wanted:
        raise ValueError(f"{store} has no band nodes")

    for tid in wanted:
        nodes = band_nodes(dt, timeid=tid)
        dropped = set(deselect_bands or ())
        keep = [n for n in nodes if int(dt[n].ds.attrs["bandid"]) not in dropped]
        if dropped:
            log.info("Dropping bands %s", sorted(dropped))

        # A fully flagged band carries WSUM == 0. pfb leaves such nodes in the
        # tree and its restore skips them, so they may not even carry the
        # product variable; keeping one would either abort the fit below or
        # give a dead band a weight.
        dead = [n for n in keep if "WSUM" in dt[n].ds and not float(np.sum(dt[n].ds.WSUM.values)) > 0]
        if dead:
            log.info(
                "Skipping fully flagged bands %s (WSUM == 0)",
                [int(dt[n].ds.attrs["bandid"]) for n in dead],
            )
            keep = [n for n in keep if n not in set(dead)]
        if not keep:
            raise ValueError(f"No bands left at timeid {tid} after dropping deselected and fully flagged bands")

        for node in keep:
            if column not in dt[node].ds:
                available = sorted(v for v in PRODUCT_VARS.values() if v in dt[node].ds)
                if partition_nodes(dt, node) and not available:
                    hint = "run spimple mosaic to combine its partitions"
                elif available:
                    scales = ", ".join(
                        f"{name} for --flux-scale {scale}"
                        for scale, name in (("apparent", "BIMAGE"), ("intrinsic", "IMAGE"), ("mixed", "KIMAGE"))
                        if name in available
                    )
                    hint = (
                        f"the tree carries {scales}. Note pfb restore defaults to --outputs kK, "
                        "which writes KIMAGE only"
                    )
                else:
                    hint = "the tree carries no restored product at all"
                raise ValueError(f"{node} has no {column}; {hint}")
        datasets = [dt[n].ds for n in keep]
        if len(datasets) < 2:
            raise ValueError("Can't produce alpha map from a single band image")

        psfparsf = check_homogeneous(datasets)
        log.info("Fitting at a resolution of %s pixels", psfparsf[0])

        for node in keep:
            if any(dt[f"{node}/{p}"].ds.attrs.get("beam_includes_n") for p in partition_nodes(dt, node)):
                log.warning(
                    "BEAM in this tree is B/n, not the bare primary beam (beam_includes_n is set). "
                    "The fitted flux is biased by n, about 0.2 percent at a 5 degree field edge. "
                    "Correct with B = BEAM * n; see docs/wiki/design-decisions.md"
                )
                break

        cube = np.stack([ds[column].values for ds in datasets])  # (nband, ncorr, ny, nx)
        beam = np.stack([ds.BEAM.values for ds in datasets])
        wsums = np.stack([np.atleast_1d(ds.WSUM.values) for ds in datasets])
        freqs = np.array([float(ds.attrs["freq_out"]) for ds in datasets])
        nband, ncorr, ny, nx = cube.shape
        ref = datasets[0]
        cell_deg = np.rad2deg(float(ref.attrs["cell_rad"]))

        # a local, so a second timeid does not inherit the first one's default
        nu_ref = ref_freq if ref_freq is not None else float(np.sum(freqs * wsums[:, 0]) / np.sum(wsums[:, 0]))
        log.info("Reference frequency is %3.2e Hz", nu_ref)

        # band weights: WSUM, else 1/RMS^2, else equal
        if np.all(wsums[:, 0] > 0):
            weights = wsums[:, 0] / wsums[:, 0].max()
        elif all("RMS" in ds for ds in datasets):
            rms = np.array([float(np.atleast_1d(ds.RMS.values)[0]) for ds in datasets])
            weights = np.where(rms > 0, 1.0 / rms**2, 0.0)
            weights /= weights.max()
        else:
            weights = np.ones(nband, dtype=np.float64)
        log.info("Channel weights: %s", weights)

        # threshold: the band RMS if the tree carries it, else a normalised
        # RESIDUAL, else a dynamic range cut
        rms_values = None
        source = None
        if all("RMS" in ds for ds in datasets):
            rms_values = np.array([float(np.atleast_1d(ds.RMS.values)[0]) for ds in datasets])
            source = "the stored RMS"
        elif all("RESIDUAL" in ds for ds in datasets):
            rms_values = np.array(
                [float(np.std(ds.RESIDUAL.values / np.atleast_1d(ds.WSUM.values)[:, None, None])) for ds in datasets]
            )
            source = "the stored RESIDUAL"
        if rms_values is not None:
            rms_mfs = float(np.sum(rms_values * weights) / weights.sum())
            threshold_val = threshold * rms_mfs
            log.info("Threshold is %s times the rms from %s", threshold, source)
        else:
            finite = cube[np.isfinite(cube)]
            threshold_val = float(finite.max()) / max_dr if finite.size else 0.0
            log.info("No rms available. Setting threshold from a max dynamic range of %s", max_dr)
        log.info("Threshold set to %s Jy", threshold_val)

        # the fit runs on Stokes I; other correlations are carried in the header only
        image = cube[:, 0]
        pbeam = beam[:, 0]
        fit_beam = np.ones_like(pbeam) if flux_scale == "intrinsic" else pbeam

        masked = np.where(pbeam > pb_min, image, np.nan)
        minimage = np.nanmin(masked, axis=0)
        maskindices = np.argwhere(np.isfinite(minimage) & (minimage > threshold_val))
        if not maskindices.size:
            raise ValueError(
                "No components found above threshold. Try lowering your threshold. "
                f"Max of the image is {np.nanmax(image):3.2e}"
            )
        fitcube = image[:, maskindices[:, 0], maskindices[:, 1]].T
        beam_comps = fit_beam[:, maskindices[:, 0], maskindices[:, 1]].T

        ncomps = fitcube.shape[0]
        cchunks = max(1, ncomps // nthreads)
        log.info("Fitting %s components", ncomps)
        alpha, alpha_err, i0, i0_err = fit_spi_components(
            da.from_array(fitcube.astype(np.float64), chunks=(cchunks, nband)),
            da.from_array(weights.astype(np.float64), chunks=(nband,)),
            da.from_array(freqs.astype(np.float64), chunks=(nband,)),
            np.float64(nu_ref),
            beam=da.from_array(beam_comps.astype(np.float64), chunks=(cchunks, nband)),
        ).compute()
        log.info("Done. Writing output.")

        maps = {}
        for name, values in (
            ("alpha", alpha),
            ("alpha_err", alpha_err),
            ("I0", i0),
            ("I0_err", i0_err),
        ):
            plane = np.full((ny, nx), np.nan, dtype=np.float64)
            plane[maskindices[:, 0], maskindices[:, 1]] = values
            maps[name] = plane

        irec = maps["I0"][None] * (freqs[:, None, None] / nu_ref) ** maps["alpha"][None]
        fit_diff = np.full_like(irec, np.nan)
        rows, cols = maskindices[:, 0], maskindices[:, 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            fit_diff[:, rows, cols] = image[:, rows, cols] / np.where(
                fit_beam[:, rows, cols] > 0, fit_beam[:, rows, cols], np.nan
            )
        fit_diff[:, rows, cols] -= irec[:, rows, cols]

        radec = (float(ref.attrs["ra"]), float(ref.attrs["dec"]))
        time_out = ref.attrs.get("time_out")

        def _write(letter, suffix, data, freq, ref=ref, tid=tid, psfparsf=psfparsf, radec=radec, time_out=time_out):
            """Write one product. data is always (nband, ncorr, ny, nx)."""
            if letter not in products:
                return
            hdr = set_wcs(
                cell_deg,
                cell_deg,
                nx,
                ny,
                radec,
                freq,
                unit="Jy/beam",
                gausspar=psfparsf[0],
                ms_time=time_out,
                time_is_unix=True,
                l0=float(ref.attrs.get("l0", 0.0)),
                m0=float(ref.attrs.get("m0", 0.0)),
            )
            name = f"{output_filename}_time{tid}.{suffix}.fits"
            save_fits(name, data, hdr, dtype=out_dtype, yx_order=True)
            log.info("Wrote %s", name)

        _write("a", "alpha", maps["alpha"][None, None], nu_ref)
        _write("e", "alpha_err", maps["alpha_err"][None, None], nu_ref)
        _write("i", "I0", maps["I0"][None, None], nu_ref)
        _write("k", "I0_err", maps["I0_err"][None, None], nu_ref)
        _write("I", "Irec_cube", irec[:, None], freqs)
        _write("d", "fit_diff", fit_diff[:, None], freqs)
        _write("b", "power_beam", pbeam[:, None], freqs)

    log.info("All done here")
