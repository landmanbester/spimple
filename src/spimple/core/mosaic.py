#!/usr/bin/env python

import multiprocessing
from pathlib import Path

import numpy as np

from spimple.utils.datatree import PRODUCT_VARS, band_nodes, open_store, partition_nodes, write_node
from spimple.utils.logging import get_logger, log_options
from spimple.utils.mosaic import stitch_band
from spimple.utils.render import dt2fits

log = get_logger("MOSAIC")


def mosaic(
    store: str,
    output_filename: str | None = None,
    eta: float = 1e-3,
    products: str = "aik",
    fits_outputs: str = "I",
    fits_output_folder: str | None = None,
    out_dtype: str = "f4",
    nthreads: int | None = None,
    nworkers: int = 1,
):
    """
    Combine the image space partitions of a datatree into band mean images.

    Only meaningful for a store written by spimple init from more than one
    pointing. A tree from pfb-imaging mosaics in visibility space, so its band
    nodes arrive already populated and there is nothing here to combine.
    """
    log_options(log, **locals())

    if not nthreads:
        nthreads = multiprocessing.cpu_count()

    store = str(store)
    dt = open_store(store)
    nodes = band_nodes(dt)
    if not nodes:
        raise ValueError(f"{store} has no band nodes")

    letters = tuple(k for k in ("a", "i", "k") if k in products)
    combined = 0
    for node in nodes:
        # A pfb tree has partition children too, but they hold visibility-space
        # arrays and a BEAM -- never the image-space products with a footprint
        # mask that this command combines. Test for what we consume, not for
        # the mere presence of a child.
        parts = [
            p
            for p in partition_nodes(dt, node)
            if "MASK" in dt[f"{node}/{p}"].ds and PRODUCT_VARS["a"] in dt[f"{node}/{p}"].ds
        ]
        if not parts:
            raise ValueError(
                f"{store}/{node} has no image-space partitions to combine and its band products are "
                "already populated; a pfb-imaging tree mosaics in visibility space and needs no "
                "spimple mosaic step"
            )
        datasets = [dt[f"{node}/{p}"].ds for p in parts]

        apparent = np.stack([ds[PRODUCT_VARS["a"]].values for ds in datasets])
        beams = np.stack([ds.BEAM.values for ds in datasets])
        masks = np.stack([ds.MASK.values[0] for ds in datasets])
        mixed = np.stack([ds[PRODUCT_VARS["k"]].values for ds in datasets]) if "k" in letters else None
        wsums = np.stack([np.atleast_1d(ds.WSUM.values) for ds in datasets])
        rms = np.stack([np.atleast_1d(ds.RMS.values) for ds in datasets]) if "RMS" in datasets[0] else None

        out = stitch_band(apparent, beams, masks, mixed=mixed, rms=rms, wsums=wsums, eta=eta)

        ref = dt[node].ds
        data_vars = {
            "IMAGE": (("corr", "y", "x"), out["IMAGE"].astype(out_dtype)),
            "BIMAGE": (("corr", "y", "x"), out["BIMAGE"].astype(out_dtype)),
            "BEAM": (("corr", "y", "x"), out["BEAM"].astype(out_dtype)),
            "SPATIALWGT": (("corr", "y", "x"), out["SPATIALWGT"].astype(out_dtype)),
            "WSUM": (("corr",), out["WSUM"].astype(np.float64)),
            "PSFPARSF": (("corr", "bpar"), ref.PSFPARSF.values),
        }
        if "KIMAGE" in out:
            data_vars["KIMAGE"] = (("corr", "y", "x"), out["KIMAGE"].astype(out_dtype))
        if "RMS" in out:
            data_vars["RMS"] = (("corr",), out["RMS"].astype(np.float64))

        write_node(
            store,
            node,
            data_vars,
            {"eta": float(eta), "nparts": len(parts)},
            {"corr": list(ref.corr.values), "bpar": list(ref.bpar.values)},
        )
        combined += 1
        log.info("Combined %d partitions for %s", len(parts), node)

    log.info("Combined %d bands", combined)

    if fits_outputs:
        base = output_filename or str(Path(store).with_suffix(""))
        folder = fits_output_folder or str(Path(store).parent / "fits")
        Path(folder).mkdir(parents=True, exist_ok=True)
        oname = f"{folder}/{Path(base).name}"
        for letter in ("a", "i", "k"):
            if letter in fits_outputs or letter.upper() in fits_outputs:
                dt2fits(
                    store,
                    PRODUCT_VARS[letter],
                    oname,
                    otype=out_dtype,
                    do_mfs=letter in fits_outputs,
                    do_cube=letter.upper() in fits_outputs,
                )

    log.info("Mosaic completed successfully")
