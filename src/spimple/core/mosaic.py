#!/usr/bin/env python

import multiprocessing
from pathlib import Path

from astropy.io import fits
import numpy as np
import ray

from spimple.utils.fits import expand_image_patterns, set_wcs
from spimple.utils.logging import get_logger, log_options
from spimple.utils.mosaic import mosaic_info, project, stitch_images

log = get_logger("MOSAIC")


def mosaic(
    images: list[str],
    output_filename: str,
    beam_model: str | None = None,
    band: str = "L",
    ref_image: str | None = None,
    padding: float = 0.1,
    method: str = "interp",
    nthreads: int = 1,
    nworkers: int = 1,
    out_dtype: str = "f4",
    convolve: bool = False,
    redo_project: bool = False,
    debug: bool = False,
):
    """
    Mosaic multiple FITS images together onto a common coordinate grid.

    This function takes multiple FITS images and combines them into a single
    mosaic image using interpolation to handle different coordinate systems
    and spatial coverage.
    """
    log_options(log, **locals())

    images = expand_image_patterns(images)

    # ray init
    if not nthreads:
        nthreads = multiprocessing.cpu_count() // 2

    ray.init(
        num_cpus=nworkers,
        logging_level="INFO",
        ignore_reinit_error=True,
        local_mode=debug,
    )

    path = Path(output_filename)
    if not path.parent.exists():
        log.info("Creating output directory: %s", path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)

    # project images
    log.info("Generating reference header")
    if isinstance(images, str):
        image_list = sorted(Path().glob(images))
    else:
        image_list = []
        for img in images:
            imgs = sorted(Path().glob(img))
            if not imgs:
                raise RuntimeError(f"Nothing found at {img}")
            image_list.extend(imgs)

    ref_wcs, ufreqs, out_names = mosaic_info(image_list, output_filename)

    nyo, nxo = ref_wcs.array_shape
    nchano = ufreqs.size
    log.info("Output image will be of shape (%s, %s, %s)", nchano, nxo, nyo)

    # check if projection has been done
    do_project = False
    if not redo_project:
        for name in out_names:
            if not Path(name).is_dir():
                do_project = True
                break
    else:
        do_project = True

    if do_project:
        log.info("Projecting images onto common wcs")
        tasks = []
        for imnum, im in enumerate(image_list):
            fut = project.remote(im, imnum, ref_wcs, beam_model, output_filename)
            tasks.append(fut)

        # Process tasks as they complete
        remaining_tasks = tasks.copy()
        while remaining_tasks:
            # Wait for at least 1 task to complete
            ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

            # Process the completed task
            for task in ready:
                result = ray.get(task)
                log.info("Completed: %s", result)

    log.info("Solving linear system")
    outim = np.zeros((nchano, nxo, nyo))
    outwgt = np.zeros((nchano, nxo, nyo))
    tasks = []
    for freq in ufreqs:
        fut = stitch_images.remote(freq, out_names)
        tasks.append(fut)

    # Process tasks as they complete
    remaining_tasks = tasks.copy()
    while remaining_tasks:
        # Wait for at least 1 task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)

        # Process the completed task
        for task in ready:
            image, weight, info, freq = ray.get(task)
            log.info("Conjugate gradient completed after %s iterations for freq = %s", info, freq)
            c = np.nonzero(ufreqs == freq)[0]
            outim[c] = image
            outwgt[c] = outwgt

    # Create output header
    cell_x = np.abs(ref_wcs.wcs.cdelt[0])
    cell_y = np.abs(ref_wcs.wcs.cdelt[1])
    ra = ref_wcs.wcs.crval[0] * np.pi / 180
    dec = ref_wcs.wcs.crval[1] * np.pi / 180
    out_hdr = set_wcs(
        cell_x,
        cell_y,
        nxo,
        nyo,
        (ra, dec),
        ufreqs,
        unit="Jy/beam",
        GuassPar=None,
        ms_time=None,
        header=True,
        casambm=False,
    )

    # Save output

    hdu = fits.PrimaryHDU(header=out_hdr)
    hdu.data = outim
    hdu.writeto(output_filename, overwrite=True)
    log.info("Saved mosaic to %s", output_filename)

    # Save weight map
    weight_filename = output_filename.replace(".fits", "_weights.fits")
    hdu.data = outwgt
    hdu.writeto(weight_filename, overwrite=True)
    log.info("Saved weight map to %s", weight_filename)

    log.info("Mosaic completed successfully")

    ray.shutdown()
