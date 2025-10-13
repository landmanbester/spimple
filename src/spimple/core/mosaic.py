#!/usr/bin/env python

import multiprocessing
from pathlib import Path
import time

from astropy.io import fits
import numpy as np
import pyscilog
import ray

from spimple.fits import set_wcs
from spimple.utils import mosaic_info, project, stitch_images

pyscilog.init("spimple")
log = pyscilog.get_logger("MOSAIC")


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
    # logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"mosaic_{timestamp}.log"
    pyscilog.log_to_file(logname)
    print(f"Logs will be written to {logname}", file=log)

    # ray init
    if not nthreads:
        nthreads = multiprocessing.cpu_count() // 2

    ray.init(
        num_cpus=nworkers,
        logging_level="INFO",
        ignore_reinit_error=True,
        local_mode=debug,
    )

    print("Input Options:", file=log)
    print(f"     {'images':>25} = {images}", file=log)
    print(f"     {'output_filename':>25} = {output_filename}", file=log)
    print(f"     {'beam_model':>25} = {beam_model}", file=log)
    print(f"     {'band':>25} = {band}", file=log)
    print(f"     {'ref_image':>25} = {ref_image}", file=log)
    print(f"     {'padding':>25} = {padding}", file=log)
    print(f"     {'method':>25} = {method}", file=log)
    print(f"     {'nthreads':>25} = {nthreads}", file=log)
    print(f"     {'nworkers':>25} = {nworkers}", file=log)
    print(f"     {'out_dtype':>25} = {out_dtype}", file=log)
    print(f"     {'convolve':>25} = {convolve}", file=log)
    print(f"     {'redo_project':>25} = {redo_project}", file=log)
    print(f"     {'debug':>25} = {debug}", file=log)

    path = Path(output_filename)
    if not path.parent.exists():
        print(f"Creating output directory: {path.parent}", file=log)
        path.parent.mkdir(parents=True, exist_ok=True)

    # project images
    print("Generating reference header", file=log)
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
    print(f"Output image will be of shape ({nchano}, {nxo}, {nyo})", file=log)

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
        print("Projecting images onto common wcs", file=log)
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
                print(f"Completed: {result}", file=log)

    print("Solving linear system", file=log)
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
            print(
                f"Conjugate gradient completed after {info} iterations for freq = {freq}",
                file=log,
            )
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
    print(f"Saved mosaic to {output_filename}", file=log)

    # Save weight map
    weight_filename = output_filename.replace(".fits", "_weights.fits")
    hdu.data = outwgt
    hdu.writeto(weight_filename, overwrite=True)
    print(f"Saved weight map to {weight_filename}", file=log)

    print("Mosaic completed successfully", file=log)

    ray.shutdown()
