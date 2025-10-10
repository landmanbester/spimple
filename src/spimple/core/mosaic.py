#!/usr/bin/env python

import pyscilog

pyscilog.init("spimple")
log = pyscilog.get_logger("MOSAIC")
import argparse
import multiprocessing
from pathlib import Path
import time

from astropy.io import fits
import numpy as np
from omegaconf import OmegaConf
import ray

from spimple.fits import set_wcs
from spimple.utils import mosaic_info, project, stitch_images, str2bool


def mosaic():
    """
    Mosaic multiple FITS images together onto a common coordinate grid.

    This command-line tool takes multiple FITS images and combines them into
    a single mosaic image using interpolation to handle different coordinate
    systems and spatial coverage.
    """
    parser = argparse.ArgumentParser(
        description="Mosaic multiple FITS images together.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-images",
        "--images",
        nargs="+",
        type=str,
        required=True,
        help="List of FITS images to mosaic together",
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        type=str,
        required=True,
        help="Path to output mosaic FITS file",
    )
    parser.add_argument(
        "-bm",
        "--beam-model",
        default=None,
        type=str,
        help="Fits beam model to use. \n"
        "Use power_beam_maker to make power beam "
        "corresponding to image. ",
    )
    parser.add_argument(
        "-band",
        "--band",
        type=str,
        default="l",
        help="Band to use with JimBeam. L, UHF or S",
    )
    parser.add_argument(
        "-ref-image",
        "--ref-image",
        type=str,
        default=None,
        help="Reference image to define the output coordinate system. \n"
        "If not provided, an optimal reference will be attempted.",
    )
    parser.add_argument(
        "-padding",
        "--padding",
        type=float,
        default=0.1,
        help="Padding factor for FFTs. \nDefault is 0.1 (10%% padding).",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        default="interp",
        choices=["interp", "adaptive", "exact"],
        help="Reprojection method, see reproject for details. \n"
        "Options: interp, adaptive, exact",
    )
    parser.add_argument(
        "-nthreads",
        "--nthreads",
        default=1,
        type=int,
        help="Number of threads to use per worker.",
    )
    parser.add_argument(
        "-nworkers",
        "--nworkers",
        default=1,
        type=int,
        help="Number of workers to use for parallel processing.",
    )
    parser.add_argument(
        "-otype",
        "--out_dtype",
        default="f4",
        type=str,
        help="Data type of output. Default is single precision",
    )
    parser.add_argument(
        "-convolve",
        "--convolve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Flag to convolve images to common resolution before projection. \n"
        "If no psf-pars are passed in the lowest resolution will be determined "
        "automatically. ",
    )
    parser.add_argument(
        "-redo",
        "--redo-project",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Force re-projection even if output exists. ",
    )
    parser.add_argument(
        "-debug",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run everyting in local mode to assist with debugging.",
    )

    opts = parser.parse_args()
    opts = OmegaConf.create(vars(opts))
    OmegaConf.set_struct(opts, True)

    # logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logname = f"mosaic_{timestamp}.log"
    pyscilog.log_to_file(logname)
    print(f"Logs will be written to {logname}", file=log)

    # ray init
    if not opts.nthreads:
        opts.nthreads = multiprocessing.cpu_count() // 2

    ray.init(
        num_cpus=opts.nworkers,
        logging_level="INFO",
        ignore_reinit_error=True,
        local_mode=opts.debug,
    )

    print("Input Options:", file=log)
    for key, val in opts.items():
        print(f"     {key:>25} = {val}", file=log)

    path = Path(opts.output_filename)
    if not path.parent.exists():
        print(f"Creating output directory: {path.parent}", file=log)
        path.parent.mkdir(parents=True, exist_ok=True)

    # project images
    print("Generating reference header", file=log)
    if isinstance(opts.images, str):
        image_list = sorted(Path().glob(opts.images))
    else:
        image_list = []
        for images in opts.images:
            imgs = sorted(Path().glob(images))
            if not imgs:
                raise RuntimeError(f"Nothing found at {images}")
            image_list.extend(imgs)

    ref_wcs, ufreqs, out_names = mosaic_info(image_list, opts.output_filename)

    nyo, nxo = ref_wcs.array_shape
    nchano = ufreqs.size
    print(f"Output image will be of shape ({nchano}, {nxo}, {nyo})", file=log)

    # check if projection has been done
    do_project = False
    if not opts.redo_project:
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
            fut = project.remote(
                im, imnum, ref_wcs, opts.beam_model, opts.output_filename
            )
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
                f"Conjugate gradient completed after {info} "
                f"iterations for freq = {freq}",
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
    hdu.writeto(opts.output_filename, overwrite=True)
    print(f"Saved mosaic to {opts.output_filename}", file=log)

    # Save weight map
    weight_filename = opts.output_filename.replace(".fits", "_weights.fits")
    hdu.data = outwgt
    hdu.writeto(weight_filename, overwrite=True)
    print(f"Saved weight map to {weight_filename}", file=log)

    print("Mosaic completed successfully", file=log)

    ray.shutdown()
