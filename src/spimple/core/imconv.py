#!/usr/bin/env python

import multiprocessing
from pathlib import Path

from astropy.io import fits
from katbeam import JimBeam
import numpy as np
import pyscilog

from spimple.core.fits import data_from_header, load_fits, save_fits
from spimple.core.utils import convolve2gaussres

pyscilog.init("spimple")
log = pyscilog.get_logger("IMCONV")


def imconv(
    image: list[str],
    output_filename: Path,
    products: str = "i",
    psf_pars: tuple[float, float, float] | None = None,
    nthreads: int | None = None,
    circ_psf: bool = False,
    dilate: float = 1.05,
    beam_model: Path | None = None,
    band: str = "L",
    pb_min: float = 0.05,
    padding_frac: float = 0.5,
    out_dtype: str = "f4",
):
    pyscilog.log_to_file("image_convolver.log")

    if not nthreads:
        nthreads = multiprocessing.cpu_count()

    print("Input Options:", file=log)
    print(f"     {'image':>25} = {image}", file=log)
    print(f"     {'output_filename':>25} = {output_filename}", file=log)
    print(f"     {'products':>25} = {products}", file=log)
    print(f"     {'psf_pars':>25} = {psf_pars}", file=log)
    print(f"     {'nthreads':>25} = {nthreads}", file=log)
    print(f"     {'circ_psf':>25} = {circ_psf}", file=log)
    print(f"     {'dilate':>25} = {dilate}", file=log)
    print(f"     {'beam_model':>25} = {beam_model}", file=log)
    print(f"     {'band':>25} = {band}", file=log)
    print(f"     {'pb_min':>25} = {pb_min}", file=log)
    print(f"     {'padding_frac':>25} = {padding_frac}", file=log)
    print(f"     {'out_dtype':>25} = {out_dtype}", file=log)

    # read coords from fits file
    hdr = fits.getheader(image[0])
    l_coord, ref_l = data_from_header(hdr, axis=1)
    l_coord -= ref_l
    m_coord, ref_m = data_from_header(hdr, axis=2)
    m_coord -= ref_m

    if hdr["NAXIS"] > 2:
        if "CTYPE4" in hdr and hdr["CTYPE4"].lower() == "freq":
            freq_axis = 4
        elif hdr["CTYPE3"].lower() == "freq":
            freq_axis = 3
        else:
            raise ValueError("Freq axis must be 3rd or 4th")
        freqs, ref_freq = data_from_header(hdr, axis=freq_axis)
    else:
        freqs = np.array([0])

    nchan = freqs.size
    gausspari = ()
    if freqs.size > 1:
        for i in range(1, nchan + 1):
            key = "BMAJ" + str(i)
            if key in hdr:
                emaj = hdr[key]
                emin = hdr["BMIN" + str(i)]
                try:
                    pa = hdr["BPA" + str(i)]
                except Exception:
                    pa = hdr["PA" + str(i)]
                gausspari += ((emaj, emin, pa),)
    elif "BMAJ" in hdr:
        emaj = hdr["BMAJ"]
        emin = hdr["BMIN"]
        pa = hdr["BPA"]
        # using key of 1 for consistency with fits standard
        gausspari = ((emaj, emin, pa),)

    if len(gausspari) == 0 and psf_pars is None:
        print("No psf parameters in fits file and none passed in.", file=log)
        raise ValueError("No psf parameters in fits file and none passed in.")

    if len(gausspari) == 0:
        print(
            "No psf parameters in fits file. Convolving model to resolution specified by psf-pars.",
            file=log,
        )
        gaussparf = tuple(psf_pars)
    elif psf_pars is None:  # type: ignore[unreachable]
        gfi = gausspari[0]
        gaussparf = list(gfi)
        # take the largest ones
        for gp in gausspari:
            gaussparf[0] = np.maximum(gaussparf[0], gp[0] * dilate)
            gaussparf[1] = np.maximum(gaussparf[1], gp[1] * dilate)
        gaussparf = tuple(gaussparf)
        if gaussparf[0] > gfi[0] * dilate or gaussparf[1] > gfi[1] * dilate:
            print(
                "Warning - largest clean beam does not correspond to "
                "band 0. You may want to consider removing such bands.",
                file=log,
            )
    else:
        gaussparf = tuple(psf_pars)
        for gp in gausspari:
            if gp[0] > gaussparf[0]:
                raise ValueError("Target resolution cannot be smaller than original. Axis 0")
            if gp[1] > gaussparf[1]:
                raise ValueError("Target resolution cannot be smaller than original. Axis 1")

    if circ_psf:
        e = np.maximum(gaussparf[0], gaussparf[1])
        gaussparf_list = list(gaussparf)
        gaussparf_list[0] = 1.05 * e
        gaussparf_list[1] = 1.05 * e
        gaussparf_list[2] = 0.0
        gaussparf = tuple(gaussparf_list)

    emaj = gaussparf[0]
    emin = gaussparf[1]
    pa = gaussparf[2]
    print(f"Using emaj = {emaj:3.2e}, emin = {emin:3.2e}, PA = {pa:3.2e}", file=log)

    # update header
    if freqs.size > 1:
        for i in range(1, nchan + 1):
            hdr["BMAJ" + str(i)] = gaussparf[0]
            hdr["BMIN" + str(i)] = gaussparf[1]
            hdr["BPA" + str(i)] = gaussparf[2]
    else:
        hdr["BMAJ"] = gaussparf[0]
        hdr["BMIN"] = gaussparf[1]
        hdr["BPA"] = gaussparf[2]

    # coodinate grid
    xx, yy = np.meshgrid(l_coord, m_coord, indexing="ij")

    # convolve image
    imagei = load_fits(image[0], dtype=np.float32).squeeze()
    if imagei.ndim == 2:
        imagei = imagei[None, :, :]
    if imagei.ndim != 3:
        raise ValueError("Unsupported number of image dimensions")
    print("Convolving image", file=log)

    image_out, gausskern = convolve2gaussres(imagei, xx, yy, gaussparf, nthreads, gausspari, padding_frac)

    # load beam and correct
    if beam_model is not None:
        if str(beam_model).endswith(".fits"):
            bhdr = fits.getheader(beam_model)
            l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
            l_coord_beam -= ref_lb
            if not np.array_equal(l_coord_beam, l_coord):
                raise ValueError(
                    "l coordinates of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
            m_coord_beam -= ref_mb
            if not np.array_equal(m_coord_beam, m_coord):
                raise ValueError(
                    "m coordinates of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
            if not np.array_equal(freqs, freqs_beam):
                raise ValueError(
                    "Freqs of beam model do not match those of image. "
                    "Use power_beam_maker to interpolate to fits header."
                )

            beam_image = load_fits(beam_model, dtype=np.float32).squeeze()
        elif str(beam_model) == "JimBeam":
            if band.lower() == "l":
                beam = JimBeam("MKAT-AA-L-JIM-2020")
            elif band.lower() == "uhf":
                beam = JimBeam("MKAT-AA-UHF-JIM-2020")
            elif band.lower() == "s":
                beam = JimBeam("MKAT-AA-S-JIM-2020")
            else:
                raise ValueError(f"Unknown beam model for katbeam in band {band}")

            beam_image = np.zeros(image_out.shape, dtype=out_dtype)
            for v in range(freqs.size):
                beam_image[v] = beam.I(xx, yy, freqs[v] / 1e6)  # freqs in MHz
        else:
            raise ValueError(f"Unknown beam model {beam_model}")

        image_out = np.where(beam_image >= pb_min, image_out / beam_image, 0.0)

    outfile = str(output_filename)

    # save images
    # save clean beam
    if "c" in products:
        name = outfile + ".clean_psf.fits"
        save_fits(name, gausskern, hdr, dtype=out_dtype)
        print(f"Wrote clean psf to {name}", file=log)

    if "i" in products:
        name = outfile + ".convolved.fits"
        save_fits(name, image_out, hdr, dtype=out_dtype)
        print(f"Wrote convolved image to {name}", file=log)

    if "b" in products:
        if "beam_image" not in locals():
            raise ValueError("Cannot write power beam: no beam model provided")
        name = outfile + ".power_beam.fits"
        save_fits(name, beam_image, hdr, dtype=out_dtype)
        print(f"Wrote average power beam to {name}", file=log)

    if "w" in products:
        if "beam_image" not in locals():
            raise ValueError("Cannot write spatial weight: no beam model provided")
        name = outfile + ".spatial_weight.fits"
        save_fits(name, beam_image**2, hdr, dtype=out_dtype)
        print(f"Wrote spatial weight to {name}", file=log)

    print("All done here", file=log)
