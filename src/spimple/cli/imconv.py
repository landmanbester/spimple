from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListStr,
    StimelaMeta,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

File = NewType("File", Path)


@stimela_cab(
    name="imconv",
    info="Convolve images to a common resolution. Deprecated in favour of spimple init.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="convolved-image",
    info="Convolved image, written when products contains 'i'.",
    implicit="{current.output-filename}.convolved.fits",
)
def imconv(
    images: Annotated[
        ListStr,
        typer.Option(
            ...,
            parser=parse_list_str,
            help="Images to convolve",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Basename of output products",
        ),
    ],
    products: Annotated[
        str,
        typer.Option(
            help="Outputs to write, as a string of letters. "
            "c is the restoring beam used for convolution. "
            "i is the convolved image. "
            "b is the average power beam. "
            "w is the beam-squared weight image used for mosaicing.",
        ),
    ] = "i",
    psf_pars: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="Beam parameters matching FWHM of restoring beam specified as emaj emin pa. "
            "By default these are taken from the fits header",
        ),
    ] = None,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads to use. Defaults to all available",
        ),
    ] = None,
    circ_psf: Annotated[
        bool,
        typer.Option(
            help="Flag to convolve with a circularised beam instead of an elliptical one",
        ),
    ] = False,
    dilate: Annotated[
        float,
        typer.Option(
            help="Dilate the psf-pars in fits header by this amount. Sometimes required for stability.",
        ),
    ] = 1.05,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Fits beam model to use. Use binterp to make a power beam matching the image.",
        ),
    ] = None,
    band: Annotated[
        str,
        typer.Option(
            help="Band to use with JimBeam. L, UHF or S",
        ),
    ] = "L",
    pb_min: Annotated[
        float,
        typer.Option(
            help="Set image to zero where primary beam falls below this value",
        ),
    ] = 0.05,
    padding_frac: Annotated[
        float,
        typer.Option(
            help="Padding fraction for FFTs (half on either side)",
        ),
    ] = 0.5,
    out_dtype: Annotated[
        str,
        typer.Option(
            help="Data type of output. Default is single precision",
        ),
    ] = "f4",
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Convolve images to a common resolution.
    Deprecated in favour of spimple init.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                imconv,
                dict(
                    images=images,
                    output_filename=output_filename,
                    products=products,
                    psf_pars=psf_pars,
                    nthreads=nthreads,
                    circ_psf=circ_psf,
                    dilate=dilate,
                    beam_model=beam_model,
                    band=band,
                    pb_min=pb_min,
                    padding_frac=padding_frac,
                    out_dtype=out_dtype,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.imconv import imconv as imconv_core  # noqa: E402

            # Call the core function with all parameters
            imconv_core(
                images,
                output_filename,
                products=products,
                psf_pars=psf_pars,
                nthreads=nthreads,
                circ_psf=circ_psf,
                dilate=dilate,
                beam_model=beam_model,
                band=band,
                pb_min=pb_min,
                padding_frac=padding_frac,
                out_dtype=out_dtype,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("spimple")
    if image is None:
        raise RuntimeError("No Container URL in spimple metadata.")

    run_in_container(
        imconv,
        dict(
            images=images,
            output_filename=output_filename,
            products=products,
            psf_pars=psf_pars,
            nthreads=nthreads,
            circ_psf=circ_psf,
            dilate=dilate,
            beam_model=beam_model,
            band=band,
            pb_min=pb_min,
            padding_frac=padding_frac,
            out_dtype=out_dtype,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
