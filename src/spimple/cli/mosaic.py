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
    name="mosaic",
    info="Reproject and combine multiple images into a mosaic.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="mosaic-image",
    info="Mosaicked image on the common coordinate grid.",
    implicit="{current.output-filename}",
)
def mosaic(
    images: Annotated[
        ListStr,
        typer.Option(
            ...,
            parser=parse_list_str,
            help="List of FITS images to mosaic together",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Path of the output mosaic FITS file",
        ),
    ],
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Beam dataset to apply. Use binterp to make a power beam matching the images.",
        ),
    ] = None,
    band: Annotated[
        str,
        typer.Option(
            help="Band to use with JimBeam. L, UHF or S",
        ),
    ] = "L",
    ref_image: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Reference image defining the output coordinate system. "
            "An optimal reference is derived when this is not provided.",
        ),
    ] = None,
    padding: Annotated[
        float,
        typer.Option(
            help="Padding factor for FFTs.",
        ),
    ] = 0.1,
    method: Annotated[
        Literal["interp", "adaptive", "exact"],
        typer.Option(
            help="Reprojection method, see reproject for details.",
        ),
    ] = "interp",
    nthreads: Annotated[
        int,
        typer.Option(
            help="Number of threads to use per worker.",
        ),
    ] = 1,
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of workers to use for parallel processing.",
        ),
    ] = 1,
    out_dtype: Annotated[
        str,
        typer.Option(
            help="Data type of output. Default is single precision",
        ),
    ] = "f4",
    convolve: Annotated[
        bool,
        typer.Option(
            help="Flag to convolve images to common resolution before projection. "
            "If no psf-pars are passed in the lowest resolution will be determined automatically.",
        ),
    ] = False,
    redo_project: Annotated[
        bool,
        typer.Option(
            help="Force re-projection even if output exists.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            help="Run everything in local mode to assist with debugging.",
        ),
    ] = False,
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
    Reproject and combine multiple images into a mosaic.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                mosaic,
                dict(
                    images=images,
                    output_filename=output_filename,
                    beam_model=beam_model,
                    band=band,
                    ref_image=ref_image,
                    padding=padding,
                    method=method,
                    nthreads=nthreads,
                    nworkers=nworkers,
                    out_dtype=out_dtype,
                    convolve=convolve,
                    redo_project=redo_project,
                    debug=debug,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.mosaic import mosaic as mosaic_core  # noqa: E402

            # Call the core function with all parameters
            mosaic_core(
                images,
                output_filename,
                beam_model=beam_model,
                band=band,
                ref_image=ref_image,
                padding=padding,
                method=method,
                nthreads=nthreads,
                nworkers=nworkers,
                out_dtype=out_dtype,
                convolve=convolve,
                redo_project=redo_project,
                debug=debug,
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
        mosaic,
        dict(
            images=images,
            output_filename=output_filename,
            beam_model=beam_model,
            band=band,
            ref_image=ref_image,
            padding=padding,
            method=method,
            nthreads=nthreads,
            nworkers=nworkers,
            out_dtype=out_dtype,
            convolve=convolve,
            redo_project=redo_project,
            debug=debug,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
