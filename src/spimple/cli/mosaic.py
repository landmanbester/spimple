from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="mosaic",
    info="Combine the partitions of a datatree into band mean images.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="datatree",
    info="Datatree store with the band mean images populated.",
    implicit="{current.store}",
)
def mosaic(
    store: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Datatree store written by spimple init",
        ),
    ],
    output_filename: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Basename for the rendered FITS files",
        ),
    ] = None,
    eta: Annotated[
        float,
        typer.Option(
            help="Tikhonov floor keeping the solve finite where no partition covers",
        ),
    ] = 0.001,
    products: Annotated[
        str,
        typer.Option(
            help="Products to combine. a is apparent. i is intrinsic. k is mixed",
        ),
    ] = "aik",
    fits_outputs: Annotated[
        str,
        typer.Option(
            help="Products to render as FITS. Lowercase is MFS and uppercase is a cube",
        ),
    ] = "I",
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Folder the rendered FITS are written to",
        ),
    ] = None,
    out_dtype: Annotated[
        str,
        typer.Option(
            help="Data type of output. Default is single precision",
        ),
    ] = "f4",
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads to use per worker",
        ),
    ] = None,
    nworkers: Annotated[
        int,
        typer.Option(
            help="Number of workers to use for parallel processing",
        ),
    ] = 1,
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
    Combine the partitions of a datatree into band mean images.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                mosaic,
                dict(
                    store=store,
                    output_filename=output_filename,
                    eta=eta,
                    products=products,
                    fits_outputs=fits_outputs,
                    fits_output_folder=fits_output_folder,
                    out_dtype=out_dtype,
                    nthreads=nthreads,
                    nworkers=nworkers,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.mosaic import mosaic as mosaic_core  # noqa: E402

            # Call the core function with all parameters
            mosaic_core(
                store,
                output_filename=output_filename,
                eta=eta,
                products=products,
                fits_outputs=fits_outputs,
                fits_output_folder=fits_output_folder,
                out_dtype=out_dtype,
                nthreads=nthreads,
                nworkers=nworkers,
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
            store=store,
            output_filename=output_filename,
            eta=eta,
            products=products,
            fits_outputs=fits_outputs,
            fits_output_folder=fits_output_folder,
            out_dtype=out_dtype,
            nthreads=nthreads,
            nworkers=nworkers,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
