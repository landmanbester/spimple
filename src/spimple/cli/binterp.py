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
    name="binterp",
    info="Interpolate and create power beam.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="power-beam",
    info="Interpolated power beam cube on the image coordinate grid.",
    implicit="{current.output-filename}",
)
def binterp(
    images: Annotated[
        ListStr,
        typer.Option(
            ...,
            parser=parse_list_str,
            help="Fits images providing the coordinates to interpolate to",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Path of the output beam cube",
        ),
    ],
    ms: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="Measurement sets used to make the image. "
            "Used to get parallactic angles when doing primary beam correction.",
        ),
    ] = None,
    field: Annotated[
        int,
        typer.Option(
            help="Field ID",
        ),
    ] = 0,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Fits beam model to use. "
            "The expected on-disk pattern is path_to_beam/name_corr_re.fits and the matching _im.fits. "
            "Provide only the path up to name, for example /home/user/beams/meerkat_lband. "
            "Patterns matching corr are determined automatically. "
            "Only real and imaginary beam models are currently supported.",
        ),
    ] = None,
    sparsify_time: Annotated[
        int,
        typer.Option(
            help="Used to select a subset of time",
        ),
    ] = 10,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads to use. Defaults to all available",
        ),
    ] = None,
    corr_type: Annotated[
        Literal["linear", "circular"],
        typer.Option(
            help="Correlation type i.e. linear or circular",
        ),
    ] = "linear",
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
    Interpolate and create power beam.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                binterp,
                dict(
                    images=images,
                    output_filename=output_filename,
                    ms=ms,
                    field=field,
                    beam_model=beam_model,
                    sparsify_time=sparsify_time,
                    nthreads=nthreads,
                    corr_type=corr_type,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.binterp import binterp as binterp_core  # noqa: E402

            # Call the core function with all parameters
            binterp_core(
                images,
                output_filename,
                ms=ms,
                field=field,
                beam_model=beam_model,
                sparsify_time=sparsify_time,
                nthreads=nthreads,
                corr_type=corr_type,
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
        binterp,
        dict(
            images=images,
            output_filename=output_filename,
            ms=ms,
            field=field,
            beam_model=beam_model,
            sparsify_time=sparsify_time,
            nthreads=nthreads,
            corr_type=corr_type,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
