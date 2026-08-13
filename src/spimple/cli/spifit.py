from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListInt,
    StimelaMeta,
    parse_list_int,
    parse_upath,
    stimela_cab,
    stimela_output,
)

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="spifit",
    info="Fit a spectral index model to the band images of a datatree.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="alpha-map",
    info="Fitted spectral index map, written when products contains 'a'.",
    implicit="{current.output-filename}_time0.alpha.fits",
)
def spifit(
    store: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Datatree store to fit. Written by spimple init or by pfb-imaging",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Basename of the output FITS products",
        ),
    ],
    flux_scale: Annotated[
        Literal["apparent", "intrinsic", "mixed"],
        typer.Option(
            ...,
            help="Flux scale to fit. Apparent uses BIMAGE. Intrinsic uses IMAGE. Mixed uses KIMAGE",
        ),
    ],
    products: Annotated[
        str,
        typer.Option(
            help="Products to write, as a string of letters. "
            "a is the alpha map. "
            "e is the alpha error map. "
            "i is the I0 map. "
            "k is the I0 error map. "
            "I is the reconstructed cube. "
            "d is the difference between the data and the fit. "
            "b is the average power beam.",
        ),
    ] = "aeikId",
    threshold: Annotated[
        float,
        typer.Option(
            help="Multiple of the residual rms below which pixels are not fitted",
        ),
    ] = 10.0,
    max_dr: Annotated[
        float,
        typer.Option(
            help="Maximum dynamic range used to set the threshold when no rms is available",
        ),
    ] = 1000.0,
    pb_min: Annotated[
        float,
        typer.Option(
            help="Beam floor below which pixels are excluded from the fit",
        ),
    ] = 0.15,
    deselect_bands: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Band ids to exclude from the fit",
        ),
    ] = None,
    ref_freq: Annotated[
        float | None,
        typer.Option(
            help="Reference frequency in Hz. Defaults to the weighted mean of the band frequencies",
        ),
    ] = None,
    timeid: Annotated[
        int | None,
        typer.Option(
            help="Restrict the fit to one time id. Every time id is fitted by default",
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
    Fit a spectral index model to the band images of a datatree.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                spifit,
                dict(
                    store=store,
                    output_filename=output_filename,
                    flux_scale=flux_scale,
                    products=products,
                    threshold=threshold,
                    max_dr=max_dr,
                    pb_min=pb_min,
                    deselect_bands=deselect_bands,
                    ref_freq=ref_freq,
                    timeid=timeid,
                    out_dtype=out_dtype,
                    nthreads=nthreads,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.spifit import spifit as spifit_core  # noqa: E402

            # Call the core function with all parameters
            spifit_core(
                store,
                output_filename,
                flux_scale,
                products=products,
                threshold=threshold,
                max_dr=max_dr,
                pb_min=pb_min,
                deselect_bands=deselect_bands,
                ref_freq=ref_freq,
                timeid=timeid,
                out_dtype=out_dtype,
                nthreads=nthreads,
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
        spifit,
        dict(
            store=store,
            output_filename=output_filename,
            flux_scale=flux_scale,
            products=products,
            threshold=threshold,
            max_dr=max_dr,
            pb_min=pb_min,
            deselect_bands=deselect_bands,
            ref_freq=ref_freq,
            timeid=timeid,
            out_dtype=out_dtype,
            nthreads=nthreads,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
