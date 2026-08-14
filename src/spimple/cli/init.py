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

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="init",
    info="Ingest FITS images into a pfb-imaging style datatree",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="Directory",
    name="datatree",
    info="Datatree store consumed by spifit and mosaic.",
    implicit="{current.output-filename}_I.dt",
)
def init(
    images: Annotated[
        ListStr,
        typer.Option(
            ...,
            parser=parse_list_str,
            help="List of FITS model or restored image files to ingest",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Basename of the output datatree store",
        ),
    ],
    residual: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="List of FITS residual files matching the images",
        ),
    ] = None,
    psf_pars: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="Target resolution as emaj emin pa in degrees. By default the lowest resolution of the inputs is used",
        ),
    ] = None,
    circ_psf: Annotated[
        bool,
        typer.Option(
            help="Force a circular target beam",
        ),
    ] = False,
    dilate: Annotated[
        float,
        typer.Option(
            help="Safety factor applied when the target resolution is derived from the inputs",
        ),
    ] = 1.05,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Primary beam to apply. Use JimBeam, a bds zarr store or a FITS cube",
        ),
    ] = None,
    band: Annotated[
        Literal["L", "UHF", "S"],
        typer.Option(
            help="Band to use with JimBeam. L, UHF or S",
        ),
    ] = "L",
    pb_min: Annotated[
        float,
        typer.Option(
            help="Beam floor below which the intrinsic image is zeroed",
        ),
    ] = 0.15,
    padding_frac: Annotated[
        float,
        typer.Option(
            help="Padding used for the FFT based convolution",
        ),
    ] = 0.5,
    products: Annotated[
        str,
        typer.Option(
            help="Restored products to store. a is apparent. i is intrinsic. k is mixed",
        ),
    ] = "aik",
    channel_weights_keyword: Annotated[
        str,
        typer.Option(
            help="Header keyword supplying the per band weight sum",
        ),
    ] = "WSCVWSUM",
    freq_tol: Annotated[
        float | None,
        typer.Option(
            help="Frequencies within this many Hz are treated as one band",
        ),
    ] = None,
    fits_outputs: Annotated[
        str,
        typer.Option(
            help="Products to render as FITS. Lowercase is MFS and uppercase is a cube",
        ),
    ] = "",
    fits_output_folder: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Folder the rendered FITS are written to",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Replace an existing datatree store",
        ),
    ] = False,
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
    Ingest FITS images into a pfb-imaging style datatree
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                init,
                dict(
                    images=images,
                    output_filename=output_filename,
                    residual=residual,
                    psf_pars=psf_pars,
                    circ_psf=circ_psf,
                    dilate=dilate,
                    beam_model=beam_model,
                    band=band,
                    pb_min=pb_min,
                    padding_frac=padding_frac,
                    products=products,
                    channel_weights_keyword=channel_weights_keyword,
                    freq_tol=freq_tol,
                    fits_outputs=fits_outputs,
                    fits_output_folder=fits_output_folder,
                    overwrite=overwrite,
                    out_dtype=out_dtype,
                    nthreads=nthreads,
                    nworkers=nworkers,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.init import init as init_core  # noqa: E402

            # Call the core function with all parameters
            init_core(
                images,
                output_filename,
                residual=residual,
                psf_pars=psf_pars,
                circ_psf=circ_psf,
                dilate=dilate,
                beam_model=beam_model,
                band=band,
                pb_min=pb_min,
                padding_frac=padding_frac,
                products=products,
                channel_weights_keyword=channel_weights_keyword,
                freq_tol=freq_tol,
                fits_outputs=fits_outputs,
                fits_output_folder=fits_output_folder,
                overwrite=overwrite,
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
        init,
        dict(
            images=images,
            output_filename=output_filename,
            residual=residual,
            psf_pars=psf_pars,
            circ_psf=circ_psf,
            dilate=dilate,
            beam_model=beam_model,
            band=band,
            pb_min=pb_min,
            padding_frac=padding_frac,
            products=products,
            channel_weights_keyword=channel_weights_keyword,
            freq_tol=freq_tol,
            fits_outputs=fits_outputs,
            fits_output_folder=fits_output_folder,
            overwrite=overwrite,
            out_dtype=out_dtype,
            nthreads=nthreads,
            nworkers=nworkers,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
