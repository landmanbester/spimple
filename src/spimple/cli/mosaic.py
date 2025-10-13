from pathlib import Path
from typing import Annotated, Literal

from hip_cargo import stimela_cab, stimela_output
from hip_cargo.callbacks import expand_patterns
import typer


@stimela_cab(
    name="mosaic",
    info="Reproject and combine multiple images into a mosaic.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(name="output_filename", dtype="File", info="{current.output_filename}.fits")
def mosaic(
    images: Annotated[
        list[str], typer.Option(..., callback=expand_patterns, help="List of FITS images to mosaic together")
    ],
    output_filename: Annotated[Path, typer.Option(..., help="Path to output mosaic FITS file")],
    beam_model: Annotated[
        str | None,
        typer.Option(help="Fits beam model to use. Use power_beam_maker to make power beam corresponding to image."),
    ] = None,
    band: Annotated[str, typer.Option(help="Band to use with JimBeam. L, UHF or S")] = "L",
    ref_image: Annotated[
        str | None,
        typer.Option(
            help="Reference image to define the output coordinate system. "
            "If not provided, an optimal reference will be attempted."
        ),
    ] = None,
    padding: Annotated[float, typer.Option(help="Padding factor for FFTs.")] = 0.1,
    method: Annotated[
        Literal["interp", "adaptive", "exact"], typer.Option(help="Reprojection method, see reproject for details.")
    ] = "interp",
    nthreads: Annotated[int, typer.Option(help="Number of threads to use per worker.")] = 1,
    nworkers: Annotated[int, typer.Option(help="Number of workers to use for parallel processing.")] = 1,
    out_dtype: Annotated[str, typer.Option(help="Data type of output. Default is single precision")] = "f4",
    convolve: Annotated[
        bool,
        typer.Option(
            "--convolve",
            help="Flag to convolve images to common resolution before projection. "
            "If no psf-pars are passed in the lowest resolution will be determined "
            "automatically.",
        ),
    ] = False,
    redo_project: Annotated[
        bool, typer.Option("--redo-project", help="Force re-projection even if output exists.")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Run everything in local mode to assist with debugging.")
    ] = False,
):
    """
    Mosaic multiple FITS images together onto a common coordinate grid.

    This tool takes multiple FITS images and combines them into a single mosaic
    image using interpolation to handle different coordinate systems and spatial
    coverage.
    """
    # Lazy import the core implementation
    from spimple.core.mosaic import mosaic as mosaic_core

    # Convert Path to string for core function
    output_filename_str = str(output_filename)

    # Call the core function with all parameters
    mosaic_core(
        images=images,
        output_filename=output_filename_str,
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
