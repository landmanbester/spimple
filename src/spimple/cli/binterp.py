from pathlib import Path
from typing import Annotated, Literal

from hip_cargo import stimela_cab, stimela_output
from hip_cargo.callbacks import expand_patterns
import typer


@stimela_cab(
    name="binterp",
    info="Interpolate and create power beam.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(name="output_filename", dtype="File", info="{current.output_filename}.fits")
def binterp(
    image: Annotated[
        list[str],
        typer.Option(..., callback=expand_patterns, help="A fits image providing the coordinates to interpolate to"),
    ],
    output_filename: Annotated[Path, typer.Option(..., help="Path to output directory")],
    ms: Annotated[
        str | None,
        typer.Option(
            help="Measurement sets used to make the image. "
            "Used to get paralactic angles if doing primary beam correction. "
            "Pass as comma-separated string."
        ),
    ] = None,
    field: Annotated[int, typer.Option(help="Field ID")] = 0,
    beam_model: Annotated[
        str | None,
        typer.Option(
            help="Fits beam model to use. "
            "It is assumed that the pattern is path_to_beam/name_corr_re/im.fits. "
            "Provide only the path up to name e.g. /home/user/beams/meerkat_lband. "
            "Patterns matching corr are determined automatically. "
            "Only real and imaginary beam models currently supported."
        ),
    ] = None,
    sparsify_time: Annotated[int, typer.Option(help="Used to select a subset of time")] = 10,
    nthreads: Annotated[int | None, typer.Option(help="Number of threads to use. Defaults to all available")] = None,
    corr_type: Annotated[
        Literal["linear", "circular"], typer.Option(help="Correlation type i.e. linear or circular")
    ] = "linear",
):
    """
    Interpolate a primary beam model onto the coordinate grid of a FITS image.

    This tool extracts spatial and frequency coordinates from an input FITS image,
    interpolates the primary beam pattern using optional measurement set and beam
    model information, and saves the resulting beam cube to the specified output file.
    """
    # Lazy import the core implementation
    from spimple.core.binterp import binterp as binterp_core

    # Parse ms if provided as comma-separated string
    ms_list = None
    if ms is not None:
        ms_list = [x.strip() for x in ms.split(",")]

    # Convert Path types to strings for core function
    beam_model_str = str(beam_model) if beam_model is not None else None
    output_filename_str = str(output_filename)

    # Call the core function with all parameters
    binterp_core(
        image=image,
        output_filename=output_filename_str,
        ms=ms_list,
        field=field,
        beam_model=beam_model_str,
        sparsify_time=sparsify_time,
        nthreads=nthreads,
        corr_type=corr_type,
    )
