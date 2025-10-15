from pathlib import Path
from typing import Annotated, NewType

from hip_cargo import stimela_cab, stimela_output
import typer

File = NewType("File", Path)


@stimela_cab(
    name="imconv",
    info="Homogenise resolution with Gaussian convolution",
    policies={"pass_missing_as_none": True},
)
@stimela_output(name="output_filename", dtype="File", info="{current.output_filename}.fits")
def imconv(
    image: Annotated[list[str], typer.Option(..., help="Image to convolve")],
    output_filename: Annotated[File, typer.Option(..., parser=File, help="Path to output directory")],
    products: Annotated[
        str,
        typer.Option(
            help="Outputs to write. Letters correspond to: \n"
            "c - restoring beam used for convolution \n"
            "i - convolved image \n"
            "b - average power beam \n"
            "w - beam**2 weight image to use for mosaicing"
        ),
    ] = "i",
    psf_pars: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="Beam parameters matching FWHM of restoring beam "
            "specified as emaj emin pa. "
            "By default these are taken from the fits header"
        ),
    ] = None,
    nthreads: Annotated[int | None, typer.Option(help="Number of threads to use. Defaults to all available")] = None,
    circ_psf: Annotated[
        bool, typer.Option("--circ-psf", help="Flag to convolve with a circularised beam instead of an elliptical one")
    ] = False,
    dilate: Annotated[
        float, typer.Option(help="Dilate the psf-pars in fits header by this amount. Sometimes required for stability.")
    ] = 1.05,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=File, help="Fits beam model to use. Use power_beam_maker to make power beam corresponding to image."
        ),
    ] = None,
    band: Annotated[str, typer.Option(help="Band to use with JimBeam. L, UHF or S")] = "L",
    pb_min: Annotated[float, typer.Option(help="Set image to zero where primary beam falls below this value")] = 0.05,
    padding_frac: Annotated[float, typer.Option(help="Padding fraction for FFTs (half on either side)")] = 0.5,
    out_dtype: Annotated[str, typer.Option(help="Data type of output. Default is single precision")] = "f4",
):
    """
    Convolve images to a common resolution with optional primary beam correction.
    """
    # Lazy import the core implementation
    from spimple.core.imconv import imconv as imconv_core

    # Call the core function with all parameters
    imconv_core(
        image=image,
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
    )
