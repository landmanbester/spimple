from pathlib import Path
from typing import Annotated, Literal, NewType

from hip_cargo import stimela_cab, stimela_output
from hip_cargo.callbacks import expand_patterns
import typer

File = NewType("File", Path)
MS = NewType("MS", Path)


@stimela_cab(
    name="spifit",
    info="Fit spectral index map.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(name="output_filename", dtype="File", info="Path to output directory + prefix", required=True)
def spifit(
    image: Annotated[list[str], typer.Option(..., callback=expand_patterns, help="Image to process")],
    output_filename: Annotated[File, typer.Option(..., parser=File, help="Path to output directory + prefix")],
    residual: Annotated[list[str] | None, typer.Option(callback=expand_patterns, help="Image to process")] = None,
    psf_pars: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="PSF (beam) parameters matching FWHM of restoring beam specified as "
            "emaj emin pa. Taken from the fits header by default."
        ),
    ] = None,
    circ_psf: Annotated[bool, typer.Option("--circ-psf", help="Flag to use circular restoring PSF (beam)")] = False,
    threshold: Annotated[
        float,
        typer.Option(
            help="Multiple of the rms in the residual to threshold on. Only components above threshold*rms will be fit."
        ),
    ] = 10,
    maxDR: Annotated[
        float,
        typer.Option(
            help="Maximum dynamic range used to determine the threshold. Only used when residual is not available."
        ),
    ] = 1000,
    nthreads: Annotated[int | None, typer.Option(help="Number of threads to use. Defaults to all")] = None,
    pb_min: Annotated[float, typer.Option(help="Don't fit components where the primary beam is less than this")] = 0.15,
    products: Annotated[
        str,
        typer.Option(
            help="""
Outputs to write. Letter correspond to:

a - alpha map

e - alpha error map

i - I0 map

k - I0 error map

I - reconstructed cube form alpha and I0

c - restoring beam used for convolution

m - convolved model

r - convolved residual

b - average power beam

d - difference between data and fitted model

Default is to write all of them
"""
        ),
    ] = "aeikIcmrbd",
    padding_frac: Annotated[float, typer.Option(help="Padding factor for FFT's.")] = 0.5,
    dont_convolve: Annotated[
        bool, typer.Option("--dont-convolve", help="Disable convolution with clean PSF (beam)")
    ] = False,
    channel_weights_keyword: Annotated[str, typer.Option(help="Header keyword for channel weight")] = "WSCIMWG",
    channel_freqs: Annotated[
        str | None,
        typer.Option(
            help="Optional channel frequencies to overwrite fits coordinates. "
            "Has to be passed in as a comma separated string. "
            "Stimela dtype: List[float]"
        ),
    ] = None,
    ref_freq: Annotated[
        float | None, typer.Option(help="Optional reference frequency to overwrite default taken from fits")
    ] = None,
    out_dtype: Annotated[str, typer.Option(help="dtype of output images")] = "f4",
    add_convolved_residuals: Annotated[
        bool,
        typer.Option("--add_convolved_residuals", help="Flag to add the convolved residuals to the convolved model"),
    ] = False,
    ms: Annotated[
        MS | None, typer.Option(parser=MS, help="Optional path to MS used to get the paralactic angle rotation")
    ] = None,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=File,
            help="Beam model to use. This can be provided as fits files in which case"
            "it assumes the path/to/beam_folder/name_corr_re/im.fits pattern. "
            "Also accepts JimBeam in which case it will get the beam from katbeam.",
        ),
    ] = None,
    sparsify_time: Annotated[
        int, typer.Option(help="Subsample PA by this many integrations when computing PA during beam interpolation.")
    ] = 10,
    corr_type: Annotated[Literal["linear", "circular"], typer.Option(help="Correlation type")] = "linear",
    band: Annotated[str, typer.Option(help="Band to use with JimBeam. L, UHF or S")] = "L",
    deselect_bands: Annotated[
        str | None, typer.Option(help="Optional bands to select. Has to be passed in as a comma separated string."
                                      "Stimela dtype: List[int]")
    ] = None,
):
    """
    Fit spectral index models to image cubes with optional convolution and primary beam correction.
    """
    # Lazy import the core implementation
    from spimple.core.spifit import spifit as spifit_core

    # Parse channel_freqs if provided as comma-separated string
    channel_freqs_list = None
    if channel_freqs is not None:
        channel_freqs_list = [float(x.strip()) for x in channel_freqs.split(",")]

    # Parse deselect_bands if provided as comma-separated string
    deselect_bands_list = None
    if deselect_bands is not None:
        deselect_bands_list = [int(x.strip()) for x in deselect_bands.split(",")]

    # Call the core function with all parameters
    spifit_core(
        image=image,
        output_filename=output_filename,
        residual=residual,
        psf_pars=psf_pars,
        circ_psf=circ_psf,
        threshold=threshold,
        maxDR=maxDR,
        nthreads=nthreads,
        pb_min=pb_min,
        products=products,
        padding_frac=padding_frac,
        dont_convolve=dont_convolve,
        channel_weights_keyword=channel_weights_keyword,
        channel_freqs=channel_freqs_list,
        ref_freq=ref_freq,
        out_dtype=out_dtype,
        add_convolved_residuals=add_convolved_residuals,
        ms=ms,
        beam_model=beam_model,
        sparsify_time=sparsify_time,
        corr_type=corr_type,
        band=band,
        deselect_bands=deselect_bands_list,
    )
