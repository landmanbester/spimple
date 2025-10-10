import typer
from typing_extensions import Annotated, Literal
from pathlib import Path
from hip_cargo.callbacks import expand_patterns


def spifit(
    image: Annotated[list[str],
                    typer.Option(...,
                                callback=expand_patterns,
                                help="Image to process")],
    output_filename: Annotated[Path,
                               typer.Option(...,
                                            help="Path to output directory + prefix")],
    residual: Annotated[list[str] | None,
                        typer.Option(callback=expand_patterns,
                                     help="Image to process")] = None,
    psf_pars: Annotated[tuple[float,float,float] | None,
                        typer.Option(help="PSF (beam) parameters matching FWHM of restoring beam specified as "
                                          "emaj emin pa. Taken from the fits header by default.")] = None,
    circ_psf: Annotated[bool, 
                        typer.Option("--circ-psf",
                                     help="Flag to use circular restoring PSF (beam)")] = False,
    threshold: Annotated[float,
                         typer.Option(help="Multiple of the rms in the residual to threshold on. "
                                           "Only components above threshold*rms will be fit.")] = 10,
    maxDR: Annotated[float,
                     typer.Option(help="Maximum dynamic range used to determine the threshold. "
                                       "Only used when residual is not available.")] = 1000,
    nthreads: Annotated[int | None,
                        typer.Option(help="Number of threads to use. Defaults to all")] = None,
    pfb_min: Annotated[float,
                       typer.Option(help="Don't fit components where the primary beam is less than this")] = 0.15,
    products: Annotated[str,
                        typer.Option(help="Outputs to write. Letter correspond to: \n"
                                          "a - alpha map \n"
                                          "e - alpha error map \n"
                                          "i - I0 map \n"
                                          "k - I0 error map \n"
                                          "I - reconstructed cube form alpha and I0 \n"
                                          "c - restoring beam used for convolution \n"
                                          "m - convolved model \n"
                                          "r - convolved residual \n"
                                          "b - average power beam \n"
                                          "d - difference between data and fitted model \n"
                                          "Default is to write all of them")] = "aeikIcmrbd",
    padding_frac: Annotated[float,
                            typer.Option(help="Padding factor for FFT's.")] = 0.5,
    dont_convolve: Annotated[bool,
                             typer.Option("--dont-convolve",
                                          help="Disable convolution with clean PSF (beam)")] = False,
    channel_weights_keyword: Annotated[str,
                                       typer.Option(help="Header for channel weight")] = "WSCIMWG",
    channel_freqs: Annotated[str | None,
                             typer.Option(help="Optional channel frequencies to overwrite fits coordinates."
                                               "Has to be passed in as a comma separated string.")] = None,
    ref_freq: Annotated[float | None,
                        typer.Option(help="Optional reference frequency to overwrite default taken from fits")] = None,
    out_dtype: Annotated[str,
                         typer.Option(help="dtype of output images")] = "f4",
    add_convolved_residuals: Annotated[bool,
                                       typer.Option("--add_convolved_residuals",
                                                    help="Flag to add the convolved residuals to the "
                                                         "convolved model")] = False,
    ms: Annotated[Path | None,
                  typer.Option(help="Optional path to MS used to get the paralactic angle rotation")] = None,
    beam_model: Annotated[Path | None,
                          typer.Option(help="Beam model to use. This can be provided as fits files in which case"
                                            "it assumes the path/to/beam_folder/name_corr_re/im.fits pattern. "
                                            "Also accepts JimBeam in which case it will get the beam from katbeam.")] = None,
    sparsify_time: Annotated[int,
                             typer.Option(help="Subsample PA by this many integrations when computing PA during "
                                               "beam interpolation.")] = 10,
    corr_type: Annotated[Literal["linear","circular"],
                         typer.Option(help="Correlation type")] = "linear",
    band: Annotated[str,
                    typer.Option(help="Band to use with JimBeam. L, UHF or S")] = "L",
    deselect_bands: Annotated[str | None,
                              typer.Option(help="Optional bands to select. "
                                                "Has to be passed in as a comma separated string")] = None,
):
    print(image)
    print(output_filename)
    print(psf_pars)
    pass
