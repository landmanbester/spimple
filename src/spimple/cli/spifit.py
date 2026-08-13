from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListFloat,
    ListInt,
    ListStr,
    StimelaMeta,
    parse_list_float,
    parse_list_int,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

File = NewType("File", Path)


@stimela_cab(
    name="spifit",
    info="Fit spectral index map.",
    policies={"pass_missing_as_none": True},
)
@stimela_output(
    dtype="File",
    name="alpha-map",
    info="Fitted spectral index map, written when products contains 'a'.",
    implicit="{current.output-filename}.alpha.fits",
)
def spifit(
    images: Annotated[
        ListStr,
        typer.Option(
            ...,
            parser=parse_list_str,
            help="Images to process",
        ),
    ],
    output_filename: Annotated[
        File,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Basename of output products",
        ),
    ],
    residual: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="Residual images matching the input images",
        ),
    ] = None,
    psf_pars: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            help="PSF (beam) parameters matching FWHM of restoring beam specified as emaj emin pa. "
            "Taken from the fits header by default.",
        ),
    ] = None,
    circ_psf: Annotated[
        bool,
        typer.Option(
            help="Flag to use circular restoring PSF (beam)",
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            help="Multiple of the rms in the residual to threshold on. "
            "Only components above threshold*rms will be fit.",
        ),
    ] = 10,
    max_dr: Annotated[
        float,
        typer.Option(
            help="Maximum dynamic range used to determine the threshold. Only used when residual is not available.",
        ),
    ] = 1000,
    nthreads: Annotated[
        int | None,
        typer.Option(
            help="Number of threads to use. Defaults to all",
        ),
    ] = None,
    pb_min: Annotated[
        float,
        typer.Option(
            help="Don't fit components where the primary beam is less than this",
        ),
    ] = 0.15,
    products: Annotated[
        str,
        typer.Option(
            help="Outputs to write, as a string of letters. "
            "a is the alpha map. "
            "e is the alpha error map. "
            "i is the I0 map. "
            "k is the I0 error map. "
            "I is the cube reconstructed from alpha and I0. "
            "c is the restoring beam used for convolution. "
            "m is the convolved model. "
            "r is the convolved residual. "
            "b is the average power beam. "
            "d is the difference between data and fitted model.",
        ),
    ] = "aeikIcmrbd",
    padding_frac: Annotated[
        float,
        typer.Option(
            help="Padding factor for FFT's.",
        ),
    ] = 0.5,
    dont_convolve: Annotated[
        bool,
        typer.Option(
            help="Disable convolution with clean PSF (beam)",
        ),
    ] = False,
    channel_weights_keyword: Annotated[
        str,
        typer.Option(
            help="Header for channel weight",
        ),
    ] = "WSCIMWG",
    channel_freqs: Annotated[
        ListFloat | None,
        typer.Option(
            parser=parse_list_float,
            help="Optional channel frequencies overriding the fits coordinates.",
        ),
    ] = None,
    ref_freq: Annotated[
        float | None,
        typer.Option(
            help="Optional reference frequency to overwrite default taken from fits",
        ),
    ] = None,
    out_dtype: Annotated[
        str,
        typer.Option(
            help="dtype of output images",
        ),
    ] = "f4",
    add_convolved_residuals: Annotated[
        bool,
        typer.Option(
            help="Flag to add the convolved residuals to the convolved model",
        ),
    ] = False,
    ms: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="Optional measurement sets used to get the parallactic angle rotation.",
        ),
    ] = None,
    beam_model: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="Beam model to use. "
            "For fits files the expected pattern is path/to/beam_folder/name_corr_re.fits and its _im.fits pair. "
            "JimBeam is also accepted, in which case the beam comes from katbeam.",
        ),
    ] = None,
    sparsify_time: Annotated[
        int,
        typer.Option(
            help="Subsample PA by this many integrations when computing PA during beam interpolation.",
        ),
    ] = 10,
    corr_type: Annotated[
        Literal["linear", "circular"],
        typer.Option(
            help="Correlation type",
        ),
    ] = "linear",
    band: Annotated[
        str,
        typer.Option(
            help="Band to use with JimBeam. L, UHF or S",
        ),
    ] = "L",
    deselect_bands: Annotated[
        ListInt | None,
        typer.Option(
            parser=parse_list_int,
            help="Optional bands to discard from the fit.",
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
    Fit spectral index map.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                spifit,
                dict(
                    images=images,
                    output_filename=output_filename,
                    residual=residual,
                    psf_pars=psf_pars,
                    circ_psf=circ_psf,
                    threshold=threshold,
                    max_dr=max_dr,
                    nthreads=nthreads,
                    pb_min=pb_min,
                    products=products,
                    padding_frac=padding_frac,
                    dont_convolve=dont_convolve,
                    channel_weights_keyword=channel_weights_keyword,
                    channel_freqs=channel_freqs,
                    ref_freq=ref_freq,
                    out_dtype=out_dtype,
                    add_convolved_residuals=add_convolved_residuals,
                    ms=ms,
                    beam_model=beam_model,
                    sparsify_time=sparsify_time,
                    corr_type=corr_type,
                    band=band,
                    deselect_bands=deselect_bands,
                ),
            )

            # Lazy import the core implementation
            from spimple.core.spifit import spifit as spifit_core  # noqa: E402

            # Call the core function with all parameters
            spifit_core(
                images,
                output_filename,
                residual=residual,
                psf_pars=psf_pars,
                circ_psf=circ_psf,
                threshold=threshold,
                max_dr=max_dr,
                nthreads=nthreads,
                pb_min=pb_min,
                products=products,
                padding_frac=padding_frac,
                dont_convolve=dont_convolve,
                channel_weights_keyword=channel_weights_keyword,
                channel_freqs=channel_freqs,
                ref_freq=ref_freq,
                out_dtype=out_dtype,
                add_convolved_residuals=add_convolved_residuals,
                ms=ms,
                beam_model=beam_model,
                sparsify_time=sparsify_time,
                corr_type=corr_type,
                band=band,
                deselect_bands=deselect_bands,
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
            images=images,
            output_filename=output_filename,
            residual=residual,
            psf_pars=psf_pars,
            circ_psf=circ_psf,
            threshold=threshold,
            max_dr=max_dr,
            nthreads=nthreads,
            pb_min=pb_min,
            products=products,
            padding_frac=padding_frac,
            dont_convolve=dont_convolve,
            channel_weights_keyword=channel_weights_keyword,
            channel_freqs=channel_freqs,
            ref_freq=ref_freq,
            out_dtype=out_dtype,
            add_convolved_residuals=add_convolved_residuals,
            ms=ms,
            beam_model=beam_model,
            sparsify_time=sparsify_time,
            corr_type=corr_type,
            band=band,
            deselect_bands=deselect_bands,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
