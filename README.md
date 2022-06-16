# spimple
Spectral index fitting made simple.
This module provides three executables:

* ```spimple-spifit```
* ```spimple-imconv```
* ```spimple-binterp```

## Spectral index fitting
The ```spimple-spifit``` executable fits a spectral index model to an
image cube. The convolution to a common resolution and primary beam
correction can optionally be performed on the fly. Usage is as follows:
```
usage: spimple-spifit [-h] [-model MODEL] [-residual RESIDUAL] -o OUTPUT_FILENAME [-pp PSF_PARS [PSF_PARS ...]] [-cp [CIRC_PSF]] [-th THRESHOLD] [-maxDR MAXDR] [-nthreads NTHREADS] [-pb-min PB_MIN]
                      [-products PRODUCTS] [-pf PADDING_FRAC] [-dc [DONT_CONVOLVE]] [-cw CHANNEL_WEIGHTS [CHANNEL_WEIGHTS ...]] [-rf REF_FREQ] [-otype OUT_DTYPE] [-acr [ADD_CONVOLVED_RESIDUALS]]
                      [-ms MS [MS ...]] [-f FIELD] [-bm BEAM_MODEL] [-st SPARSIFY_TIME] [-ct CORR_TYPE] [-band BAND]

Simple spectral index fitting tool.

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL, --model MODEL
  -residual RESIDUAL, --residual RESIDUAL
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Path to output directory + prefix.
  -pp PSF_PARS [PSF_PARS ...], --psf-pars PSF_PARS [PSF_PARS ...]
                        Beam parameters matching FWHM of restoring beam specified as emaj emin pa.
                        By default these are taken from the fits header of the residual image.
  -cp [CIRC_PSF], --circ-psf [CIRC_PSF]
                        Passing this flag will convolve with a circularised beam instead of an elliptical one
  -th THRESHOLD, --threshold THRESHOLD
                        Multiple of the rms in the residual to threshold on.
                        Only components above threshold*rms will be fit.
  -maxDR MAXDR, --maxDR MAXDR
                        Maximum dynamic range used to determine the threshold above which components need to be fit.
                        Only used if residual is not passed in.
  -nthreads NTHREADS, --nthreads NTHREADS
                        Number of threads to use.
                        Default of zero means use all threads
  -pb-min PB_MIN, --pb-min PB_MIN
                        Set image to zero where pb falls below this value
  -products PRODUCTS, --products PRODUCTS
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
                        Default is to write all of them
  -pf PADDING_FRAC, --padding-frac PADDING_FRAC
                        Padding factor for FFT's.
  -dc [DONT_CONVOLVE], --dont-convolve [DONT_CONVOLVE]
                        Passing this flag bypasses the convolution by the clean beam
  -cw CHANNEL_WEIGHTS [CHANNEL_WEIGHTS ...], --channel_weights CHANNEL_WEIGHTS [CHANNEL_WEIGHTS ...]
                        Per-channel weights to use during fit to frequency axis.
                         Only has an effect if no residual is passed in (for now).
  -rf REF_FREQ, --ref-freq REF_FREQ
                        Reference frequency where the I0 map is sought.
                        Will overwrite in fits headers of output.
  -otype OUT_DTYPE, --out_dtype OUT_DTYPE
                        Data type of output. Default is single precision
  -acr [ADD_CONVOLVED_RESIDUALS], --add-convolved-residuals [ADD_CONVOLVED_RESIDUALS]
                        Flag to add in the convolved residuals before fitting components
  -ms MS [MS ...], --ms MS [MS ...]
                        Mesurement sets used to make the image.
                        Used to get paralactic angles if doing primary beam correction
  -f FIELD, --field FIELD
                        Field ID
  -bm BEAM_MODEL, --beam-model BEAM_MODEL
                        Fits beam model to use.
                        It is assumed that the pattern is path_to_beam/name_corr_re/im.fits.
                        Provide only the path up to name e.g. /home/user/beams/meerkat_lband.
                        Patterns mathing corr are determined automatically.
                        Only real and imaginary beam models currently supported.
  -st SPARSIFY_TIME, --sparsify-time SPARSIFY_TIME
                        Used to select a subset of time
  -ct CORR_TYPE, --corr-type CORR_TYPE
                        Correlation typ i.e. linear or circular.
  -band BAND, --band BAND
                        Band to use with JimBeam. L or UHF

```


## Image convolutions and primary beam correction
The ```spimple-imconv``` executable convolves images to a common resolution
and optionally performs a primary beam correction. Usage is as follows:
```
usage: spimple-imconv [-h] -image IMAGE -o OUTPUT_FILENAME [-pp PSF_PARS [PSF_PARS ...]] [-nthreads NTHREADS] [-cp] [-bm BEAM_MODEL] [-band BAND] [-pb-min PB_MIN] [-pf PADDING_FRAC] [-otype OUT_DTYPE]

Convolve images to a common resolution.

optional arguments:
  -h, --help            show this help message and exit
  -image IMAGE, --image IMAGE
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Path to output directory.
  -pp PSF_PARS [PSF_PARS ...], --psf-pars PSF_PARS [PSF_PARS ...]
                        Beam parameters matching FWHM of restoring beam specified as emaj emin pa.
                        By default these are taken from the fits header of the image.
  -nthreads NTHREADS, --nthreads NTHREADS
                        Number of threads to use.
                        Default of zero means use all threads
  -cp, --circ-psf       Passing this flag will convolve with a circularised beam instead of an elliptical one
  -bm BEAM_MODEL, --beam-model BEAM_MODEL
                        Fits beam model to use.
                        Use power_beam_maker to make power beam corresponding to image.
  -band BAND, --band BAND
                        Band to use with JimBeam. L or UHF
  -pb-min PB_MIN, --pb-min PB_MIN
                        Set image to zero where pb falls below this value
  -pf PADDING_FRAC, --padding-frac PADDING_FRAC
                        Padding fraction for FFTs (half on either side)
  -otype OUT_DTYPE, --out_dtype OUT_DTYPE
                        Data type of output. Default is single precision
```


## Power beam interpolation
The ```spimple-binterp``` executable allows primary beam interpolation
and conversion to a power beam. Usage is as follows:
```
usage: spimple-binterp [-h] -image IMAGE [-ms MS [MS ...]] [-f FIELD] -o OUTPUT_FILENAME [-bm BEAM_MODEL] [-st SPARSIFY_TIME] [-nthreads NTHREADS] [-ct CORR_TYPE]

Beam intrepolation tool.

optional arguments:
  -h, --help            show this help message and exit
  -image IMAGE, --image IMAGE
  -ms MS [MS ...], --ms MS [MS ...]
                        Mesurement sets used to make the image.
                        Used to get paralactic angles if doing primary beam correction
  -f FIELD, --field FIELD
                        Field ID
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Path to output directory.
  -bm BEAM_MODEL, --beam-model BEAM_MODEL
                        Fits beam model to use.
                        It is assumed that the pattern is path_to_beam/name_corr_re/im.fits.
                        Provide only the path up to name e.g. /home/user/beams/meerkat_lband.
                        Patterns mathing corr are determined automatically.
                        Only real and imaginary beam models currently supported.
  -st SPARSIFY_TIME, --sparsify-time SPARSIFY_TIME
                        Used to select a subset of time
  -nthreads NTHREADS, --nthreads NTHREADS
                        Number of threads to use.
                        Default of zero means use all threads
  -ct CORR_TYPE, --corr-type CORR_TYPE
                        Correlation typ i.e. linear or circular.
```
