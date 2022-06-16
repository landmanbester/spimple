# spimple
Spectral index fitting made simple.
This module provides three executables:

* ```spimple-imconv```
* ```spimple-spifit```
* ```spimple-binterp```

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
