---
type: reference
title: FITS and beam conventions
description: The header conventions spimple's I/O assumes, and the beam-model layouts it expects.
tags: [fits, wcs, beams, conventions]
timestamp: 2026-08-13
last_verified_commit: b7bfbc4
---

# FITS and beam conventions

What `utils/fits.py` and `utils/beam.py` assume about their inputs. Read before touching
either, or before building a test fixture.

## The frequency axis lives on CTYPE3 *or* CTYPE4

Both conventions occur in real WSClean and CASA output, and spimple is **not uniform**
about which it expects:

| Consumer | Convention |
|---|---|
| `imconv`, `spifit` | Detect it: check `CTYPE4` first, fall back to `CTYPE3`, else raise. |
| `utils/mosaic.mosaic_info` | Hardcodes `data_from_header(hdr, axis=3)` and reads `NAXIS3`/`NAXIS4` directly. **Requires frequency on CTYPE3.** |

`tests/conftest.py` fixtures both — `image_cube` (CTYPE4) and `image_cube_ctype3`. A test
that hands a CTYPE4 cube to `mosaic_info` fails in confusing ways.

### The numpy ↔ FITS axis inversion

astropy maps numpy axes to FITS axes in reverse. A numpy array of shape `(a, b, c, d)`
becomes `NAXIS1=d, NAXIS2=c, NAXIS3=b, NAXIS4=a`. So:

- frequency on **CTYPE4** → frequency is numpy axis **0** → shape `(nchan, 1, ny, nx)`
- frequency on **CTYPE3** → frequency is numpy axis **1** → shape `(1, nchan, ny, nx)`

`load_fits` then applies its own transpose, `(1, 0, 3, 2)`, so what the core code receives
is `(NAXIS3, NAXIS4, NAXIS1, NAXIS2)`. Getting this backwards is the single easiest way to
write a fixture that loads without error and means something different from what you
intended.

## `data_from_header` returns CRVAL, not the band centre

```python
freqs, ref_freq = data_from_header(hdr, axis=4)
```

`freqs` is the full coordinate array; `ref_freq` is **`CRVAL<axis>`** — the value at the
reference pixel. With the usual `CRPIX=1` that is the *first* channel, not the middle one.
Do not assume `ref_freq == freqs[freqs.size // 2]`.

## Per-channel restoring beams

Cubes carry one set of beam keywords per channel, one-indexed:

```
BMAJ1, BMIN1, BPA1
BMAJ2, BMIN2, BPA2
...
```

plus a scalar `BMAJ`/`BMIN`/`BPA` for the MFS beam. `imconv` and `spifit` read the
per-channel set to work out each channel's native resolution before convolving to a common
one. Some headers spell the position angle `PA<n>` instead of `BPA<n>`; the code tries
`BPA<n>` first and falls back.

Units are **degrees** throughout the header. `Gaussian2D` takes its position angle in
radians — see the docstring on `add_beampars`.

### Flux scaling on convolution

The data are **Jy/beam**. Convolving from a native beam to a coarser common beam therefore
*multiplies* the summed pixel values by the ratio of beam areas — it does not conserve the
pixel sum, and the peak generally rises rather than falls.
`tests/test_imconv.py::test_convolution_scales_flux_by_the_beam_area_ratio` pins that
ratio, and is the check that catches a normalisation regression in `convolve2gaussres`.

## Beam models

### FITS real/imaginary pattern

`--beam-model` is a path *prefix*, not a file. The loader globs
`<prefix>*_*.fits` in the prefix's parent directory and matches on the tail of each name:

```
/home/user/beams/meerkat_lband_xx_re.fits
/home/user/beams/meerkat_lband_xx_im.fits
/home/user/beams/meerkat_lband_yy_re.fits
/home/user/beams/meerkat_lband_yy_im.fits
```

so you pass `--beam-model /home/user/beams/meerkat_lband`. Correlation names come from
`corr_type`: `linear` → `XX`/`YY`, `circular` → `LL`/`RR`. Anything else raises `KeyError`.
The power beam is `(|corr1|² + |corr2|²) / 2`.

The beam header must use `CUNIT1`/`CUNIT2` of `deg`, and its `l`/`m` extent must cover the
image's — a beam smaller than the field raises "The supplied beam is not large enough".

**This path is currently broken.** See `design-decisions.md` § "The FITS primary-beam path
is internally inconsistent".

### JimBeam

`imconv` and `spifit` accept the literal string `JimBeam` as `--beam-model`, which pulls an
analytic MeerKAT beam from `katbeam` instead of reading files. The `--band` option selects
the model:

| `--band` | katbeam model |
|---|---|
| `L` | `MKAT-AA-L-JIM-2020` |
| `UHF` | `MKAT-AA-UHF-JIM-2020` |
| `S` | `MKAT-AA-S-JIM-2020` |

`binterp` has **no** JimBeam path and no `--band` option, despite its CLI docs having
historically implied otherwise.

### Measurement sets and parallactic angle

Passing `--ms` makes the beam path compute parallactic angles and unflagged-visibility
counts per time, so the beam is averaged over the observation rather than evaluated at a
single instant. `--sparsify-time` subsamples the time axis to keep that affordable, and
`--field` picks the field whose phase centre is used. Antenna positions are checked for
consistency across multiple measurement sets.


## The DataTree path uses different I/O

Everything above describes the FITS-to-FITS commands (`imconv`, `binterp`). The DataTree
path — `init`, `mosaic`, `spifit` — differs deliberately:

- **Arrays are (Y, X).** `load_fits` keeps its legacy `(1, 0, 3, 2)` transpose into (X, Y)
  for the FITS-to-FITS commands. The tree path instead uses **`load_cube`**, which returns
  `(nband, ncorr, ny, nx)` with the FITS raster untouched, and **`save_fits(yx_order=True)`**
  to write it back.
- **`load_cube` resolves the frequency-axis fork in one place.** It detects `CTYPE3` versus
  `CTYPE4` itself and moves frequency to axis 0. Previously `imconv`, `spifit` and
  `mosaic_info` each re-implemented that detection, and inconsistently — the trap this page
  and `CLAUDE.md` both warned about. Pinned by asserting `load_cube` returns identical
  arrays for the `image_cube` and `image_cube_ctype3` fixtures.
- **Beams come from `utils/beamsource.py`,** which supports `JimBeam`, a FITS cube and a
  meerkat-beams `.bds.zarr`, each evaluated on the partition's own grid around its own
  phase centre and then reprojected. The FITS backend goes through `reproject_interp`, so
  the pixel-exact coordinate match `imconv`/`spifit` demand is no longer required — a
  `binterp` output cube is a valid `init --beam-model`.
- **The MS-derived DDE path (`utils/beam.py`) is now reachable only from `binterp`.** It is
  deliberately not an `init` backend, and its documented defects are unchanged.

See [datatree-contract.md](datatree-contract.md) for the store layout itself.
