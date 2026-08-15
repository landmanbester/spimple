---
type: reference
title: The DataTree contract with pfb-imaging
description: The store layout spimple reads and writes, its invariants, and how a spimple tree differs from a pfb-imaging one.
tags: [datatree, zarr, pfb-imaging, interop, conventions]
timestamp: 2026-08-15
last_verified_commit: af2d642
---

# The DataTree contract with pfb-imaging

`spimple spifit` and `spimple mosaic` operate on an `xarray.DataTree` store, not on loose
FITS files. The layout is the one `pfb imager` writes, so a tree produced by pfb-imaging is
consumed by spimple **directly**, with no ingest step.

`src/spimple/utils/datatree.py` is the single owner of this layout. Everything else goes
through it, so the contract has exactly one place to drift from. Pinned by
`tests/test_datatree.py` and by the synthetic `pfb_tree` fixture in `tests/conftest.py`.

## The two workflows

```
pfb origin     pfb imager -> pfb deconv -> pfb restore --gausspar ...
                                                  |
                                                  v  <out>_I.dt
                                            spimple spifit

FITS origin    spimple init --images ... -o out        -> out_I.dt
               spimple mosaic --store out_I.dt         (only if >1 partition)
               spimple spifit --store out_I.dt
```

**`spimple mosaic` is never part of a pfb workflow.** `pfb imager` mosaics in *visibility*
space — its partitions are summed into the band node during gridding — so a pfb tree
arrives with its band nodes already populated. `mosaic` exists solely to combine the
*image-space* partitions that `spimple init` creates from multi-pointing FITS input. It
refuses a tree whose partitions carry no `MASK`/`BIMAGE`, which is exactly the pfb case.

## Layout

```
out_I.dt/                                    # xr.open_datatree(..., engine="zarr")
  (root attrs)
      product, nband, ntime, nx, ny, cell_rad
      origin = "spimple-init"                # absent on a pfb tree

  band{b:04d}_time{t:04d}/                   # one output image
      attrs:
          bandid, timeid, freq_out, freq_nominal, time_out,
          ra, dec, cell_rad, l0, m0, pb_min
      vars:
          IMAGE      (corr, y, x)   intrinsic, at PSFPARSF
          BIMAGE     (corr, y, x)   apparent,  at PSFPARSF
          KIMAGE     (corr, y, x)   intrinsic model + apparent residual
          BEAM       (corr, y, x)
          WSUM       (corr,)
          RMS        (corr,)        spimple only; absent on a pfb tree
          PSFPARSF   (corr, bpar)   the common target resolution
          SPATIALWGT (corr, y, x)   written by mosaic only

      part{p:04d}/                           # one input pointing
          attrs:
              field_name, ra0, dec0, freq_out, psfparsn,
              beam_includes_n
          vars:
              IMAGE, BIMAGE, KIMAGE, BEAM, MASK  (corr, y, x)
              WSUM, RMS (corr,), PSFPARSF (corr, bpar)
```

`bpar` is the coordinate `["BMAJ", "BMIN", "BPA"]`; `corr` carries the Stokes labels.

`init` writes the band-node **image** variables only when the band has exactly one
partition. With several partitions the band node carries its attrs and `PSFPARSF` alone
until `mosaic` runs — combining is mosaic's job, and init must not guess a band-level
image.

## Invariants

Each of these is asserted in `utils/datatree.py` or pinned by a test. They are where a
spimple tree could silently diverge from a pfb one.

| Invariant | Statement |
|---|---|
| **Axis order** | Every image variable is `(corr, y, x)`. No transposes on the tree path; `save_fits` is called with `yx_order=True`. The only test that catches a violation is the orientation test in `tests/test_render.py` — everything else passes while transposed. |
| **Resolution units** | `PSFPARSN`/`PSFPARSF` are `(emaj, emin, pa)` in **pixels, pixels, radians**, matching pfb. FITS `BMAJ`/`BMIN` (degrees) are divided by `cell_deg`; `BPA` goes through `deg2rad`. `set_wcs` converts back with `unit2deg=cell_x`. |
| **Flux scale** | `IMAGE` intrinsic, `BIMAGE` apparent, `KIMAGE` mixed — pfb's three scales, same names (`PRODUCT_VARS`). All are Jy/beam at `PSFPARSF`, already normalised. `spifit` fits `IMAGE` or `BIMAGE` only; `KIMAGE` mixes scales and is rejected (D18). |
| **No native-resolution copy** | spimple never persists a native `MODEL`/`RESIDUAL`: the input FITS is the archive. pfb's "divide the stored `RESIDUAL` by `WSUM`" step therefore has no spimple analogue. |
| **Beam semantics** | On a spimple tree `BEAM` is the bare primary beam and `beam_includes_n` is False. On a pfb tree `BEAM` is `B/n` with the flag True — see design-decisions.md. |
| **Grid** | Every partition under a band node is on the union grid. Native-grid arrays are never stored: `reproject_interp` conserves surface brightness, not flux, so a Jy/pixel model would not survive it. Convolving first makes everything Jy/beam, after which reprojection is sound. |
| **Time** | `time_out` is **unix seconds**, so headers are rendered with `set_wcs(time_is_unix=True)`. |
| **Band ordering** | Consumers sort band nodes by `bandid`, never by `freq_out`. Effective frequencies are data-dependent and asymmetric flagging can invert two bands. |
| **Attribute merging** | `to_zarr(mode="a")` replaces a group's attrs wholesale, so `write_node` starts from the group's existing attrs and merges. |

## What spimple reads, and what it ignores

A pfb tree carries much more than spimple needs. spimple reads the band-node products
above and nothing else; it must never *require* the rest:

- ignored: `DIRTY`, `BDIRTY`, `PSF`, `PSFHAT`, `PSFPARSN`, `MODEL`, `BRESIDUAL`, `NOISE`,
  and every visibility-space array under `part####` (`VIS`, `WEIGHT`, `UVW`, `FREQ`,
  `MASK` in its vis-space sense);
- read opportunistically: `RESIDUAL`, used only to estimate a threshold rms when the tree
  carries no `RMS` variable. Its absence is never an error.

## Sources

- pfb-imaging `docs/wiki/imager-pipeline.md` (tree layout) and
  `docs/wiki/image-and-beam-orientation.md` §6 (the reprojection construction), branch
  `dev-0.1.0` at commit `b84407b`.
- `src/spimple/utils/datatree.py`, `src/spimple/utils/render.py`,
  `src/spimple/core/init.py`, `src/spimple/core/mosaic.py`, `src/spimple/core/spifit.py`.
- `tests/test_datatree.py`, `tests/test_render.py`, and the `pfb_tree` fixture in
  `tests/conftest.py`.
