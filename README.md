# spimple

Radio astronomy image post-processing, made simple.

`spimple` works with the same `xarray.DataTree` store that
[pfb-imaging](https://github.com/ratt-ru/pfb-imaging) writes, so a tree produced by pfb is
consumed directly. FITS input from any other imager is ingested into the same layout by
`spimple init`.

| Command | What it does |
|---|---|
| `spimple init` | Ingests FITS images into a datatree, grouping them into partitions by phase centre and into bands by frequency, and homogenising resolution. |
| `spimple mosaic` | Combines a datatree's image-space partitions into band mean images. Only needed for multi-pointing input from `init`. |
| `spimple spifit` | Fits a spectral index model to the band images of a datatree. |
| `spimple binterp` | Interpolates a primary beam model onto an image's coordinate grid. FITS in, FITS out. |
| `spimple imconv` | **Deprecated.** Convolves images to a common resolution. Use `spimple init`. |

### The two workflows

```bash
# images from pfb-imaging: no ingest step at all
pfb restore --output-filename out --gausspar 0.00222 0.00178 0.0 --outputs iI  # degrees: 8.0 x 6.4 arcsec
spimple spifit --store out_I.dt --flux-scale intrinsic --output-filename spi

# FITS from any other imager
spimple init --images "field*-model.fits" --residual "field*-residual.fits" \
             --output-filename out --beam-model JimBeam
spimple mosaic --store out_I.dt            # only if the input had several pointings
spimple spifit --store out_I.dt --flux-scale apparent --output-filename spi
```

`--flux-scale` is required and has no default: the two scales mean different things, and
which products a tree even holds depends on how it was made. Watch out for `pfb restore`'s
default of `--outputs kK`, which writes `KIMAGE` only: that product is an intrinsic model
plus an apparent residual, so it is not fittable and `spifit` refuses it. Ask restore for
`--outputs i` or `--outputs a`, as above.

`pfb imager` mosaics in visibility space, so its band nodes arrive already populated —
**`spimple mosaic` is never part of a pfb workflow.**

It follows the [hip-cargo](https://github.com/landmanbester/hip-cargo) package format, so
every command is available both on the command line and as a
[Stimela](https://github.com/caracal-pipeline/stimela) cab.

## Installation

There are two install modes.

```bash
pip install spimple          # lightweight
pip install spimple[full]    # full scientific stack
```

The **lightweight** install pulls only `hip-cargo`. It is still enough to *run* any
command: when the scientific stack is missing, each command automatically re-dispatches
itself into the project's container image. Use it for Stimela recipes and for CI machines
that only need to launch containers.

The **full** install adds the scientific stack so commands execute natively. Use it for
local work.

## Usage

Every command self-documents:

```bash
spimple --help
spimple spifit --help
```

### Build a datatree from FITS

```bash
spimple init \
    --images "field*-model.fits" \
    --residual "field*-residual.fits" \
    --output-filename out/field \
    --beam-model JimBeam \
    --psf-pars 0.00222 0.00178 0.0
```

Writes `out/field_I.dt`, the product suffix coming from the input STOKES axis. Files
sharing a phase centre and grid become one partition; each distinct frequency becomes a
band. `--psf-pars` is `emaj emin pa` in **degrees** (the example above is 8.0 x 6.4
arcsec); omit it to take the lowest resolution of
the inputs. `--beam-model` accepts `JimBeam`, a FITS beam cube, or a meerkat-beams
`.bds.zarr` store.

### Combine a multi-pointing datatree

```bash
spimple mosaic --store out/field_I.dt --fits-outputs iI
```

Solves `(sum_p B_p^2 + eta) S = sum_p B_p A_p` per band to populate the band mean images.
Skip this for single-pointing input: `init` writes the band products itself.

### Fit a spectral index map

```bash
spimple spifit \
    --store out/field_I.dt \
    --output-filename out/field \
    --flux-scale apparent \
    --products aeikI
```

`--products` selects outputs: `a` alpha map, `e` alpha error, `i` I0 map, `k` I0 error,
`I` reconstructed cube, `d` data minus fitted model, `b` average power beam. Files are
named `<output-filename>_time{t}.<product>.fits`. Unfitted pixels are `NaN`, not zero.

`--pb-min` is an **all-bands** cut: a pixel is fitted only where every band's beam clears
the floor, so the band with the smallest beam — the highest frequency one — sets the
footprint. No pixel is ever fitted from a subset of the bands.

`--threshold` is an SNR cut in multiples of the residual rms, and it is applied to the
**apparent** flux on both scales — the intrinsic path puts the beam back before comparing.
The rms available to `spifit` is always apparent, so testing the beam-corrected `IMAGE`
against it would cut at `threshold × B` and admit the field edge at a few sigma. Both
`--flux-scale` runs therefore apply the same cut, and fit the same pixels wherever the
tree's `BIMAGE` and `IMAGE × BEAM` agree.

`--flux-scale` is **required** and picks which stored product to fit: `apparent`
(`BIMAGE`, fitted with the beam in the model) or `intrinsic` (`IMAGE`, already
beam-corrected, fitted with `BEAM²` weights). The two are the same weighted least-squares
problem written two ways, so they agree to machine precision on a self-consistent tree —
run both as a check on the tree's products. On a `pfb restore` tree expect close rather
than exact agreement: it builds the two products in a way that does not leave `IMAGE`
exactly `BIMAGE / BEAM`, which
[`docs/wiki/design-decisions.md`](docs/wiki/design-decisions.md) quantifies. `KIMAGE` is not fittable and there is no
`mixed` scale; see the note above. The tree must already be at one resolution —
`spimple init` or `pfb restore --gausspar` does that.

### Interpolate a primary beam

```bash
spimple binterp \
    --images "image-cube.fits" \
    --output-filename out/power_beam.fits \
    --beam-model /path/to/beams/meerkat_lband
```

`--beam-model` is a path *prefix*; the loader expects `<prefix>_xx_re.fits` and its
`_im.fits`, `_yy_*` pair alongside. See
[`docs/wiki/fits-and-beams.md`](docs/wiki/fits-and-beams.md) for the full convention. The
output cube can be fed straight to `spimple init --beam-model`.

List-valued options (`--images`, `--residual`, `--ms`, `--deselect-bands`) take a
comma-separated string and accept glob patterns.

## Breaking changes in this release

- `spifit` and `mosaic` take `--store`, a datatree, instead of FITS images. Their
  FITS-specific options are gone; convolution, beams, frequencies and weights now come
  from the tree.
- `spifit --flux-scale` is required, with no default, and no longer accepts `mixed`.
  `KIMAGE` mixes an intrinsic model with an apparent residual, so fitting it biased the
  flux towards the field edge; `spifit` now errors out naming the `pfb restore` rerun.
- `spifit --pb-min` now cuts on every band rather than on each band separately. Pixels
  whose beam fell below the floor in only some bands were previously fitted anyway, using
  the whole band stack including the bands that failed. The fitted footprint shrinks to
  the highest-frequency band's, and alpha near the field edge changes.
- `spifit --threshold` now cuts on apparent flux on both flux scales. `--flux-scale
  intrinsic` previously compared the beam-corrected `IMAGE` against an apparent rms, an
  effective cut of `threshold × B`, so its fitted footprint was always a strict superset of
  the apparent one with a low-SNR skirt towards the field edge. The intrinsic footprint
  shrinks to the apparent one's, up to the near-threshold pixels where a pfb tree's
  `BIMAGE` and `IMAGE × BEAM` differ; alpha on the pixels both runs already fitted is
  unchanged.
- `spifit --flux-scale intrinsic` now weights each pixel by `BEAM²` instead of fitting
  with unity weights, which makes it agree with `--flux-scale apparent` exactly. Fitted
  spectral indices and their errors change wherever the beam varies across the band.
- `imconv` is deprecated and warns on use. It will be removed in a following release.
- `Gaussian2D` was producing kernels 8.51 % too wide, so every convolution now homogenises
  to the resolution actually requested. Output changes accordingly.
- `set_wcs` wrote a `CRVAL3` one channel too high for any multi-channel cube, and raised
  `IndexError` for a two-channel one. Both fixed.

## Running in a container

Every command takes `--backend`:

```bash
spimple init --backend docker --images "image.fits" --output-filename out/field
```

`auto` (the default) tries to run natively and falls back to a detected container runtime.
`native` forces in-process execution. `docker`, `podman`, `apptainer` and `singularity`
dispatch straight into that runtime. `--always-pull-images` forces a fresh pull.

Path options are UPath-backed, so they also accept `s3://`, `gs://` and `az://` URIs.

## Using spimple from Stimela

Cab definitions ship inside the package, so a recipe can include them directly:

```yaml
_include:
  - (spimple.cabs)init.yml
  - (spimple.cabs)spifit.yml
```

The cabs are generated from the CLI source and pinned to a versioned container image, so
a recipe and the command line stay in step.

## Development

```bash
uv sync --extra full --group dev
uv run pre-commit install

uv run pytest tests/                      # everything
uv run pytest -m "not slow"               # skip the end-to-end command tests
uv run ruff format . && uv run ruff check . --fix
```

The cab definitions under `src/spimple/cabs/` are **generated** — never edit them by hand.
A pre-commit hook regenerates them, and `tests/test_roundtrip.py` asserts every CLI wrapper
regenerates byte-for-byte from its own cab.

Contributor guidance lives in [`CLAUDE.md`](CLAUDE.md) and `.claude/rules/`; the
conventions this code assumes about FITS headers and beam models, along with the ledger of
design decisions and known defects, live in [`docs/wiki/`](docs/wiki/index.md).

## License

MIT — see [LICENSE](LICENSE).
