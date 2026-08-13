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
pfb restore --output-filename out --gausspar 8.0 6.4 0.0
spimple spifit --store out_I.dt --output-filename spi

# FITS from any other imager
spimple init --images "field*-model.fits" --residual "field*-residual.fits" \
             --output-filename out --beam-model JimBeam
spimple mosaic --store out_I.dt            # only if the input had several pointings
spimple spifit --store out_I.dt --output-filename spi
```

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
    --psf-pars 8.0 6.4 0.0
```

Writes `out/field_I.dt`, the product suffix coming from the input STOKES axis. Files
sharing a phase centre and grid become one partition; each distinct frequency becomes a
band. `--psf-pars` is `emaj emin pa` in degrees; omit it to take the lowest resolution of
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

`--flux-scale` picks which stored product to fit: `apparent` (`BIMAGE`, the default,
fitted together with the beam), `intrinsic` (`IMAGE`, already beam-corrected) or `mixed`
(`KIMAGE`). The tree must already be at one resolution — `spimple init` or
`pfb restore --gausspar` does that.

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
