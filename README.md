# spimple

Radio astronomy image post-processing, made simple.

`spimple` provides four commands for working with FITS image cubes produced by radio
interferometric imagers:

| Command | What it does |
|---|---|
| `spimple spifit` | Fits a spectral index model to an image cube, optionally convolving to a common resolution and applying a primary beam correction on the fly. |
| `spimple imconv` | Convolves images to a common resolution, optionally with primary beam correction. |
| `spimple binterp` | Interpolates a primary beam model onto an image's coordinate grid. |
| `spimple mosaic` | Reprojects and combines multiple images onto a common coordinate grid. |

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

### Fit a spectral index map

```bash
spimple spifit \
    --images "image-cube.fits" \
    --residual "residual-cube.fits" \
    --output-filename out/field \
    --products aeikI
```

`--products` is a string of letters selecting outputs: `a` alpha map, `e` alpha error,
`i` I0 map, `k` I0 error, `I` reconstructed cube, `c` restoring beam, `m` convolved model,
`r` convolved residual, `b` average power beam, `d` data minus fitted model.

### Convolve to a common resolution

```bash
spimple imconv \
    --images "image-*.fits" \
    --output-filename out/convolved \
    --psf-pars 8.0 6.4 0.0 \
    --products ic
```

`--psf-pars` is `emaj emin pa` in degrees. Omit it to take the resolution from the FITS
header. Note the data are Jy/beam, so convolving to a coarser beam scales pixel values by
the ratio of beam areas.

### Interpolate a primary beam

```bash
spimple binterp \
    --images "image-cube.fits" \
    --output-filename out/power_beam.fits \
    --beam-model /path/to/beams/meerkat_lband
```

`--beam-model` is a path *prefix*; the loader expects `<prefix>_xx_re.fits` and its
`_im.fits`, `_yy_*` pair alongside. See
[`docs/wiki/fits-and-beams.md`](docs/wiki/fits-and-beams.md) for the full convention.

### Mosaic

```bash
spimple mosaic \
    --images "field*-image.fits" \
    --output-filename out/mosaic.fits \
    --beam-model /path/to/beam.bds.zarr
```

List-valued options (`--images`, `--residual`, `--ms`, `--channel-freqs`,
`--deselect-bands`) take a comma-separated string and accept glob patterns.

## Running in a container

Every command takes `--backend`:

```bash
spimple imconv --backend docker --images "image.fits" --output-filename out/conv
```

`auto` (the default) tries to run natively and falls back to a detected container runtime.
`native` forces in-process execution. `docker`, `podman`, `apptainer` and `singularity`
dispatch straight into that runtime. `--always-pull-images` forces a fresh pull.

Path options are UPath-backed, so they also accept `s3://`, `gs://` and `az://` URIs.

## Using spimple from Stimela

Cab definitions ship inside the package, so a recipe can include them directly:

```yaml
_include:
  - (spimple.cabs)spifit.yml
  - (spimple.cabs)imconv.yml
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
