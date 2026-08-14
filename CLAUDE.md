# CLAUDE.md - Project Context for Claude Code

## Project Overview

**spimple** is a radio astronomy image post-processing suite: spectral index fitting,
resolution homogenisation, primary beam interpolation and mosaicking. It follows the
[hip-cargo](https://github.com/landmanbester/hip-cargo) package format — a lightweight CLI
install with auto-generated [Stimela](https://github.com/caracal-pipeline/stimela) cab
definitions and containerised execution. The project prioritises **simplicity and
minimalism** over feature completeness.

Five commands, each a thin `cli/` wrapper over a `core/` implementation. `init`, `mosaic`
and `spifit` operate on an `xarray.DataTree` store laid out exactly as `pfb imager` writes
it, so a pfb tree is consumed directly — see `docs/wiki/datatree-contract.md`.

| Command | What it does |
|---|---|
| `spimple init` | Ingests FITS into a datatree: partitions by phase centre, bands by frequency, resolution homogenised, beams attached, partitions reprojected onto a union grid. |
| `spimple mosaic` | Combines a datatree's image-space partitions into band mean images. Never needed for a pfb tree, which mosaics in visibility space. |
| `spimple spifit` | Fits a spectral index model to the band images of a datatree. |
| `spimple binterp` | Interpolates a primary beam model onto an image's coordinate grid. FITS in, FITS out; the only home of the MS-derived DDE beam path. |
| `spimple imconv` | **Deprecated**, subsumed by `init`. Removal in a follow-up PR. |

*Detailed architecture, Python standards and CI rules are modularised into
`.claude/rules/` for progressive disclosure. Read the relevant file before editing the
matching files.*

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/spimple/**` — layout, install modes, container fallback, cab generation. |
| `docs/wiki/datatree-contract.md` | anything reading or writing a `.dt` store — the layout, its invariants, and what spimple ignores in a pfb tree. |
| `.claude/rules/python-standards.md` | any `**/*.py` — type hints, lazy imports, Typer syntax, hip-cargo types. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**` — round-trip tests, fixtures, commits. |

## LLM wiki (`docs/wiki/`)

Deep internal knowledge — the FITS and beam conventions this code assumes, and the
design-decisions ledger with its known defects — lives in `docs/wiki/`. Start at
`docs/wiki/index.md`. Consult it before touching `utils/fits.py` or `utils/beam.py`,
and before "fixing" something that looks wrong: it may be a documented decision, or a
known-broken path someone has already diagnosed.

**Maintenance rule:** any change that invalidates a wiki page updates the page and its
`last_verified_commit` stamp in the same session.

**Specs and plans are ephemeral.** The brainstorming and planning skills write to
`docs/superpowers/specs/` and `docs/superpowers/plans/`; that directory is gitignored and
never committed. Before finishing a branch, fold durable knowledge into `docs/wiki/` and
let the spec and plan die with the branch. Wiki pages cite code, tests and commits — never
spec or plan paths.

## Mandatory Development Workflow

**Always run linting after adding or modifying any code:**

```bash
uv run ruff format . && uv run ruff check . --fix
```

**Setup:**

```bash
uv sync --extra full --group dev
uv run pre-commit install
```

**Tests:**

```bash
uv run pytest tests/          # everything
uv run pytest -m "not slow"   # skip the end-to-end command tests
```

## Core Dependencies

* Minimise external dependencies.
* The lightweight install provides the CLI and cab definitions only (sole dependency:
  `hip-cargo`). It is always sufficient to *invoke* any command, because the wrappers
  fall back to the container when native imports fail.
* The scientific stack is optional via `pip install spimple[full]`.
* Development uses a single `dev` dependency group; the stack stays behind the `full`
  extra (`uv sync --extra full --group dev`).

## Working Effectively (notes for agents)

Lessons from real sessions in this repo. They are cheaper than rediscovery.

* **The cab's `command:` targets `core/`, not `cli/`.** Generated cabs carry
  `command: spimple.core.<mod>.<fn>`, so **Stimela never executes the CLI wrapper**. Any
  logic you leave in the wrapper — argument parsing, glob expansion, defaulting — is
  invisible to every recipe. Put it in `core/`.
* **Never name a CLI parameter `image`.** `generate-function` emits a local
  `image = get_container_image("spimple")` in the container-fallback path, which shadows
  it and passes the container URL to the command as its input. All four commands use
  `images`.
* **Help text has to survive the cab round-trip.** The generator re-wraps `help=` at
  `". "` boundaries into a YAML scalar. Embedded `\n`, a bare `: `, or a leading `{`
  produce invalid YAML and `generate-cabs` fails outright. Write plain short sentences.
  Put `{current.*}` templates in `@stimela_output`'s `implicit=`, never `info=`.
* **Verify behaviour before asserting it.** Several tests here were first written against
  plausible-sounding assumptions that turned out wrong — convolution *raises* the summed
  pixel value (Jy/beam scales with beam area), and unfitted `spifit` pixels are `NaN`,
  not zero, so an `!= 0` mask silently measures nothing. Probe the real output first.
* **Ray needs `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`** under `uv run`, defaulted in
  `src/spimple/__init__.py`. Without it Ray hands itself the project directory as a URI
  and `ray.init` dies, taking `spimple mosaic` with it.

## Project Structure

```
spimple/
├── src/spimple/
│   ├── __init__.py
│   ├── _container_image.py   # CONTAINER_IMAGE — single source of truth for the image tag
│   ├── cabs/                 # AUTO-GENERATED Stimela YAMLs. Never hand-edit.
│   ├── cli/                  # Lightweight Typer wrappers; what generate-cabs parses
│   │   └── __init__.py       # Builds the Typer app, registers subcommands
│   ├── core/                 # Real implementations. Heavy deps live here.
│   └── utils/                # Shared helpers
│       ├── beam.py           # Primary beam interpolation
│       ├── convolution.py    # Gaussian restoring-beam convolution
│       ├── fits.py           # FITS I/O, WCS, pattern expansion
│       ├── logging.py        # get_logger / log_options
│       └── mosaic.py         # Reprojection and stitching
├── tests/
├── docs/wiki/                # LLM wiki
├── Dockerfile
├── pyproject.toml
├── tbump.toml
├── cliff.toml
└── .pre-commit-config.yaml
```
