# Architectural Rules & Domain Logic

Read this when editing `src/spimple/**` files.

## 1. Package Layout

```
spimple/
├── src/spimple/
│   ├── __init__.py
│   ├── _container_image.py    # CONTAINER_IMAGE — single source of truth for the image tag
│   ├── cli/                   # Lightweight Typer wrappers. THIS is what generate-cabs parses.
│   │   ├── __init__.py        # Builds the Typer `app` and registers subcommands
│   │   └── onboard.py         # One file per subcommand (delete onboard once setup is done)
│   ├── core/                  # Real implementations. Heavy deps live here.
│   │   ├── __init__.py
│   │   └── onboard.py         # Mirrors cli/onboard.py — same function name, no decorators
│   └── cabs/                  # AUTO-GENERATED Stimela YAMLs. Do NOT hand-edit.
│       ├── __init__.py
│       └── onboard.yml
├── tests/
│   ├── test_install.py
│   └── test_roundtrip.py      # Guards the CLI → cab → CLI round-trip
├── Dockerfile                 # Builds the image referenced by _container_image.py
├── pyproject.toml
├── tbump.toml                 # Release tooling — updates _container_image.py + cabs
├── .pre-commit-config.yaml    # Runs generate-cabs on every commit
└── .github/workflows/
    ├── ci.yml
    ├── publish.yml             # PyPI on tag push
    ├── publish-container.yml   # ghcr.io on tag + every push to main
    └── update-cabs.yml         # Regenerates cabs on merge to main
```

### Role of each directory

| Directory | What lives there | What does NOT live there |
|---|---|---|
| `cli/` | Thin Typer wrappers with `@stimela_cab` (and optional `@stimela_output`). One file per command. **Imports from `core/` must be lazy** (inside the function body). | Heavy imports at module top. Business logic. NumPy / pandas / domain libs. |
| `core/` | The actual implementation. Type-hinted function with the same name as the CLI wrapper, but **no Typer / hip-cargo decorators**. Free to import anything. | Typer. `@stimela_cab`. UI concerns. `typer.Exit(...)`. |
| `cabs/` | Generated `<command>.yml` files. Committed to source control. Loaded by Stimela. | Anything you wrote by hand. Drift from `cli/*.py`. |

### Adding a new command

1. Create `src/spimple/cli/<name>.py` with a `@stimela_cab`-decorated
   Typer function. Lazily import the core implementation inside the function.
2. Create `src/spimple/core/<name>.py` with the real implementation —
   same function name, no decorators, free to import heavy deps.
3. Register the new command in `src/spimple/cli/__init__.py` (next to
   the existing `onboard` registration; mirror its pattern).
4. Commit. The pre-commit hook regenerates `src/spimple/cabs/<name>.yml`
   automatically.

**Never** create files under `cabs/` by hand. They are derived artefacts.

## 2. Lightweight vs Full Installation

This package supports two install modes. The split is what makes the
container-fallback pattern below work.

| Mode | Command | What it pulls | When to use |
|---|---|---|---|
| **Lightweight** | `pip install spimple` | `hip-cargo` + `typer` only | Cab consumers (Stimela), CI machines that only need to dispatch commands into containers, anyone who already has the project's container image available. |
| **Full** | `pip install spimple[full]` | Lightweight + everything listed under `[project.optional-dependencies].full` in `pyproject.toml` | Local development; native (non-container) execution. |

The lightweight install is **always sufficient to invoke any command** because
the generated CLI wrappers fall back to running the same command inside the
project's container when native imports fail (see §3).

### When you add a heavy dep

- Add it to `[project.optional-dependencies].full` in `pyproject.toml`, **not**
  to the top-level `dependencies`. The top-level list must stay tiny so the
  lightweight install remains lightweight.
- Import it **only from inside `core/`**. Never import it from `cli/` at module
  scope.

## 3. Container Fallback & Backends

Every generated CLI wrapper in `cli/*.py` follows the same shape (this is
emitted by `hip-cargo generate-function`, but the pattern matters when you
write a new command by hand too):

```python
def my_command(...):
    if backend == "native" or backend == "auto":
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist
            preflight_remote_must_exist(my_command, dict(...))
            from spimple.core.my_command import my_command as my_command_core
            my_command_core(...)
            return
        except ImportError:
            if backend == "native":
                raise
    # Heavy deps missing OR backend explicitly chose a container → run in container.
    from hip_cargo.utils.config import get_container_image
    from hip_cargo.utils.runner import run_in_container
    image = get_container_image("spimple")
    run_in_container(my_command, dict(...), image=image, backend=backend, ...)
```

### How `--backend` flows

Every command auto-grows two parameters via `hip-cargo generate-function`:

| Flag | Values | Effect |
|---|---|---|
| `--backend` | `auto` (default), `native`, `apptainer`, `singularity`, `docker`, `podman` | `auto` tries native then falls back to a detected container runtime. `native` forces in-process execution and surfaces the `ImportError` if `[full]` is not installed. The explicit backends skip the native attempt entirely and dispatch into the matching runtime. |
| `--always-pull-images` | bool | Forces a fresh `pull` before each container run. |

Both flags are decorated with `StimelaMeta(skip=True)` so they appear in the
Python CLI but **not** in the generated cab YAML — Stimela manages container
execution on its own side and doesn't need them.

### Image resolution

The image tag is owned by `src/spimple/_container_image.py`:

```python
CONTAINER_IMAGE = "ghcr.io/landmanbester/spimple:latest"
```

Three things keep this in sync — do not bypass them:

1. **Feature branches:** Edit `_container_image.py` by hand to point at your
   branch tag (e.g. `:my-feature`). The `publish-container.yml` workflow builds
   and pushes that tag on every push of the PR.
2. **Merge to `main`:** The `update-cabs.yml` workflow resets the
   tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **Releases:** `tbump <version>` rewrites the tag to the semantic version and
   regenerates cabs as a `before_commit` hook.

### Remote URIs (S3 / GCS / Azure)

Path-typed parameters (`File`, `Directory`, `MS`, `URI`) accept both local
paths and remote URIs (`s3://...`, `gs://...`, `az://...`). When the path is
remote:

- `_resolve_mounts` skips it (nothing to bind-mount).
- `preflight_remote_must_exist` checks existence via fsspec.
- `run_in_container` forwards the matching credentials (`AWS_*`, `~/.aws`,
  `GOOGLE_APPLICATION_CREDENTIALS`, `~/.config/gcloud`, `AZURE_*`, `~/.azure`).

Users who want native remote access install the right extra: `pip install
hip-cargo[s3]`, `[gcs]`, or `[azure]`. Without it, the wrapper's existing
`try/except ImportError` routes them into the container, which already has the
backends.

## 4. Cab Generation is Automatic

**The `src/spimple/cabs/*.yml` files are generated artefacts. Never edit
them by hand and never run `hip-cargo generate-cabs` manually.**

Three automated paths keep them in sync with `cli/*.py`:

1. **Pre-commit hook** (`.pre-commit-config.yaml`): on every commit, runs
   `hip-cargo generate-cabs --module src/spimple/cli/*.py --output-dir
   src/spimple/cabs`. If it modifies files, pre-commit will "fail" the
   commit — re-run `git add -u && git commit` to include the updates.
2. **`update-cabs.yml` workflow**: on merge to `main`, resets the
   container tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **`tbump`**: on release, rewrites the container tag to the version and
   regenerates cabs.

If you ever see a cab YAML in a diff that wasn't generated by one of these
three paths, that's a bug — revert it and edit the corresponding `cli/*.py`
instead.

> **Heads-up:** `generate-cabs` resolves the `image:` field from the *installed*
> package metadata. Activate the project venv (so `spimple` is importable)
> before committing — otherwise the regenerated cab is written without an
> `image:` field.

### How CLI source maps to cab YAML

- `@stimela_cab(name=..., info=...)` → the cab's name and top-level info.
- `@stimela_output(...)` → entries under `outputs:` in the cab.
- Each Typer parameter → an entry under `inputs:` (dtype inferred from the type
  hint, `info` from `help=`, defaults from `= ...`).
- `Annotated[..., StimelaMeta(skip=True)]` → omitted from the cab (used for
  `--backend`, `--always-pull-images`, etc.).
- `Annotated[..., StimelaMeta(metadata={"rich_help_panel": "Inputs", "tunable":
  True})]` → flows into the cab's `metadata:` dict.
- Inline comments after `Annotated[...]` rows are preserved through the round
  trip — they show up as `# noqa: ...` or similar on the matching cab field.

---

## 5. spimple specifics

### The `utils/` package sits at package level, not under `core/`

`core/` holds one module per command — the real implementation the cab's `command:`
points at. Anything shared between commands lives in `src/spimple/utils/`:

| Module | Contents |
|---|---|
| `utils/fits.py` | `to4d`, `data_from_header`, `load_fits`, `save_fits`, `set_wcs`, `add_beampars`, `set_header_info`, `expand_image_patterns` |
| `utils/convolution.py` | `Gaussian2D`, `get_padding_info`, `convolve2gaussres` |
| `utils/beam.py` | `extract_dde_info`, `make_power_beam`, `interpolate_beam` |
| `utils/mosaic.py` | `mosaic_info`, `project`, `stitch_images`, `conjugate_gradient` |
| `utils/logging.py` | `get_logger`, `log_options` |

This mirrors `pfb_imaging/utils/`. `utils/__init__.py` is deliberately empty — import from
the specific module, never add a re-export shim.

### Core signatures are the Stimela contract

Generated cabs carry `command: spimple.core.<mod>.<fn>`. Stimela imports and calls that
function **directly** and never runs the `cli/` wrapper. Two consequences:

1. A `core/` signature must accept exactly what its cab declares. When you rename a CLI
   parameter, rename the core parameter in the same commit.
2. Behaviour that must apply to recipe runs — glob expansion, defaulting, validation —
   belongs in `core/`. The wrapper is for Typer and container dispatch only.

### The beam path takes explicit arguments

`interpolate_beam`, `extract_dde_info` and `make_power_beam` take keyword-only parameters
(`beam_model`, `ms`, `field`, `sparsify_time`, `corr_type`, `nthreads`). They previously
took a duck-typed `opts` object that callers faked with a throwaway class, which hid a
real `AttributeError` in `spifit` for years. **Never reintroduce an `opts` parameter.** A
missing argument must be a `TypeError` at the call site.

### Logging

Every core command opens with `log_options(log, **locals())` as its **first statement** —
`locals()` there is exactly the parameter list, so the logged options cannot drift from
the signature. Anything that reorders statements above it silently corrupts that log.
`pyscilog` was retired; use `spimple.utils.logging.get_logger`.
