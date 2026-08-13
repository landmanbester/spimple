---
type: reference
title: Design decisions and known defects
description: The decision ledger for the hip-cargo port, plus the defects diagnosed but not fixed.
tags: [architecture, decisions, known-issues]
timestamp: 2026-08-13
last_verified_commit: f3c2726
---

# Design decisions and known defects

## Registry and automation setup

`.github/workflows/update-cabs.yml` authenticates with `actions/create-github-app-token`
using `secrets.APP_CLIENT_ID` and `secrets.APP_PRIVATE_KEY`. The App is installed on this
repository. If those secrets are ever rotated away, the workflow fails on every merge to
`main` and cabs stop being reset to the `:latest` tag; local pre-commit and `tbump` keep
them in sync regardless.

**GHCR package access.** `ghcr.io/landmanbester/spimple` is a *user*-namespace package. A
user package created by a manual `docker push` is not linked to any repository, and
`GITHUB_TOKEN` then cannot write to it — `publish-container.yml` fails with
`denied: permission_denied: write_package` even though the repo's default workflow
permissions are `write` and the job requests `packages: write`. The fix is a one-time
settings change, not a workflow change: package settings → Manage Actions access → add the
`spimple` repository with the **Write** role. (`ratt-ru/pfb-imaging` never hit this because
an org package inherits its repository's access.)

## Decisions

| ID | Decision | Why |
|---|---|---|
| D1 | `utils/` sits at package level, not under `core/`. | `core/` is one module per command — the thing a cab's `command:` points at. Shared helpers are not commands. Mirrors `pfb_imaging/utils/`, and is what the broken `from spimple.utils import ...` lines predating the port were reaching for. |
| D2 | The beam path takes explicit keyword-only arguments; no `opts` object. | The duck-typed `opts` hid a real bug: `spifit` built a throwaway `BeamOpts` that never set `field` (nor `nthreads`), so `spimple spifit --beam-model … --ms …` raised `AttributeError` deep in `extract_dde_info`. Explicit parameters make that a `TypeError` at the call site. |
| D3 | `pyscilog` retired for stdlib `logging`; `log_options(log, **locals())` as the first statement of each core command. | Matches pfb-imaging and meerkat-beams, drops a dependency, and `locals()` at the first statement cannot drift from the signature. Note ruff's G004 forbids f-strings in logging calls, so messages use lazy `%` formatting. |
| D4 | `psf_pars` stays `tuple[float, float, float]`. | The canonical hip-cargo type for a fixed-length triple; it round-trips as `Optional[Tuple[float, float, float]]`, as `pfb-imaging`'s identical `restore.gausspar` does. `ListStr`/`ListInt`/`ListFloat` are for genuinely variable-length lists Typer cannot express as one option. |
| D5 | Breaking CLI renames: `--image` → `--images` everywhere, `--maxDR` → `--max-dr`, `--pfb-min` → `--pb-min`. | `--images` is not cosmetic — see D6. `--maxDR` was mixedCase; `--pfb-min` was a typo for the `--pb-min` that `imconv` already used. |
| D6 | No CLI parameter may be named `image`. | `generate-function` emits a local `image = get_container_image("spimple")` in the container-fallback path. A parameter of that name is shadowed and receives the container URL as its input file. |
| D7 | `requires-python = ">=3.10, <3.14"`; `.python-version` pins **3.11** for development. | `uv lock --extra full` resolves across the range, so the spec's `<3.13` fallback was not needed. 3.10 remains supported for the lightweight install; 3.11 is the dev interpreter, matching pfb-imaging and meerkat-beams. Note this bump is what exposed D9. |
| D8 | `set_wcs` uses `astropy.time.Time`, not `casacore.quanta`. | Removes the only direct `python-casacore` import (it still arrives transitively via `codex-africanus[complete]`). Pinned by four regression vectors in `tests/test_fits.py` captured from the casacore implementation; they agree to better than 2.4e-7 s, far below the whole-second truncation applied. |
| D9 | `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` is defaulted in `src/spimple/__init__.py`. | Ray ≥ 2.43 auto-injects a `runtime_env` when it detects `uv run` and hands itself the project directory as a URI. `ray.init` then dies with "… is not a valid URI", taking `spimple mosaic` out entirely. |
| D10 | Two targeted `noqa`s are load-bearing: `N816` on `iFs`, `E402` on the four deferred imports in `cli/__init__.py`. | Preferred over widening the project-wide ignore list. Both rules only became live when the ruff config was replaced, which is why they were added in that commit and not earlier — under the old config they would have been unused suppressions tripping `RUF100`. |

## Bugs found and fixed during the port

All pre-existing; none introduced by the port.

- **`expand_image_patterns` could not resolve absolute globs.** It used `Path().glob()`,
  which raises `NotImplementedError: Non-relative patterns are unsupported` for any
  pattern starting with `/` — and `--images "/data/*.fits"` is the normal case. Now
  `glob.glob()`. Regression tests in `tests/test_fits.py`.
- **`utils/beam.py` globbed for `"**_**.fits"`.** Python 3.11+ rejects `**` inside a path
  component, so the entire FITS primary-beam path raised `ValueError` before doing any
  work. Now `"*_*.fits"`. Exposed by D7.
- **`core/mosaic.py` wrote an all-zero weight map.** `outwgt[c] = outwgt` where it meant
  `= weight`; single-channel input silently produced zeros, multi-channel raised a
  broadcast `ValueError`. Its only lint signal (`RUF059` on the unused `weight`) had been
  masked by D3's `**locals()`.
- **`core/mosaic.py` re-globbed its already-resolved input list** through `Path().glob`,
  which additionally rejects absolute paths. `expand_image_patterns` had already done that
  work; the block is gone.
- **`set_wcs` stamped `ORIGIN = "pfb-imaging"`** into every FITS file spimple wrote.
- **`utils/mosaic.project` indexed the frequency array with the correlation index.**
  Both `beamo((freq[c], ll, mm))` and the per-slice `freq` attribute used `c` where
  they meant `f`, so with one correlation every channel silently got channel 0's beam
  and channel 0's frequency metadata. The neighbouring `image[c, f]` shows the intended
  convention. Found by Copilot's review of #49.
- **The MS-driven parallactic-angle path counted flags wrongly and read out of bounds.**
  `_unflagged_counts` derived each timeslot's upper bound from `time_idx[i + 1]`, which
  overran on the final iteration (a numba kernel with bounds checking off, so it read
  adjacent memory rather than raising), and the caller passed `flags.astype(np.int32)`,
  making `~flags` a bitwise-not (`~0 == -1`) instead of a logical inversion — the
  'unflagged count' came out negative. Each timeslot is now counted over its own rows
  via `np.unique(..., return_counts=True)`, which is also the correct semantics under
  `--sparsify-time`: a subsampled slot counts only its own rows, not the span to the
  next sampled slot. `FLAG_ROW` is cast to `bool`. Covered by four unit tests in
  `tests/test_beam.py` — the kernel is testable without a measurement set, which is how
  the bug survived. Found by Copilot's review of #49.

## Known defects, diagnosed but not fixed

These are out of the port's scope. The diagnosis is recorded so nobody has to redo it.

### The FITS primary-beam path is internally inconsistent

`make_power_beam` cannot work for any input. `load_fits` transposes `(1, 0, 3, 2)`, and the
code then drops axis 0 as the correlation axis — which requires the frequency axis on
`NAXIS4`. But a few lines later it reads the frequency metadata from `CTYPE3`/`NAXIS3`/
`CRVAL3`. No beam cube satisfies both, so the path raises whatever you feed it. Settling
it needs a decision on the intended on-disk beam layout, which needs real MeerKAT beam
files. Covered by a skip in `tests/test_binterp.py`.

This also means `binterp` has no JimBeam support at all, despite `imconv` and `spifit`
accepting `JimBeam` as a `--beam-model` value.

### `mosaic` cannot run without a beam

`utils/mosaic.project` calls `xr.open_zarr(beam)` unconditionally, so `beam_model=None`
dies with an opaque zarr `GroupNotFoundError` rather than skipping beam weighting. A
hermetic end-to-end test needs a synthetic meerkat-beams `.bds.zarr` fixture. Covered by a
skip in `tests/test_mosaic.py`, which does cover `mosaic_info`.

### `--channel-weights-keyword` is dead on the residual path

`spifit` accepts `channel_weights_keyword` (default `WSCIMWG`) but the residual-weighting
branch hardcodes `WSCVWSUM`. The option has no effect there.

### Stray `print()` calls in `utils/beam.py`

`make_power_beam` still prints directly rather than logging. The pyscilog retirement (D3)
covered `core/*.py` only.
