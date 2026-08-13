# Testing & CI/CD Guidelines

Read this when editing `tests/**` or `.github/workflows/**` files.

## 1. Round-Trip Tests

The round-trip test in `tests/test_roundtrip.py` is **not optional** — it is how
this project guarantees that `cli/*.py` and `cabs/*.yml` agree. It runs:

```
cli/<cmd>.py  ──(generate-cabs)──►  cabs/<cmd>.yml  ──(generate-function)──►  <cmd>.py
```

…then asserts the regenerated `<cmd>.py` is byte-identical (after `ruff format`)
to the original `cli/<cmd>.py`. If you write a CLI wrapper in a shape that
hip-cargo cannot round-trip, the test fails and the cab is unreliable. **Fix the
source, not the test.**

Add a new round-trip case to `tests/test_roundtrip.py` whenever you add a new
command under `cli/`.

## 2. Test Infrastructure

- Use `tempfile.TemporaryDirectory()` for isolated temp files. No test artifacts
  should ever be written to the repo directory; tests must clean up after
  themselves.
- For remote-URI behaviour, prefer fsspec's built-in `memory://` protocol — it
  needs no credentials and is fast.
- Any test hitting a real S3/GCS/Azure endpoint must be opt-in (gated on an env
  var) and excluded from required CI checks.

## 3. Mandatory Dev Workflow

After every code change run:

```bash
uv run ruff format . && uv run ruff check . --fix
```

This is non-negotiable — the pre-commit hook and CI both enforce it, and
generated code is formatted with the same configuration, so divergence breaks
the round-trip.

## 4. Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `<type>: <description>` (`feat`, `fix`, `refactor`, `perf`, `docs`, `test`,
  `ci`, `deps`, `chore`). Imperative mood, first line under 72 chars.
- The `update-cabs` bot uses `[skip checks]` to bypass required status checks;
  **do not** add that tag to human commits.

---

## 5. spimple specifics

### Fixtures

`tests/conftest.py` builds synthetic FITS cubes under `tmp_path` — no downloads, no
`tests/data/`, nothing written into the repo tree. Available fixtures:

| Fixture | What it is |
|---|---|
| `image_cube` | 4-channel power-law cube, spectral index `-0.7`, frequency on **CTYPE4** |
| `image_cube_ctype3` | the same data with frequency on **CTYPE3** |
| `residual_cube` | noise-only residual on the same grid, carrying `WSCVWSUM` |
| `beam_params` | `(emaj, emin, pa)` in degrees, coarser than every channel's native beam |
| `true_alpha` | the injected spectral index |

Both frequency-axis conventions are fixtured because astropy maps numpy axes to FITS axes
in reverse, and the code is not uniform: `mosaic_info` reads `axis=3` and `NAXIS3`, while
`imconv` and `spifit` detect the axis. A test that picks the wrong fixture fails in
confusing ways.

### Assert what the code does, not what you assume it does

Several tests here were first written against plausible but wrong assumptions. Probe the
real output before writing an assertion:

* Convolving to a **coarser** beam **increases** the summed pixel value, because the data
  are Jy/beam and the sum scales with beam area. `test_convolution_scales_flux_by_the_beam_area_ratio`
  pins that ratio, which is the meaningful regression check.
* `spifit` writes **NaN** for pixels below the fitting threshold, not zero. Mask with
  `np.isfinite(...)`; a bare `!= 0` lets NaN through and the assertion measures nothing.
* FITS data comes back big-endian (`>f8`). Compare `dtype.kind` and `dtype.itemsize`, not
  the dtype object.

### The round-trip test is not optional

`tests/test_roundtrip.py` globs `src/spimple/cli/*.py`, regenerates each wrapper through
its cab, and compares byte-for-byte. A new command is picked up automatically. If a
wrapper cannot round-trip, the committed cab is unreliable — **fix the source, never the
test**. See `python-standards.md` §6 for the help-text constraints it imposes.

### Known-skipped tests

Two end-to-end tests are `@pytest.mark.skip` with reasons in the skip message, both for
pre-existing breakage rather than anything the port introduced: the FITS primary-beam path
(`test_binterp.py`) and full `mosaic` (`test_mosaic.py`). Read
`docs/wiki/design-decisions.md` before attempting either — the diagnosis is already
written down.

### The generate-cabs pre-commit hook

The hook shells out via `uv run`, because it executes with the ambient PATH and there is
no venv on it during `git commit`. If cabs drift, the hook rewrites them and pre-commit
"fails" the commit — re-run `git add -u && git commit`.

### update-cabs needs a GitHub App

`.github/workflows/update-cabs.yml` authenticates with `secrets.APP_CLIENT_ID` /
`secrets.APP_PRIVATE_KEY`. Until that App is installed on this repo the workflow fails on
every merge to `main`. See `docs/wiki/design-decisions.md` for the setup steps.
