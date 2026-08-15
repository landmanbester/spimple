---
type: reference
title: Design decisions and known defects
description: The decision ledger for the hip-cargo port, plus the defects diagnosed but not fixed.
tags: [architecture, decisions, known-issues]
timestamp: 2026-08-15
last_verified_commit: a42fa00
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
| D11 | No pfb-imaging dependency; the tree layout is a documented cross-repo contract instead. | `pfb_imaging.utils.misc` imports jax, numba, numexpr, daskms and skimage at module scope, so importing `convolve2gaussres` from it would drag in `pfb-imaging[full]`. spimple ports the handful of helpers it needs and owns the schema in `utils/datatree.py`. |
| D12 | Partitions are derived from the data's own identity — phase centre plus grid — not from user labels. | A recipe should not have to spell out a grouping that the headers already state. Rounded to a thousandth of a pixel so header round-tripping cannot split a partition. |
| D13 | `init` homogenises resolution; `mosaic` combines. `spifit` and `mosaic` both assume an already-homogenised tree. | Convolution lives in one place. For a pfb-origin tree the equivalent upstream step is `pfb restore --gausspar`, and `spifit` refuses with a message naming it. |
| D14 | `init` writes the restore-named products `IMAGE`/`BIMAGE`/`KIMAGE` plus `PSFPARSF`, and never persists a native-resolution `MODEL`/`RESIDUAL`. | Identical variable names to `pfb restore` means `spifit` needs no origin-specific branching. The input FITS is already the archive of the native-resolution data, so copying it into the store buys nothing. |
| D15 | The band beam after mosaicking is `sum B^2 / sum B`, the beam-weighted mean — deliberately not pfb's WSUM-weighted mean. | It is the reduction consistent with the `B^2`-weighted solve that produced `IMAGE`, so `BEAM * IMAGE` is exactly the beam-weighted mean apparent image. pfb's D28 warns specifically against reducing a quantity and its weighting by different rules at the same level. Reduces to `B` where partitions agree and to `B_p` where only `p` covers. |
| D16 | The mosaic normal equations are solved directly, not by conjugate gradient. | `(sum_p mask_p B_p^2 + eta) S = sum_p B_p A_p` is diagonal — purely elementwise — so the exact solution is one division and CG only iterates towards it. `conjugate_gradient` is retained for a future non-diagonal formulation and pinned against the direct solve by a test. `--cg-max-iter`/`--cg-tol` were dropped; `--eta` stays. |
| D17 | `fit_spi_components` is vendored into `utils/fit_spi.py` and its weights carry a component axis, so `--flux-scale intrinsic` weights by `B^2`. | Image-plane noise is flat in the *apparent* image, so `IMAGE = BIMAGE / BEAM` has variance `sigma_v^2 / B(v,p)^2` and its inverse-variance weight is `B(v,p)^2 / sigma_v^2`. The beam narrows with frequency, so that weight does not factorise into a `(chan,)` vector — africanus's signature cannot express it, which is why the intrinsic path previously fitted with unity weights. With `B^2` weights the intrinsic fit is *algebraically the same normal equations* as the apparent fit with the beam in the model (`sum_v w_v (d_app - B_v M_v)^2 = sum_v (w_v B_v^2)(d_int - M_v)^2`), so the two scales become a consistency check on `BIMAGE`, `IMAGE` and `BEAM` agreeing in the tree rather than two differently-weighted estimators. Pinned to machine precision in `tests/test_fit_spi.py` and end-to-end in `test_intrinsic_and_apparent_scales_give_the_same_alpha`. |
| D18 | `spifit` has no `--flux-scale mixed`; `KIMAGE` is not fittable and asking for it, or handing over a tree carrying only it, is an early error naming the `pfb restore` rerun. | `KIMAGE` is an intrinsic model plus an *apparent* residual, so its signal and its noise sit on different flux scales and no single beam relates them. Fitting it with the beam in the model — what the removed `mixed` path did — drove `I0` up roughly as `1/B` towards the field edge; fitting it without one mis-weights the residual instead. There is no correct setting, only two wrong ones, so the option is gone rather than fixed. It has to fail early and loudly because `pfb restore` defaults to `--outputs kK`, which writes `KIMAGE` *only*, so the tree a user most likely has is exactly the one that cannot be fitted. `init` and `mosaic` still carry `KIMAGE` through the tree (D14); it is only the fit that refuses it. |
| D19 | `spifit`'s `--pb-min` is an all-bands cut: a pixel is fitted only where **every** band's `BEAM` clears the floor. | The old mask was `np.nanmin(np.where(pbeam > pb_min, image, np.nan), axis=0)`, whose `nanmin` discards exactly the bands the `where` had just masked out. A pixel therefore survived unless it failed the cut in *every* band, and was then handed to the fitter with the full band stack — including the bands whose beam was below the floor. In the apparent path those bands enter the model with a near-zero beam; in the intrinsic path they get a `B^2` weight near zero (D17) after `pfb restore` has already divided by that same small beam. Either way the field edge is fitted from a bandwidth it does not really have, which biases alpha. The beam narrows with frequency, so the highest-frequency band sets the cut. Invisible in `pfb_tree`, whose beam is band-independent, so `tests/test_spifit.py` builds a `1/freq` beam fixture; against the old code that test found fitted pixels whose band-minimum beam was `0.041` at `pb_min = 0.15`. |


## Bugs found and fixed during the DataTree refactor

- **`Gaussian2D` produced kernels 8.51 % too wide.** The exponent was
  `exp(-fwhm_conv * R)` where the FWHM parametrisation requires `exp(-4 ln2 * R)`, i.e. a
  coefficient of `0.5 * fwhm_conv**2`. Requesting a 10-pixel FWHM yielded 10.851. Verified
  against pfb's `gaussian2d`, which matches to 1.7e-16 once the width is corrected, so the
  position-angle parametrisations — which differ in form — are algebraically identical.
  Every `imconv`/`spifit` convolution before this had homogenised to a resolution 8.51 %
  coarser than requested. Pinned by `tests/test_convolution.py`.
- **`set_wcs` read the reference frequency one channel too high.** It indexed the frequency
  array with the *one-based* `CRPIX3`, so `CRVAL3` named the wrong channel, and a
  two-channel cube raised `IndexError` — two bands being `spifit`'s documented minimum. It
  survived because the only caller passing an array of frequencies was on the end-to-end
  mosaic path the tests skipped. Pinned by `tests/test_fits.py`.
- **`convolve2gaussres` tested `gausspari in [None, ()]`.** That is an elementwise
  comparison for a numpy array and raises "truth value of an array is ambiguous". Every
  legacy caller passed tuples of tuples, so it only surfaced once `restore_products` began
  passing arrays.
- **`ref_wcs.array_shape` was unpacked in both orders** — `(nyo, nxo)` in `core/mosaic.py`
  and `(nxo, nyo)` in `utils/mosaic.py`, invisible only for square outputs. Both call sites
  are gone with the rewrite.

## A regression introduced and reverted, worth remembering

Fixing the `Gaussian2D` width, its **support** was also changed from `nsigma * FWHM` to
`nsigma * sigma`, to match both its docstring and pfb's `gaussian2d`. That broke the
*deconvolution* path badly enough to displace sources.

`convolve2gaussres` divides by the kernel's transform when `gausspari` is given. A
Gaussian's transform decays to about 1e-14 by Nyquist; truncating the kernel at 5 sigma
leaves a step of `exp(-12.5) = 3.7e-6` of peak whose spectral ripple is many orders of
magnitude larger than that floor. The division amplifies it into ringing that moves the
peak to *twice* its offset from the image centre. At 5 FWHM (11.8 sigma) the step is about
4e-31 and the division is clean.

Caught only by running `spimple init` end to end — a source at `x=20` rendered at `x=24` —
because no unit test covered `gausspari != gaussparf`. That gap is now pinned by
`test_convolve2gaussres_preserves_position_when_deconvolving`. **The same latent issue
exists in pfb-imaging**, whose `gaussian2d` uses `nsigma * sigma_maj` while
`restore_products` performs the identical division: reported as
[ratt-ru/pfb-imaging#312](https://github.com/ratt-ru/pfb-imaging/issues/312), with the
displacement measured at every image size from 32 to 1024 pixels.

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

### `BEAM` from pfb-imaging is `B/n`, not the primary beam

pfb's D22 folds the wgridder's geometric Jacobian into the stored beam, so a pfb tree's
`BEAM` is the effective image-plane response `B/n` with `n = sqrt(1 - l^2 - m^2)`.
Partitions carry `beam_includes_n: True` to say so. `spimple spifit` therefore applies
`B/n` where it means `B`: about 0.2 % at a 5-degree field-of-view edge, 1.5 % at 10
degrees. `spifit` logs a warning naming the correction (`B = BEAM * n`, evaluated on the
tree's own grid from `cell_rad`, `l0`, `m0`) rather than silently absorbing it. Deferred to
a follow-up PR by decision, not oversight.

### Two cab-contract issues in the DataTree commands

Both are declaration problems, not defects in the science path, and are deferred to a
follow-up PR.

1. **`init`'s cab advertises `implicit="{current.output-filename}_I.dt"`**, but `core.init`
   derives the store suffix from the FITS STOKES axis, so a `Q` or `XXYY` input produces
   `..._Q.dt` while the cab tells Stimela to expect `..._I.dt`. Correct for the Stokes I
   case, which is everything spimple is normally pointed at. Fixing it needs either an
   explicit `--product` option or a declared output that carries no suffix.
2. **`--beam-model` is declared `File` but accepts the literal `JimBeam`**, so Stimela may
   try to validate or stage the sentinel as a path. `imconv` has always done the same, so
   changing it either diverges from that convention for one command or changes both.

### The `.bds.zarr` beam backend is unverified against a real file

`utils/beamsource._bds_beam` assumes the `l_beam`/`m_beam`/`chan`/`BEAM` names the
pre-refactor `utils/mosaic.project` used, with a fallback to the `X`/`Y` spelling pfb's
orientation wiki documents — both spellings have been in use. No real meerkat-beams dataset
was available, and the test fixture only pins the reader against our own writer. Run it
against a real `.bds.zarr` before advertising the backend.

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

### The `B^2` fit weight ignores how many pointings covered a pixel

For a band mosaicked from overlapping pointings the true inverse variance of `IMAGE` is
`SPATIALWGT = sum_p B_p^2 + eta`, not `BEAM^2 = (sum_p B_p^2 / sum_p B_p)^2`. Two
identical overlapping pointings give `SPATIALWGT = 2 B^2` but `BEAM^2 = B^2`, so the
overlap is under-weighted by exactly the factor the mosaic bought. `BEAM^2` is used
anyway because it is what makes the intrinsic and apparent fits identical (D17), and
because the apparent path shares the same implicit flat-noise assumption — moving to
`SPATIALWGT` means changing both paths together. The two coincide for single-pointing
bands, which is every tree the tests cover.

### Stray `print()` calls in `utils/beam.py`

`make_power_beam` still prints directly rather than logging. The pyscilog retirement (D3)
covered `core/*.py` only.
