# Python Standards & CLI Implementation Guidelines

Read this when editing or creating any `**/*.py` files.

## 1. Type Hints and Modern Python

- **Python 3.10+.** Use modern syntax (`X | Y`, `list[int]`, etc.).
- **Type hints on every function signature.**
- Use `from typing import Annotated` for Typer parameter annotations.

## 2. Lazy Imports in `cli/`

Heavy imports live in `core/` only. CLI wrappers under `cli/` must import from
`core/` **inside the function body**, never at module scope. This keeps the
lightweight install fast and lets the container-fallback pattern work (see
`architecture.md` §3).

- **fsspec backends stay lazy.** Never import `s3fs`, `gcsfs`, or `adlfs`
  directly. fsspec loads the matching backend on demand when a remote UPath is
  first accessed.

## 3. Typer Option / Argument Syntax (CRITICAL)

**Never pass `None` as the positional default to `typer.Option()`** — it raises
`AttributeError`. Follow these exact patterns:

- **Required:** `Annotated[T, typer.Option(..., help="...")]` (no `= default`).
- **Optional w/ default:** `Annotated[T, typer.Option(help="...")] = default`.
- **Optional None:** `Annotated[T | None, typer.Option(help="...")] = None`.

## 4. hip-cargo Types

- **Comma-separated lists:** use `ListInt`, `ListFloat`, `ListStr` from
  `hip_cargo`, with their matching `parse_list_*` parsers. Typer cannot natively
  handle variable-length lists as a single option; these `NewType` wrappers wrap
  `str` for Typer but parse into `list[int]` etc. at runtime.
- **UPath-backed path types:** `File`, `Directory`, `MS`, `URI` are
  `NewType(..., UPath)`. Generated CLIs use `parser=parse_upath` so the same
  signature accepts local paths and remote URIs. User functions receive a
  `universal_pathlib.UPath` and call `.open()` / `.exists()` directly.
- **The `stimela` metadata dict:** `Annotated[..., StimelaMeta(...)]` overrides
  inferred cab metadata. Use `StimelaMeta(skip=True)` to exclude a parameter from
  the generated cab YAML entirely.

## 5. Architectural Style

- Prefer functional, explicit-over-implicit code. Use classes when state or
  polymorphism genuinely helps.
- Keep `core/` implementations straightforward; let exceptions propagate. Use
  `typer.Exit(code=1)` for CLI errors in `cli/` (never in `core/`).
- Use Google-style docstrings (document Args, Returns, Raises). Keep them
  concise. Add short inline comments only when intent isn't obvious.

---

## 6. spimple specifics

### Ruff configuration

```toml
[tool.ruff]
line-length = 120
target-version = "py310"
extend-exclude = ["*.md"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
# domain names: l/m coords, Stokes I0/Ix, Gaussian2D, emaj/emin
ignore = ["E741", "N802", "N803", "N806"]
```

Do not re-add the old hand-tuned rule block. Two targeted suppressions are load-bearing
and must stay:

* `# noqa: N816` on `iFs` in `utils/convolution.py` — the `Fs`/`iFs` fftshift aliases are
  a domain idiom; only `iFs` trips mixedCase.
* `# noqa: E402` on the four deferred command imports in `cli/__init__.py` — they sit
  below the `app` construction deliberately, as in `pfb-imaging`.

### CLI help text must survive the cab round-trip

`tests/test_roundtrip.py` regenerates every wrapper from its cab and compares
line-by-line, so `help=` strings must match hip-cargo's canonical formatting exactly.

* The generator re-wraps help at `". "` sentence boundaries, one sentence per line. A
  single sentence rendering longer than 120 chars makes E501 unfixable and the whole
  regenerated file comes out unformatted (line-count mismatch). Split long help up.
* **No embedded `\n`, no bare `: `, no leading `{`.** Each produces invalid YAML and
  `generate-cabs` fails outright rather than degrading. Write plain sentences; describe
  letter-coded options as "a is the alpha map. e is the alpha error map." and so on.
* Avoid mid-help abbreviations containing periods (`e.g.`) — false sentence boundaries.
* `{current.*}` templates belong in `@stimela_output(implicit=...)`, never `info=`.

### Never name a parameter `image`

`generate-function` emits `image = get_container_image("spimple")` in the
container-fallback path. A parameter called `image` is shadowed by it and receives the
container URL. Use `images` (all four commands already do).

### hip-cargo types

Fixed-length tuples stay tuples: `psf_pars: tuple[float, float, float] | None` round-trips
as `Optional[Tuple[float, float, float]]`. `ListStr`/`ListInt`/`ListFloat` are for
genuinely variable-length lists, which Typer cannot express as a single option. Path
options use the `File` NewType with `parser=parse_upath` so they accept remote URIs.
