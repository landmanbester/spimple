---
type: index
title: spimple LLM wiki
description: Progressive-disclosure listing of the in-repo knowledge bundle.
timestamp: 2026-08-13
last_verified_commit: b7bfbc4
---

# spimple LLM wiki

In-repo knowledge bundle in the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
style: plain markdown plus YAML frontmatter, readable by humans without tools and by
agents without SDKs. The primary reader is an LLM agent; humans are a close second.

**This is the canonical reference for what is implemented**, including what is
*deliberately* not implemented and what is known broken. Specs and plans are ephemeral
process artefacts under `docs/superpowers/`, are gitignored, and must never be cited.

**Verification contract:** every page's frontmatter carries `last_verified_commit` — the
commit its claims were last checked against. To assess staleness:

```bash
git diff <stamp>..HEAD -- <files the page covers>
```

**Maintenance rule** (also in `CLAUDE.md`): if your change invalidates or extends a page,
update the page and refresh its stamp in the same session.

## Pages

| Page | Covers | Read when |
|---|---|---|
| [design-decisions.md](design-decisions.md) | The decision ledger, known defects, and the outstanding GitHub App setup | Before "fixing" something that looks wrong, or changing structure |
| [datatree-contract.md](datatree-contract.md) | The DataTree store layout shared with pfb-imaging, its invariants, and which pfb variables spimple reads | Before touching `utils/datatree.py`, `core/init.py`, `core/mosaic.py`, `core/spifit.py`, or anything that writes to a store |
| [fits-and-beams.md](fits-and-beams.md) | Frequency-axis conventions, per-channel beam keywords, beam-model layouts, JimBeam bands | Touching `utils/fits.py`, `utils/beam.py`, or any FITS fixture |

## Not covered here

- **How to edit this codebase** (linting, commit format, Typer patterns, test fixtures):
  `.claude/rules/*.md` — harness instructions, kept separately.
- **The hip-cargo format itself**: <https://github.com/landmanbester/hip-cargo>.
- **Release mechanics** (tbump, git-cliff, the `update-cabs` workflow): summarised in
  [design-decisions.md](design-decisions.md).
