# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.6] - 2026-08-13

### CI

- **deps**: Bump actions/cache from 4 to 5 (#40)
- **deps**: Bump actions/checkout from 5 to 6 (#39)
- **deps**: Bump docker/build-push-action from 5 to 6 (#37)
- **deps**: Bump actions/checkout from 4 to 5 (#35)
- **deps**: Bump actions/attest-build-provenance from 1 to 3 (#34)
- **deps**: Bump astral-sh/setup-uv from 6 to 7 (#33)

### Changed

- Replace pyscilog with stdlib logging
- Extract shared helpers into a package-level utils/

### Dependencies

- **deps-dev**: Update uv-build requirement (#41)
- **deps**: Update uv-build requirement (#36)

### Fixed

- Pass beam settings explicitly instead of a duck-typed opts object
- **mosaic**: Write per-frequency weights to the weight map
- **fits**: Resolve absolute glob patterns in expand_image_patterns
- Unbreak imports and drop the removed expand_patterns callback

### Other

- Prefer trusted publisher (#38)


## [0.0.6] - 2026-08-13

### CI

- **deps**: Bump actions/setup-python from 5 to 6 (#31)
- **deps**: Bump actions/checkout from 4 to 5 (#29)
- **deps**: Bump astral-sh/setup-uv from 4 to 6 (#28)

### Other

- Bump to 0.0.6
- Convert to use hip-cargo structure [WIP] (#32)

* basic cli

* restructure package directories

* add generate_cabs.py

* replace argparse versions by clean function definitions for core apps

* fix formatting issues

* add cab definitions

* install full package in CI

* Add Dockerfile and move cabs into src directory

* add docker-publish workflow

* fix docker-publish worklow
- Auto-lint with ruff (#30)

* auto-lint with ruff

* do not require tests on push

* fix beam_cube_dde_dask import

* Update spimple/fits.py

Co-authored-by: Athanaseus Javas Ramaila <aramaila@ska.ac.za>

---------

Co-authored-by: Athanaseus Javas Ramaila <aramaila@ska.ac.za>


## [0.0.5] - 2025-08-11

### Other

- Bump to 0.0.5
- Fix CI issues and add initial mosaicing functionality (#26)

* add missing imports, don't use dask by default

* allow 3.8

* select specific axes instead of squeezing in binterp

* avoid zero chunks in spifit

* create new header with correct freq axis in binterp (#24)

* create new header with correct freq axis in binterp

* switch to pyproject.toml

* update project to use poetry and upgrade testing matrix

* actally add pyproject.toml

* pin python between 3.10 and 3.12 and fix dask-ms version

* convert ant_scale to dask array if using dask for beam interpolation

* 📝 Add docstrings to `binterp-fixes` (#25)

Docstrings generation was requested by @landmanbester.

* https://github.com/landmanbester/spimple/pull/21#issuecomment-2948774990

The following files were modified:

* `spimple/apps/power_beam_maker.py`
* `spimple/apps/spi_fitter.py`
* `spimple/utils.py`

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>

* Update .github/workflows/publish.yml

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>

* make pytest a dev dependency and remove unnecessary return statement

* add option to imconv write out beam**2 image for mosaicing

* allow PA and BPA keywords in header

* test claude code fix

* add mosaic app

* comment katpoint import

* split out fits utils into a separate file. Don't use load_fits/save_fits in mosaic app

* add warning if intrepolating outside beam freq limits

* PEP 621 compliance + tbump

* switch to uv instead of poetry

* fix import issues

* add dependabot

---------

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>
- Fixes for spimple-binterp (#21)

* add missing imports, don't use dask by default

* allow 3.8

* select specific axes instead of squeezing in binterp

* avoid zero chunks in spifit

* create new header with correct freq axis in binterp (#24)

* create new header with correct freq axis in binterp

* switch to pyproject.toml

* update project to use poetry and upgrade testing matrix

* actally add pyproject.toml

* pin python between 3.10 and 3.12 and fix dask-ms version

* convert ant_scale to dask array if using dask for beam interpolation

* 📝 Add docstrings to `binterp-fixes` (#25)

Docstrings generation was requested by @landmanbester.

* https://github.com/landmanbester/spimple/pull/21#issuecomment-2948774990

The following files were modified:

* `spimple/apps/power_beam_maker.py`
* `spimple/apps/spi_fitter.py`
* `spimple/utils.py`

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>

* Update .github/workflows/publish.yml

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>

* make pytest a dev dependency and remove unnecessary return statement

* add option to imconv write out beam**2 image for mosaicing

* allow PA and BPA keywords in header

* test claude code fix

---------

Co-authored-by: coderabbitai[bot] <136622811+coderabbitai[bot]@users.noreply.github.com>
- Add katbeam S band option (#20)

* add S band to katbeam options

* debug

* add residual after fit as output

* fix uninitialised beam image when using katbeam

* np.float64 -> float in spimple-spifit parser

---------

Co-authored-by: landmanbester <lbester@ska.ac.za>
Co-authored-by: Athanaseus Javas Ramaila <aramaila@ska.ac.za>
- Ostrich (#15)

* Ignoring invalid division by zero or nan
An ostrich approach to #11

* Dropzweights (#14)

* remove full nan slices

* don't acr by default

* look for max of model only at unflagged locations

* drop invalid gausspars

* accommodate different keys for different header

* idx -> fidx

* fresq -> freqs

* Exclude any bands that might be awful with `-ds`

* pep8 indetation level

* syntax error fix

* Update spi_fitter.py

* Update spi_fitter.py

---------

Co-authored-by: landmanbester <lbester@ska.ac.za>

* Fix freq dimension check
Fix empty gauspar checking

---------

Co-authored-by: Athanaseus Javas Ramaila <aramaila@ska.ac.za>
Co-authored-by: landmanbester <lbester@ska.ac.za>
- Dropzweights (#14)

* remove full nan slices

* don't acr by default

* look for max of model only at unflagged locations

* drop invalid gausspars

* accommodate different keys for different header

* idx -> fidx

* fresq -> freqs

* Exclude any bands that might be awful with `-ds`

* pep8 indetation level

* syntax error fix

* Update spi_fitter.py

* Update spi_fitter.py

---------

Co-authored-by: landmanbester <lbester@ska.ac.za>
- `model` & `residual` can be a list of single frequency images (#12)

* Allow list of model/residual image to be specified

* hdr of cube

* Channel weights from headers

* get info from residual header

* BUmp version

* Update ci.yml

* python_requires >=3.8

* Update ci.yml
- Add option for manual freqs in the spi-fitter (#9)

* Add option for manual freqs in the spi-fitter

* Notify users that cw is optional
- Fix channel weights issue. Fix typo in log


## [0.0.3] - 2022-07-08

### Other

- Update imconv usage docs
- Bump version
- Slightly dilate emaj and emin by default. Use double precision for stability
- Typo
- Take maximum instead of average when doing circ-psf in imconv
- Make target resolution is > initial resolution


## [0.0.2] - 2022-06-17

### Other

- Bump version


## [0.0.1] - 2022-06-17

### Other

- Move apps to submodule
- Correct token attempt 2
- Correct secret name
- Rudimentary docs for executables
- Add imconv description to readme
- Add JimBeam option to image convolver
- Manually add extra axis if  single band image added in
- Format strings
- Add convolve2guassres test
- Add github workflows
- Add cleanup of spi_fitter
- Restructure image convolver
- Initial commit


[0.0.6]: https://github.com/landmanbester/spimple/compare/v0.0.6...v0.0.6
[0.0.6]: https://github.com/landmanbester/spimple/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/landmanbester/spimple/compare/v0.0.3...v0.0.5
[0.0.3]: https://github.com/landmanbester/spimple/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/landmanbester/spimple/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/landmanbester/spimple/releases/tag/v0.0.1
