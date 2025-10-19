# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Spimple is a radio astronomy image post-processing tool for spectral index fitting. The package provides four main command-line executables:

- `spimple-spifit`: Fits spectral index models to image cubes with optional convolution and primary beam correction
- `spimple-imconv`: Convolves images to common resolution with optional primary beam correction
- `spimple-binterp`: Primary beam interpolation and conversion to power beams
- `spimple-mosaic`: Mosaics multiple FITS images together onto a common coordinate grid

## Architecture

The codebase follows a simple structure:
- `/spimple/apps/`: Contains the four main application entry points (spi_fitter.py, image_convolver.py, power_beam_maker.py, mosaic.py)
- `/spimple/utils.py`: Core utility functions for image processing, convolution, and FITS handling
- All applications use `pyscilog` for logging
- Heavy computation relies on `dask` arrays and `africanus` radio astronomy tools

Key dependencies: astropy, dask, africanus, katbeam, pyscilog, reproject

## Development Commands

**Install Poetry (if not already installed):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Install dependencies:**
```bash
poetry install --with dev
```

**Build package:**
```bash
poetry build
```

**Run tests:**
```bash
poetry run pytest tests/
```

**Run single test:**
```bash
poetry run pytest tests/test_convolve2gaussres.py
```

**Test specific function:**
```bash
poetry run pytest tests/test_convolve2gaussres.py::test_convolve2gaussres
```

**Add new dependency:**
```bash
poetry add package-name
```

**Add development dependency:**
```bash
poetry add --group dev package-name
```

**Install production dependencies only:**
```bash
poetry install --only main
```

## Key Implementation Notes

- All three apps follow hip-cargo pattern and use typer to create lightweight CLI interfaces with lazy imports to heavy dependencies
- Image processing uses FITS format via astropy
- Convolution operations use FFT-based methods with dask arrays for memory efficiency
- Primary beam models support real/imaginary FITS patterns and JimBeam models
- Spectral index fitting uses africanus.model.spi components with weighted least squares
