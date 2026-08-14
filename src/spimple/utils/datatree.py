"""The DataTree schema layer.

The only module that knows the store layout. Everything else goes through it,
so the cross-repo contract with pfb-imaging has exactly one owner. See
``docs/wiki/datatree-contract.md``.

Node naming, array order, units and iteration order all matter and are pinned
by ``tests/test_datatree.py``:

* nodes are ``band{bandid:04d}_time{timeid:04d}`` with ``part{pid:04d}`` children;
* image variables are ``(corr, y, x)``;
* ``PSFPARSN``/``PSFPARSF`` are ``(emaj, emin, pa)`` in pixels, pixels, radians;
* bands are iterated sorted by ``bandid``, never by ``freq_out`` -- effective
  frequencies are data-dependent and asymmetric flagging can invert two bands.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

# CLI letter -> band-node variable, matching pfb's restore contract. The B
# prefix follows the tree's convention for beam-attenuated quantities.
PRODUCT_VARS = {"a": "BIMAGE", "i": "IMAGE", "k": "KIMAGE"}

BPAR = ["BMAJ", "BMIN", "BPA"]


def band_node_name(bandid: int, timeid: int) -> str:
    """Return the group path of a band node."""
    return f"band{bandid:04d}_time{timeid:04d}"


def partition_node_name(pid: int) -> str:
    """Return the group name of a partition child."""
    return f"part{pid:04d}"


def create_store(url: str, attrs: dict, overwrite: bool = False) -> None:
    """Create the store root, carrying the root attributes.

    Args:
        url: Store path.
        attrs: Root attributes.
        overwrite: Replace an existing store.

    Raises:
        FileExistsError: If the store exists and overwrite is False.
    """
    if Path(url).exists() and not overwrite:
        raise FileExistsError(f"{url} exists, pass overwrite to replace it")
    xr.Dataset(attrs=attrs).to_zarr(url, mode="w", consolidated=False)


def open_store(url: str) -> xr.DataTree:
    """Open a store lazily, without chunking.

    ``consolidated=False`` matches how every write here is made; without it
    xarray tries the consolidated metadata first, fails, and warns on every
    open.
    """
    return xr.open_datatree(url, engine="zarr", chunks=None, consolidated=False)


def write_node(url: str, node: str, data_vars: dict, attrs: dict, coords: dict) -> None:
    """Write or extend one group, merging attributes.

    ``to_zarr(mode="a")`` replaces a group's attrs wholesale, so an incremental
    write must start from whatever the group already carries. Callers therefore
    pass only the attrs they are adding.

    Args:
        url: Store path.
        node: Group path relative to the root, e.g. ``band0000_time0000/part0000``.
        data_vars: Mapping suitable for ``xr.Dataset``.
        attrs: Attributes to merge into the group's existing attributes.
        coords: Coordinates for the dataset.
    """
    existing = {}
    try:
        existing = dict(xr.open_zarr(url, group=node, chunks=None, consolidated=False).attrs)
    except (FileNotFoundError, KeyError, OSError):
        pass
    xr.Dataset(data_vars, coords=coords, attrs={**existing, **attrs}).to_zarr(
        url, group=node, mode="a", consolidated=False
    )


def timeids(dt: xr.DataTree) -> list[int]:
    """Return the sorted distinct timeids present in the tree."""
    return sorted({int(dt[name].ds.attrs["timeid"]) for name in dt.children if name.startswith("band")})


def band_nodes(dt: xr.DataTree, timeid: int | None = None) -> list[str]:
    """Return band-node names sorted by bandid, optionally filtered by timeid."""
    names = [name for name in dt.children if name.startswith("band")]
    if timeid is not None:
        names = [name for name in names if int(dt[name].ds.attrs["timeid"]) == timeid]
    return sorted(names, key=lambda name: int(dt[name].ds.attrs["bandid"]))


def partition_nodes(dt: xr.DataTree, node: str) -> list[str]:
    """Return the partition child names of a band node, sorted by pid."""
    return sorted(name for name in dt[node].children if name.startswith("part"))


def require_vars(ds: xr.Dataset, names: Sequence[str], node: str, hint: str) -> None:
    """Raise unless every named variable is present.

    Args:
        ds: The band or partition dataset.
        names: Required variable names.
        node: Node path, for the message.
        hint: What the user should do, named explicitly.

    Raises:
        ValueError: If any variable is missing.
    """
    missing = [name for name in names if name not in ds]
    if missing:
        raise ValueError(f"{node} has no {', '.join(missing)}; {hint}")


def check_homogeneous(datasets: Sequence[xr.Dataset]) -> np.ndarray:
    """Return the common PSFPARSF, raising unless every band agrees.

    A spectral index fit needs every band at one resolution. A tree is
    homogeneous when ``spimple init`` or ``pfb restore --gausspar`` produced it.

    Args:
        datasets: The band datasets, in bandid order.

    Returns:
        The shared ``(ncorr, 3)`` resolution in pixels, pixels, radians.

    Raises:
        ValueError: If PSFPARSF is absent anywhere, or the bands disagree.
    """
    for i, ds in enumerate(datasets):
        if "PSFPARSF" not in ds:
            raise ValueError(
                f"band {i} has no PSFPARSF; run spimple init or pfb restore --gausspar to "
                "homogenise the resolution before fitting"
            )
    ref = np.asarray(datasets[0].PSFPARSF.values, dtype=float)
    for i, ds in enumerate(datasets[1:], 1):
        pars = np.asarray(ds.PSFPARSF.values, dtype=float)
        if not np.allclose(ref, pars, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"band {i} is at a different resolution to band 0 ({pars[0]} vs {ref[0]} pixels); "
                "run spimple init or pfb restore --gausspar to homogenise them"
            )
    return ref


def psfpars_from_header(hdr, nband: int, ncorr: int, cell_deg: float) -> np.ndarray:
    """Synthesise PSFPARSN from a FITS header's beam cards.

    FITS input carries no PSF array, so the native resolution comes from the
    per-plane ``BMAJ{i}``/``BMIN{i}``/``BPA{i}`` cards, falling back to the
    scalar ``BMAJ``/``BMIN``/``BPA`` broadcast over every band.

    Args:
        hdr: FITS header.
        nband: Number of bands.
        ncorr: Number of correlations.
        cell_deg: Cell size in degrees.

    Returns:
        ``(nband, ncorr, 3)`` in pixels, pixels, radians.

    Raises:
        ValueError: If the header carries no beam parameters at all.
    """
    pars = np.full((nband, 3), np.nan, dtype=np.float64)
    for band in range(nband):
        key = f"BMAJ{band + 1}"
        if key in hdr:
            pa = hdr.get(f"BPA{band + 1}", hdr.get(f"PA{band + 1}", 0.0))
            pars[band] = (hdr[key] / cell_deg, hdr[f"BMIN{band + 1}"] / cell_deg, np.deg2rad(pa))
    if np.isnan(pars).any():
        if "BMAJ" not in hdr:
            raise ValueError("No BMAJ in the header and no per-plane BMAJ cards; pass psf-pars explicitly")
        scalar = (hdr["BMAJ"] / cell_deg, hdr["BMIN"] / cell_deg, np.deg2rad(hdr.get("BPA", 0.0)))
        pars = np.where(np.isnan(pars), np.array(scalar)[None], pars)
    return np.tile(pars[:, None, :], (1, ncorr, 1))
