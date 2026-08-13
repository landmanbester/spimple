"""The container-image indirection hip-cargo resolves cabs and fallback through."""

from importlib.metadata import version

from hip_cargo import get_container_image


def test_container_image_is_resolvable():
    image = get_container_image("spimple")

    assert image == "ghcr.io/landmanbester/spimple:latest"


def test_cabs_subpackage_is_importable():
    """Stimela's `_include: (spimple.cabs)imconv.yml` needs this to be a package."""
    import spimple.cabs

    assert spimple.cabs is not None


def test_version_strings_agree():
    import spimple

    assert spimple.__version__ == version("spimple")
