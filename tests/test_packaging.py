"""The container-image indirection hip-cargo resolves cabs and fallback through."""

from importlib.metadata import version

from hip_cargo import get_container_image


def test_container_image_is_resolvable():
    """The image resolves to this project's GHCR repository, whatever the tag.

    The tag is deliberately not pinned here: the documented workflow rewrites it
    to the branch name on feature branches and to the version on release, so
    asserting `:latest` would fail anyone following that process.
    """
    image = get_container_image("spimple")

    assert image is not None, "get_container_image returned None; is _container_image.py present?"
    repository, _, tag = image.rpartition(":")
    assert repository == "ghcr.io/landmanbester/spimple"
    assert tag, "the image must carry an explicit tag so stimela can match it to a cab"


def test_cabs_subpackage_is_importable():
    """Stimela's `_include: (spimple.cabs)imconv.yml` needs this to be a package."""
    import spimple.cabs

    assert spimple.cabs is not None


def test_version_strings_agree():
    import spimple

    assert spimple.__version__ == version("spimple")
