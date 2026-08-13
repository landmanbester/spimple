"""Guards the import-time breakage class that shipped in the partial port.

Each of these was a real ModuleNotFoundError on the pre-port tree.
"""

import importlib

import pytest

CORE_MODULES = [
    "spimple.core.binterp",
    "spimple.core.imconv",
    "spimple.core.mosaic",
    "spimple.core.spifit",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_cli_app_imports():
    from spimple.cli import app

    assert app is not None
