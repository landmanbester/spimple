"""Every command must at least be reachable and describe itself."""

import re

import pytest
from typer.testing import CliRunner

from spimple.cli import app

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI styling from help output.

    rich styles each option name as several spans, so a coloured
    "--images" contains no literal "--images" substring. Colour is on in
    CI and off in a plain local run, which is exactly the kind of
    difference that passes locally and fails on the runner.
    """
    return _ANSI.sub("", text)


COMMANDS = ["init", "spifit", "imconv", "binterp", "mosaic"]

runner = CliRunner()


def test_group_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_init_help_lists_the_key_options(monkeypatch):
    # rich truncates option names to the terminal width, so widen it or
    # --output-filename renders as --output-filen...
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    out = plain(result.stdout)
    for flag in ("--images", "--output-filename", "--psf-pars", "--beam-model", "--nworkers"):
        assert flag in out


def test_spifit_help_shows_the_store_and_flux_scale_options(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["spifit", "--help"])

    assert result.exit_code == 0
    out = plain(result.stdout)
    assert "--store" in out
    assert "--flux-scale" in out
    assert "--beam-model" not in out
