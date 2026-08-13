"""Every command must at least be reachable and describe itself."""

import pytest
from typer.testing import CliRunner

from spimple.cli import app

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
    for flag in ("--images", "--output-filename", "--psf-pars", "--beam-model", "--nworkers"):
        assert flag in result.stdout


def test_spifit_help_shows_the_store_and_flux_scale_options(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["spifit", "--help"])

    assert result.exit_code == 0
    assert "--store" in result.stdout
    assert "--flux-scale" in result.stdout
    assert "--beam-model" not in result.stdout
