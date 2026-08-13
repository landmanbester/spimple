"""Every command must at least be reachable and describe itself."""

import pytest
from typer.testing import CliRunner

from spimple.cli import app

COMMANDS = ["spifit", "imconv", "binterp", "mosaic"]

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
