"""Round-trip: cli/*.py -> cabs/*.yml -> cli/*.py, byte for byte.

This is how the project guarantees the committed cabs match the CLI source. If
a wrapper is written in a shape hip-cargo cannot regenerate, the cab is
unreliable -- fix the source, never the test.

Covers every command under cli/. A new command is picked up automatically.
"""

import tempfile
from pathlib import Path

import pytest
from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function

COMMANDS = sorted(p.stem for p in Path("src/spimple/cli").glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("command", COMMANDS)
def test_roundtrip(command):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cab_dir = tmpdir / "cabs"
        cab_dir.mkdir()

        cli_module = Path(f"src/spimple/cli/{command}.py")
        assert cli_module.exists(), f"missing CLI module: {cli_module}"

        generate_cabs(module=[cli_module], output_dir=cab_dir, image=None)

        cab_file = cab_dir / f"{command}.yml"
        assert cab_file.exists(), f"generate-cabs did not produce {cab_file}"

        generated_file = tmpdir / f"{command}_roundtrip.py"
        generate_function(cab_file, generated_file, config_file=Path("pyproject.toml"))
        assert generated_file.exists(), "generate-function produced no output"

        generated_code = generated_file.read_text()
        compile(generated_code, str(generated_file), "exec")

        original_lines = cli_module.read_text().splitlines()
        generated_lines = generated_code.splitlines()

        assert len(original_lines) == len(generated_lines), (
            f"Line count mismatch for {command}: original has {len(original_lines)} lines, "
            f"generated has {len(generated_lines)} lines"
        )
        for i, (orig, gen) in enumerate(zip(original_lines, generated_lines, strict=True), 1):
            assert orig == gen, f"Line {i} differs in {command}:\n  Original:  {orig}\n  Generated: {gen}"
