# scripts/generate_cabs.py
"""Generate all Stimela cab definitions."""

from pathlib import Path
import subprocess

CLI_MODULES = ["spimple.cli.imconv", "spimple.cli.spifit", "spimple.cli.binterp", "spimple.cli.mosaic"]

# Cab definitions are stored in the package to be included in distribution
CABS_DIR = Path("src/spimple/cabs")
CABS_DIR.mkdir(exist_ok=True)

for module in CLI_MODULES:
    cmd_name = module.split(".")[-1]
    output = CABS_DIR / f"{cmd_name}.yml"

    print(f"Generating {output}...")
    subprocess.run(["cargo", "generate-cab", module, str(output)], check=True)

print("✓ All cabs generated")
