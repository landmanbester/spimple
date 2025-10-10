# scripts/generate_cabs.py
"""Generate all Stimela cab definitions."""
import subprocess
from pathlib import Path

CLI_MODULES = [
    "spimple.cli.imconv",
    "spimple.cli.spifit", 
    "spimple.cli.binterp",
    "spimple.cli.mosaic"
]

CABS_DIR = Path("cabs")
CABS_DIR.mkdir(exist_ok=True)

for module in CLI_MODULES:
    cmd_name = module.split(".")[-1]
    output = CABS_DIR / f"{cmd_name}.yml"
    
    print(f"Generating {output}...")
    subprocess.run([
        "cargo", "generate-cab",
        module,
        str(output)
    ], check=True)

print("✓ All cabs generated")