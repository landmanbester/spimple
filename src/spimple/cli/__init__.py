import typer
from typing import ParamSpec, TypeVar, Callable

# Main app
app = typer.Typer(
    name="spimple",
    help="spimple: Radio interferometry image post-processing tools",
    no_args_is_help=True,
)

@app.callback()
def callback():
    """
    spimple: Radio interferometry image post-processing tools
    """
    pass

# Import and register commands
from spimple.cli.spifit import spifit
app.command(name="spifit")(spifit)

from spimple.cli.imconv import imconv
app.command(name="imconv")(imconv)

from spimple.cli.binterp import binterp
app.command(name="binterp")(binterp)

from spimple.cli.mosaic import mosaic
app.command(name="mosaic")(mosaic)