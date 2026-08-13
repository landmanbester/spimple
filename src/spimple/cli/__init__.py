import typer

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


# Import and register commands
from spimple.cli.spifit import spifit  # noqa: E402

app.command(name="spifit")(spifit)

from spimple.cli.imconv import imconv  # noqa: E402

app.command(name="imconv")(imconv)

from spimple.cli.binterp import binterp  # noqa: E402

app.command(name="binterp")(binterp)

from spimple.cli.mosaic import mosaic  # noqa: E402

app.command(name="mosaic")(mosaic)
