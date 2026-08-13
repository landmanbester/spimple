"""Project logging.

Stdlib-based, matching meerkat_beams.utils and pfb_imaging.utils.logging.
Handlers are attached once per logger so repeated get_logger calls in the
same process do not duplicate output.
"""

import logging
import sys

_FORMAT = "%(name)s - %(asctime)s %(levelname)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return the namespaced project logger for a command.

    Args:
        name: Short command tag, e.g. "IMCONV".

    Returns:
        A logger named ``spimple.<name>`` with a stdout handler attached.
    """
    log = logging.getLogger(f"spimple.{name}")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return log


def log_options(log: logging.Logger, **options) -> None:
    """Log a command's resolved input options, one aligned line each.

    Args:
        log: Logger to write to.
        options: Parameter name/value pairs, typically ``**locals()`` called as
            the first statement of a command so it cannot drift from the signature.
    """
    log.info("Input Options:")
    if not options:
        return
    width = max(len(key) for key in options)
    for key, value in options.items():
        log.info("  %s = %s", key.rjust(width), value)
