"""The stdlib-logging replacement for pyscilog."""

import logging

from spimple.utils.logging import get_logger, log_options


def test_get_logger_is_namespaced():
    log = get_logger("IMCONV")

    assert log.name == "spimple.IMCONV"
    assert log.level == logging.DEBUG


def test_get_logger_is_idempotent():
    first = get_logger("SPIFIT")
    second = get_logger("SPIFIT")

    assert first is second
    assert len(first.handlers) == len(second.handlers)


def test_log_options_emits_one_line_per_option(caplog):
    log = get_logger("MOSAIC")

    with caplog.at_level(logging.INFO, logger="spimple.MOSAIC"):
        log_options(log, band="L", nthreads=4)

    messages = [record.message for record in caplog.records]
    assert messages[0] == "Input Options:"
    assert any("band = L" in message for message in messages)
    assert any("nthreads = 4" in message for message in messages)
