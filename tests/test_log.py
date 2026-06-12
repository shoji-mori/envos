"""
Tests for envos/log.py (P2-C: simplified logging).

Covers:
- setup(): idempotent — calling it N times leaves exactly 1 StreamHandler
- setup(logfile=...): adds exactly 1 FileHandler; the file is created
- setup(file_level=...): file handler uses independent level
- update_logfile(): emits DeprecationWarning and still configures logging
- StandardFormatter: no 'is this used?' artefact in output
"""

import logging
import pathlib
import warnings

import pytest


def _count_file_handlers(lg):
    """Return the number of FileHandler instances on a logger."""
    return sum(1 for h in lg.handlers if isinstance(h, logging.FileHandler))


def _count_stream_handlers(lg):
    """Return the number of StreamHandler (but not FileHandler) instances."""
    return sum(
        1 for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )


# ---------------------------------------------------------------------------
# setup() idempotency: calling it 3 times must not accumulate handlers.
# ---------------------------------------------------------------------------

def test_setup_idempotent_stream_handler():
    """
    Calling setup() three times must leave exactly one StreamHandler on the
    'envos' logger (no handler proliferation).
    """
    import envos.log as log

    log.setup(level="INFO")
    log.setup(level="WARNING")
    log.setup(level="INFO")

    assert _count_stream_handlers(log.logger) == 1, (
        f"Expected exactly 1 StreamHandler after 3 setup() calls, "
        f"got {_count_stream_handlers(log.logger)}"
    )


def test_setup_stream_level():
    """setup(level='DEBUG') sets the stream handler to DEBUG."""
    import envos.log as log

    log.setup(level="DEBUG")

    stream_handlers = [
        h for h in log.logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers, "No StreamHandler found after setup()"
    for h in stream_handlers:
        assert h.level == logging.DEBUG

    # Restore
    log.setup(level="INFO")


# ---------------------------------------------------------------------------
# setup(logfile=...): file creation and single FileHandler.
# ---------------------------------------------------------------------------

def test_setup_with_logfile_creates_file(tmp_path):
    """setup(logfile=...) must create the log file and add exactly 1 FileHandler."""
    import envos.log as log

    logfile = tmp_path / "envos.log"
    log.setup(logfile=logfile)

    assert logfile.exists(), "Log file must be created by setup(logfile=...)"
    assert _count_file_handlers(log.logger) == 1, (
        f"Expected exactly 1 FileHandler, got {_count_file_handlers(log.logger)}"
    )

    # Cleanup
    log.setup(level="INFO")


def test_setup_with_logfile_idempotent(tmp_path):
    """Calling setup(logfile=...) twice leaves exactly 1 FileHandler."""
    import envos.log as log

    logfile = tmp_path / "envos.log"
    log.setup(logfile=logfile)
    log.setup(logfile=logfile)

    assert _count_file_handlers(log.logger) == 1, (
        f"Expected exactly 1 FileHandler after 2 setup() calls, "
        f"got {_count_file_handlers(log.logger)}"
    )

    # Cleanup
    log.setup(level="INFO")


def test_setup_without_logfile_has_no_file_handler():
    """setup() without logfile must not add any FileHandler."""
    import envos.log as log

    log.setup(level="INFO")

    assert _count_file_handlers(log.logger) == 0, (
        f"Expected 0 FileHandlers when logfile=None, "
        f"got {_count_file_handlers(log.logger)}"
    )


# ---------------------------------------------------------------------------
# setup(file_level=...): independent file handler level.
# ---------------------------------------------------------------------------

def test_setup_file_level_independent(tmp_path):
    """
    setup(level='WARNING', logfile=..., file_level='DEBUG') sets stream to WARNING
    and file handler to DEBUG independently.
    """
    import envos.log as log

    logfile = tmp_path / "envos.log"
    log.setup(level="WARNING", logfile=logfile, file_level="DEBUG")

    stream_handlers = [
        h for h in log.logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    file_handlers = [h for h in log.logger.handlers if isinstance(h, logging.FileHandler)]

    assert stream_handlers, "No StreamHandler found"
    assert stream_handlers[0].level == logging.WARNING, (
        f"Stream handler should be WARNING, got {stream_handlers[0].level}"
    )
    assert file_handlers, "No FileHandler found"
    assert file_handlers[0].level == logging.DEBUG, (
        f"File handler should be DEBUG, got {file_handlers[0].level}"
    )

    # Cleanup
    log.setup(level="INFO")


# ---------------------------------------------------------------------------
# update_logfile(): DeprecationWarning + functional forwarding to setup().
# ---------------------------------------------------------------------------

def test_update_logfile_emits_deprecation_warning(tmp_path):
    """update_logfile() must emit a DeprecationWarning."""
    import envos.log as log

    logfile = tmp_path / "dep.log"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log.update_logfile(filepath=logfile)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "update_logfile() must emit a DeprecationWarning"

    # Cleanup
    log.setup(level="INFO")


def test_update_logfile_still_configures_logging(tmp_path):
    """update_logfile(filepath=...) still creates the log file via setup()."""
    import envos.log as log

    logfile = tmp_path / "dep_test.log"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        log.update_logfile(filepath=logfile)

    assert logfile.exists(), "update_logfile() must create the log file"
    assert _count_file_handlers(log.logger) == 1, (
        f"Expected 1 FileHandler after update_logfile(), "
        f"got {_count_file_handlers(log.logger)}"
    )

    # Cleanup
    log.setup(level="INFO")


# ---------------------------------------------------------------------------
# StandardFormatter: no debug artifact in output.
# ---------------------------------------------------------------------------

def test_standard_formatter_no_debug_artifact():
    """
    StandardFormatter must not append 'is this used?' to formatted output.
    """
    import envos.log as log

    formatter = log.StandardFormatter()
    record = logging.LogRecord(
        name="envos",
        level=logging.CRITICAL,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    result = formatter.format(record)
    assert "is this used?" not in result, (
        f"Debug artifact 'is this used?' found in formatted output: {result!r}"
    )
