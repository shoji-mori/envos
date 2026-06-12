"""
Tests for envos/log.py (P1-5).

Covers:
- unset_logfile: file handlers are properly removed without increasing handler count
- set_level: stream handler level is correctly changed
- change_rundir: log file destination moves to new directory
"""

import logging
import pathlib

import pytest


def _count_file_handlers(lg):
    """Return the number of FileHandler instances on a logger."""
    return sum(1 for h in lg.handlers if isinstance(h, logging.FileHandler))


def _count_stream_handlers(lg):
    """Return the number of StreamHandler (but not FileHandler) instances on a logger."""
    return sum(
        1 for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )


# ---------------------------------------------------------------------------
# Handler non-proliferation: calling update_logfile twice must NOT add
# extra file handlers (B-30 / P1-5 item 1).
# ---------------------------------------------------------------------------

def test_update_logfile_no_handler_proliferation(tmp_path):
    """
    Calling update_logfile() twice must not accumulate extra file handlers.
    After both calls, exactly one FileHandler should exist on the 'envos' logger.
    """
    import envos.log as log

    logfile = tmp_path / "test.log"

    # Call update_logfile twice with the same file
    log.update_logfile(name="envos", filepath=logfile)
    log.update_logfile(name="envos", filepath=logfile)

    lg = log.loggers["envos"]
    assert _count_file_handlers(lg) == 1, (
        f"Expected exactly 1 FileHandler after two update_logfile calls, "
        f"got {_count_file_handlers(lg)}"
    )

    # Clean up: remove the file handler
    log.unset_logfile("envos")


def test_unset_logfile_removes_all_file_handlers(tmp_path):
    """
    unset_logfile removes all FileHandlers without touching StreamHandlers.
    """
    import envos.log as log

    logfile = tmp_path / "cleanup_test.log"
    log.update_logfile(name="envos", filepath=logfile)

    lg = log.loggers["envos"]
    stream_count_before = _count_stream_handlers(lg)

    log.unset_logfile("envos")

    assert _count_file_handlers(lg) == 0, "All FileHandlers should be removed"
    assert _count_stream_handlers(lg) == stream_count_before, (
        "StreamHandlers should not be affected by unset_logfile"
    )


# ---------------------------------------------------------------------------
# set_level: stream handler level is updated correctly (B-30 / P1-5 item 2)
# ---------------------------------------------------------------------------

def test_set_level_stream_handler(tmp_path):
    """
    set_level("envos", "DEBUG", target="stream") sets the stream handler to DEBUG.
    """
    import envos.log as log

    lg = log.loggers["envos"]

    # Save original level so we can restore it
    original_levels = [h.level for h in lg.handlers if isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.FileHandler)]

    log.set_level("envos", "DEBUG", target="stream")

    for h in lg.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            assert h.level == logging.DEBUG, (
                f"Stream handler level should be DEBUG, got {h.level}"
            )

    # Restore
    for h, lv in zip(
        [h for h in lg.handlers if isinstance(h, logging.StreamHandler)
         and not isinstance(h, logging.FileHandler)],
        original_levels,
    ):
        h.setLevel(lv)


# ---------------------------------------------------------------------------
# change_rundir: log file destination moves (B-29 / P1-5 item 3)
# ---------------------------------------------------------------------------

def test_change_rundir_moves_log_file(tmp_path):
    """
    After change_rundir(), the FileHandler points to the new directory.
    """
    import envos.log as log

    dir_a = tmp_path / "run_A"
    dir_a.mkdir()
    dir_b = tmp_path / "run_B"
    dir_b.mkdir()

    logfile_a = dir_a / "envos.log"

    # Set up a file handler pointing at dir_a
    log.update_logfile(name="envos", filepath=logfile_a)

    lg = log.loggers["envos"]
    assert _count_file_handlers(lg) == 1

    # Move to dir_b
    log.change_rundir(dir_b)

    # Now the file handler should point to dir_b/envos.log
    file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, "Should still have exactly 1 FileHandler after change_rundir"

    new_path = pathlib.Path(file_handlers[0].baseFilename)
    assert new_path.parent == dir_b.resolve(), (
        f"FileHandler should point into {dir_b}, but points to {new_path.parent}"
    )

    # Clean up
    log.unset_logfile("envos")


# ---------------------------------------------------------------------------
# StandardFormatter: no debug artifact in output (D-65 / P1-5 item 4)
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
