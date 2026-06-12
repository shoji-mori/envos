"""
Tests for envos/config.py and envos/__init__.py (P1-4 / P2-C).

Covers:
- Config(logfile=...) creates the log file without crashing (B-12)
- Config generated twice does not accumulate extra file handlers (B-12)
- Config(logfile=None) does not add a FileHandler (idempotency / P2-C)
- level_stdout wiring: stream handler level is set correctly (追補-72)
- from envos import * succeeds (B-13) -- also tested in test_smoke.py
"""

import logging
import pathlib

import pytest


def _count_file_handlers(lg):
    return sum(1 for h in lg.handlers if isinstance(h, logging.FileHandler))


def _count_stream_handlers(lg):
    return sum(
        1 for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )


# ---------------------------------------------------------------------------
# B-12: Config(logfile=...) must not crash and must create the log file.
# ---------------------------------------------------------------------------

def test_config_with_logfile_creates_file(tmp_path):
    """
    Config(run_dir=tmp, logfile=...) must not raise and must produce the file.
    """
    import envos.log as log
    from envos.config import Config

    logfile = tmp_path / "envos.log"

    config = Config(run_dir=str(tmp_path), logfile=str(logfile))

    assert logfile.exists(), "Log file should be created after Config(logfile=...)"

    # Cleanup: reset to no-file state
    log.setup(level="INFO")


def test_config_twice_no_handler_proliferation(tmp_path):
    """
    Creating Config twice with a logfile must not accumulate FileHandlers.
    After both Config() calls, the 'envos' logger should have exactly one
    FileHandler.
    """
    import envos.log as log
    from envos.config import Config

    logfile = tmp_path / "envos.log"

    Config(run_dir=str(tmp_path), logfile=str(logfile))
    Config(run_dir=str(tmp_path), logfile=str(logfile))

    lg = log.logger
    n = _count_file_handlers(lg)
    assert n == 1, (
        f"Expected exactly 1 FileHandler after two Config() calls, got {n}"
    )

    # Cleanup
    log.setup(level="INFO")


def test_config_without_logfile_no_file_handler(tmp_path):
    """
    Config(logfile=None) must not add a FileHandler (idempotency / P2-C).
    """
    import envos.log as log
    from envos.config import Config

    # Ensure we start with no file handler
    log.setup(level="INFO")
    assert _count_file_handlers(log.logger) == 0

    Config(run_dir=str(tmp_path))

    assert _count_file_handlers(log.logger) == 0, (
        "Config without logfile must not add a FileHandler"
    )


# ---------------------------------------------------------------------------
# 追補-72: level_stdout wires the stream handler level.
# ---------------------------------------------------------------------------

def test_config_level_stdout_sets_stream_handler(tmp_path):
    """
    Config(level_stdout="DEBUG") must set the stream handler to DEBUG level.
    """
    import envos.log as log
    from envos.config import Config

    Config(run_dir=str(tmp_path), level_stdout="DEBUG")

    lg = log.logger
    stream_handlers = [
        h for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers, "No StreamHandler found on the envos logger"
    for h in stream_handlers:
        assert h.level == logging.DEBUG, (
            f"Stream handler level should be DEBUG after Config(level_stdout='DEBUG'), "
            f"got {h.level}"
        )

    # Restore to INFO
    log.setup(level="INFO")


# ---------------------------------------------------------------------------
# B-13: from envos import * succeeds (main test is in test_smoke.py;
# this one additionally verifies that column_density is importable as a
# top-level name after the star import).
# ---------------------------------------------------------------------------

def test_from_envos_import_star_column_density():
    """
    After 'from envos import *', column_density must be available.
    """
    ns = {}
    exec("from envos import *", ns)
    assert "column_density" in ns, (
        "'column_density' should be in namespace after 'from envos import *'"
    )
    assert "read_mg" not in ns, (
        "'read_mg' must NOT be in namespace (it was removed from __all__ in P1-4)"
    )
