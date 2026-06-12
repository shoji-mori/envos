"""
Tests for envos/tools.py (P1-2).

Covers: shell(), filecopy(), dataclass_str(), savefile().
"""

import logging
import subprocess
from pathlib import Path

import pytest

from envos.tools import shell, filecopy


# ---------------------------------------------------------------------------
# Helper: capture envos logger records despite propagate=False
# ---------------------------------------------------------------------------

class _ListHandler(logging.Handler):
    """Minimal in-memory log handler."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _make_envos_handler(level=logging.DEBUG):
    import envos.log as _log
    h = _ListHandler()
    h.setLevel(level)
    _log.logger.addHandler(h)
    return h


def _remove_envos_handler(h):
    import envos.log as _log
    _log.logger.removeHandler(h)


# ---------------------------------------------------------------------------
# shell()
# ---------------------------------------------------------------------------

def test_shell_false_raises():
    """shell('false') must raise CalledProcessError (returncode != 0)."""
    with pytest.raises(subprocess.CalledProcessError):
        shell("false")


def test_shell_error_keyword_raises():
    """shell with a matching error_keyword must raise CalledProcessError."""
    with pytest.raises(subprocess.CalledProcessError):
        shell("echo ERROR x", error_keyword="ERROR")


def test_shell_error_keyword_list_raises():
    """shell with error_keyword as a list raises on any match."""
    with pytest.raises(subprocess.CalledProcessError):
        shell("echo FATAL something", error_keyword=["ERROR", "FATAL"])


def test_shell_ok_no_raise_and_logs():
    """shell('echo ok') completes without exception and logs the output line."""
    h = _make_envos_handler()
    try:
        shell("echo ok", log=True)
    finally:
        _remove_envos_handler(h)

    messages = [r.getMessage() for r in h.records]
    assert any("ok" in m for m in messages), (
        f"Expected 'ok' to appear in logger output, got: {messages}"
    )


def test_shell_dryrun_no_execution():
    """shell with dryrun=True logs the command but does not execute it."""
    h = _make_envos_handler()
    try:
        shell("false", dryrun=True)  # 'false' would fail if actually run
    finally:
        _remove_envos_handler(h)

    messages = [r.getMessage() for r in h.records]
    assert any("dryrun" in m for m in messages), (
        f"Expected 'dryrun' in log messages, got: {messages}"
    )


def test_shell_skip_error_no_raise():
    """shell with skip_error=True must warn but not raise on error."""
    shell("false", skip_error=True)  # should not raise


def test_shell_no_simple_arg():
    """shell() must not accept a 'simple' keyword argument."""
    with pytest.raises(TypeError):
        shell("echo hi", simple=True)  # type: ignore[call-arg]


def test_shell_returns_none():
    """shell() return value is None on success."""
    result = shell("echo hello")
    assert result is None


# ---------------------------------------------------------------------------
# filecopy()
# ---------------------------------------------------------------------------

def test_filecopy_skips_existing_dst(tmp_path):
    """
    When destination already exists and error_already_exist=False,
    filecopy() must log a warning and return early without copying.
    """
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("source content")
    dst.write_text("original content")

    h = _make_envos_handler(logging.WARNING)
    try:
        filecopy(str(src), str(dst), error_already_exist=False)
    finally:
        _remove_envos_handler(h)

    # Destination must not have been overwritten
    assert dst.read_text() == "original content", (
        "filecopy should not overwrite existing destination when error_already_exist=False"
    )
    messages = [r.getMessage() for r in h.records]
    assert any("already exists" in m for m in messages), (
        f"Expected 'already exists' warning, got: {messages}"
    )


def test_filecopy_raises_if_dst_exists_and_error_flag(tmp_path):
    """When destination exists and error_already_exist=True, raise Exception."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("source content")
    dst.write_text("original content")

    with pytest.raises(Exception, match="already exists"):
        filecopy(str(src), str(dst), error_already_exist=True)


def test_filecopy_copies_file(tmp_path):
    """filecopy() copies a file when destination does not exist."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")

    filecopy(str(src), str(dst))

    assert dst.exists()
    assert dst.read_text() == "hello"


def test_filecopy_src_missing_raises(tmp_path):
    """filecopy() raises FileNotFoundError when source does not exist."""
    src = tmp_path / "nonexistent.txt"
    dst = tmp_path / "dst.txt"

    with pytest.raises(FileNotFoundError):
        filecopy(str(src), str(dst))
