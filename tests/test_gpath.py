"""
Tests for envos/gpath.py (P1-3).

Covers: set_radmcdir() global variable bug fix.
"""

from pathlib import Path

import pytest

import envos.gpath as gpath


# ---------------------------------------------------------------------------
# P1-3: set_radmcdir should update radmc_dir, not run_dir
# ---------------------------------------------------------------------------

def test_set_radmcdir_updates_radmc_dir(tmp_path):
    """
    After set_radmcdir(path), gpath.radmc_dir must equal Path(path).
    """
    target = tmp_path / "radmc_test"
    target.mkdir()

    gpath.set_radmcdir(str(target))

    assert gpath.radmc_dir == Path(str(target)), (
        f"gpath.radmc_dir should be {target}, got {gpath.radmc_dir}"
    )


def test_set_radmcdir_does_not_change_run_dir(tmp_path):
    """
    set_radmcdir() must not modify gpath.run_dir.
    """
    run_dir_before = gpath.run_dir

    target = tmp_path / "radmc_only"
    target.mkdir()
    gpath.set_radmcdir(str(target))

    assert gpath.run_dir == run_dir_before, (
        f"gpath.run_dir should be unchanged ({run_dir_before}), "
        f"but got {gpath.run_dir}"
    )


def test_set_radmcdir_accepts_path_object(tmp_path):
    """set_radmcdir() accepts a pathlib.Path argument."""
    target = tmp_path / "radmc_path_obj"
    target.mkdir()

    gpath.set_radmcdir(target)  # pass Path, not str

    assert gpath.radmc_dir == target
