"""
Tests for envos.models (P1-8).

Covers:
  - save_pickle respects the current gpath.run_dir (A-9 lazy-reference fix).
  - calc_midplane_average works without AttributeError (B-28 fix).
  - PowerlawDisk.get_Sigma raises ValueError for unknown tail type.
"""

import os
import pytest
import numpy as np
from pathlib import Path

from envos.config import Config
from envos.model_generator import ModelGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path, subdir="run_a", **kwargs):
    """Helper: create a small UCM Config rooted at tmp_path/subdir."""
    run_dir = tmp_path / subdir
    defaults = dict(
        run_dir=str(run_dir),
        rau_in=10,
        rau_out=100,
        dr_to_r=0.2,
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _make_model(config):
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()


# ---------------------------------------------------------------------------
# P2-A-2: save_pickle writes to the directory given by the explicit argument
# (the implicit run_dir fallback was removed; dirpath is now required).
# ---------------------------------------------------------------------------

def test_save_pickle_uses_given_dirpath(tmp_path):
    """
    save_pickle writes to the directory supplied via dirpath; building paths
    from a second Config's run_path proves there is no shared global state.
    """
    config_a = _make_config(tmp_path, subdir="run_a")
    config_b = _make_config(tmp_path, subdir="run_b")

    model = _make_model(config_b)
    fname = "test_model.pkl"
    model.save_pickle(fname, dirpath=config_b.run_path)

    # File should exist under run_b (the directory we passed)
    expected = Path(config_b.run_dir) / fname
    assert expected.exists(), (
        f"save_pickle should write to the given dirpath ({config_b.run_dir}), "
        f"but file not found there; run_a would be {Path(config_a.run_dir) / fname}"
    )
    # Sanity: definitely NOT in run_a
    unexpected = Path(config_a.run_dir) / fname
    assert not unexpected.exists(), (
        "save_pickle incorrectly wrote to run_a"
    )


def test_save_pickle_requires_target():
    """save_pickle without dirpath or filepath raises ValueError."""
    from envos.models import CircumstellarModel
    with pytest.raises(ValueError):
        CircumstellarModel().save_pickle("m.pkl")


# ---------------------------------------------------------------------------
# P1-8 item 2: calc_midplane_average must not raise AttributeError (B-28 fix)
# ---------------------------------------------------------------------------

def test_calc_midplane_average(tmp_path):
    """
    calc_midplane_average() should set *_mid attributes without crashing.
    Previously called self.take_midplane_average() which did not exist (B-28).
    """
    config = _make_config(tmp_path)
    model = _make_model(config)

    model.calc_midplane_average()

    for vn in ("rhogas", "vr", "vt", "vp"):
        attr = vn + "_mid"
        assert hasattr(model, attr), f"model.{attr} should be set by calc_midplane_average"
        val = getattr(model, attr)
        assert val is not None, f"model.{attr} should not be None"


# ---------------------------------------------------------------------------
# P1-8 item 3: PowerlawDisk.get_Sigma ValueError for unknown tail
# ---------------------------------------------------------------------------

def test_powerlawdisk_bad_tail_raises(tmp_path):
    """PowerlawDisk with an unsupported tail type raises ValueError."""
    from envos.grid import Grid
    from envos.models import PowerlawDisk
    from envos import nconst as nc

    config = _make_config(tmp_path)
    grid = Grid(rau_lim=[10, 100], dr_to_r=0.2)
    Ms = 0.3 * nc.Msun
    Rd = 100 * nc.au

    with pytest.raises(ValueError, match="tail"):
        PowerlawDisk(grid, Ms, Rd, tail="bogus")
