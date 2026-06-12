"""
Smoke tests for the envos package (P0-3).

These tests verify that basic imports and core workflows function correctly.
All tests are fast and require no radmc3d binary.
"""

import subprocess
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# (a) Basic import
# ---------------------------------------------------------------------------

def test_import_envos():
    """import envos must succeed without error."""
    import envos  # noqa: F401


# ---------------------------------------------------------------------------
# (b) Config creation
# ---------------------------------------------------------------------------

def test_config_creation(tmp_path):
    """Config with basic parameters can be created without error."""
    from envos.config import Config

    config = Config(
        run_dir=str(tmp_path),
        rau_in=10,
        rau_out=100,
        dr_to_r=0.1,
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
    )
    assert config.rau_in == 10
    assert config.rau_out == 100
    assert config.CR_au == 100


# ---------------------------------------------------------------------------
# (c) ModelGenerator -> calc_kinematic_structure -> get_model
# ---------------------------------------------------------------------------

def test_model_generator_rhogas(tmp_path):
    """
    ModelGenerator produces a model whose rhogas is finite, non-negative,
    and contains at least one non-zero value.
    """
    from envos.config import Config
    from envos.model_generator import ModelGenerator

    config = Config(
        run_dir=str(tmp_path),
        rau_in=10,
        rau_out=100,
        dr_to_r=0.1,
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
    )
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    model = mg.get_model()

    rhogas = model.rhogas
    assert rhogas is not None, "rhogas should not be None"
    assert np.isfinite(rhogas).all(), "rhogas should be finite everywhere"
    assert (rhogas >= 0).all(), "rhogas should be non-negative everywhere"
    assert (rhogas > 0).any(), "rhogas should be non-zero somewhere"


# ---------------------------------------------------------------------------
# (d) Grid standalone creation
# ---------------------------------------------------------------------------

def test_grid_standalone():
    """Grid can be constructed directly with rau_lim and dr_to_r."""
    from envos.grid import Grid

    grid = Grid(rau_lim=[10, 100], dr_to_r=0.1)
    assert hasattr(grid, "rc_ax"), "grid should have rc_ax"
    assert hasattr(grid, "tc_ax"), "grid should have tc_ax"
    assert len(grid.rc_ax) > 0, "rc_ax should be non-empty"


# ---------------------------------------------------------------------------
# (e) PhysicalParameters: three parameter combinations
# ---------------------------------------------------------------------------

def _assert_ppar_complete(pp):
    """Helper: all six key attributes must be non-None."""
    for attr in ("Mdot", "cs", "t", "CR", "jmid", "Omega"):
        val = getattr(pp, attr, None)
        assert val is not None, f"PhysicalParameters.{attr} should not be None"


def test_physical_parameters_T_CR_Ms():
    """T + CR_au + Ms_Msun combination fills all key attributes."""
    from envos.physical_params import PhysicalParameters

    pp = PhysicalParameters(T=10, CR_au=100, Ms_Msun=0.3)
    _assert_ppar_complete(pp)


def test_physical_parameters_T_t_yr_Omega():
    """T + t_yr + Omega combination fills all key attributes."""
    from envos.physical_params import PhysicalParameters

    pp = PhysicalParameters(T=10, t_yr=1e5, Omega=1e-13)
    _assert_ppar_complete(pp)


def test_physical_parameters_Mdot_Ms_jmid():
    """Mdot_smpy + Ms_Msun + jmid combination fills all key attributes."""
    from envos.physical_params import PhysicalParameters

    pp = PhysicalParameters(Mdot_smpy=4.5e-6, Ms_Msun=0.2, jmid=1e20)
    _assert_ppar_complete(pp)


# ---------------------------------------------------------------------------
# (f) from envos import *  -- B-13 fixed in P1-4
# ---------------------------------------------------------------------------

def test_from_envos_import_star():
    """
    'from envos import *' must succeed.

    B-13 fix (P1-4): removed 'read_mg' from __all__ and added
    'from . import column_density' so all listed names resolve.
    """
    # exec isolates the wildcard import so pytest can catch the exception
    exec("from envos import *", {})
