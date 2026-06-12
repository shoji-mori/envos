"""
Shared pytest configuration and fixtures.

This file is loaded automatically by pytest before any tests run.
"""

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before any pyplot import

import pytest
from pathlib import Path
from envos.config import Config
from envos.model_generator import ModelGenerator


# ---------------------------------------------------------------------------
# G1 model fixture (session-scoped, shared across all tests)
# Parameters match PLAN.md P0-4 G1 spec:
#   rau_in=10, rau_out=1000, nr=30, ntheta=20, nphi=1,
#   CR_au=100, Ms_Msun=0.3, T=10, cavangle_deg=45
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def g1_model(tmp_path_factory):
    """
    Session-scoped fixture that builds the G1 UCM model.

    Parameters are identical to the golden data generation in make_golden.py
    so tests share the exact same construction path.
    Returns a CircumstellarModel instance.
    """
    run_dir = tmp_path_factory.mktemp("g1_run")
    config = Config(
        run_dir=str(run_dir),
        rau_in=10,
        rau_out=1000,
        nr=30,
        ntheta=20,
        nphi=1,
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
    )
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()
