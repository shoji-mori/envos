"""
tests/test_examples.py — P1-19 examples verification.

Strategy:
- Syntax check: py_compile each example file.
- Model pipeline: re-create the same Config used in each example but with a
  small grid (dr_to_r=0.2 etc.) so the test runs fast.  Only the
  ModelGenerator → calc_kinematic_structure → get_model path is exercised
  (no radmc3d, no plot functions, no streamline tracing).
- All slow tests are marked @pytest.mark.slow.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import envos

# Root of the repository (two levels up from this file's directory)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths to the example files (relative to REPO_ROOT)
EXAMPLE_FILES = [
    REPO_ROOT / "examples" / "make_model.py",
    REPO_ROOT / "examples" / "make_userdefined_model.py",
    REPO_ROOT / "examples" / "trace_streamline.py",
    REPO_ROOT / "examples" / "trace_streamline_with_column_density.py",
    REPO_ROOT / "example_run.py",
]


# ---------------------------------------------------------------------------
# Syntax checks (fast, no marks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filepath", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_syntax(filepath):
    """Each example file must compile without syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(filepath)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"py_compile failed for {filepath}:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Model pipeline tests (slow)
# ---------------------------------------------------------------------------


def _make_small_config(tmp_path, **overrides):
    """Return a Config with a tiny grid suitable for fast tests."""
    defaults = dict(
        run_dir=str(tmp_path),
        rau_in=10,
        rau_out=500,
        dr_to_r=0.2,
        aspect_ratio=1,
        inenv="UCM",
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
    )
    defaults.update(overrides)
    return envos.Config(**defaults)


def _assert_model_ok(model):
    """Assert that the model rhogas array is valid."""
    assert model.rhogas is not None
    assert np.isfinite(model.rhogas).all(), "rhogas contains non-finite values"
    assert (model.rhogas >= 0).all(), "rhogas contains negative values"
    assert (model.rhogas > 0).any(), "rhogas is all zero"


@pytest.mark.slow
def test_make_model_pipeline(tmp_path):
    """
    Mirrors examples/make_model.py: ModelGenerator → calc_kinematic_structure
    → get_model.  No radmc3d, no plot calls.
    """
    config = _make_small_config(tmp_path)
    mg = envos.ModelGenerator(config)
    mg.calc_kinematic_structure()
    model = mg.get_model()
    _assert_model_ok(model)


@pytest.mark.slow
def test_make_userdefined_model_pipeline(tmp_path):
    """
    Mirrors examples/make_userdefined_model.py: user-defined grid + density/
    velocity fields fed into the generator.  Tests set_gas_density,
    set_gas_velocity, and set_dust_density.  No radmc3d.

    dr_to_r is supplied so that ModelGenerator.init_from_config can build an
    initial grid; it is immediately replaced by the user-defined axes via
    mg.set_grid() — which is the pattern the example uses.
    """
    import numpy as np

    config = envos.Config(
        run_dir=str(tmp_path),
        rau_in=10,
        rau_out=500,
        dr_to_r=0.3,        # needed for init_from_config to build a grid
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
    )
    mg = envos.ModelGenerator(config)

    # Replace the auto-built grid with a small user-defined grid
    ri = np.geomspace(10, 500, 10) * envos.nc.au
    ti = np.linspace(0, np.pi, 11)
    pi = np.linspace(0, 2 * np.pi, 6)
    mg.set_grid(ri=ri, ti=ti, pi=pi)

    rr, tt, pp = mg.get_meshgrid()
    rho = 1e-17 * np.sin(tt) ** 2
    vr = np.zeros(rr.shape)
    vt = np.zeros(rr.shape)
    vp = np.sqrt(envos.nc.G * config.Ms_Msun * envos.nc.Msun / rr)

    mg.set_gas_density(rho=rho)
    mg.set_gas_velocity(vr=vr, vt=vt, vp=vp)

    # Explicitly set dust density (mirrors P1-19 fix for make_userdefined_model.py)
    mg.model.set_dust_density(f_dg=config.f_dg)

    model = mg.get_model()
    _assert_model_ok(model)
    assert model.rhodust is not None, "rhodust should be set after set_dust_density"
    assert np.allclose(model.rhodust, model.rhogas * config.f_dg), (
        "rhodust should equal rhogas * f_dg"
    )


@pytest.mark.slow
def test_trace_streamline_pipeline(tmp_path):
    """
    Mirrors examples/trace_streamline.py model-generation portion.
    No radmc3d, no streamline tracing, no plots.
    """
    config = _make_small_config(
        tmp_path,
        rau_out=1000,
        dr_to_r=0.2,
    )
    mg = envos.ModelGenerator(config)
    mg.calc_kinematic_structure()
    model = mg.get_model()
    _assert_model_ok(model)


@pytest.mark.slow
def test_trace_streamline_with_column_density_pipeline(tmp_path):
    """
    Mirrors examples/trace_streamline_with_column_density.py model-generation
    portion.  No radmc3d, no streamline tracing, no plots.
    """
    config = _make_small_config(
        tmp_path,
        rau_out=1000,
        dr_to_r=0.2,
    )
    mg = envos.ModelGenerator(config)
    mg.calc_kinematic_structure()
    model = mg.get_model()
    _assert_model_ok(model)


@pytest.mark.slow
def test_example_run_pipeline(tmp_path):
    """
    Mirrors example_run.py model-generation portion.
    No radmc3d, no obs simulation, no plots.
    """
    config = _make_small_config(
        tmp_path,
        nphi=4,
        dr_to_r=0.2,
    )
    mg = envos.ModelGenerator(config)
    mg.calc_kinematic_structure()
    model = mg.get_model()
    _assert_model_ok(model)
