"""
Tests for envos.physical_params (P1-6).

Covers:
  - Three valid parameter combinations produce fully-filled attributes.
  - Insufficient-parameter calls raise ValueError with a helpful message.
  - calc_dependent_params is gone (module no longer exports it).
"""

import pytest
from envos.physical_params import PhysicalParameters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_ppar_complete(pp):
    """All six key attributes must be non-None and finite."""
    import numpy as np
    for attr in ("Mdot", "cs", "t", "CR", "jmid", "Omega"):
        val = getattr(pp, attr, None)
        assert val is not None, f"PhysicalParameters.{attr} should not be None"
        assert np.isfinite(val), f"PhysicalParameters.{attr} should be finite"


# ---------------------------------------------------------------------------
# Valid combinations
# ---------------------------------------------------------------------------

def test_T_CR_Ms():
    """T + CR_au + Ms_Msun fills all key attributes."""
    pp = PhysicalParameters(T=10, CR_au=100, Ms_Msun=0.3)
    _assert_ppar_complete(pp)


def test_T_t_yr_Omega():
    """T + t_yr + Omega fills all key attributes."""
    pp = PhysicalParameters(T=10, t_yr=1e5, Omega=1e-13)
    _assert_ppar_complete(pp)


def test_Mdot_Ms_jmid():
    """Mdot_smpy + Ms_Msun + jmid fills all key attributes."""
    pp = PhysicalParameters(Mdot_smpy=4.5e-6, Ms_Msun=0.2, jmid=1e20)
    _assert_ppar_complete(pp)


# ---------------------------------------------------------------------------
# Insufficient parameters → ValueError
# ---------------------------------------------------------------------------

def test_no_params_raises():
    """PhysicalParameters() with no arguments raises ValueError."""
    with pytest.raises(ValueError, match="insufficient parameters"):
        PhysicalParameters()


def test_only_T_raises():
    """PhysicalParameters(T=10) alone raises ValueError (Ms/t missing)."""
    with pytest.raises(ValueError, match="insufficient parameters"):
        PhysicalParameters(T=10)


def test_T_Ms_no_jmid_raises():
    """T + Ms_Msun but no CR/jmid/Omega raises ValueError."""
    with pytest.raises(ValueError, match="insufficient parameters"):
        PhysicalParameters(T=10, Ms_Msun=0.3)


# ---------------------------------------------------------------------------
# calc_dependent_params must be gone
# ---------------------------------------------------------------------------

def test_calc_dependent_params_removed():
    """calc_dependent_params must not exist in the module (P1-6 deletion)."""
    import envos.physical_params as pp_module
    assert not hasattr(pp_module, "calc_dependent_params"), (
        "calc_dependent_params should have been deleted (P1-6)"
    )
