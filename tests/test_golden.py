"""
Golden data regression tests for envos (P0-4).

These tests rebuild the physical models and compare the results to
pre-computed reference data stored in tests/golden/.

G1 and G3 tests are fast (no TSC solve required).
G2 test is @pytest.mark.slow because it requires the TSC solution
(tscsol.pkl is loaded from tests/golden/ to avoid re-solving).

Re-generation procedure:
    MPLBACKEND=Agg /home/user/envos/.venv/bin/python tests/make_golden.py
Commit the updated *.npz / *.json / tscsol.pkl together.

Physical-change policy: if any golden value changes, record the relative
change in the PR body and add the `physics-change` label.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.golden_helpers import (
    GOLDEN_DIR,
    G3A_KWARGS,
    G3B_KWARGS,
    G3_ATTRS,
    build_g1_model,
    build_g2_model,
    build_g3,
    build_g4_model,
    extract_g1_arrays,
    extract_g4_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_g1_golden():
    """Load pre-computed G1 arrays from tests/golden/g1_ucm.npz."""
    return dict(np.load(GOLDEN_DIR / "g1_ucm.npz"))


def load_g2_golden():
    """Load pre-computed G2 arrays from tests/golden/g2_ucm_tsc.npz."""
    return dict(np.load(GOLDEN_DIR / "g2_ucm_tsc.npz"))


def load_g3_golden(suffix):
    """Load pre-computed G3 scalars from tests/golden/g3{suffix}_ppar.json."""
    with open(GOLDEN_DIR / f"g3{suffix}_ppar.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# G1: UCM model array regression
# ---------------------------------------------------------------------------

def test_g1_arrays_match_golden(tmp_path):
    """
    UCM model (G1) arrays must match stored golden values to rtol=1e-10.

    Covers: rhogas, vr, vt, vp, rc_ax, tc_ax.
    """
    golden = load_g1_golden()
    model = build_g1_model(tmp_path)
    result = extract_g1_arrays(model)

    for name in golden:
        np.testing.assert_allclose(
            result[name],
            golden[name],
            rtol=1e-10,
            err_msg=f"G1 mismatch in array '{name}'",
        )


# ---------------------------------------------------------------------------
# G3: PhysicalParameters scalar regression
# ---------------------------------------------------------------------------

def test_g3a_scalars_match_golden():
    """
    G3(a): T+CR_au+Ms_Msun — all six key scalars match golden to rtol=1e-10.

    This combination does not depend on nc.year, so it should be invariant
    under P1-1 (year correction).
    """
    golden = load_g3_golden("a")
    result = build_g3(G3A_KWARGS)

    for attr in G3_ATTRS:
        np.testing.assert_allclose(
            result[attr],
            golden[attr],
            rtol=1e-10,
            err_msg=f"G3(a) mismatch in scalar '{attr}'",
        )


def test_g3b_scalars_match_golden():
    """
    G3(b): Mdot_smpy+Ms_Msun+CR_au — all six key scalars match golden to rtol=1e-10.

    This combination uses nc.smpy (= nc.Msun / nc.year), so it will change
    when P1-1 (year correction) is applied.  That PR must regenerate this file
    and record the change rate.
    """
    golden = load_g3_golden("b")
    result = build_g3(G3B_KWARGS)

    for attr in G3_ATTRS:
        np.testing.assert_allclose(
            result[attr],
            golden[attr],
            rtol=1e-10,
            err_msg=f"G3(b) mismatch in scalar '{attr}'",
        )


# ---------------------------------------------------------------------------
# G2: UCM+TSC model array regression  (@slow — loads tscsol.pkl)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# G4: UCM + powerlaw disk array regression  (P1-9 physics-change: replace-mode)
# ---------------------------------------------------------------------------

def test_g4_arrays_match_golden(tmp_path):
    """
    UCM + powerlaw disk model (G4) arrays must match stored golden values to rtol=1e-10.

    This golden was generated with the replace-mode disk synthesis introduced
    in P1-9 (D1): ``rho[cond] = disk.rho[cond]`` where ``cond = rho < disk.rho``.
    The previous additive method (``rho += disk.rho``) produced up to ~50% higher
    density in disk-dominated cells; that difference is the physics change recorded
    in this PR.

    Covers: rhogas, vr, vt, vp, rc_ax, tc_ax.
    """
    golden = dict(np.load(GOLDEN_DIR / "g4_ucm_disk.npz"))
    model = build_g4_model(tmp_path)
    result = extract_g4_arrays(model)

    for name in golden:
        np.testing.assert_allclose(
            result[name],
            golden[name],
            rtol=1e-10,
            err_msg=f"G4 mismatch in array '{name}'",
        )


@pytest.mark.slow
def test_g2_arrays_match_golden(tmp_path):
    """
    UCM+TSC model (G2) arrays must match stored golden values to rtol=1e-10.

    The TSC solution table (tscsol.pkl) is loaded from tests/golden/ by
    pointing storage_dir there, so this test does NOT re-solve the ODE
    system.  If tscsol.pkl is missing the test will re-solve (slow) or fail
    if the solver encounters numerical issues.
    """
    golden = load_g2_golden()
    # Use GOLDEN_DIR as storage_dir so tsc.read_table() picks up tscsol.pkl
    model = build_g2_model(tmp_path, GOLDEN_DIR)
    result = extract_g1_arrays(model)

    for name in golden:
        np.testing.assert_allclose(
            result[name],
            golden[name],
            rtol=1e-10,
            err_msg=f"G2 mismatch in array '{name}'",
        )
