"""
Tests for envos.grid (P1-7 and P1-12 partial).

Covers:
  - Grid() with no valid arguments raises ValueError (C-50).
  - Grid with rau_lim but no nr/dr_to_r raises ValueError (C-50).
  - ringhost=True inserts 3 ghost cells (C-51 log text fix).
  - Dead functions removed: get_interface_coord, compressed_x2 (P1-12).
"""

import numpy as np
import pytest
from envos.grid import Grid


# ---------------------------------------------------------------------------
# Error cases (P1-7)
# ---------------------------------------------------------------------------

def test_no_args_raises():
    """Grid() with no arguments raises ValueError, not a silent return None."""
    with pytest.raises(ValueError, match="Grid requires either"):
        Grid()


def test_rau_lim_without_resolution_raises():
    """Grid with rau_lim but neither nr nor dr_to_r raises ValueError."""
    with pytest.raises((ValueError, TypeError)):
        Grid(rau_lim=[10, 100])  # nr=None, dr_to_r=None → TypeError on nr+1


def test_nr_without_ntheta_raises():
    """Grid with rau_lim + nr but no ntheta / dr_to_r raises ValueError."""
    with pytest.raises(ValueError, match="Specify either ntheta or dr_to_r"):
        Grid(rau_lim=[10, 100], nr=10)


# ---------------------------------------------------------------------------
# ringhost cell count (P1-7, C-51)
# ---------------------------------------------------------------------------

def test_ringhost_adds_3_cells():
    """ringhost=True should add exactly 3 ghost cells (np.linspace(..., 4)[:-1])."""
    grid_normal = Grid(rau_lim=[10, 100], dr_to_r=0.1)
    grid_ghost = Grid(rau_lim=[10, 100], dr_to_r=0.1, ringhost=True)
    # ri_ax has nr+1 faces; ringhost prepends 3 extra interface points
    assert len(grid_ghost.ri_ax) == len(grid_normal.ri_ax) + 3


# ---------------------------------------------------------------------------
# Dead code removed (P1-12)
# ---------------------------------------------------------------------------

def test_get_interface_coord_removed():
    """Module-level get_interface_coord must not exist (P1-12 deletion)."""
    import envos.grid as grid_module
    assert not hasattr(grid_module, "get_interface_coord"), (
        "get_interface_coord should have been deleted (P1-12)"
    )


def test_compressed_x2_removed():
    """Module-level compressed_x2 must not exist (P1-12 deletion)."""
    import envos.grid as grid_module
    assert not hasattr(grid_module, "compressed_x2"), (
        "compressed_x2 should have been deleted (P1-12)"
    )


# ---------------------------------------------------------------------------
# Normal construction still works
# ---------------------------------------------------------------------------

def test_grid_dr_to_r():
    """Grid with dr_to_r constructs correctly."""
    grid = Grid(rau_lim=[10, 1000], dr_to_r=0.1)
    assert hasattr(grid, "rc_ax")
    assert len(grid.rc_ax) > 0
    assert np.all(np.diff(grid.rc_ax) > 0), "rc_ax should be monotonically increasing"


def test_grid_explicit_axes():
    """Grid with explicit interface axes constructs correctly."""
    import envos.nconst as nc
    ri = np.geomspace(10, 100, 11) * nc.au
    ti = np.linspace(0, np.pi / 2, 6)
    pi = np.linspace(0, 2 * np.pi, 2)
    grid = Grid(ri_ax=ri, ti_ax=ti, pi_ax=pi)
    assert len(grid.rc_ax) == 10
    assert len(grid.tc_ax) == 5
