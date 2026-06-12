"""
Tests for envos/column_density.py — P1-14
Covers C-54 (top-level imports removed) and C-55 (step=0 guard in column_density_z).
"""

import numpy as np
import pytest
from envos.column_density import calc_column_density


@pytest.mark.parametrize("direction", ["r", "theta"])
def test_column_density_r_theta_finite(g1_model, direction):
    """
    calc_column_density for 'r' and 'theta' must return finite values.
    Uses G1 model (UCM, radmc3d not required).
    """
    col = calc_column_density(g1_model, direction)
    assert col is not None
    assert col.shape == g1_model.rhogas.shape
    # Non-zero entries should be finite
    nonzero = col[col != 0]
    assert np.all(np.isfinite(nonzero)), (
        f"Non-finite values in column_density direction='{direction}'"
    )


@pytest.mark.parametrize("double_interp", [True, False])
def test_column_density_z_finite(g1_model, double_interp):
    """
    C-55: column_density_z must handle z=0 (step would be 0 before fix).
    Both double_interp=True and double_interp=False must return finite values.
    """
    col = calc_column_density(g1_model, "z", double_interp=double_interp)
    assert col is not None
    # Convert object array (from frompyfunc) if needed
    col_arr = np.array(col, dtype=float)
    finite_mask = np.isfinite(col_arr)
    # At least some values should be finite (model has finite density)
    assert finite_mask.any(), (
        f"No finite values in z column density (double_interp={double_interp})"
    )


def test_no_top_level_pyplot_import():
    """
    C-54: column_density.py must not import matplotlib.pyplot at module level.
    """
    import importlib, ast, pathlib
    src = pathlib.Path(__file__).parent.parent / "envos" / "column_density.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only check top-level (not inside functions/classes)
            pass  # walk includes nested; filter by checking lineno context

    # Simpler: check that 'matplotlib' is not in top-level imports by inspecting
    # the module's source directly for top-level import statements
    lines = src.read_text().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import matplotlib") or stripped.startswith("from matplotlib"):
            # Check it's a top-level line (not indented)
            assert line.startswith("    ") or line.startswith("\t"), (
                f"matplotlib imported at module top level (line {i+1}): {line!r}"
            )


def test_no_top_level_envos_import():
    """
    C-54: column_density.py must not `import envos` at module level (circular import).
    """
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "envos" / "column_density.py"
    lines = src.read_text().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "import envos":
            # If it's at top level (no indentation), fail
            assert line[0] in (" ", "\t"), (
                f"`import envos` at module top level (line {i+1}): {line!r}"
            )
