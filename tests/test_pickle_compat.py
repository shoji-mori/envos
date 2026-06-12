"""
P2-B: Pickle compatibility test.

Verifies that the legacy pickle fixture (created before the obs.py → obs/
package split) can be loaded and round-trips correctly after the split.

The fixture was serialised with Cube.__module__ == "envos.obs".
After the split, Cube lives in envos.obs.data but is re-exported through
envos.obs.__init__, so `envos.obs.Cube` still resolves — meaning the old
pickle must still load without errors.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from envos.obs import Cube, read_obsdata

# The fixture was pickled under numpy 2.x, whose arrays reference the
# numpy._core modules that do not exist in numpy 1.x (the latest numpy
# available on Python 3.9). Loading is therefore impossible there for
# reasons outside envos's control; the module-path compatibility this
# file protects is fully exercised on numpy 2.x environments.
pytestmark = pytest.mark.skipif(
    int(np.__version__.split(".")[0]) < 2,
    reason="legacy fixture requires numpy >= 2 to unpickle (numpy._core)",
)

FIXTURE = Path(__file__).parent / "data" / "cube_legacy.pkl"


def test_fixture_exists():
    """The legacy fixture file must be present in the repository."""
    assert FIXTURE.exists(), f"Missing fixture: {FIXTURE}"


def test_load_via_read_obsdata():
    """read_obsdata('.pkl') must return a Cube instance from the legacy pickle."""
    obj = read_obsdata(str(FIXTURE))
    assert isinstance(obj, Cube), f"Expected Cube, got {type(obj)}"


def test_legacy_cube_shape():
    """Loaded Cube must have the expected shape (5, 6, 7)."""
    obj = read_obsdata(str(FIXTURE))
    assert obj.Ippv.shape == (5, 6, 7), (
        f"Unexpected shape: {obj.Ippv.shape}"
    )


def test_legacy_cube_axes():
    """Loaded Cube must have the expected axis ranges."""
    obj = read_obsdata(str(FIXTURE))
    np.testing.assert_allclose(obj.xau[0], -50.0)
    np.testing.assert_allclose(obj.xau[-1], 50.0)
    np.testing.assert_allclose(obj.yau[0], -50.0)
    np.testing.assert_allclose(obj.yau[-1], 50.0)
    np.testing.assert_allclose(obj.vkms[0], -5.0)
    np.testing.assert_allclose(obj.vkms[-1], 5.0)


def test_legacy_cube_dpc():
    """Loaded Cube must retain dpc=140.0."""
    obj = read_obsdata(str(FIXTURE))
    assert obj.dpc == 140.0


def test_legacy_cube_data_values():
    """Loaded Cube data must match deterministic Gaussian formula."""
    obj = read_obsdata(str(FIXTURE))
    nx, ny, nv = 5, 6, 7
    xau = np.linspace(-50.0, 50.0, nx)
    yau = np.linspace(-50.0, 50.0, ny)
    vkms = np.linspace(-5.0, 5.0, nv)
    X, Y, V = np.meshgrid(xau, yau, vkms, indexing="ij")
    expected = np.exp(-X**2 / 2000.0 - Y**2 / 2000.0 - V**2 / 20.0)
    np.testing.assert_allclose(obj.Ippv, expected, rtol=1e-10)


def test_legacy_cube_is_envos_obs_cube():
    """The loaded object's class must be the same as envos.obs.Cube (re-export check)."""
    obj = read_obsdata(str(FIXTURE))
    from envos.obs import Cube as ObsCube
    assert type(obj) is ObsCube


def test_legacy_cube_round_trip(tmp_path):
    """Re-pickling a loaded legacy Cube and reloading must yield identical data."""
    obj = read_obsdata(str(FIXTURE))
    new_pkl = tmp_path / "re_saved.pkl"
    pd.to_pickle(obj, str(new_pkl))
    obj2 = read_obsdata(str(new_pkl))
    np.testing.assert_array_equal(obj.Ippv, obj2.Ippv)
    np.testing.assert_array_equal(obj.xau, obj2.xau)
    np.testing.assert_array_equal(obj.yau, obj2.yau)
    np.testing.assert_array_equal(obj.vkms, obj2.vkms)
