"""
Tests for the P2-A path-management redesign.

Covers:
  - Config is the single source of truth for paths (run/radmc/fig/log/storage).
  - Multiple Config instances in the same process route their outputs to the
    correct, independent directories (no shared global state).
  - The legacy envos.gpath.* attributes still resolve but emit a
    DeprecationWarning, and reflect the most recently constructed Config.
  - storage_path three-stage resolution (explicit / legacy / packaged).
"""

import importlib
import warnings
from pathlib import Path

import pytest

from envos.config import Config
from envos.model_generator import ModelGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(run_dir, **kwargs):
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


# ---------------------------------------------------------------------------
# Config path properties
# ---------------------------------------------------------------------------

def test_config_path_properties_defaults():
    """Default path rules mirror the legacy gpath behaviour."""
    c = Config()  # no run_dir
    assert c.run_path == Path("./run")
    assert c.radmc_path == Path("./run") / "radmc"
    assert c.fig_path == Path("./run") / "fig"
    assert c.log_path == Path("./run") / "log.dat"


def test_config_path_properties_explicit(tmp_path):
    """Explicit run_dir drives the derived paths; explicit overrides win."""
    c = Config(run_dir=str(tmp_path / "r"), radmc_dir=str(tmp_path / "rad"))
    assert c.run_path == tmp_path / "r"
    assert c.radmc_path == tmp_path / "rad"        # explicit override
    assert c.fig_path == tmp_path / "r" / "fig"    # derived from run


def test_config_does_not_create_dirs(tmp_path):
    """Constructing a Config must not create the run directory."""
    run_dir = tmp_path / "not_created_yet"
    Config(run_dir=str(run_dir))
    assert not run_dir.exists(), "Config must not create directories at construction"


# ---------------------------------------------------------------------------
# Multi-run test: two Configs in one process write to their own directories
# ---------------------------------------------------------------------------

def test_multiple_run_dirs_route_correctly(tmp_path):
    """
    Build and save a model with Config(run_dir=A), then with Config(run_dir=B);
    each artifact must land in its own directory and nowhere else.
    """
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"

    config_a = _small_config(dir_a)
    mg_a = ModelGenerator(config_a)
    mg_a.calc_kinematic_structure()
    model_a = mg_a.get_model()
    model_a.save_pickle("model.pkl", dirpath=config_a.run_path)
    mg_a.save()  # defaults to config_a.run_path

    config_b = _small_config(dir_b)
    mg_b = ModelGenerator(config_b)
    mg_b.calc_kinematic_structure()
    model_b = mg_b.get_model()
    model_b.save_pickle("model.pkl", dirpath=config_b.run_path)
    mg_b.save()

    # Each directory has exactly its own artifacts.
    assert (dir_a / "model.pkl").exists()
    assert (dir_a / "mg.pkl").exists()
    assert (dir_b / "model.pkl").exists()
    assert (dir_b / "mg.pkl").exists()

    # No cross-contamination: A's model is not in B and vice versa beyond
    # what we explicitly wrote (both names are identical, so verify counts).
    assert sorted(p.name for p in dir_a.glob("*.pkl")) == ["mg.pkl", "model.pkl"]
    assert sorted(p.name for p in dir_b.glob("*.pkl")) == ["mg.pkl", "model.pkl"]


# ---------------------------------------------------------------------------
# gpath deprecation shim
# ---------------------------------------------------------------------------

def test_gpath_run_dir_deprecation_warning(tmp_path):
    """envos.gpath.run_dir works but emits a DeprecationWarning."""
    import envos.gpath as gpath

    _small_config(tmp_path / "active")  # registers itself
    with pytest.warns(DeprecationWarning):
        value = gpath.run_dir
    assert Path(value) == (tmp_path / "active")


def test_gpath_reflects_latest_config(tmp_path):
    """The shim reflects the most recently constructed Config."""
    import envos.gpath as gpath

    _small_config(tmp_path / "first")
    _small_config(tmp_path / "second")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert Path(gpath.run_dir) == (tmp_path / "second")
        assert Path(gpath.radmc_dir) == (tmp_path / "second" / "radmc")


def test_gpath_home_dir_compat():
    """home_dir stays the package parent directory (mori2023.py compat)."""
    import envos.gpath as gpath

    with pytest.warns(DeprecationWarning):
        home = gpath.home_dir
    assert Path(home) == Path(gpath.__file__).parents[1]


def test_gpath_unknown_attribute_raises():
    """Accessing an unknown attribute raises AttributeError (not a warning)."""
    import envos.gpath as gpath

    with pytest.raises(AttributeError):
        gpath.does_not_exist


def test_gpath_get_no_warning(recwarn):
    """The internal _get() helper resolves without a DeprecationWarning."""
    import envos.gpath as gpath

    gpath._get("logfile")
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in recwarn.list
    )


# ---------------------------------------------------------------------------
# storage_path three-stage resolution
# ---------------------------------------------------------------------------

def test_storage_path_explicit(tmp_path):
    """Stage 1: explicit storage_dir is used verbatim."""
    c = Config(storage_dir=str(tmp_path / "mystore"))
    assert c.storage_path == tmp_path / "mystore"


def test_storage_path_legacy_repo_root(tmp_path, monkeypatch):
    """Stage 2: a repository-root storage/ directory is preferred if present."""
    import envos.config as config_mod

    # Simulate the package living under <root>/envos with <root>/storage present.
    pkg_dir = tmp_path / "envos"
    pkg_dir.mkdir()
    fake_file = pkg_dir / "config.py"
    fake_file.write_text("")
    legacy = tmp_path / "storage"
    legacy.mkdir()

    monkeypatch.setattr(config_mod, "__file__", str(fake_file))
    c = Config()  # no storage_dir
    assert c.storage_path == legacy


def test_storage_path_packaged_default():
    """Stage 3: with no explicit/legacy dir, fall back to packaged storage."""
    # In this repo the package-internal storage exists; default resolves to it.
    c = Config()
    sp = c.storage_path
    assert sp.name == "storage"
    assert sp.is_dir()
    # Must contain the bundled data files.
    assert (sp / "dustkappa_MRN20.inp").exists()
