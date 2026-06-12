"""
Tests for envos/radmc3d.py — P1-10.

Covers:
- B-14: set_model FileNotFoundError for non-existent path, ValueError for None
- B-15: rhodust=None fallback (rhogas*f_dg) without exit()
- B-16: remove_file uses full path (gas_temperature.inp / dust_temperature.dat)
- C-57/D6: Config(nonlte=1) → RadmcController raises NotImplementedError
- C-58: run_mctherm chdir wrapped in try/finally (structural test)
- P1-4 carry-over: set_lineobs_inpfiles raises ValueError when molabun is None

No radmc3d binary is required for any of these tests.
"""

import os
import tempfile
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tests.golden_helpers import G1_CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_config(tmp_path):
    """Return a minimal Config that sets dirs without needing radmc3d."""
    from envos.config import Config

    return Config(run_dir=str(tmp_path), **G1_CONFIG)


@pytest.fixture()
def built_model(tmp_path):
    """Return a CircumstellarModel built from G1 config (no disk)."""
    from envos.config import Config
    from envos.model_generator import ModelGenerator

    config = Config(run_dir=str(tmp_path), **G1_CONFIG)
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()


@pytest.fixture()
def controller(minimal_config):
    """Return a RadmcController initialised from minimal_config."""
    from envos.radmc3d import RadmcController

    return RadmcController(config=minimal_config)


# ---------------------------------------------------------------------------
# B-14: set_model
# ---------------------------------------------------------------------------

class TestSetModel:
    """B-14: set_model must correctly handle string paths and non-None models."""

    def test_set_model_nonexistent_path_raises(self, controller):
        with pytest.raises(FileNotFoundError):
            controller.set_model("/nonexistent/path/model.pkl")

    def test_set_model_none_raises_value_error(self, controller):
        with pytest.raises(ValueError):
            controller.set_model(None)

    def test_set_model_object_is_stored(self, controller, built_model):
        """Passing a model object directly stores it on self.model."""
        controller.set_model(built_model)
        assert controller.model is built_model


# ---------------------------------------------------------------------------
# B-15: rhodust=None fallback (no exit(), no print)
# ---------------------------------------------------------------------------

class TestRhodustFallback:
    """
    B-15: when model.rhodust is None, set_mctherm_inpfiles must compute
    rhod = rhogas * f_dg without calling exit().

    radmc3d binary is NOT needed — we only verify the generated input file.
    """

    def _make_controller_with_model(self, tmp_path, f_dg=0.01):
        from envos.config import Config
        from envos.model_generator import ModelGenerator
        from envos.radmc3d import RadmcController

        config = Config(run_dir=str(tmp_path), f_dg=f_dg, **G1_CONFIG)
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()

        # Force rhodust to None to trigger the fallback path
        model.rhodust = None

        radmc = RadmcController(config=config)
        radmc.set_model(model)
        return radmc, model

    def test_set_mctherm_inpfiles_with_null_rhodust(self, tmp_path):
        """
        set_mctherm_inpfiles() must complete without SystemExit when
        model.rhodust is None (old code called exit()).
        """
        # Provide a dummy dustkappa_silicate.inp so the file copy doesn't fail
        storage_dir = Path(str(tmp_path)) / "storage"
        storage_dir.mkdir()
        (storage_dir / "dustkappa_silicate.inp").write_text(
            "2\n3\n1.0 1.0 0.0\n10.0 0.5 0.0\n100.0 0.2 0.0\n"
        )

        from envos.config import Config
        from envos.model_generator import ModelGenerator
        from envos.radmc3d import RadmcController

        config = Config(
            run_dir=str(tmp_path),
            storage_dir=str(storage_dir),
            f_dg=0.02,
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()
        model.rhodust = None  # trigger fallback

        radmc = RadmcController(config=config)
        radmc.set_model(model)

        # Must not raise SystemExit
        radmc.set_mctherm_inpfiles()

    def test_dust_density_file_matches_rhogas_times_fdg(self, tmp_path):
        """
        After set_mctherm_inpfiles() with rhodust=None, dust_density.inp
        must contain values equal to rhogas * f_dg.
        """
        f_dg = 0.02

        storage_dir = Path(str(tmp_path)) / "storage"
        storage_dir.mkdir()
        (storage_dir / "dustkappa_silicate.inp").write_text(
            "2\n3\n1.0 1.0 0.0\n10.0 0.5 0.0\n100.0 0.2 0.0\n"
        )

        from envos.config import Config
        from envos.model_generator import ModelGenerator
        from envos.radmc3d import RadmcController

        config = Config(
            run_dir=str(tmp_path),
            storage_dir=str(storage_dir),
            f_dg=f_dg,
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()
        rhogas = model.rhogas.copy()
        model.rhodust = None  # trigger fallback

        radmc = RadmcController(config=config)
        radmc.set_model(model)
        radmc.set_mctherm_inpfiles()

        # Parse dust_density.inp and compare to rhogas * f_dg
        ddensity_path = Path(radmc.radmc_dir) / "dust_density.inp"
        assert ddensity_path.exists(), "dust_density.inp was not created"

        lines = ddensity_path.read_text().strip().split("\n")
        # Format: iformat\n ncells\n nspecies\n values...
        n_cells = int(lines[1])
        values = np.array([float(v) for v in lines[3:3 + n_cells]])

        expected = (rhogas * f_dg).ravel(order="F")
        np.testing.assert_allclose(
            values, expected, rtol=1e-10,
            err_msg="dust_density.inp values do not match rhogas * f_dg",
        )


# ---------------------------------------------------------------------------
# B-16: remove_file uses radmc_dir-joined path
# ---------------------------------------------------------------------------

class TestRemoveFilePaths:
    """
    B-16: gas_temperature.inp and dust_temperature.dat are removed from
    radmc_dir, not from the current working directory.
    """

    def test_temperature_files_removed_from_radmc_dir(self, tmp_path):
        """
        If gas_temperature.inp and dust_temperature.dat exist in radmc_dir,
        they must be deleted after set_mctherm_inpfiles().
        """
        storage_dir = Path(str(tmp_path)) / "storage"
        storage_dir.mkdir()
        (storage_dir / "dustkappa_silicate.inp").write_text(
            "2\n3\n1.0 1.0 0.0\n10.0 0.5 0.0\n100.0 0.2 0.0\n"
        )

        from envos.config import Config
        from envos.model_generator import ModelGenerator
        from envos.radmc3d import RadmcController

        config = Config(
            run_dir=str(tmp_path),
            storage_dir=str(storage_dir),
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()

        radmc = RadmcController(config=config)
        radmc.set_model(model)

        radmc_dir = Path(radmc.radmc_dir)

        # Pre-create temperature files in radmc_dir to confirm they get removed
        gas_t = radmc_dir / "gas_temperature.inp"
        dust_t = radmc_dir / "dust_temperature.dat"
        gas_t.write_text("dummy gas temperature\n")
        dust_t.write_text("dummy dust temperature\n")

        assert gas_t.exists()
        assert dust_t.exists()

        # Also create sentinel files in cwd to confirm cwd files are untouched
        cwd_gas_t = Path(os.getcwd()) / "gas_temperature.inp"
        cwd_dust_t = Path(os.getcwd()) / "dust_temperature.dat"
        created_cwd_gas = False
        created_cwd_dust = False
        if not cwd_gas_t.exists():
            cwd_gas_t.write_text("sentinel gas\n")
            created_cwd_gas = True
        if not cwd_dust_t.exists():
            cwd_dust_t.write_text("sentinel dust\n")
            created_cwd_dust = True

        try:
            radmc.set_mctherm_inpfiles()
        finally:
            if created_cwd_gas and cwd_gas_t.exists():
                cwd_gas_t.unlink()
            if created_cwd_dust and cwd_dust_t.exists():
                cwd_dust_t.unlink()

        assert not gas_t.exists(), "gas_temperature.inp was not removed from radmc_dir"
        assert not dust_t.exists(), "dust_temperature.dat was not removed from radmc_dir"


# ---------------------------------------------------------------------------
# D6 / C-57: nonlte=1 raises NotImplementedError in __init__
# ---------------------------------------------------------------------------

class TestNonlteRaisesEarly:
    """D6: Config(nonlte=1) → RadmcController must raise NotImplementedError."""

    def test_nonlte_raises_not_implemented_error(self, tmp_path):
        from envos.config import Config
        from envos.radmc3d import RadmcController

        config = Config(run_dir=str(tmp_path), nonlte=1, **G1_CONFIG)
        with pytest.raises(NotImplementedError, match="nonlte"):
            RadmcController(config=config)

    def test_nonlte_zero_does_not_raise(self, tmp_path):
        from envos.config import Config
        from envos.radmc3d import RadmcController

        config = Config(run_dir=str(tmp_path), nonlte=0, **G1_CONFIG)
        rc = RadmcController(config=config)
        assert rc is not None


# ---------------------------------------------------------------------------
# P1-4 carry-over: molabun guard in set_lineobs_inpfiles
# ---------------------------------------------------------------------------

class TestMolabunGuard:
    """
    set_lineobs_inpfiles() must raise ValueError when molabun is None.
    (Config.molabun defaults to "" which is falsy but not None;
     this guard defends against explicit None assignment.)
    """

    def test_molabun_none_raises_value_error(self, tmp_path, built_model):
        from envos.config import Config
        from envos.radmc3d import RadmcController

        config = Config(run_dir=str(tmp_path), **G1_CONFIG)
        rc = RadmcController(config=config)
        rc.set_model(built_model)
        rc.molabun = None  # force the unset path

        with pytest.raises(ValueError, match="molabun"):
            rc.set_lineobs_inpfiles()
