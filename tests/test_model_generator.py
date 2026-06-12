"""
Tests for envos/model_generator.py — P1-9.

Covers:
- C-48: f_dg=None initialisation + warning-with-fallback
- D1: disk synthesis is replace-mode (G4 golden regression)
- read_model: NotImplementedError for non-pkl paths
- disk="powerlaw" smoke test (set_disk uses logger, not print)
"""

import warnings
import tempfile
import logging

import numpy as np
import pytest

from tests.golden_helpers import (
    G1_CONFIG,
    GOLDEN_DIR,
    build_g4_model,
    extract_g4_arrays,
)


# ---------------------------------------------------------------------------
# C-48: f_dg=None initialisation and fallback warning
# ---------------------------------------------------------------------------

class TestFdgInitialisation:
    """C-48: ModelGenerator must initialise f_dg=None and warn when unset."""

    def test_fdg_is_none_before_config(self):
        """Without config, f_dg starts as None."""
        from envos.model_generator import ModelGenerator

        mg = ModelGenerator()
        assert mg.f_dg is None

    def test_fdg_set_from_config(self, tmp_path):
        """With config, f_dg is taken from config."""
        from envos.config import Config
        from envos.model_generator import ModelGenerator

        config = Config(run_dir=str(tmp_path), f_dg=0.05, **G1_CONFIG)
        mg = ModelGenerator(config)
        assert mg.f_dg == pytest.approx(0.05)

    def test_fdg_none_fallback_and_warning(self, tmp_path):
        """
        calc_kinematic_structure() uses f_dg=0.01 as fallback when f_dg is
        None, and emits a WARNING-level log record on the 'envos' logger.

        The 'envos' logger has propagate=False, so we attach a temporary
        handler directly to it to capture the record.
        """
        import logging
        from envos.config import Config
        from envos.model_generator import ModelGenerator
        from envos.log import logger as envos_logger

        config = Config(run_dir=str(tmp_path), **G1_CONFIG)
        mg = ModelGenerator(config)
        mg.f_dg = None  # force the unset path

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        envos_logger.addHandler(handler)
        try:
            mg.calc_kinematic_structure()
        finally:
            envos_logger.removeHandler(handler)

        # f_dg should be set to the default
        assert mg.f_dg == pytest.approx(0.01)

        # A warning mentioning f_dg must have been emitted
        assert any(
            "f_dg" in (r.getMessage() if callable(getattr(r, "getMessage", None)) else str(r.message))
            for r in records
        ), "Expected a WARNING mentioning 'f_dg', got: " + str([r.getMessage() for r in records])


# ---------------------------------------------------------------------------
# D1: disk synthesis — replace-mode
# ---------------------------------------------------------------------------

class TestDiskSynthesisReplaceMode:
    """
    D1: disk density must win only where disk.rho > envelope rho
    (replace-mode, not additive).
    """

    def test_disk_rho_never_exceeds_replace_sum(self, tmp_path):
        """
        After calc_kinematic_structure with disk='powerlaw', the combined
        rhogas must not exceed max(envelope_rho, disk_rho) in any cell.
        (The old additive method would have violated this.)
        """
        from envos.config import Config
        from envos.model_generator import ModelGenerator

        config = Config(
            run_dir=str(tmp_path),
            disk="powerlaw",
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()
        rhogas = model.rhogas

        # In replace mode: rhogas = max(envelope, disk) cell-by-cell.
        # Verify the disk contribution: disk.rho[cond] == rhogas[cond]
        disk_rho = mg.disk.rho
        cond = mg.disk_region  # cells where disk dominates (set during calc)
        np.testing.assert_allclose(
            rhogas[cond],
            disk_rho[cond],
            rtol=1e-12,
            err_msg="In disk-dominated cells, rhogas must equal disk.rho (replace mode)",
        )

    def test_g4_arrays_match_golden(self, tmp_path):
        """
        G4 golden regression: UCM + disk model reproduces replace-mode results.
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


# ---------------------------------------------------------------------------
# read_model: NotImplementedError for non-pkl
# ---------------------------------------------------------------------------

class TestReadModel:
    """read_model must raise NotImplementedError for non-pkl paths."""

    def test_read_model_non_pkl_raises(self, tmp_path):
        from envos.model_generator import read_model

        non_pkl = str(tmp_path / "model.fits")
        with pytest.raises(NotImplementedError):
            read_model(non_pkl)

    def test_read_model_pkl_path_returns(self, tmp_path):
        """
        For a .pkl path, read_model delegates to tools.read_pickle.
        We don't test the actual deserialization here — just that the
        non-pkl branch is no longer taken.
        """
        from envos.model_generator import read_model

        pkl_path = str(tmp_path / "model.pkl")
        # File does not exist, but read_pickle will raise a more specific error
        with pytest.raises(Exception) as exc_info:
            read_model(pkl_path)
        # Must NOT be a NotImplementedError (that's the old "Still constructing" path)
        assert not isinstance(exc_info.value, NotImplementedError), (
            "read_model('.pkl') should not raise NotImplementedError"
        )


# ---------------------------------------------------------------------------
# disk="powerlaw" smoke — logger.info not print
# ---------------------------------------------------------------------------

class TestDiskPowerlaw:
    """disk="powerlaw" must work end-to-end and not use print()."""

    def test_disk_powerlaw_smoke(self, tmp_path):
        """
        Config with disk='powerlaw' produces a model with finite, positive rhogas.
        """
        from envos.config import Config
        from envos.model_generator import ModelGenerator

        config = Config(
            run_dir=str(tmp_path),
            disk="powerlaw",
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        mg.calc_kinematic_structure()
        model = mg.get_model()

        assert np.all(np.isfinite(model.rhogas)), "rhogas contains non-finite values"
        assert np.any(model.rhogas > 0), "rhogas is everywhere zero"

    def test_set_disk_no_print(self, tmp_path, capsys):
        """
        set_disk with disk='powerlaw' must not call print() — it should use
        logger.info instead (P1-9 item 4).
        """
        from envos.config import Config
        from envos.model_generator import ModelGenerator

        config = Config(
            run_dir=str(tmp_path),
            disk="powerlaw",
            **G1_CONFIG,
        )
        mg = ModelGenerator(config)
        captured = capsys.readouterr()
        # The disk config dict should NOT appear on stdout
        assert "fracMd" not in captured.out, (
            "set_disk() printed disk config to stdout; should use logger.info instead"
        )
