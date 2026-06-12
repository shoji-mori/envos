"""
Tests for envos/streamline.py — P1-13
Covers B-23 (global run_dir), B-24 (mutable default), B-25 (mirror deletion),
and print→logger.debug.
"""

import os
import pytest
from envos.streamline import calc_streamline, StreamlineCalculator2


def test_variables_not_accumulated(g1_model):
    """
    B-24: calling calc_streamline twice must not accumulate variables across calls.
    The mutable default `variables=[]` was the root cause; now `variables=None` + copy.
    """
    streams1 = calc_streamline(g1_model, r0_au=500, theta0_deg=60)
    streams2 = calc_streamline(g1_model, r0_au=500, theta0_deg=60)

    # Each call should produce the same number of variables (rhogas only for g1_model)
    assert len(streams1) == 1
    assert len(streams2) == 1
    nvar1 = len(streams1[0].variables)
    nvar2 = len(streams2[0].variables)
    assert nvar1 == nvar2, (
        f"Variable count changed between calls: {nvar1} vs {nvar2}. "
        "This indicates mutable default accumulation was not fixed."
    )


def test_save_writes_file(g1_model, tmp_path):
    """
    B-23: save=True with dpath=tmp_path must produce at least one .txt file
    without raising NameError from the old `global run_dir`.
    """
    streams = calc_streamline(
        g1_model,
        r0_au=500,
        theta0_deg=60,
        save=True,
        dpath=tmp_path,
    )
    files = list(tmp_path.iterdir())
    assert len(files) > 0, "save=True produced no output files"
    assert any(f.suffix == ".txt" for f in files), "Expected .txt files from save"


def test_mirror_argument_removed():
    """
    D5: StreamlineCalculator2 must no longer accept a `mirror` argument.
    """
    import inspect
    sig = inspect.signature(StreamlineCalculator2.__init__)
    assert "mirror" not in sig.parameters, (
        "mirror argument should have been removed (D5 decision)"
    )


def test_mirror_symmetry_attribute_removed(g1_model):
    """
    D5: StreamlineCalculator2 instances must not have a mirror_symmetry attribute.
    """
    from envos import nconst as nc
    import numpy as np
    slc = StreamlineCalculator2(
        g1_model,
        t_eval=np.arange(10, 1e5, 100) * nc.yr,
    )
    assert not hasattr(slc, "mirror_symmetry"), (
        "mirror_symmetry attribute should have been removed (D5 decision)"
    )
