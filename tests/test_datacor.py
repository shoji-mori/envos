"""
Tests for envos/datacor.py — P1-15
Covers C-56: symmetric preprocessing and ranges-axis-drop fix.
"""

import numpy as np
import pytest
from envos.datacor import calc_datacor
from envos.obs import Cube


def _make_cube(shape=(5, 5, 7), seed=42):
    """Build a synthetic Cube without radmc3d."""
    rng = np.random.default_rng(seed)
    nx, ny, nv = shape
    xau = np.linspace(-100.0, 100.0, nx)
    yau = np.linspace(-100.0, 100.0, ny)
    vkms = np.linspace(-3.0, 3.0, nv)
    # Positive data so ZNCC is well-defined
    Ippv = rng.uniform(0.5, 1.5, shape)
    return Cube(Ippv=Ippv, xau=xau, yau=yau, vkms=vkms)


class TestZNCC:
    def test_self_correlation_is_one(self):
        """ZNCC of a cube with itself must equal 1.0."""
        cube = _make_cube()
        res = calc_datacor(cube, cube, ranges=[(-100, 100), (-100, 100), (-3, 3)])
        assert abs(res - 1.0) < 1e-10, f"Self-correlation expected 1.0, got {res}"

    def test_self_correlation_one_axis_range(self):
        """ZNCC self-correlation with only 1 axis range specified must also be 1.0."""
        cube = _make_cube()
        # ranges shorter than number of axes — C-56 fix (no axis drop)
        res = calc_datacor(cube, cube, ranges=[(-100, 100)])
        assert abs(res - 1.0) < 1e-10, (
            f"Self-correlation with 1-axis range expected 1.0, got {res}"
        )

    def test_one_axis_consistent_with_three_axes(self):
        """
        C-56: specifying ranges for 1 axis vs all 3 axes must give the same result
        when the extra ranges span the identical data (same grid points selected).
        With ranges=[(-50,50)], xau is filtered to [-50,0,50].
        With ranges=[(-50,50), None, None], the extra None ranges default to full axes.
        Without the fix, the None range caused a `continue` and axis dropout → crash.
        We verify the 1-axis call succeeds and produces a valid ZNCC value.
        """
        cube1 = _make_cube(seed=1)
        cube2 = _make_cube(seed=2)

        # 1-axis range — before the fix, the remaining axes would be dropped → crash
        res_1ax = calc_datacor(cube1, cube2, ranges=[(-50, 50)])
        assert np.isfinite(res_1ax), (
            f"1-axis range returned non-finite ZNCC: {res_1ax}"
        )
        assert -1.0 <= res_1ax <= 1.0, (
            f"1-axis range returned out-of-range ZNCC: {res_1ax}"
        )

        # 3-axis range with same x restriction and full y/v ranges
        # (use slightly wider bounds so all y and v points are included)
        res_3ax = calc_datacor(
            cube1, cube2, ranges=[(-50, 50), (-101, 101), (-4, 4)]
        )
        # Both restrict xau to [-50, 0, 50]; y and v axes are the same full extent
        assert abs(res_3ax - res_1ax) < 1e-10, (
            f"1-axis range ({res_1ax:.6f}) differs from 3-axis range ({res_3ax:.6f}). "
            "The axes_newgrid must be identical when extra ranges cover the full axes."
        )

    def test_no_range_full_axes(self):
        """calc_datacor with empty ranges should work (uses all axes)."""
        cube = _make_cube()
        res = calc_datacor(cube, cube, ranges=[])
        assert abs(res - 1.0) < 1e-10


class TestPreprocessSymmetry:
    def test_symmetric_preprocess(self):
        """
        C-56: preprocess_func must be applied symmetrically.
        With a preprocess that returns a constant, im1 and im2 must be scaled
        by factors computed from the *original* (unmodified) data.
        """
        cube1 = _make_cube(seed=10)
        cube2 = _make_cube(seed=20)

        call_log = []

        def tracking_preprocess(im_a, im_b, axes):
            """Record which (hash of) im_a was seen for each call."""
            call_log.append(im_a.mean())
            return np.ones(1)  # multiplicative identity

        calc_datacor(
            cube1, cube2,
            ranges=[(-100, 100), (-100, 100), (-3, 3)],
            preprocess_func=tracking_preprocess,
        )
        # Two calls: one for fac1 (im_a=im1_original) and one for fac2 (im_a=im2_original)
        assert len(call_log) == 2, (
            f"Expected preprocess_func called twice, got {len(call_log)}"
        )
        # The two im_a means must differ (one is from cube1, one from cube2)
        assert abs(call_log[0] - call_log[1]) > 0.01, (
            "Both preprocess calls received the same im_a — "
            "the asymmetric bug (modified im1 used for im2 factor) was not fixed."
        )
