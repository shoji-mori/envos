"""
Tests for P1-16, P1-17, P1-18: plot_tools fixes.

All tests run under Agg backend (set in conftest.py).
Tests verify:
  - P1-16: get_subgrid_peaks, add_mass_estimate_plot, add_peaks (plot_funcs.py)
  - P1-17: plot_velocity_midplane_profile, plot_losvelocity_midplane_map,
           plot_angular_velocity_midplane_profile (physical_structure.py)
  - P1-18: plot_pvdiagram, plot_image with refimage and contour=False (obs_output.py)
"""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt

import envos
import envos.gpath as gpath
from envos.plot_tools import plot_funcs as pfun
from envos.plot_tools import obs_output as pobs
from envos.plot_tools import physical_structure as pphys
from envos.obs import PVmap, Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close_fig():
    """Close any open matplotlib figures to avoid state leakage."""
    plt.close("all")


def _make_synthetic_gaussian_pv(nx=40, nv=40, peak_x=20.0, peak_v=1.0,
                                 sigma_x=10.0, sigma_v=0.3):
    """
    Create a synthetic 2D Gaussian PV diagram centred at (peak_x, peak_v).

    Returns (xau, vkms, Ipv) as numpy arrays.
    """
    xau = np.linspace(-100, 100, nx)
    vkms = np.linspace(-3, 3, nv)
    xx, vv = np.meshgrid(xau, vkms, indexing="ij")
    Ipv = np.exp(-0.5 * ((xx - peak_x) / sigma_x) ** 2
                  - 0.5 * ((vv - peak_v) / sigma_v) ** 2)
    return xau, vkms, Ipv


# ---------------------------------------------------------------------------
# P1-16 tests
# ---------------------------------------------------------------------------

class TestGetSubgridPeaks:
    """P1-16: get_subgrid_peaks returns correct peak within ±1 cell."""

    def test_peak_within_one_cell(self):
        """
        Quantitative check: the subgrid peak must lie within ±1 cell of the
        true Gaussian centre in both x and v dimensions.
        """
        peak_x_true = 20.0
        peak_v_true = 1.0
        nx, nv = 40, 40
        xau, vkms, Ipv = _make_synthetic_gaussian_pv(
            nx=nx, nv=nv, peak_x=peak_x_true, peak_v=peak_v_true
        )
        dx = xau[1] - xau[0]
        dv = vkms[1] - vkms[0]

        result = pfun.get_subgrid_peaks(xau, vkms, Ipv, num_peak_level=1,
                                         threshold_rel=0.3, num_peaks=4)
        assert result is not None, "get_subgrid_peaks should not return None"
        assert len(result) >= 1, "at least one peak level should be returned"
        assert len(result[0]) >= 1, "at least one peak in level 0"

        top_peak = result[0][0]
        assert abs(top_peak.x1 - peak_x_true) <= dx, (
            f"x1={top_peak.x1:.2f} not within ±1 cell ({dx:.2f}) of true peak {peak_x_true}"
        )
        assert abs(top_peak.x2 - peak_v_true) <= dv, (
            f"x2={top_peak.x2:.2f} not within ±1 cell ({dv:.2f}) of true peak {peak_v_true}"
        )

    def test_peak_value_is_scalar_float(self):
        """Peak.value must be a Python float (not a numpy 2D array)."""
        xau, vkms, Ipv = _make_synthetic_gaussian_pv()
        result = pfun.get_subgrid_peaks(xau, vkms, Ipv, num_peak_level=1,
                                         threshold_rel=0.3, num_peaks=4)
        assert result is not None
        for peak in result[0]:
            assert isinstance(peak.value, float), (
                f"peak.value should be a float, got {type(peak.value)}"
            )

    def test_returns_none_on_flat_image(self):
        """Flat (all-zero) PV returns None (no peaks found)."""
        xau = np.linspace(-50, 50, 20)
        vkms = np.linspace(-2, 2, 20)
        Ipv = np.zeros((20, 20))
        result = pfun.get_subgrid_peaks(xau, vkms, Ipv, num_peak_level=1)
        assert result is None, "flat PV should return None"


class TestAddMassEstimatePlot:
    """P1-16: add_mass_estimate_plot does not raise NameError for partial flags."""

    def setup_method(self):
        plt.figure()

    def teardown_method(self):
        _close_fig()

    def test_mass_ip_only_no_error(self):
        """mass_ip=True, mass_vp=False must not raise NameError (txt_Mvp was undefined)."""
        xau, vkms, Ipv = _make_synthetic_gaussian_pv()
        # Should complete without NameError
        pfun.add_mass_estimate_plot(
            xau, vkms, Ipv,
            Ms_Msun=0.3,
            rCR_au=100,
            mass_ip=True,
            mass_vp=False,
        )

    def test_mass_vp_only_no_error(self):
        """mass_ip=False, mass_vp=True must not raise NameError (txt_Mip was undefined)."""
        xau, vkms, Ipv = _make_synthetic_gaussian_pv()
        pfun.add_mass_estimate_plot(
            xau, vkms, Ipv,
            Ms_Msun=0.3,
            rCR_au=100,
            f_crit=0.5,
            mass_ip=False,
            mass_vp=True,
        )

    def test_both_flags_no_error(self):
        """mass_ip=True, mass_vp=True must also work."""
        xau, vkms, Ipv = _make_synthetic_gaussian_pv()
        pfun.add_mass_estimate_plot(
            xau, vkms, Ipv,
            Ms_Msun=0.3,
            rCR_au=100,
            f_crit=0.5,
            mass_ip=True,
            mass_vp=True,
        )


# ---------------------------------------------------------------------------
# P1-17 tests  (require g1_model fixture)
# ---------------------------------------------------------------------------

class TestPlotVelocityMidplaneProfile:
    """P1-17: plot_velocity_midplane_profile runs under Agg and saves a PDF."""

    def teardown_method(self):
        _close_fig()

    def test_runs_midplane_slice(self, g1_model, tmp_path):
        """midplane_average=False should not raise (was: NameError/TypeError on get_argmid)."""
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            plt.figure()
            # midplane_average=False: test that the fixed save logic (P1-17 fix 4)
            # and get_argmid() call (P1-17 fix 1) don't crash
            pphys.plot_velocity_midplane_profile(g1_model, midplane_average=False, save=True)
            # PDF should have been created
            pdfs = list(tmp_path.glob("*.pdf"))
            assert len(pdfs) >= 1, "Expected at least one PDF to be written"
        finally:
            gpath.make_dirs(fig=fig_dir_orig)

    def test_runs_midplane_average(self, g1_model, tmp_path):
        """midplane_average=True (the default) should also complete without error."""
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            plt.figure()
            pphys.plot_velocity_midplane_profile(g1_model, midplane_average=True, save=True)
            pdfs = list(tmp_path.glob("*.pdf"))
            assert len(pdfs) >= 1
        finally:
            gpath.make_dirs(fig=fig_dir_orig)


class TestPlotLosvelocityMidplaneMap:
    """P1-17: plot_losvelocity_midplane_map runs under Agg (coordinate fix B-33)."""

    def teardown_method(self):
        _close_fig()

    def test_runs_without_error(self, g1_model, tmp_path):
        """Function should complete and save a PDF without NameError or ValueError.

        The G1 model has Tgas=None (no radmc3d run). We supply a synthetic Tgas
        (uniform 10 K) to satisfy the plot function's requirement, which is not
        part of the fix under test. The fix (B-33) is in the x,y coordinate
        computation.
        """
        import copy
        model = copy.copy(g1_model)
        if model.Tgas is None:
            model.Tgas = np.full_like(model.rhogas, 10.0)

        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            plt.figure()
            pphys.plot_losvelocity_midplane_map(model)
            pdfs = list(tmp_path.glob("*.pdf"))
            assert len(pdfs) >= 1
        finally:
            gpath.make_dirs(fig=fig_dir_orig)


class TestPlotAngularVelocityMidplaneProfile:
    """P1-17: plot_angular_velocity_midplane_profile runs without UnboundLocalError (B-32)."""

    def teardown_method(self):
        _close_fig()

    def test_runs_without_error(self, g1_model, tmp_path):
        """Removing trailing comma on Omega prevents tuple / UnboundLocalError."""
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            plt.figure()
            pphys.plot_angular_velocity_midplane_profile(g1_model, save=True)
            pdfs = list(tmp_path.glob("*.pdf"))
            assert len(pdfs) >= 1
        finally:
            gpath.make_dirs(fig=fig_dir_orig)


# ---------------------------------------------------------------------------
# P1-18 tests
# ---------------------------------------------------------------------------

class TestPlotPvdiagram:
    """P1-18: plot_pvdiagram completes on a PV with no detectable peaks (None guard)."""

    def teardown_method(self):
        _close_fig()

    def test_no_peaks_pv_no_crash(self, tmp_path):
        """
        A PV where get_subgrid_peaks returns None (below-threshold data).
        With contour=True the peaks scatter call must be skipped (B-39 fix).
        Using a very-low-signal PV with distinct min/max so contour is valid.
        """
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            nx, nv = 20, 20
            xau = np.linspace(-100, 100, nx)
            vkms = np.linspace(-2, 2, nv)
            # Use a gradient image: no local peaks above threshold_rel=0.7
            # (peak_local_max with threshold_rel=0.7 on a linear gradient finds no peaks)
            xx, vv = np.meshgrid(xau, vkms, indexing="ij")
            Ipv = (xx - xx.min()) / (xx.max() - xx.min())  # linear ramp, no local max
            pv = PVmap(Ipv=Ipv, xau=xau, vkms=vkms)
            # contour=True to exercise the peaks guard (B-39)
            pobs.plot_pvdiagram(pv, contour=True, out="test_pv.pdf")
        finally:
            gpath.make_dirs(fig=fig_dir_orig)

    def test_gaussian_pv_with_peaks(self, tmp_path):
        """A Gaussian PV has a detectable peak; plot_pvdiagram should complete."""
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            xau, vkms, Ipv = _make_synthetic_gaussian_pv()
            pv = PVmap(Ipv=Ipv, xau=xau, vkms=vkms)
            pobs.plot_pvdiagram(pv, contour=True, out="test_pv_gaussian.pdf")
        finally:
            gpath.make_dirs(fig=fig_dir_orig)


class TestPlotImageWithRefimage:
    """P1-18: plot_image(im, refimage=ref, contour=False) completes (B-38)."""

    def teardown_method(self):
        _close_fig()

    def test_refimage_contour_false_no_crash(self, tmp_path):
        """
        When contour=False, _contopt was undefined before the fix.
        Now _contopt is defined unconditionally, so refimage contour works.
        """
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            nx, ny = 20, 20
            xau = np.linspace(-100, 100, nx)
            yau = np.linspace(-100, 100, ny)
            data = np.random.default_rng(42).random((nx, ny))
            data_ref = np.random.default_rng(7).random((nx, ny))

            im = Image(data=data, xau=xau, yau=yau)
            ref = Image(data=data_ref, xau=xau, yau=yau)

            pobs.plot_image(im, refimage=ref, contour=False, save=True,
                            out="test_image.pdf")
            pdfs = list(tmp_path.glob("*.pdf"))
            assert len(pdfs) >= 1
        finally:
            gpath.make_dirs(fig=fig_dir_orig)

    def test_no_refimage_contour_false(self, tmp_path):
        """No refimage with contour=False should also work (baseline sanity)."""
        fig_dir_orig = gpath.fig_dir
        gpath.make_dirs(fig=tmp_path)
        try:
            nx, ny = 20, 20
            xau = np.linspace(-100, 100, nx)
            yau = np.linspace(-100, 100, ny)
            data = np.random.default_rng(42).random((nx, ny))
            im = Image(data=data, xau=xau, yau=yau)
            pobs.plot_image(im, contour=False, save=True, out="test_image_noref.pdf")
        finally:
            gpath.make_dirs(fig=fig_dir_orig)
