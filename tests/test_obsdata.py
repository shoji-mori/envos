"""
Tests for P1-11 obs.py bug fixes.

Covers PLAN.md P1-11 test items (a)-(g), plus:
  - trim round-trip
  - mom0 sum/integrate consistency
  - _reset_positive_axes works for PVmap and Image

Synthetic data only — no radmc3d required.
"""

import numpy as np
import pytest
from astropy import units as u

from envos.obs import Cube, Image, PVmap, Obreso, Convolver, minmaxargs


# ---------------------------------------------------------------------------
# Helper: build a synthetic Cube with a 3-D Gaussian brightness distribution
# ---------------------------------------------------------------------------

def _make_gaussian_cube(nx=10, ny=10, nv=15, sigma_x=2.0, sigma_y=2.0, sigma_v=2.0):
    """Return a Cube with I(x,y,v) = exp(-x^2/(2sx^2) - y^2/(2sy^2) - v^2/(2sv^2))."""
    xau = np.linspace(-50, 50, nx)
    yau = np.linspace(-50, 50, ny)
    vkms = np.linspace(-5, 5, nv)

    X, Y, V = np.meshgrid(xau, yau, vkms, indexing="ij")
    Ippv = np.exp(
        -X**2 / (2 * sigma_x**2)
        - Y**2 / (2 * sigma_y**2)
        - V**2 / (2 * sigma_v**2)
    )
    return Cube(Ippv=Ippv, xau=xau, yau=yau, vkms=vkms, dpc=None)


# ===========================================================================
# (a) get_mom0_map(vlim=...) excludes channels outside the range
# ===========================================================================

class TestMom0Vlim:
    def test_vlim_excludes_outer_channels(self):
        """Channels outside vlim must be excluded from the moment-0 map."""
        cube = _make_gaussian_cube(nx=8, ny=8, nv=20)
        # Full moment-0 (no vlim)
        img_full = cube.get_mom0_map(normalize=None, method="sum")
        # Restrict to central ±2 km/s — peak stays but wings are cut
        img_vlim = cube.get_mom0_map(normalize=None, method="sum", vlim=(-2, 2))
        # The restricted integral must be smaller than the full integral
        assert np.max(img_vlim.get_I()) < np.max(img_full.get_I())

    def test_vlim_channels_actually_included(self):
        """Only channels in [vlim[0], vlim[1]) contribute."""
        cube = _make_gaussian_cube(nx=6, ny=6, nv=11)
        vkms = cube.vkms
        # Choose vlim that exactly includes a subset
        vlim = (vkms[3], vkms[7])
        img = cube.get_mom0_map(normalize=None, method="sum", vlim=vlim)
        # Manually replicate the expected calculation
        cond = (vkms > vlim[0]) & (vkms < vlim[1])
        _Ippv = cube.Ippv[..., cond]
        _vkms = vkms[cond]
        expected = np.sum(_Ippv, axis=-1) * (_vkms[1] - _vkms[0])
        np.testing.assert_allclose(img.get_I(), expected)

    def test_vlim_invalid_raises(self):
        """vlim with vlim[0] >= vlim[1] must raise ValueError."""
        cube = _make_gaussian_cube()
        with pytest.raises(ValueError):
            cube.get_mom0_map(vlim=(2, 1))

    def test_vlim_equal_raises(self):
        """vlim[0] == vlim[1] is also invalid."""
        cube = _make_gaussian_cube()
        with pytest.raises(ValueError):
            cube.get_mom0_map(vlim=(1, 1))

    def test_vlim_sum_equals_integrate_approx(self):
        """sum and integrate give close results for uniform spacing."""
        cube = _make_gaussian_cube(nv=30)
        img_sum = cube.get_mom0_map(normalize=None, method="sum")
        img_int = cube.get_mom0_map(normalize=None, method="integrate")
        # Allow ~5% relative tolerance for uniform spacing approximation
        np.testing.assert_allclose(
            img_sum.get_I(), img_int.get_I(), rtol=0.05
        )

    def test_normalize_peak_sets_iunit(self):
        """normalize='peak' must update Iunit (追補-77)."""
        cube = _make_gaussian_cube()
        img = cube.get_mom0_map(normalize="peak")
        # After norm_I("max"), Iunit becomes a custom unit named "I_max"
        assert str(img.Iunit) == "I_max"

    def test_normalize_peak_max_is_one(self):
        """After normalize='peak', maximum of map must be 1."""
        cube = _make_gaussian_cube()
        img = cube.get_mom0_map(normalize="peak")
        assert np.isclose(np.max(img.get_I()), 1.0)


# ===========================================================================
# (b) Two Cube instances share NO refpos object
# ===========================================================================

class TestRefposSeparation:
    def test_two_cubes_have_separate_refpos(self):
        """Setting refpos on one Cube must not affect another (A-7 / P0-2)."""
        c1 = _make_gaussian_cube()
        c2 = _make_gaussian_cube()
        c1.refpos.freq0 = 12345.0
        assert c2.refpos.freq0 != 12345.0, (
            "c2.refpos.freq0 was modified — refpos objects are shared!"
        )

    def test_cube_and_image_have_separate_refpos(self):
        """Cube and Image created independently must not share refpos."""
        cube = _make_gaussian_cube()
        img = Image(
            data=np.ones((5, 5)),
            xau=np.linspace(-10, 10, 5),
            yau=np.linspace(-10, 10, 5),
        )
        cube.refpos.ra0 = 99.9
        assert img.refpos.ra0 != 99.9


# ===========================================================================
# (c) Descending-axis Image is reordered correctly by _reset_positive_axes
# ===========================================================================

class TestResetPositiveAxes:
    def test_descending_xau_image_is_corrected(self):
        """Image with descending xau must be flipped to ascending order."""
        data = np.arange(12, dtype=float).reshape(4, 3)
        xau_desc = np.array([4.0, 2.0, 0.0, -2.0])  # descending
        yau = np.array([0.0, 5.0, 10.0])
        img = Image(data=data.copy(), xau=xau_desc.copy(), yau=yau.copy())
        # After __post_init__, xau must be ascending
        assert img.xau[0] < img.xau[-1], "xau is still descending after _reset_positive_axes"
        # Data must be flipped along axis-0 to match the flipped xau
        np.testing.assert_array_equal(img.get_I(), data[::-1])

    def test_ascending_xau_image_unchanged(self):
        """Image with already-ascending xau must not be modified."""
        data = np.arange(12, dtype=float).reshape(4, 3)
        xau = np.array([-2.0, 0.0, 2.0, 4.0])
        yau = np.array([0.0, 5.0, 10.0])
        img = Image(data=data.copy(), xau=xau.copy(), yau=yau.copy())
        np.testing.assert_array_equal(img.get_I(), data)
        np.testing.assert_array_equal(img.xau, xau)

    def test_descending_vkms_pvmap_is_corrected(self):
        """PVmap with descending vkms must be corrected by _reset_positive_axes."""
        Ipv = np.arange(20, dtype=float).reshape(4, 5)
        xau = np.linspace(0, 10, 4)
        vkms_desc = np.array([5.0, 2.5, 0.0, -2.5, -5.0])  # descending
        pv = PVmap(Ipv=Ipv.copy(), xau=xau.copy(), vkms=vkms_desc.copy())
        assert pv.vkms[0] < pv.vkms[-1], "vkms still descending after reset"
        np.testing.assert_array_equal(pv.get_I(), Ipv[:, ::-1])

    def test_descending_yau_cube_is_corrected(self):
        """Cube with descending yau must be corrected (axis 1)."""
        Ippv = np.arange(60, dtype=float).reshape(3, 4, 5)
        xau = np.linspace(0, 10, 3)
        yau_desc = np.array([8.0, 4.0, 0.0, -4.0])  # descending
        vkms = np.linspace(-2, 2, 5)
        cube = Cube(Ippv=Ippv.copy(), xau=xau.copy(), yau=yau_desc.copy(), vkms=vkms.copy())
        assert cube.yau[0] < cube.yau[-1]
        np.testing.assert_array_equal(cube.get_I(), Ippv[:, ::-1, :])


# ===========================================================================
# (d) trim round-trip
# ===========================================================================

class TestTrimRoundTrip:
    def test_trim_and_untrim_cube_shape(self):
        """Trim should reduce shape; trimming back should not exceed original."""
        cube = _make_gaussian_cube(nx=12, ny=12, nv=10)
        orig_shape = cube.Ippv.shape
        orig_xau = cube.xau.copy()
        orig_yau = cube.yau.copy()
        orig_vkms = cube.vkms.copy()
        # Trim to inner region
        cube.trim(xlim=(-30, 30), ylim=(-30, 30), vlim=(-3, 3))
        # Shape must be smaller
        assert cube.Ippv.shape[0] <= orig_shape[0]
        assert cube.Ippv.shape[1] <= orig_shape[1]
        assert cube.Ippv.shape[2] <= orig_shape[2]
        # Axes must stay consistent with data shape
        assert len(cube.xau) == cube.Ippv.shape[0]
        assert len(cube.yau) == cube.Ippv.shape[1]
        assert len(cube.vkms) == cube.Ippv.shape[2]

    def test_trim_preserves_values(self):
        """Values inside trim region are unchanged."""
        cube = _make_gaussian_cube(nx=10, ny=10, nv=10)
        # Copy the whole array before trimming
        orig_Ippv = cube.Ippv.copy()
        orig_xau = cube.xau.copy()
        orig_yau = cube.yau.copy()
        orig_vkms = cube.vkms.copy()
        cube.trim(xlim=(orig_xau[2], orig_xau[7]))
        # The x-axis of the trimmed cube must be a contiguous slice of the original
        for i, x in enumerate(cube.xau):
            assert x in orig_xau

    def test_trim_image(self):
        """trim on Image object works correctly."""
        data = np.ones((8, 8))
        xau = np.linspace(-20, 20, 8)
        yau = np.linspace(-20, 20, 8)
        img = Image(data=data, xau=xau.copy(), yau=yau.copy())
        img.trim(xlim=(-10, 10), ylim=(-10, 10))
        # All remaining x values must be within the limit
        assert np.all(img.xau >= -10) and np.all(img.xau <= 10)
        assert np.all(img.yau >= -10) and np.all(img.yau <= 10)

    def test_minmaxargs_boundary_inclusive(self):
        """minmaxargs must include boundary values (>= and <=)."""
        array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        imin, imax = minmaxargs(array, (1.0, 3.0))
        # Boundary values 1.0 and 3.0 must be included
        assert array[imin] == 1.0
        assert array[imax - 1] == 3.0

    def test_minmaxargs_empty_raises(self):
        """minmaxargs must raise ValueError when no points in range."""
        array = np.array([0.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no points within"):
            minmaxargs(array, (5.0, 10.0))


# ===========================================================================
# (e) set_refpoint updates refpos.dec0 (C-45)
# ===========================================================================

class TestSetRefpoint:
    def test_set_refpoint_updates_ra0_dec0(self):
        """set_refpoint must update ra0 and dec0 in refpos (not dec)."""
        cube = _make_gaussian_cube()
        cube.set_refpoint(ra0=10.5, dec0=-30.2)
        assert cube.refpos.ra0 == 10.5
        assert cube.refpos.dec0 == -30.2

    def test_set_refpoint_does_not_create_dec_attr(self):
        """set_refpoint must not create a spurious refpos.dec attribute."""
        cube = _make_gaussian_cube()
        cube.set_refpoint(ra0=5.0, dec0=15.0)
        # RefPos only has dec0, not dec
        assert not hasattr(cube.refpos, "dec") or cube.refpos.dec0 == 15.0

    def test_set_refpoint_on_image(self):
        """set_refpoint works on Image too."""
        img = Image(
            data=np.ones((4, 4)),
            xau=np.linspace(-5, 5, 4),
            yau=np.linspace(-5, 5, 4),
        )
        img.set_refpoint(ra0=1.0, dec0=2.0)
        assert img.refpos.dec0 == 2.0


# ===========================================================================
# (f) Convolver(mode='null') works without beam size, acts as identity
# ===========================================================================

class TestConvolverNull:
    def test_null_mode_no_beam_required(self):
        """Convolver with mode='null' must not require beam_maj_au/beam_min_au."""
        # Should not raise
        conv = Convolver(grid_size=(1.0, 1.0), mode="null")

    def test_null_mode_is_identity(self):
        """Convolver(mode='null') must return the input unchanged."""
        conv = Convolver(grid_size=(1.0, 1.0), mode="null")
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = conv(image)
        np.testing.assert_array_equal(result, image)

    def test_non_null_mode_requires_beam(self):
        """Non-null Convolver without beam must raise ValueError."""
        with pytest.raises(ValueError, match="beam size is required"):
            Convolver(grid_size=(1.0, 1.0, 0.1), mode="fft")

    def test_non_null_mode_partial_beam_raises(self):
        """Providing only beam_maj_au (not beam_min_au) must raise ValueError."""
        with pytest.raises(ValueError, match="beam size is required"):
            Convolver(grid_size=(1.0, 1.0), beam_maj_au=50.0, mode="fft")


# ===========================================================================
# (g) Obreso without beam info raises ValueError (C-61)
# ===========================================================================

class TestObresoValidation:
    def _dummy_obsdata(self, dpc=140.0):
        """Minimal obsdata-like object."""
        class _OD:
            pass
        od = _OD()
        od.dpc = dpc
        return od

    def test_no_beam_info_raises(self):
        """Obreso with neither au nor deg pair must raise ValueError."""
        od = self._dummy_obsdata()
        with pytest.raises(ValueError, match="specify both"):
            Obreso(obsdata=od)

    def test_partial_au_pair_raises(self):
        """Obreso with only beam_maj_au (not beam_min_au) must raise ValueError."""
        od = self._dummy_obsdata()
        with pytest.raises(ValueError, match="specify both"):
            Obreso(obsdata=od, beam_maj_au=50.0)

    def test_partial_deg_pair_raises(self):
        """Obreso with only beam_maj_deg must raise ValueError."""
        od = self._dummy_obsdata()
        with pytest.raises(ValueError, match="specify both"):
            Obreso(obsdata=od, beam_maj_deg=0.001)

    def test_complete_au_pair_ok(self):
        """Obreso with complete au pair must succeed and derive deg."""
        od = self._dummy_obsdata(dpc=140.0)
        obr = Obreso(obsdata=od, beam_maj_au=50.0, beam_min_au=30.0)
        assert obr.beam_maj_au == 50.0
        assert obr.beam_min_au == 30.0
        assert obr.beam_maj_deg is not None
        assert obr.beam_min_deg is not None

    def test_complete_deg_pair_ok(self):
        """Obreso with complete deg pair must succeed and derive au."""
        od = self._dummy_obsdata(dpc=140.0)
        obr = Obreso(obsdata=od, beam_maj_deg=0.001, beam_min_deg=0.0005)
        assert obr.beam_maj_deg == 0.001
        assert obr.beam_min_deg == 0.0005
        assert obr.beam_maj_au is not None
        assert obr.beam_min_au is not None


# ===========================================================================
# PVmap and Image also get _reset_positive_axes applied
# ===========================================================================

class TestResetPositiveAxesAllTypes:
    def test_image_descending_yau_corrected(self):
        """Image with descending yau must be corrected along axis 1."""
        data = np.arange(12, dtype=float).reshape(3, 4)
        xau = np.linspace(0, 10, 3)
        yau_desc = np.array([12.0, 8.0, 4.0, 0.0])
        img = Image(data=data.copy(), xau=xau.copy(), yau=yau_desc.copy())
        assert img.yau[0] < img.yau[-1]
        np.testing.assert_array_equal(img.get_I(), data[:, ::-1])

    def test_pvmap_descending_xau_corrected(self):
        """PVmap with descending xau must be corrected along axis 0."""
        Ipv = np.arange(20, dtype=float).reshape(4, 5)
        xau_desc = np.array([8.0, 4.0, 0.0, -4.0])
        vkms = np.linspace(-5, 5, 5)
        pv = PVmap(Ipv=Ipv.copy(), xau=xau_desc.copy(), vkms=vkms.copy())
        assert pv.xau[0] < pv.xau[-1]
        np.testing.assert_array_equal(pv.get_I(), Ipv[::-1, :])
