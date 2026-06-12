"""
Tests for P2-D: the rewritten FITS I/O (envos/obs/fits_io.py).

Covers:
  * round-trip save_fits -> read_*_fits for Cube / Image / PVmap, checking
    axes, data, beam and refpos (including a non-zero ra0/dec0/freq0 case);
  * CASA-style robustness: NAXIS=4 with a Stokes axis, BUNIT='Jy/beam',
    deg axes and a (descending) FREQ axis, read with read_cube_fits;
  * descending-frequency input is re-ordered to ascending vkms;
  * read_obsdata header dispatch;
  * the B-19 .. B-22 symptoms no longer occur.

Synthetic data only -- no radmc3d required.
"""

import os
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits as afits

from envos.obs import (
    Cube,
    Image,
    PVmap,
    save_fits,
    read_cube_fits,
    read_image_fits,
    read_pv_fits,
    read_obsdata,
    au2deg,
)

DPC = 140.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CASA_CUBE = os.path.join(DATA_DIR, "casa_cube.fits")


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

def _axes(nx=8, ny=7, nv=9):
    xau = np.linspace(-40.0, 40.0, nx)
    yau = np.linspace(-35.0, 35.0, ny)
    vkms = np.linspace(-4.0, 4.0, nv)
    return xau, yau, vkms


def _gaussian_cube(nx=8, ny=7, nv=9, freq0=230e9, ra0=83.81, dec0=-5.37, beam=True):
    xau, yau, vkms = _axes(nx, ny, nv)
    X, Y, V = np.meshgrid(xau, yau, vkms, indexing="ij")
    Ippv = np.exp(-X**2 / 200 - Y**2 / 200 - V**2 / 8) + 0.1
    cube = Cube(Ippv=Ippv, xau=xau, yau=yau, vkms=vkms, dpc=DPC)
    cube.refpos.ra0 = ra0
    cube.refpos.dec0 = dec0
    cube.refpos.freq0 = freq0
    if beam:
        cube.set_obs_resolution(
            beam_maj_au=20.0, beam_min_au=10.0, beam_pa_deg=30.0
        )
    return cube


# ===========================================================================
# Round-trip: Cube
# ===========================================================================

class TestRoundTripCube:
    def test_cube_roundtrip(self, tmp_path):
        cube = _gaussian_cube()
        fp = tmp_path / "cube.fits"
        save_fits(cube, fp)
        out = read_cube_fits(fp, dpc=DPC)

        np.testing.assert_allclose(out.xau, cube.xau, atol=1e-6)
        np.testing.assert_allclose(out.yau, cube.yau, atol=1e-6)
        np.testing.assert_allclose(out.vkms, cube.vkms, atol=1e-9)
        np.testing.assert_allclose(out.Ippv, cube.Ippv, rtol=1e-6, atol=1e-7)
        # beam
        np.testing.assert_allclose(out.obreso.beam_maj_au, 20.0, rtol=1e-6)
        np.testing.assert_allclose(out.obreso.beam_min_au, 10.0, rtol=1e-6)
        np.testing.assert_allclose(out.obreso.beam_pa_deg, 30.0)
        # refpos (non-zero ra0/dec0/freq0)
        np.testing.assert_allclose(out.refpos.ra0, 83.81)
        np.testing.assert_allclose(out.refpos.dec0, -5.37)
        np.testing.assert_allclose(out.refpos.freq0, 230e9)

    def test_cube_roundtrip_no_beam(self, tmp_path):
        cube = _gaussian_cube(beam=False)
        fp = tmp_path / "cube_nb.fits"
        save_fits(cube, fp)
        out = read_cube_fits(fp, dpc=DPC)
        np.testing.assert_allclose(out.Ippv, cube.Ippv, rtol=1e-6, atol=1e-7)
        assert out.obreso is None


# ===========================================================================
# Round-trip: Image
# ===========================================================================

class TestRoundTripImage:
    def test_image_roundtrip(self, tmp_path):
        xau, yau, _ = _axes()
        X, Y = np.meshgrid(xau, yau, indexing="ij")
        data = np.exp(-X**2 / 200 - Y**2 / 200) + 0.1
        img = Image(data, xau=xau, yau=yau, dpc=DPC)
        img.refpos.ra0 = 83.81
        img.refpos.dec0 = -5.37
        img.refpos.freq0 = 345e9

        fp = tmp_path / "img.fits"
        save_fits(img, fp)
        out = read_image_fits(fp, dpc=DPC)

        np.testing.assert_allclose(out.xau, img.xau, atol=1e-6)
        np.testing.assert_allclose(out.yau, img.yau, atol=1e-6)
        np.testing.assert_allclose(out.data, img.data, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out.refpos.ra0, 83.81)
        np.testing.assert_allclose(out.refpos.dec0, -5.37)
        np.testing.assert_allclose(out.refpos.freq0, 345e9)


# ===========================================================================
# Round-trip: PVmap
# ===========================================================================

class TestRoundTripPV:
    def test_pv_roundtrip(self, tmp_path):
        xau, _, vkms = _axes()
        X, V = np.meshgrid(xau, vkms, indexing="ij")
        Ipv = np.exp(-X**2 / 200 - V**2 / 8) + 0.1
        pv = PVmap(Ipv, xau=xau, vkms=vkms, dpc=DPC)
        pv.refpos.freq0 = 230e9
        pv.set_obs_resolution(beam_maj_au=20.0, beam_min_au=10.0, beam_pa_deg=30.0)

        fp = tmp_path / "pv.fits"
        save_fits(pv, fp)
        out = read_pv_fits(fp, dpc=DPC)

        np.testing.assert_allclose(out.xau, pv.xau, atol=1e-6)
        np.testing.assert_allclose(out.vkms, pv.vkms, atol=1e-9)
        np.testing.assert_allclose(out.Ipv, pv.Ipv, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out.refpos.freq0, 230e9)
        np.testing.assert_allclose(out.obreso.beam_maj_au, 20.0, rtol=1e-6)

    def test_pv_fits_ctype_is_offset(self, tmp_path):
        """PVmap first axis must be written as CTYPE1='OFFSET' (CASA style)."""
        xau, _, vkms = _axes()
        Ipv = np.ones((len(xau), len(vkms)))
        pv = PVmap(Ipv, xau=xau, vkms=vkms, dpc=DPC)
        fp = tmp_path / "pv_ctype.fits"
        save_fits(pv, fp)
        with afits.open(fp) as hdul:
            assert hdul[0].header["CTYPE1"] == "OFFSET"
            assert hdul[0].header["CTYPE2"] == "VRAD"


# ===========================================================================
# CASA-style robustness (NAXIS=4, Stokes, Jy/beam, deg, RESTFRQ)
# ===========================================================================

class TestCasaCube:
    def test_casa_cube_reads(self):
        cube = read_cube_fits(CASA_CUBE, dpc=DPC)
        # Stokes (length-1) axis squeezed out -> 3-D cube.
        assert cube.Ippv.ndim == 3
        assert cube.Ippv.shape == (6, 5, 7)

    def test_casa_jy_per_beam_converted(self):
        cube = read_cube_fits(CASA_CUBE, dpc=DPC)
        assert cube.Iunit == u.Jy / u.pix

        # Reconstruct the expected Jy/pixel data from the raw file.
        with afits.open(CASA_CUBE) as hdul:
            h = hdul[0].header
            raw = np.asarray(hdul[0].data).T  # -> (nx, ny, stokes, freq)
        factor = np.abs(h["CDELT1"] * h["CDELT2"]) / (
            np.pi * h["BMAJ"] * h["BMIN"] / (4 * np.log(2))
        )
        expected = raw[:, :, 0, :] * factor  # drop Stokes, apply factor
        # With CRVAL4 at the highest frequency and CDELT4 < 0, the channels are
        # already ordered from highest freq (lowest vkms) to lowest freq
        # (highest vkms), i.e. ascending vkms, so no spectral flip is applied.
        np.testing.assert_allclose(cube.Ippv, expected, rtol=1e-6)

    def test_casa_vkms_ascending_and_correct(self):
        cube = read_cube_fits(CASA_CUBE, dpc=DPC)
        # vkms must be monotonically ascending even though FREQ is descending.
        assert np.all(np.diff(cube.vkms) > 0)
        # Reference channel (CRPIX4=1, CRVAL4 = restfrq + 3*dnu) maps to a known
        # velocity; check the channel spacing magnitude via radio convention.
        with afits.open(CASA_CUBE) as hdul:
            h = hdul[0].header
        import envos.nconst as nc
        dnu = abs(h["CDELT4"])
        dv_expected = nc.c / 1e5 * dnu / h["RESTFRQ"]
        np.testing.assert_allclose(np.diff(cube.vkms), dv_expected, rtol=1e-9)

    def test_casa_beam_populated(self):
        cube = read_cube_fits(CASA_CUBE, dpc=DPC)
        assert cube.obreso is not None
        np.testing.assert_allclose(cube.obreso.beam_maj_au, 30.0, rtol=1e-6)
        np.testing.assert_allclose(cube.obreso.beam_min_au, 20.0, rtol=1e-6)
        np.testing.assert_allclose(cube.obreso.beam_pa_deg, 45.0)

    def test_jy_per_beam_without_bmaj_raises(self, tmp_path):
        """Jy/beam without BMAJ must raise (cannot convert to Jy/pixel)."""
        hdu = afits.PrimaryHDU(np.ones((4, 4, 4), dtype=np.float32))
        h = hdu.header
        h["BUNIT"] = "Jy/beam"
        dx = au2deg(10.0, DPC)
        h["CTYPE1"] = "RA---SIN"; h["CRVAL1"] = 0.0; h["CRPIX1"] = 2; h["CDELT1"] = -dx; h["CUNIT1"] = "deg"
        h["CTYPE2"] = "DEC--SIN"; h["CRVAL2"] = 0.0; h["CRPIX2"] = 2; h["CDELT2"] = dx; h["CUNIT2"] = "deg"
        h["CTYPE3"] = "VRAD"; h["CRVAL3"] = 0.0; h["CRPIX3"] = 1; h["CDELT3"] = 500.0; h["CUNIT3"] = "m s-1"
        h["RESTFRQ"] = 230e9
        fp = tmp_path / "nobmaj.fits"
        hdu.writeto(fp)
        with pytest.raises(ValueError, match="BMAJ"):
            read_cube_fits(fp, dpc=DPC)


# ===========================================================================
# Descending-frequency / descending-axis handling
# ===========================================================================

class TestDescendingAxes:
    def test_descending_freq_synthetic_cube(self, tmp_path):
        """A cube saved then re-saved with descending freq reads back with
        ascending vkms and consistent data (axis/data stay aligned)."""
        cube = _gaussian_cube(beam=False)
        fp = tmp_path / "cube.fits"
        save_fits(cube, fp)
        # Rewrite the header with a descending spectral axis (flip CDELT3 and
        # CRVAL3 so it describes the same channels in reverse), flipping data.
        with afits.open(fp) as hdul:
            h = hdul[0].header.copy()
            data = np.asarray(hdul[0].data)
        nv = h["NAXIS3"]
        v_first = h["CRVAL3"] + (nv - 1) * h["CDELT3"]
        h["CRVAL3"] = v_first
        h["CDELT3"] = -h["CDELT3"]
        h["CRPIX3"] = 1.0
        data = data[::-1, :, :]  # FITS spectral axis is the slowest (axis 0)
        fp2 = tmp_path / "cube_desc.fits"
        afits.PrimaryHDU(data, h).writeto(fp2)

        out = read_cube_fits(fp2, dpc=DPC)
        assert np.all(np.diff(out.vkms) > 0)
        np.testing.assert_allclose(out.vkms, cube.vkms, atol=1e-9)
        np.testing.assert_allclose(out.Ippv, cube.Ippv, rtol=1e-6, atol=1e-7)


# ===========================================================================
# read_obsdata dispatch (B-20 regression)
# ===========================================================================

class TestReadObsdataDispatch:
    def test_dispatch_cube(self):
        obj = read_obsdata(CASA_CUBE)
        assert type(obj).__name__ == "Cube"

    def test_dispatch_image(self, tmp_path):
        xau, yau, _ = _axes()
        data = np.ones((len(xau), len(yau)))
        img = Image(data, xau=xau, yau=yau, dpc=DPC)
        fp = tmp_path / "img.fits"
        save_fits(img, fp)
        assert type(read_obsdata(fp)).__name__ == "Image"

    def test_dispatch_pv(self, tmp_path):
        xau, _, vkms = _axes()
        Ipv = np.ones((len(xau), len(vkms)))
        pv = PVmap(Ipv, xau=xau, vkms=vkms, dpc=DPC)
        pv.refpos.freq0 = 230e9
        fp = tmp_path / "pv.fits"
        save_fits(pv, fp)
        assert type(read_obsdata(fp)).__name__ == "PVmap"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="cannot infer file type"):
            read_obsdata("/tmp/nope.dat")


# ===========================================================================
# B-19 .. B-22 regression coverage
# ===========================================================================

class TestIssueRegressions:
    def test_b19_save_writes_to_given_path(self, tmp_path):
        """B-19: save_fits must write to the explicit filepath (no implicit
        run-dir / CWD path) and must not crash on beam-bearing data."""
        cube = _gaussian_cube()  # has a beam
        target = tmp_path / "sub" / "dir" / "out.fits"
        save_fits(cube, target)
        assert target.is_file()

    def test_b19_save_via_base_save_mode_fits(self, tmp_path):
        """B-19: BaseObsData.save(mode='fits', filepath=...) is wired up."""
        cube = _gaussian_cube(beam=False)
        target = tmp_path / "viabase.fits"
        cube.save(mode="fits", filepath=str(target))
        assert target.is_file()
        out = read_cube_fits(target, dpc=DPC)
        np.testing.assert_allclose(out.Ippv, cube.Ippv, rtol=1e-6, atol=1e-7)

    def test_b21_axes_are_one_dimensional(self, tmp_path):
        """B-21: resolved axes must be 1-D arrays (legacy code produced 2-D)."""
        cube = _gaussian_cube(beam=False)
        fp = tmp_path / "c.fits"
        save_fits(cube, fp)
        out = read_cube_fits(fp, dpc=DPC)
        assert out.xau.ndim == 1
        assert out.yau.ndim == 1
        assert out.vkms.ndim == 1

    def test_b22_frequency_axis_readable(self):
        """B-22: a FREQ spectral axis must be convertible to vkms (the legacy
        comparison `unit.lower() in ['Hz']` never matched)."""
        cube = read_cube_fits(CASA_CUBE, dpc=DPC)
        assert cube.vkms.ndim == 1
        assert np.all(np.isfinite(cube.vkms))

    def test_get_pv_map_save_requires_path(self):
        """get_pv_map(save=True) without savefile stays NotImplementedError
        (no implicit path); with savefile it writes a FITS file."""
        cube = _gaussian_cube(beam=False)
        with pytest.raises(NotImplementedError):
            cube.get_pv_map(save=True)

    def test_get_pv_map_save_with_path(self, tmp_path):
        cube = _gaussian_cube(beam=False)
        target = tmp_path / "pv_out.fits"
        pv = cube.get_pv_map(pangle_deg=0, save=True, savefile=str(target))
        assert target.is_file()
        assert type(pv).__name__ == "PVmap"


def test_no_debug_print_in_fits_io():
    """The legacy debug `print('do fits')` and friends are gone."""
    import envos.obs.fits_io as mod
    src = open(mod.__file__).read()
    assert "do fits" not in src
