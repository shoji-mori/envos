"""
envos/obs/fits_io.py — FITS input/output for observation data (P2-D rewrite).

This module reads and writes :class:`~envos.obs.data.Cube`,
:class:`~envos.obs.data.Image` and :class:`~envos.obs.data.PVmap` objects to and
from FITS files using astropy only (no spectral-cube dependency; decision D4).

Data model (internal representation)
------------------------------------
The obsdata classes store *physical* offset coordinates relative to a reference
position ``refpos = (ra0, dec0, freq0)``:

* ``xau`` : offset along the right-ascension axis, in au, increasing toward
  **west** (i.e. ``xau = -(ra - ra0)`` converted to au, already containing the
  ``cos(dec0)`` correction of the SIN projection intermediate coordinate).
* ``yau`` : offset along the declination axis, in au (toward north positive).
* ``vkms`` : line-of-sight velocity in km/s, relative to ``refpos.freq0`` using
  the radio (VRAD) convention ``v = c (freq0 - freq) / freq0``.
* ``refpos`` : the absolute reference ``(ra0 [deg], dec0 [deg], freq0 [Hz])``.

FITS writing
------------
* ``CTYPE`` : ``RA---SIN`` / ``DEC--SIN`` for the sky axes, ``VRAD`` for the
  spectral axis.  For a :class:`PVmap` the first axis is the position offset and
  is written as ``CTYPE1 = "OFFSET"`` (CASA convention) with ``CUNIT1 = "deg"``.
* ``CRVAL`` : ``(ra0, dec0, freq0)``.
* ``CUNIT`` : ``deg`` / ``deg`` / ``m s-1``.
* ``BUNIT`` : taken from ``obsdata.Iunit`` (defaults to ``Jy/pixel``).
* Beam     : ``BMAJ`` / ``BMIN`` (deg) and ``BPA`` (deg), plus ``RESTFRQ``.

FITS reading
------------
Coordinates are resolved one axis at a time from *type-specific* sub-WCS
(``wcs.sub(...)``) rather than calling ``all_pix2world`` on every axis and then
selecting columns (the broken design of the legacy implementation).  The axis
interpretation depends on the fits type:

* **cube** : ``[longitude, latitude, spectral]`` sub-WCS.
* **image** : ``[longitude, latitude]`` sub-WCS.
* **pv**   : the first axis is a *linear* offset axis resolved with
  ``wcs.sub([1])`` + ``CUNIT1``; the second axis is the spectral sub-WCS.

The spectral axis is converted to ``vkms`` using ``RESTFRQ`` (``RESTFREQ`` is
also accepted).  ``BUNIT = "Jy/beam"`` data are converted to Jy/pixel on read
(requires ``BMAJ``).  Degenerate (length-1) axes such as a Stokes axis are
squeezed out.  If ``BMAJ`` / ``BMIN`` / ``BPA`` are present the resulting
object's beam resolution (``obreso``) is populated.
"""

import numpy as np
from pathlib import Path

import astropy.io.fits as afits
from astropy.wcs import WCS

from .. import nconst as nc
from .. import tools
from ..log import logger


#########################################################
# Coordinate conversion helpers
#########################################################


def au2deg(length_au, dpc):
    """Convert a transverse length [au] at distance ``dpc`` [pc] to degrees."""
    return np.rad2deg(np.pi * length_au / (648000 * dpc))


def deg2au(length_deg, dpc):
    """Convert an angle [deg] at distance ``dpc`` [pc] to a length [au]."""
    return length_deg * 3600 * dpc


def deg_to_ra(deg):
    hour = int(deg / 15)
    minu = int((deg / 15 - hour) * 60)
    seco = (deg / 15 - hour - minu / 60) * 3600
    return hour, minu, seco


def deg_to_dec(deg):
    _deg = int(deg)
    arcmin = int(60 * (deg - _deg))
    arcsec = 3600 * (deg - _deg - arcmin / 60)
    return _deg, arcmin, arcsec


def ra_to_deg(hour, minu, sec):
    return 15 * (hour + minu / 60 + sec / 3600)


def dec_to_deg(deg, arcmin, arcsec):
    return deg + arcmin / 60 + arcsec / 3600


#########################################################
# Saving
#########################################################


def save_fits(obsdata, filepath, overwrite=True):
    """Write ``obsdata`` (Cube / Image / PVmap) to ``filepath`` as a FITS file.

    The file is written to the *explicit* ``filepath`` only — there is no
    implicit gpath / run-directory fallback (decision D4 / §0 "no implicit
    paths").
    """
    dtype = type(obsdata).__name__
    filepath = Path(filepath)
    if filepath.parent != Path(""):
        filepath.parent.mkdir(parents=True, exist_ok=True)

    dpc = obsdata.dpc
    if dpc is None:
        raise ValueError(
            "save_fits requires obsdata.dpc to be set (needed to convert "
            "the au offsets to angular FITS axes)."
        )

    refpos = obsdata.refpos
    # FITS data axis order is reversed relative to numpy (last numpy axis is
    # the fastest-varying FITS axis), so transpose the intensity array.
    hdu = afits.PrimaryHDU(np.asarray(obsdata.get_I()).T)
    header = hdu.header

    header["DTYPE"] = dtype
    header["DPC"] = float(dpc)
    header["BTYPE"] = "Intensity"
    header["BUNIT"] = _iunit_to_bunit(obsdata.Iunit)

    def _sky_axis(nax, n, ax_au, ctype, crval, west_positive):
        # CDELT in deg.  For the RA axis the internal offset increases toward
        # west, while RA decreases toward west, hence the sign flip.  CRVAL is
        # the reference sky position (ra0/dec0) which corresponds to offset 0,
        # so CRPIX is placed (with sub-pixel precision) at the offset-zero
        # crossing of the linear axis.
        dax_au = ax_au[1] - ax_au[0]
        cdelt = au2deg(dax_au, dpc)
        if west_positive:
            cdelt = -cdelt
        # offset-zero pixel (0-based) along the au axis, then 1-based for FITS.
        pix_zero = -ax_au[0] / dax_au
        header[f"CTYPE{nax}"] = ctype
        header[f"CRVAL{nax}"] = float(crval)
        header[f"CRPIX{nax}"] = float(pix_zero + 1.0)
        header[f"CDELT{nax}"] = float(cdelt)
        header[f"CUNIT{nax}"] = "deg"

    def _offset_axis(nax, ax_au):
        # Linear position-offset axis (PVmap first axis), stored in deg.
        dax_au = ax_au[1] - ax_au[0]
        header[f"CTYPE{nax}"] = "OFFSET"
        header[f"CRVAL{nax}"] = float(au2deg(ax_au[0], dpc))
        header[f"CRPIX{nax}"] = 1.0
        header[f"CDELT{nax}"] = float(au2deg(dax_au, dpc))
        header[f"CUNIT{nax}"] = "deg"

    def _spectral_axis(nax, vkms):
        # VRAD axis: store reference frequency at CRVAL and velocity step
        # converted to a frequency step about freq0.
        dv_kms = vkms[1] - vkms[0]
        v0 = vkms[0]
        header[f"CTYPE{nax}"] = "VRAD"
        header[f"CRVAL{nax}"] = float(v0 * 1e3)  # m/s
        header[f"CRPIX{nax}"] = 1.0
        header[f"CDELT{nax}"] = float(dv_kms * 1e3)  # m/s
        header[f"CUNIT{nax}"] = "m s-1"

    if dtype == "Cube":
        _sky_axis(1, obsdata.Nx, obsdata.xau, "RA---SIN", refpos.ra0, west_positive=True)
        _sky_axis(2, obsdata.Ny, obsdata.yau, "DEC--SIN", refpos.dec0, west_positive=False)
        _spectral_axis(3, obsdata.vkms)
    elif dtype == "Image":
        _sky_axis(1, obsdata.Nx, obsdata.xau, "RA---SIN", refpos.ra0, west_positive=True)
        _sky_axis(2, obsdata.Ny, obsdata.yau, "DEC--SIN", refpos.dec0, west_positive=False)
    elif dtype == "PVmap":
        _offset_axis(1, obsdata.xau)
        _spectral_axis(2, obsdata.vkms)
    else:
        raise ValueError(f"save_fits does not support obsdata type {dtype!r}")

    if refpos.freq0:
        header["RESTFRQ"] = float(refpos.freq0)

    obreso = getattr(obsdata, "obreso", None)
    if obreso is not None and obreso.beam_maj_deg is not None:
        header["BMAJ"] = float(obreso.beam_maj_deg)
        header["BMIN"] = float(obreso.beam_min_deg)
        header["BPA"] = float(obreso.beam_pa_deg if obreso.beam_pa_deg is not None else 0.0)
        if obreso.vreso_kms is not None:
            header["VRESO"] = float(obreso.vreso_kms)

    hdulist = afits.HDUList([hdu])
    hdulist.writeto(str(filepath), overwrite=overwrite)
    logger.info(f"Saved fits file: {filepath}")


def _iunit_to_bunit(iunit):
    s = str(iunit)
    # astropy renders Jy/pix as "Jy / pix"; normalise to the conventional form.
    compact = s.replace(" ", "")
    if compact in ("Jy/pix", "Jy/pixel"):
        return "Jy/pixel"
    if compact in ("Jy/beam",):
        return "Jy/beam"
    if compact in ("K",):
        return "K"
    return s


#########################################################
# Reading
#########################################################


def read_obsdata(path, mode=None):
    """Dispatch a saved obsdata file to the appropriate reader.

    Handles pickle (.pkl), joblib (.jb) and FITS (.fits).  For FITS the obsdata
    type is inferred from the header (NAXIS + CTYPE) and the matching
    ``read_*_fits`` reader is called with defaults.
    """
    path = str(path)
    if (".pkl" in path) or (mode == "pickle"):
        return tools.read_pickle(path)

    elif (".jb" in path) or (mode == "joblib"):
        import joblib

        return joblib.load(path)

    elif (".fits" in path) or (mode == "fits"):
        fitstype = _infer_fitstype(path)
        if fitstype == "cube":
            return read_cube_fits(path, dpc=1)
        elif fitstype == "image":
            return read_image_fits(path, dpc=1)
        elif fitstype == "pv":
            return read_pv_fits(path, dpc=1)
        else:
            raise ValueError(f"cannot infer obsdata type from FITS header: {path}")

    else:
        raise ValueError(f"cannot infer file type: {path}")


def _infer_fitstype(filepath):
    """Infer 'cube' / 'image' / 'pv' from the FITS header (NAXIS + CTYPE)."""
    with afits.open(filepath) as hdul:
        header = hdul[0].header
    dtype = header.get("DTYPE", None)
    if dtype is not None:
        return {"Cube": "cube", "Image": "image", "PVmap": "pv"}.get(dtype, None)

    ctypes = [str(header.get(f"CTYPE{i}", "")).upper() for i in range(1, header["NAXIS"] + 1)]
    # number of non-degenerate axes
    n_real = sum(1 for i in range(1, header["NAXIS"] + 1) if header.get(f"NAXIS{i}", 1) > 1)
    has_spectral = any(_is_spectral_ctype(c) for c in ctypes)
    has_two_sky = sum(1 for c in ctypes if _is_lon_ctype(c) or _is_lat_ctype(c)) >= 2
    has_offset = any(_is_offset_ctype(c) for c in ctypes)

    if has_two_sky and has_spectral and n_real >= 3:
        return "cube"
    if has_two_sky and not has_spectral:
        return "image"
    if (has_offset or not has_two_sky) and has_spectral:
        return "pv"
    if has_two_sky:
        return "image"
    return None


def _is_spectral_ctype(ctype):
    c = ctype.upper()
    return c.startswith(("VRAD", "VOPT", "VELO", "FREQ", "WAVE", "AWAV", "FELO"))


def _is_lon_ctype(ctype):
    c = ctype.upper()
    return c.startswith(("RA", "GLON", "ELON", "HLON", "SLON"))


def _is_lat_ctype(ctype):
    c = ctype.upper()
    return c.startswith(("DEC", "GLAT", "ELAT", "HLAT", "SLAT"))


def _is_offset_ctype(ctype):
    c = ctype.upper()
    return c.startswith(("OFFSET", "ANGLE", "DIST", "POSITION"))


def read_cube_fits(
    filepath,
    dpc,
    unit1=None,
    unit2=None,
    unit3=None,
    v0_kms=0.0,
    Iunit=None,
    freq0=None,
):
    return _read_fits(
        filepath,
        "cube",
        dpc=dpc,
        units=(unit1, unit2, unit3),
        v0_kms=v0_kms,
        Iunit=Iunit,
        freq0=freq0,
    )


def read_image_fits(
    filepath,
    dpc,
    unit1=None,
    unit2=None,
    Iunit=None,
    freq0=None,
):
    return _read_fits(
        filepath,
        "image",
        dpc=dpc,
        units=(unit1, unit2),
        v0_kms=0.0,
        Iunit=Iunit,
        freq0=freq0,
    )


def read_pv_fits(
    filepath,
    dpc,
    unit1=None,
    unit2=None,
    v0_kms=0.0,
    Iunit=None,
    freq0=None,
):
    return _read_fits(
        filepath,
        "pv",
        dpc=dpc,
        units=(unit1, unit2),
        v0_kms=v0_kms,
        Iunit=Iunit,
        freq0=freq0,
    )


def _read_fits(filepath, fitstype, dpc, units, v0_kms=0.0, Iunit=None, freq0=None):
    # Deferred import to avoid a circular import (data.py -> _base.save ->
    # fits_io, and fits_io needs the data classes here).
    from .data import Cube, Image, PVmap

    logger.info(f"Reading fits file: {filepath}")
    with afits.open(filepath) as hdul:
        header = hdul[0].header.copy()
        data = np.asarray(hdul[0].data).T  # FITS axis order -> numpy axis order

    full_wcs = WCS(header)
    naxis = header["NAXIS"]

    def _header_or(user_val, name):
        if user_val is not None:
            return user_val
        if name in header:
            v = header[name]
            return v.rstrip() if isinstance(v, str) else v
        return None

    # ---- intensity unit (BUNIT) -----------------------------------------
    Iunit = _header_or(Iunit, "BUNIT")
    if isinstance(Iunit, str) and Iunit.lower().replace(" ", "") in ("jy/beam",):
        if "BMAJ" not in header:
            raise ValueError(
                "BUNIT='Jy/beam' requires BMAJ/BMIN in the header to convert "
                "to Jy/pixel."
            )
        cdelt1 = header["CDELT1"]
        cdelt2 = header["CDELT2"]
        beam_area = np.pi * header["BMAJ"] * header["BMIN"] / (4 * np.log(2))
        data = data * np.abs(cdelt1 * cdelt2) / beam_area
        from astropy import units as u

        Iunit = u.Jy / u.pix

    # ---- reference frequency --------------------------------------------
    if freq0 is None:
        if "RESTFRQ" in header:
            freq0 = header["RESTFRQ"]
        elif "RESTFREQ" in header:
            freq0 = header["RESTFREQ"]

    # ---- resolve axes via type-specific sub-WCS -------------------------
    # Note: freq0 is assigned to refpos *after* construction (rather than via
    # the freq0 InitVar) so that the reader does not depend on the InitVar
    # ordering of the data classes.
    ra0 = dec0 = None
    if fitstype == "cube":
        xau, yau, ra0, dec0 = _read_sky_axes(header, dpc, units[0], units[1])
        vkms = _read_spectral_axis(full_wcs, header, freq0) - v0_kms
        ip = _squeeze(data, 3)
        obj = Cube(ip, xau=xau, yau=yau, vkms=vkms, dpc=dpc, Iunit=Iunit)

    elif fitstype == "image":
        xau, yau, ra0, dec0 = _read_sky_axes(header, dpc, units[0], units[1])
        ip = _squeeze(data, 2)
        obj = Image(ip, xau=xau, yau=yau, dpc=dpc, Iunit=Iunit)

    elif fitstype == "pv":
        xau = _read_offset_axis(full_wcs, header, dpc, units[0])
        vkms = _read_spectral_axis(full_wcs, header, freq0) - v0_kms
        ip = _squeeze(data, 2)
        obj = PVmap(ip, xau=xau, vkms=vkms, dpc=dpc, Iunit=Iunit)

    else:
        raise ValueError(f"unknown fitstype {fitstype!r}")

    if ra0 is not None:
        obj.refpos.ra0 = ra0
        obj.refpos.dec0 = dec0
    if freq0 is not None:
        obj.refpos.freq0 = freq0

    _add_beam_info(obj, header)
    return obj


def _squeeze(data, ndim):
    """Drop degenerate (length-1) axes so that data has ``ndim`` dimensions."""
    if data.ndim == ndim:
        return data
    newshape = tuple(s for s in data.shape if s != 1)
    if len(newshape) != ndim:
        raise ValueError(
            f"cannot reduce data of shape {data.shape} to {ndim} dimensions"
        )
    return data.reshape(newshape)


def _axis_indices(header, predicate):
    """Return 1-based FITS axis numbers whose CTYPE matches ``predicate``."""
    return [
        i
        for i in range(1, header["NAXIS"] + 1)
        if predicate(str(header.get(f"CTYPE{i}", "")))
    ]


def _read_sky_axes(header, dpc, unit1, unit2):
    """Return (xau, yau, ra0, dec0) from the longitude/latitude axes.

    xau is the west-positive RA offset in au; yau is the dec offset in au.
    ra0/dec0 are the reference sky position (CRVAL) in deg.
    """
    lon_axes = _axis_indices(header, _is_lon_ctype)
    lat_axes = _axis_indices(header, _is_lat_ctype)
    if not lon_axes or not lat_axes:
        raise ValueError("could not find longitude/latitude axes in FITS header")
    lon_ax, lat_ax = lon_axes[0], lat_axes[0]

    ra0 = header[f"CRVAL{lon_ax}"]
    dec0 = header[f"CRVAL{lat_ax}"]

    nlon = header[f"NAXIS{lon_ax}"]
    nlat = header[f"NAXIS{lat_ax}"]

    # Work in the SIN projection-plane (intermediate world) coordinate, which
    # is linear in pixel index: xi = CDELT * (pix - (CRPIX - 1)).  This is the
    # tangent-plane angular offset and is exactly invertible (matching how the
    # offsets are stored on write).  The internal xau increases toward west, so
    # for RA (CDELT < 0) the sign flip recovers a west-positive offset.
    cdelt_lon = header[f"CDELT{lon_ax}"]
    cdelt_lat = header[f"CDELT{lat_ax}"]
    crpix_lon = header[f"CRPIX{lon_ax}"]
    crpix_lat = header[f"CRPIX{lat_ax}"]

    xi_lon = cdelt_lon * (np.arange(nlon) - (crpix_lon - 1))  # deg, RA direction
    xi_lat = cdelt_lat * (np.arange(nlat) - (crpix_lat - 1))  # deg, Dec direction

    xau = -deg2au(xi_lon, dpc)  # west positive (CDELT_RA is negative)
    yau = deg2au(xi_lat, dpc)
    return xau, yau, ra0, dec0


def _read_offset_axis(full_wcs, header, dpc, unit1):
    """Return the position-offset axis (au) of a PV map (linear first axis)."""
    n = header["NAXIS1"]
    sub = full_wcs.sub([1])
    pix = np.arange(n).reshape(-1, 1)
    world = sub.wcs_pix2world(pix, 0)[:, 0]
    unit1 = unit1 if unit1 is not None else str(header.get("CUNIT1", "deg")).strip()
    u = unit1.lower().replace(" ", "")
    if u in ("deg", "degree"):
        return deg2au(world, dpc)
    elif u in ("arcsec",):
        return deg2au(world / 3600.0, dpc)
    elif u in ("au",):
        return world
    else:
        raise ValueError(f"unknown unit for PV offset axis: {unit1!r}")


def _read_spectral_axis(full_wcs, header, freq0):
    """Return the spectral axis converted to km/s about ``freq0``."""
    spec_axes = _axis_indices(header, _is_spectral_ctype)
    if not spec_axes:
        raise ValueError("could not find a spectral axis in FITS header")
    spec_ax = spec_axes[0]
    ctype = str(header[f"CTYPE{spec_ax}"]).upper()
    n = header[f"NAXIS{spec_ax}"]

    sub = full_wcs.sub([spec_ax])
    pix = np.arange(n).reshape(-1, 1)
    world = sub.wcs_pix2world(pix, 0)[:, 0]
    cunit = str(header.get(f"CUNIT{spec_ax}", "")).strip().lower().replace(" ", "")

    if ctype.startswith(("VRAD", "VOPT", "VELO", "FELO")):
        if cunit in ("ms-1", "m/s", "ms", ""):
            return world * 1e-3  # m/s -> km/s
        elif cunit in ("kms-1", "km/s", "kms"):
            return world
        else:
            raise ValueError(f"unknown velocity unit: {cunit!r}")
    elif ctype.startswith("FREQ"):
        if freq0 is None:
            raise ValueError(
                "spectral axis is FREQ but no RESTFRQ/freq0 is available to "
                "convert to velocity."
            )
        # world is in Hz (CUNIT Hz assumed); radio convention.
        return nc.c / 1e5 * (freq0 - world) / freq0
    else:
        raise ValueError(f"unsupported spectral CTYPE: {ctype!r}")


def _add_beam_info(obj, header):
    if "BMAJ" not in header:
        return
    if hasattr(obj, "vkms") and (obj.vkms is not None) and (len(obj.vkms) >= 2):
        vreso_kms = obj.vkms[1] - obj.vkms[0]
    elif "VRESO" in header:
        vreso_kms = header["VRESO"]
    else:
        vreso_kms = None
    obj.set_obs_resolution(
        beam_maj_deg=header["BMAJ"],
        beam_min_deg=header["BMIN"],
        beam_pa_deg=header.get("BPA", 0.0),
        vreso_kms=vreso_kms,
    )
