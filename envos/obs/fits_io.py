"""
envos/obs/fits_io.py — save_fits, read_obsdata, read_*_fits, read_fits,
                        and coordinate conversion functions.

Moved from envos/obs.py (P2-B split, move-only, no behaviour changes).
P2-D will rewrite this module; for now the legacy implementation is preserved
exactly as it was.
"""
import numpy as np
from pathlib import Path

import astropy
import astropy.io.fits as afits
from astropy import units as u

from .. import nconst as nc
from .. import tools
from ..log import logger


#########################################################
# Save & Read Functions
#########################################################


def save_fits(od, filepath):
    """
    save the obsdata as a fitsfile
    see IAU manual for variables used in fits:
         https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    hdu = afits.PrimaryHDU(od.get_I().T)
    ## I may need to change ppv into vyx , ...Ivyx?
    dtype = type(od).__name__
    hd = {
        "DTYPE": dtype,
        "DPC": od.dpc,
        "BTYPE": "Intensity",
        "BUNIT": "Jy/pixel",
    }

    def axis_dict(Nax, nax, dax, axname):
        if (axname == "x") or (axname == "y"):
            dic = {
                f"CTYPE{nax}": "ANGLE",
                f"NAXIS{nax}": Nax,
                f"CRVAL{nax}": 0.0,
                f"CRPIX{nax}": (Nax + 1.0) / 2.0,
                f"CDELT{nax}": au2deg(
                    dax, od.dpc
                ),  # np.rad2deg( np.pi*dax/(648000* od.dpc) ),
                f"CUNIT{nax}": "deg",
            }
        elif axname == "v":
            dic = {
                f"CTYPE{nax}": "VRAD",
                f"NAXIS{nax}": od.Nv,
                f"CRVAL{nax}": 0.0,
                f"CRPIX{nax}": (od.Nv + 1) / 2,
                f"CDELT{nax}": od.dv * 1e3,
                f"CUNIT{nax}": "m/s",
            }
        return dic

    if dtype == "Cube":
        hd.update({"NAXIS": 3})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Ny, 2, od.dy, "y"))
        hd.update(axis_dict(od.Nv, 3, od.dv, "v"))
    elif dtype == "Image":
        hd.update({"NAXIS": 2})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Ny, 2, od.dy, "y"))
    elif dtype == "PVmap":
        hd.update({"NAXIS": 2})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Nv, 2, od.dv, "v"))

    if od.beam_maj_au:
        hd.update(
            {
                "BMAJ": au2deg(od.beam_maj_au, od.dpc),
                "BMIN": au2deg(od.beam_min_au, od.dpc),
                "BPA": od.beam_pa_deg,
                "VRESO": od.vreso_kms,
            }
        )

    infonames = {
        "iline": "iline",
        "molname": "molname",
        "incl": "incl",
        "phi": "phi",
        "posang": "posang",
        "freq0": "resfrq",
    }
    hd.update(
        (_fitsn, getattr(od, _odn))
        for _odn, _fitsn in infonames.items()
        if hasattr(od, _odn)
    )
    hdu.header.update(hd)
    hdulist = afits.HDUList([hdu])
    hdulist.writeto(str(filepath), overwrite=True)


# Readers to be deleted
def read_obsdata(path, mode=None):
    path = str(path)
    if (".pkl" in path) or (mode == "pickle"):
        return tools.read_pickle(path)

    elif (".jb" in path) or (mode == "joblib"):
        import joblib

        return joblib.load(path)

    elif (".fits" in path) or (mode == "fits"):
        raise NotImplementedError(
            "FITS reading via read_obsdata is not implemented; "
            "use read_cube_fits / read_image_fits / read_pv_fits instead"
        )

    else:
        raise ValueError(f"cannot infer file type: {path}")


def read_image_fits(
    filepath, unit1="au", unit2="au", dpc=1, unit1_cm=None, unit2_cm=None
):
    return read_fits(filepath, "image", dpc=dpc, unit1=unit1, unit2=unit2)


def read_pv_fits(
    filepath, unit1="au", unit2="kms", dpc=1, unit1_cm=None, unit2_cms=None, v0_kms=0
):
    return read_fits(filepath, "pv", dpc=dpc, unit1=unit1, unit2=unit2, v0_kms=v0_kms)


def read_cube_fits(
    # filepath, unit1=None, unit2=None, unit3=None, dpc=1, unit1_cm=None, unit2_cm=None, unit3_cms=None, v0_kms=0
    filepath,
    unit1=None,
    unit2=None,
    unit3=None,
    dpc=1,
    v0_kms=0,
):
    # return read_fits(filepath, "cube", dpc=dpc)
    #    return read_fits(filepath, "cube", dpc=dpc)
    return read_fits(
        filepath, "cube", dpc=dpc, unit1=unit1, unit2=unit2, unit3=unit3, v0_kms=v0_kms
    )  # , unit1_fac=unit1_cm, unit2_fac=unit2_cm, unit3_fac=unit3_cms)


#!Constracting!

# from astropy.nddata import CCDData


def read_fits(
    filepath,
    fitstype,
    dpc,
    unit1=None,
    unit2=None,
    unit3=None,
    unit1_fac=None,
    unit2_fac=None,
    unit3_fac=None,
    Iunit=None,
    v0_kms=0,
    freq0=None,
):
    # Deferred import to avoid circular dependency (data imports fits_io via
    # BaseObsData.save, and fits_io needs data classes here).
    from .data import Cube, Image, PVmap

    logger.info(f"Reading fits file: {filepath}")
    hdul = afits.open(filepath)[0]
    header = hdul.header
    data = hdul.data.T  ## .T is due to data shape#
    #    print(data.shape)
    #    exit()
    #    print(header)
    naxis = header["NAXIS"]

    print("Warning can apper in reading fits (see wcs in astropy):")
    print("**** Warning start ****")
    wcs = astropy.wcs.WCS(header, naxis=naxis)
    print("**** Warning end ****")
    #    logger.debug("header:\n" + textwrap.fill(str(header), 80) + "\n")
    origin = 0
    centerpix = (
        wcs.wcs.crpix + origin - 1
    )  # - 1 # should be improved into the center pixcoord
    print(
        f"crpix in fits = {wcs.wcs.crpix}, origin = {origin}, crpix in code {centerpix}"
    )
    axref = wcs.all_pix2world([centerpix], origin, ra_dec_order=True)[0]
    print("Taking centerpix as ", centerpix, " gives reference position ", axref)
    print("This should be equal to CRVAL* in FITS")

    def input_header_value(usr_val, name_fits):
        if usr_val is not None:
            return usr_val
        elif name_fits in header:
            return header[name_fits].rstrip()
        else:
            return None

    Iunit = input_header_value(
        Iunit, "BUNIT"
    )  # Iunit if Iunit is not None else ( header["BUNIT"] if "BUNIT" in header else None)
    if Iunit.lower() == "jy/beam":
        data *= np.abs(header["CDELT1"] * header["CDELT2"]) / (
            np.pi * header["BMAJ"] * header["BMIN"] / (4 * np.log(2))
        )
        # data *= np.abs(header["CDELT1"] * header["CDELT2"])/( np.pi * header["BMAJ"] * header["BMIN"] )
        Iunit = u.Jy / u.pix
    unit1 = input_header_value(
        unit1, "CUNIT1"
    )  # unit1 if unit1 is not None else ( header["CUNIT1"].rstrip() if "CUNIT1" in header else None )
    unit2 = input_header_value(
        unit2, "CUNIT2"
    )  # unit2 if unit2 is not None else ( header["CUNIT2"].rstrip() if "CUNIT2" in header else None )
    unit3 = input_header_value(
        unit3, "CUNIT3"
    )  # unit3 if unit3 is not None else ( header["CUNIT3"].rstrip() if "CUNIT3" in header else None )

    def _get_ax(axnum, sub=True):
        n = header[f"NAXIS{axnum:d}"]
        pixcoord = np.tile(centerpix, (n, 1))
        pixcoord[:, axnum - 1] = np.arange(
            origin, n + origin
        )  # pixcoord in python starts from 0 but that in fits from 1 ??
        # ax = wcs.all_pix2world(pixcoord, origin, ra_dec_order=True)[:,axnum-1]
        ax = wcs.all_pix2world(pixcoord, origin, ra_dec_order=True)
        print(ax)
        # ax = wcs.pix2foc(pixcoord, origin)[:,axnum-1]
        if sub:
            axc = wcs.wcs_pix2world([centerpix], origin)[:, axnum - 1]
            return ax - axc
        else:
            return ax

    def _add_beam_info(obj, dpc):
        if "BMAJ" not in header:
            return
        if hasattr(obj, "vkms") and (obj.vkms is not None) and (len(obj.vkms) >= 2):
            vreso_kms = obj.vkms[1] - obj.vkms[0]
        else:
            vreso_kms = None
        obj.set_obs_resolution(
            beam_maj_deg=header["BMAJ"],
            beam_min_deg=header["BMIN"],
            vreso_kms=vreso_kms,
            beam_pa_deg=header["BPA"],
        )

    def _data_shape_convert(data, ndim, transpose=True):
        if len(np.shape(data)) != ndim:
            nshape = [i for i in np.shape(data) if i != 1]
            if len(nshape) == ndim:
                print(data.shape)
                ndata = np.reshape(data, nshape)
                print(data.shape)
            else:
                logger.error(f"Wait, something wrong in the data! nshape = {nshape}")
                raise Exception
        else:
            ndata = data
        return ndata

    def _lfac_to_cm(unit_name=None, unit_cm=None):
        if unit_cm is not None:
            logger.info(
                "   1st axis is interpreted as POSITION "
                f"with user-defined unit (unit = {unit_cm} cm)."
            )
            fac = unit_cm / nc.au
        elif unit_name.lower() in ["degree", "deg"]:
            fac = 3600 * dpc
        elif unit_name.lower() in ["au"]:
            fac = 1.0
        else:
            raise Exception("Unknown datatype in 1st axis")
        # print(fac)
        return fac

    def _vfac_to_kms(unit_name=None, unit_cms=None):
        if unit_cms:
            logger.info(
                "   **nd axis is interpreted as VELOCITY "
                f"with user-defined unit (unit = {unit_cms} cm/s)."
            )
            fac = unit_cms / 1e5

        elif unit_name.lower() in ["ms", "m/s"]:
            fac = 1e-3

        elif unit_name.lower() in ["kms", "km/s"]:
            fac = 1.0

        elif unit_name.lower() in ["Hz"]:  # when dnu is in Hz
            nu0 = header["RESTFRQ"]  # freq: max --> min
            fac = nc.c / 1e5 / nu0
            logger.info(
                f"Here nu0 is supposed to be {nu0/1e9} GHz. If it is wrong, please let S.M know, because it should be a bug."
            )

        else:
            raise Exception("Unknown datatype in 2nd axis")

        return fac

    if freq0 is None:
        if "RESTFREQ" in header:
            freq0 = header["RESTFRQ"]

    if fitstype.lower() == "image":
        ax1 = _get_ax(1, sub=1)  # * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2, sub=1)  # * _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        data = _data_shape_convert(data, 2)
        ax1 *= np.cos(
            axref[1] * np.pi / 180
        )  # convert "RA" to "the offset from RA0 along -RA"
        if unit1 == "deg" and unit2 == "deg":
            radec_deg = (axref[0], axref[1], ax1, ax2)
            # obj = Image(data, radec_deg=radec_deg, dpc=dpc, Iunit=Iunit, freq0=freq0)
            obj = Image(data, radecSIN_deg=radec_deg, dpc=dpc, Iunit=Iunit, freq0=freq0)

        elif unit1 == "au" and unit2 == "au":
            obj = Image(data, xau=ax1, yau=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)

    elif fitstype.lower() == "pv":
        ax1 = _get_ax(1, sub=0)  # * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2, sub=0) * _vfac_to_kms(
            unit_name=unit2, unit_cms=unit2_fac
        )  # * _get_velscale(unit_name=unit2, unit_cms=unit2_fac)
        ax2 -= v0_kms
        data = _data_shape_convert(data, 2)
        if unit1 == "deg":
            # radec_deg=(axref[0], axref[1], ax1, ax2)
            # print(radec_deg)
            obj = PVmap(
                data, xrad=np.deg2rad(ax1), vkms=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0
            )
        elif unit1 == "au":
            obj = PVmap(data, xau=ax1, vkms=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)

    elif fitstype.lower() == "cube":
        ax1 = _get_ax(1, sub=0)  # * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2, sub=0)  # * _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        ax3 = _get_ax(3, sub=0) * _vfac_to_kms(unit_name=unit3, unit_cms=unit3_fac)
        ax3 -= v0_kms
        data = _data_shape_convert(data, 3)
        if unit1 == "deg":
            radec_deg = (axref[0], axref[1], ax1, ax2)
            # obj = Cube(data, radec_deg=radec_deg, vkms=ax3, dpc=dpc, Iunit=Iunit, freq0=freq0)
            obj = Cube(
                data,
                radecSIN_deg=radec_deg,
                vkms=ax3,
                dpc=dpc,
                Iunit=Iunit,
                freq0=freq0,
            )
        elif unit1 == "au":
            obj = Cube(
                data, xau=ax1, yau=ax2, vkms=ax3, dpc=dpc, Iunit=Iunit, freq0=freq0
            )
    _add_beam_info(obj, dpc)
    return obj


#########################################################
# Conversion functions
#########################################################


def au2deg(length_au, dpc):
    return np.rad2deg(np.pi * length_au / (648000 * dpc))


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
    return 15 * (hour + minu / 60 + sec / (3600))


def dec_to_deg(deg, arcmin, arcsec):
    return deg + arcmin / 60 + arcsec / (3600)
