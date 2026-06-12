"""
envos/obs/data.py — Obreso, RefPos, Cube, Image, PVmap, minmaxargs

Also re-exports BaseObsData from the private _base sub-module so that the
public name envos.obs.data.BaseObsData is available.

Moved from envos/obs.py (P2-B split, move-only, no behaviour changes).
"""
import dataclasses
import numpy as np
from scipy import integrate, interpolate

from astropy import units as u

from .. import nconst as nc
from .. import tools
from ..log import logger
from ._base import BaseObsData  # re-exported for public access


@dataclasses.dataclass
class Obreso:
    obsdata: dataclasses.InitVar[BaseObsData] = None
    beam_maj_deg: float = None
    beam_min_deg: float = None
    beam_pa_deg: float = None
    vreso_kms: float = None
    dpc: float = None
    beam_maj_au: float = None
    beam_min_au: float = None

    def __str__(self):
        return tools.dataclass_str(self)

    def __post_init__(self, obsdata):
        self.set_dpc_from_obsdata(obsdata)
        au_pair_complete = (self.beam_maj_au is not None) and (self.beam_min_au is not None)
        deg_pair_complete = (self.beam_maj_deg is not None) and (self.beam_min_deg is not None)
        if au_pair_complete:
            self.set_beamsize_deg()
        elif deg_pair_complete:
            self.set_beamsize_au()
        else:
            raise ValueError(
                "specify both (beam_maj_au, beam_min_au) or both "
                "(beam_maj_deg, beam_min_deg); "
                f"given: beam_maj_au={self.beam_maj_au}, "
                f"beam_min_au={self.beam_min_au}, "
                f"beam_maj_deg={self.beam_maj_deg}, "
                f"beam_min_deg={self.beam_min_deg}"
            )

    def set_beamsize_au(self):
        if self.dpc is None:
            raise Exception("dpc is not set.")
        self.beam_maj_au = np.deg2rad(self.beam_maj_deg) * self.dpc * nc.pc / nc.au
        self.beam_min_au = np.deg2rad(self.beam_min_deg) * self.dpc * nc.pc / nc.au

    def set_beamsize_deg(self):
        if self.dpc is None:
            raise Exception("dpc is not set.")
        self.beam_maj_deg = np.rad2deg(self.beam_maj_au * nc.au / (self.dpc * nc.pc))
        self.beam_min_deg = np.rad2deg(self.beam_min_au * nc.au / (self.dpc * nc.pc))

    def set_dpc_from_obsdata(self, obsdata):
        if (
            (obsdata is not None)
            and (hasattr(obsdata, "dpc"))
            and (obsdata.dpc is not None)
        ):
            self.dpc = obsdata.dpc
        else:
            logger.warning("Something wrong in obsdata: dpc could not be determined")


@dataclasses.dataclass
class RefPos:
    ra0: float = 0
    dec0: float = 0
    freq0: float = 0
    vkms0: float = 0

    def radec(self):
        return np.array([self.ra0, self.dec0])


@dataclasses.dataclass
class Cube(BaseObsData):
    """
    Usually, obsdata is made by doing obsevation.
    But one can generate obsdata by reading fitsfile or radmcdata

    Coordinate:
        A standard coordinate respect to the reference position (RA, Dec, freq0).
        One can generate RA--Dec coordinate by using Cube.ra, Cube.dec, or Cube.radec.
        The refrence position is contained as refpos.
        The origin of coordinate is set to be the same as the FITS file,
        but the origin can be moved arbitrary.
        # Actually I have considered to use, e.g., x in arcsecond as the basic coordinate.
        # But I have not been sure if it is clearer than xau, so I gave up thinking...
        # Also, historically, this class has had the two coordinate, xau and ra.
        # It was very confusing, produced many bugs, and destroyed my confidence...
        # Please tell me your thoughts...
        # For now, I am partially convinced because directly obtained coordinate
        # from synthetic observation to models is physical coordinate rather than
        # observational coordinate. So this choice of the basic coordinate is somewhat reasonable.
    """

    Ippv: np.ndarray
    xau: np.ndarray = None
    yau: np.ndarray = None
    vkms: np.ndarray = None
    dpc: float = None
    obreso: Obreso = None
    sobs_info: dict = None
    Iunit: str = u.Jy / u.pix  # r"Jy pixel$^{-1}$"
    refpos: RefPos = dataclasses.field(default_factory=RefPos)  # list[float, float, float] = [0,0,0]
    radec_deg: dataclasses.InitVar[tuple] = None
    radecSIN_deg: dataclasses.InitVar[tuple] = None
    freq0: dataclasses.InitVar[float] = None
    ##
    dtype = "Cube"
    _axorder = {"xau": 0, "yau": 1, "vkms": 2}
    _Iname = "Ippv"
    _axnames = ["xau", "yau", "vkms"]

    def __post_init__(self, radec_deg, radecSIN_deg, freq0):
        if radec_deg:
            self.set_coord_from_radec(*radec_deg)
        elif radecSIN_deg:
            self.set_coord_from_radecSIN(*radecSIN_deg)
        if freq0:
            self.refpos.freq0 = freq0
        # self._complement_coord()
        self._check_data_shape()
        self._reset_positive_axes()

    def get_mom0_map(self, normalize="peak", method="sum", vlim=None):
        if self.Ippv.shape[2] == 1:
            _Ipp = self.Ippv[..., 0]
        else:
            _Ippv = self.Ippv
            _vkms = self.vkms
            if vlim is not None:
                if len(vlim) != 2 or vlim[0] >= vlim[1]:
                    raise ValueError(
                        f"vlim must be a 2-element sequence with vlim[0] < vlim[1]; got {vlim}"
                    )
                cond = (vlim[0] < _vkms) & (_vkms < vlim[1])
                _Ippv = _Ippv[..., cond]
                _vkms = _vkms[cond]

            if method == "sum":
                _Ipp = np.sum(_Ippv, axis=-1) * (_vkms[1] - _vkms[0])
            elif method == "integrate":
                _Ipp = integrate.simpson(_Ippv, x=_vkms, axis=-1)
            else:
                raise ValueError(f"Unknown method: {method!r}")

        img = Image(_Ipp, xau=self.xau, yau=self.yau, dpc=self.dpc, Iunit=self.Iunit)
        img.copy_info_from_obsdata(self)
        if normalize == "peak":
            img.norm_I("max")

        return img

    def get_pv_map(
        self, length=None, pangle_deg=0, poffset_au=0, norm=None, save=False
    ):
        # norm: None, "max", float
        if self.Ippv.shape[1] > 1:
            #            pa = pangle_deg * nc.deg2rad
            #            L =  np.sqrt(self.Lx**2 + self.Ly**2)
            #            dl = ((np.sin(pa)/self.dx)**2 + (np.cos(pa)/self.dy)**2)**(-0.5)
            #            posax = np.arange(-L/2, L/2+dl, dl)
            #            posline = self.position_line(
            #                posax, PA_deg=pangle_deg, poffset_au=poffset_au
            #            )
            L = np.sqrt(self.Lx**2 + self.Ly**2)
            posline = tools.get_position_line(
                pangle_deg, L, self.dx, self.dy, poffset_au
            )
            points = [
                [(pl[0], pl[1], v) for v in self.vkms] for pl in posline["points"]
            ]
            posax = posline["posax"]
            Ipv = interpolate.interpn(
                (self.xau, self.yau, self.vkms),
                self.Ippv,
                points,
                bounds_error=False,
                method="linear",
                fill_value=0,
            )
        else:
            posax = self.xau  # ???
            Ipv = self.Ippv[:, 0, :]

        pv = PVmap(
            Ipv,
            xau=posax,
            vkms=self.vkms,
            dpc=self.dpc,
            Iunit=self.Iunit,
            pangle_deg=pangle_deg,
            poffset_au=poffset_au,
        )
        pv.copy_info_from_obsdata(self)
        if norm is not None:
            pv.norm_I(norm)
        if save:
            raise NotImplementedError(
                "get_pv_map(save=True) is not yet implemented; "
                "will be wired to save_fits in P2-D"
            )
        # self.pv_list.append(pv)
        return pv


@dataclasses.dataclass
class Image(BaseObsData):
    data: np.ndarray
    xau: np.ndarray = None
    yau: np.ndarray = None
    refpos: RefPos = dataclasses.field(default_factory=RefPos)
    dpc: float = None
    obreso: Obreso = None
    sobs_info: dict = None
    Iunit: str = u.Jy / u.pix
    radec_deg: dataclasses.InitVar[tuple] = None
    radecSIN_deg: dataclasses.InitVar[tuple] = None
    freq0: dataclasses.InitVar[float] = None
    ##
    dtype = "Image"
    _axorder = {"xau": 0, "yau": 1}
    _Iname = "data"
    _axnames = ["xau", "yau"]

    def __post_init__(self, radec_deg, radecSIN_deg, freq0):
        if radec_deg:
            self.set_coord_from_radec(*radec_deg)
        elif radecSIN_deg:
            self.set_coord_from_radecSIN(*radecSIN_deg)

        if freq0:
            self.refpos.freq0 = freq0

        self._check_data_shape()
        self._reset_positive_axes()

    #    def offset_center_to_maximum(self):
    #        xc, yc = self.get_peak_position(interp=False)
    #        self.move_center_pos(xy_au=(xc,yc))

    def convert_perbeam_to_perpixel(self):  # this could go base
        bmaj = self.obreso.beam_maj_au  # = full width half maximum along major axis
        bmin = self.obreso.beam_min_au
        A_beam = np.pi * bmaj * bmin / (4 * np.log(2))
        A_pix = self.dx * self.dy
        self.data *= A_pix / A_beam
        self.Iunit = u.Jy / u.pix  # "Jy/pix"


@dataclasses.dataclass
class PVmap(BaseObsData):
    Ipv: np.ndarray
    xau: np.ndarray = None
    vkms: np.ndarray = None
    refpos: RefPos = dataclasses.field(default_factory=RefPos)
    dpc: float = None
    " Information for how to have made this PV "
    pangle_deg: float = None
    poffset_au: float = None
    Iunit: str = u.Jy / u.pix  # r"[Jy pixel$^{-1}$]"
    conv_info: dict = None
    obreso: Obreso = None
    # freq0: float = None
    freq0: dataclasses.InitVar[float] = None
    xrad: dataclasses.InitVar[float] = None
    ##
    dtype = "PVmap"
    _axorder = {"xau": 0, "vkms": 1}
    _Iname = "Ipv"
    _axnames = ["xau", "vkms"]

    def __post_init__(self, xrad, freq0):
        if xrad is not None:
            self.xau = xrad * self.dpc * nc.pc / nc.au
        if freq0:
            self.refpos.freq0 = freq0
        self._check_data_shape()
        self._reset_positive_axes()


#########################################################
# Tools
#########################################################


def minmaxargs(array, lim):
    indices = np.argwhere((array >= lim[0]) & (array <= lim[-1]))
    if len(indices) == 0:
        raise ValueError(f"no points within {lim}")
    imin, imax = np.take(indices, (0, -1))
    return imin, imax + 1
