"""
envos/obs/_base.py — BaseObsData (private implementation module)

BaseObsData was too large to fit in data.py under the 600-line budget, so
it is extracted here.  All public symbols are re-exported through
envos/obs/data.py and envos/obs/__init__.py; nothing outside the package
should import this module directly.
"""
import os
import re
import copy
import numpy as np
from scipy import interpolate, optimize

from astropy import units as u

from .. import nconst as nc
from .. import tools
from ..log import logger


class BaseObsData:
    au2pc = nc.au / nc.pc
    fac_deg2au = np.pi / 180 * nc.pc / nc.au

    def __str__(self):
        return tools.dataclass_str(self)

    def _reset_positive_axes(self):
        for i, name in enumerate(self._axnames):
            ax = getattr(self, name)
            if ax is None or len(ax) < 2:
                continue
            if ax[1] < ax[0]:
                setattr(self, name, ax[::-1])
                self.set_I(np.flip(self.get_I(), axis=i))

    def _check_data_shape(self):
        lens = tuple([len(ax) for ax in self.get_axes()])
        if self.get_I().shape != lens:
            logger.info("Data type error")
            raise Exception

    """
    Dynamical variables
    """

    def __getattr__(self, vname):
        dyv = self._dynamic_vars(vname)
        if dyv is not None:
            return dyv
        else:
            raise AttributeError(vname)

    def _dynamic_vars(self, key):
        if key == "dx":
            return self.xau[1] - self.xau[0]
        elif key == "dy":
            return self.yau[1] - self.yau[0]
        elif key == "dv":
            return self.vkms[1] - self.vkms[0]
        elif key == "Lx":
            return self.xau[-1] - self.xau[0]
        elif key == "Ly":
            return self.yau[-1] - self.yau[0]
        elif key == "Lv":
            return self.vkms[-1] - self.vkms[0]
        elif key == "Nx":
            return len(self.xau)
        elif key == "Ny":
            return len(self.yau)
        elif key == "Nv":
            return len(self.vkms)
        else:
            return None

    def copy(self):
        return copy.deepcopy(self)

    """
    Get and set I and axes
    """

    def get_Iname(self):
        return self._Iname

    def get_I(self):
        return getattr(self, self.get_Iname())

    def set_I(self, _I):
        return setattr(self, self.get_Iname(), _I)

    def get_Imax(self):
        return np.max(self.get_I())

    def get_Imax_pos(self, interp=False, ver=2, **kwargs):
        I = self.get_I()
        axes = self.get_axes()
        inds = np.unravel_index(np.argmax(I), I.shape)
        pos = [ax[i] for ax, i in zip(axes, inds)]
        if interp:
            for nax, (_p, _ax) in enumerate(zip(pos, axes)):
                dax = _ax[1] - _ax[0]
                b = (_p - 4 * dax, _p + 4 * dax)
                from .data import minmaxargs
                minmax = minmaxargs(_ax, b)
                I = np.take(I, range(*minmax), axis=nax)
                axes[nax] = axes[nax][minmax[0] : minmax[1]]

            if ver == 1 and len(axes) == 2:
                fun = interpolate.RectBivariateSpline(axes[0], axes[1], I)

                def f(x):
                    # print(x, -fun(x[0], x[1])[0, 0])
                    return -fun(x[0], x[1])[0, 0]

            elif ver == 2 or len(axes) != 2:
                _kwargs = {"method": "cubic"}
                _kwargs.update(kwargs)
                import scipy

                if (_kwargs["method"] in ("slenar", "cubic", "quintic")) and (
                    int(scipy.__version__[2]) < 9
                ):
                    logger.warning(
                        'Method "cubic" in RegularGridInterpolator are vailable only for scipy\'s version > 1.9.'
                    )
                    logger.warning("Just return positoin obtained from numpy.argmax")
                    return pos
                fun = interpolate.RegularGridInterpolator(axes, I, **_kwargs)

                def f(x):
                    # print(x, -fun(x)[0])
                    return -fun(x)[0]

            res = optimize.minimize(
                f,
                pos,
                tol=np.max(I) * 1e-8,
                method="Nelder-Mead",
            )
            return res.x  # np.array([ res.x[0], res.x[1] ])

        else:
            return pos

    #        return np.array([ self.xau[ip], self.yau[jp] ])

    def get_axes(self):
        return [getattr(self, _axn) for _axn in self._axnames if hasattr(self, _axn)]

    def set_axes(self, axes):
        for ax, _axn in zip(axes, self._axnames):
            if hasattr(self, _axn):
                setattr(self, _axn, ax)
        # self.calc_stdcoord_to_radec()

    def Iu(self, format="latex_inline"):
        s = f"{self.Iunit:{format}}"
        if str(self.Iunit) == "I_max":
            _s = re.search("\\\mathrm{(.*)}", s).group(1)
            return rf"${_s}$"
        else:
            return s

    """
    Functions setting important attributes
    """

    def set_dpc(self, dpc):
        self.dpc = dpc

    def set_obs_resolution(
        self,
        beam_maj_au=None,
        beam_min_au=None,
        vreso_kms=None,
        beam_pa_deg=None,
        beam_maj_deg=None,
        beam_min_deg=None,
    ):
        from .data import Obreso
        self.obreso = Obreso(
            obsdata=self,
            beam_maj_au=beam_maj_au,
            beam_min_au=beam_min_au,
            beam_maj_deg=beam_maj_deg,
            beam_min_deg=beam_min_deg,
            vreso_kms=vreso_kms,
            beam_pa_deg=beam_pa_deg,
            dpc=self.dpc,
        )

    def set_sobs_info(self, iline=None, molname=None, incl=None, phi=None, posang=None):
        """
        sobs info is nit used for actual observation data.
        """
        self.sobs_info = {
            "iline": iline,
            "molname": molname,
            "incl": incl,
            "phi": phi,
            "posang": posang,
        }
        for k, v in self.sobs_info.items():
            setattr(self, k, v)

    def copy_info_from_obsdata(self, obsdata):
        if hasattr(obsdata, "obreso"):
            self.obreso = obsdata.obreso

        if hasattr(obsdata, "sobs_info") and (obsdata.sobs_info is not None):
            self.set_sobs_info(
                iline=obsdata.iline,
                molname=obsdata.molname,
                incl=obsdata.incl,
                phi=obsdata.phi,
                posang=obsdata.posang,
            )

    """
    Functions operating I and axes
    """

    def norm_I(self, norm="max"):
        I = self.get_I()
        if norm is None:
            pass

        elif norm == "max":
            self.set_I(I / np.max(I))
            # self.Ipv /= np.max(self.Ipv)
            # self.Iunit = r"[$I_{\rm max}$]"
            self.Iunit = u.def_unit(
                "I_max", np.max(I) * self.Iunit, format={"latex": r"I_{\rm max}"}
            )  # r"[$I_{\rm max}$]"

        elif isinstance(norm, float):
            self.set_I(I / norm)
            self.Iunit = norm * self.Iunit  # f"[{norm:.2g}" + r"Jy pixel$^{-1}$]"

        else:
            raise Exception
            # return None

    def convto_Tb(self):
        if (self.Iunit == u.Jy / u.pix) and (self.refpos.freq0 is not None):
            tb = self.convert_Jyppix_to_brightness_temperature(
                self.get_I(), unit="Jyppix", freq0=self.refpos.freq0
            )
            self.set_I(tb)
            self.Iunit = u.K
        elif (self.Iunit == u.Jy / u.beam) and (self.refpos.freq0 is not None):
            tb = self.convert_Jyppix_to_brightness_temperature(
                self.get_I(), unit="Jypbeam", freq0=self.refpos.freq0
            )
            self.set_I(tb)
            self.Iunit = u.K
        else:
            raise AttributeError(
                f"Failed to convert the data into brightness temperature: Iunit={self.Iunit} and freq0={self.refpos.freq0}"
            )

    def trim(self, xlim=None, ylim=None, vlim=None):
        from .data import minmaxargs
        if xlim is not None:
            nax = self._axorder["xau"]
            imin, imax = minmaxargs(self.xau, xlim)
            self.set_I(np.take(self.get_I(), range(imin, imax), axis=nax))
            self.xau = self.xau[imin:imax]
            # self.calc_stdcoord_to_radec()

        if ylim is not None and hasattr(self, "yau"):
            nax = self._axorder["yau"]
            jmin, jmax = minmaxargs(self.yau, ylim)
            self.set_I(np.take(self.get_I(), range(jmin, jmax), axis=nax))
            self.yau = self.yau[jmin:jmax]
            # self.calc_stdcoord_to_radec()

        if vlim is not None and hasattr(self, "vkms"):
            nax = self._axorder["vkms"]
            kmin, kmax = minmaxargs(self.vkms, vlim)
            self.set_I(np.take(self.get_I(), range(kmin, kmax), axis=nax))
            self.vkms = self.vkms[kmin:kmax]

    def mask(
        self,
        fill=0,
        Imin=None,
        Imax=None,
        region=None,
    ):
        I = self.get_I()
        cond = np.ones_like(I)
        if region is not None:
            cond *= region
        if Imin is not None:
            cond *= I >= Imin
        if Imax is not None:
            cond *= I <= Imax
        I = np.where(cond, I, fill)
        self.set_I(I)

    def reverse_ax(self, nums):  # reverese_ax(1,2)
        if isinstance(nums, int):
            nums = [nums]
        for _num in nums:
            axn = self._axnames[_num]
            ax = getattr(self, axn)
            axrev = ax[::-1]
            setattr(self, axn, axrev)
            I = self.get_I()
            Irev = np.flip(I, axis=_num)
            self.set_I(Irev)

    def reversed_ax_data(self, nums):
        if isinstance(nums, int):
            nums = [nums]
        obj = self.copy()
        for _num in nums:
            axn = obj._axnames[_num]
            ax = getattr(obj, axn)
            axrev = ax[::-1]
            setattr(obj, axn, axrev)
            I = obj.get_I()
            Irev = np.flip(I, axis=_num)
            obj.set_I(Irev)
        return obj

    """
    Functions for conversion
    """

    def convert_Jyppix_to_brightness_temperature(
        self, image, unit="Jyppix", freq0=None, lam0=None
    ):
        if lam0 is not None:
            freq0 = nc.c / lam0
        lam0 = nc.c / freq0
        if unit == "Jyppix":
            pix_sr = self.dx * self.dy / self.dpc**2 * (nc.au / nc.pc) ** 2
            Inu = image * 1e-23 / pix_sr
        elif unit == "Jypbeam":
            beam_sr = (
                np.pi
                * self.obreso.beam_maj_au
                * self.obreso.beam_min_au
                / (4 * np.log(2))
                * self.dpc ** (-2)
                * (nc.au / nc.pc) ** 2
            )
            Inu = image * 1e-23 / beam_sr
        Tb_RJ = Inu * nc.c**2 / (2 * nc.kB * freq0**2)
        Tb = (
            nc.h
            * freq0
            / nc.kB
            * 1.0
            / (np.log(1.0 + 2.0 * nc.h * freq0**3 / (Inu.clip(0) * nc.c**2)))
        )
        Tb_RJ2 = (
            nc.h
            * freq0
            / nc.kB
            * 1.0
            / (-1 + np.sqrt(1 + 4.0 * nc.h * freq0**3 / (Inu.clip(0) * nc.c**2)))
        )
        # Tb_app =  nc.h * freq0/nc.kB * np.nan_to_num( 1./np.log(1. + np.nan_to_num( 2.*nc.h*freq0**3/(Inu*nc.c**2) ) ) )

        print(
            f"Tb_app is {np.max(Tb)} K , Tb_RJ is {np.max(Tb_RJ)} K, Tb_RJ2 is {np.max(Tb_RJ2)} K"
        )
        if np.isnan(np.max(Tb)):
            raise Exception
        return Tb

    """
    Functions for observational coordinates

    """

    def ra(self):
        return (
            -self.xau * self.au2pc / (self.dpc * np.cos(self.refpos.dec0 * nc.deg2rad))
            + self.refpos.ra0
        )

    def dec(self):
        return self.yau * nc.au2pc / self.dpc + self.refpos.dec0

    def radec(self):
        return self.ra(), self.dec()

    def freq(self):
        return tools.vkms_to_freq(self.vkms, self.refpos.freq0)

    def set_coord_from_radec(self, ra0_deg, dec0_deg, ra_deg, dec_deg):
        self.refpos.ra0 = ra0_deg
        self.refpos.dec0 = dec0_deg
        self.yau = self.deg2au(dec_deg - dec0_deg)
        self.xau = -np.cos(dec0_deg * nc.deg2rad) * self.deg2au(ra_deg - ra0_deg)

    def set_coord_from_radecSIN(self, ra0_deg, dec0_deg, ra_deg, dec_deg):
        self.refpos.ra0 = ra0_deg
        self.refpos.dec0 = dec0_deg
        self.yau = self.deg2au(dec_deg - dec0_deg)
        self.xau = -self.deg2au(ra_deg - ra0_deg)

    def deg2au(self, v_deg):
        return v_deg * 3600 * self.dpc  # self.fac_deg2au * self.dpc

    """
    set coorfdinate functions
    To be marged
    """

    def move_position(
        self, center_pos, ax="x", axnum=None, unit="au"
    ):  # ra=False, decl=False, deg=False, au=False):
        from .fits_io import ra_to_deg, dec_to_deg
        deg2au = 3600 * self.dpc

        if unit == "ra":
            d = ra_to_deg(*center_pos) * deg2au
        elif unit == "dec":
            d = dec_to_deg(*center_pos) * deg2au
        elif unit == "deg":
            d = center_pos * deg2au
        elif unit == "au":
            d = center_pos
        elif unit == "kms":
            d = center_pos
        elif unit == "ms":
            d = center_pos * 1e-3

        if ax == "x":
            self.xau -= d
        elif ax == "y":
            self.yau -= d
        elif ax == "v":
            self.vkms -= d

    def set_refpoint(self, ra0, dec0):
        "1. Change the reference point (ra0, dec0)"
        self.refpos.ra0 = ra0
        self.refpos.dec0 = dec0

    def move_center(self, xy_au=None, to_Imax=False, **kwargs):
        "2. Change the origin of the coordinate but not change the reference point"
        if xy_au is not None:
            self.move_position(xy_au[0], "x", "au")
            self.move_position(xy_au[1], "y", "au")
        elif to_Imax:
            pos = self.get_Imax_pos(**kwargs)
            newaxes = []
            for ax, cp in zip(self.get_axes(), pos):
                newaxes.append(ax - cp)
            self.set_axes(newaxes)

            # self.move_position(xy_au[0], "x", "au")
            # self.move_position(xy_au[1], "y", "au")

    #    def move_radec_pos(self, radec_deg=None):
    #    "3. Change the origin of the coordinate with changing the reference point"
    #        if radec_deg is not None:
    #            dxdeg = (radec_deg[0] - self.refpos.ra0) * np.cos(self.refpos.dec0*nc.deg2rad)
    #            dydeg = radec_deg[1] - self.refpos.dec0
    #            self.move_position(dxdeg, "x", "deg")
    #            self.move_position(dydeg, "y", "deg")

    """
    Functions for saving this object
    """

    def save(
        self,
        filename=None,
        basename="obsdata",
        mode="pickle",
        dpc=None,
        dirpath=None,
        filepath=None,
    ):
        # P2-A-2: the output location must be given explicitly; the implicit
        # run-directory fallback has been removed.
        if filepath is None:
            if dirpath is None:
                raise ValueError(
                    "BaseObsData.save requires either filepath or dirpath"
                )
            if filename is None:
                output_ext = {"joblib": "jb", "pickle": "pkl", "fits": "fits"}[mode]
                filename = basename + "." + output_ext
            filepath = os.path.join(dirpath, filename)

        outdir = os.path.dirname(filepath)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        if os.path.exists(filepath):
            logger.info(f"remove old file: {filepath}")
            os.remove(filepath)

        if mode == "joblib":
            import joblib

            joblib.dump(self, filepath, compress=3)
            logger.info(f"Saved joblib file: {filepath}")

        elif mode == "pickle":
            import pandas as pd
            pd.to_pickle(self, filepath)
            logger.info(f"Saved pickle file: {filepath}")

        elif mode == "fits":
            # Deferred import to avoid circular dependency:
            # fits_io imports data classes; data.py must not import fits_io at
            # module level.
            from .fits_io import save_fits
            save_fits(self, filepath)
            logger.info(f"Saved fits file: {filepath}")
