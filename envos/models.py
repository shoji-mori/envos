import os
import numpy as np
import pandas as pd
from scipy import interpolate, integrate
from dataclasses import dataclass
from typing import Any
from . import tools, cubicsolver, tsc, column_density
from .nconst import G, kB, amu, au, Msun
from . import gpath
from .log import logger


class ModelBase:
    def __str__(self):
        return f"models.{self.__class__.__name__}"

    def read_grid(self, grid):
        for k, v in grid.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.vR = self.vr * np.sin(self.tt) + self.vt * np.cos(self.tt)
        self.vz = self.vr * np.cos(self.tt) - self.vt * np.sin(self.tt)

    def save(self, basename="file", mode="pickle", filepath=None):
        """
        Any Model object can be saved by using .save function.
        basename is the filename witout extension
        mode can be choosen in "pickle",

        """
        tools.savefile(self, basename=basename, mode=mode, filepath=filepath)

    def save_arrays(self, varnames, filename):
        if isinstance(varnames, str):
            varnames = [varnames]
        varhdr = ["r[cm]", "theta[rad]", "phi[rad]"]
        varlist = [self.rr, self.tt, self.pp]
        for vn in varnames:
            if not hasattr(self, vn):
                print(f"Error: The variable name `{vn}` is not found in this instance.")
                print("       Please choose variable name within ")
                print(
                    ", ".join(
                        [
                            k
                            for k in vars(self).keys()
                            if np.shape(self.rr) == np.shape(getattr(self, k))
                        ]
                    )
                )
                raise KeyError
            v = getattr(self, vn)
            if np.shape(self.rr) == np.shape(v):
                varhdr.append(vn)
                varlist.append(v)
        tools.save_array(varlist, filename, header=" ".join(varhdr))

    def save_pickle(self, filename, filepath=None):
        if filepath is None:
            filepath = os.path.join(gpath.run_dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        pd.to_pickle(self, filepath)
        logger.info(f"Saved : {filepath}")

    def read_pickle(self, filename=None, filepath=None):
        if (filename is not None) and (filepath is None):
            filepath = os.path.join(gpath.run_dir, filename)
        tools.setattr_from_pickle(self, filepath)

    def get_midplane_profile(self, vname, dtheta=0.03, ntheta=1000, vabs=False):
        if isinstance(vname, str):
            val = getattr(self, vname)
        elif isinstance(vname, np.ndarray):
            val = vname
        val = tools.take_midplane_average(self, val, dtheta=dtheta, ntheta=ntheta, vabs=vabs)
        val = tools.take_horizontal_average(val)
        return val

    def get_gasmask(self):
        gasmask = np.where(self.rhogas > 0, 1, 0)
        return gasmask

    def get_argmid(self):
        return np.argmin(np.abs(self.tc_ax - 0.5 * np.pi ))


#        dth = np.pi/2-self.tc_ax
#        val = np.abs( getattr(self, vname) ) if vabs else getattr(self, vname)
#        func = interpolate.interp1d(dth, val, axis=1)
#        dth_new = np.linspace(-dtheta, dtheta, ntheta)
#        midvalue = integrate.simpson(func(dth_new), dth_new, axis=1)/(2*dtheta)
#        return np.average(midvalue, axis=1)


@dataclass
class CircumstellarModel(ModelBase):
    """
    Data structure of physical variable:
    rr: radial distance from the central star [cm]
    tt: polar angle [rad]
    pp: azimuthal angle [rad]
    R: cylindrical radius [cm]
    z: vertical distance from the midplane [cm]
    ppar: physical parameters
    rhogas: gas density [g/cm^3]
    rhodust: dust density [g/cm^3]
    vr: radial velocity [cm/s]
    vt: polar velocity [cm/s]
    vp: azimuthal velocity [cm/s]
    vturb: turbulent velocity [cm/s]
    heatrate: heating rate [erg/s]
    Tgas: gas temperature [K]
    Tdust: dust temperature [K]
    f_dg: dust-to-gas mass ratio
    molname: name of molecule
    radmcdir: directory of RADMC-3D
    filepath: filepath of this model
    """

    grid: Any = None
    rc_ax: np.ndarray = None
    tc_ax: np.ndarray = None
    pc_ax: np.ndarray = None
    rr: np.ndarray = None
    tt: np.ndarray = None
    pp: np.ndarray = None
    R: np.ndarray = None
    z: np.ndarray = None
    ppar: Any = None
    rhogas: np.ndarray = None
    rhodust: np.ndarray = None
    vr: np.ndarray = None
    vt: np.ndarray = None
    vp: np.ndarray = None
    vturb: np.ndarray = None
    heatrate: np.ndarray = None
    Tgas: np.ndarray = None
    Tdust: np.ndarray = None
    f_dg: float = None
    molname: str = None
    radmcdir: str = None
    filepath: str = None

    def __post_init__(self):
        logger.info("Constructing a model of circumstellar material")
        if self.filepath is not None:
            self.read_pickle(filepath=self.filepath)

        if self.grid is not None:
            self.set_grid(self.grid)

        if self.ppar is not None:
            self.set_physical_parameters(self.ppar)

    def __str__(self):
        return tools.dataclass_str(self)

    def set_physical_parameters(self, ppar):
        self.ppar = ppar

    def set_grid(self, grid):
        self.read_grid(grid)

    def set_gas_density(self, rho):
        self.rhogas = rho

    def set_gas_velocity(self, vr, vt, vp):
        self.vr = vr
        self.vt = vt
        self.vp = vp
        self.set_cylindrical_velocity()

    def set_turb_velocity(self, vturb):
        self.vturb = vturb

    def set_heatrate(self, heatrate):
        self.heatrate = heatrate

    def set_gas_temperature(self, temperature):
        self.Tgas = temperature
        self.Tdust = temperature

    #    def get_midplane_gas_temperature(self):
    #        return tools.take_midplane_average(model, value, dtheta=0.03, ntheta=1000, vabs=False)

    def set_dust_density(self, f_dg=None):
        self.f_dg = f_dg

        if hasattr(self, "rhogas"):
            self.rhodust = self.rhogas * f_dg

    def set_molname(self, molname):
        self.molname = molname

    def set_radmcdir(self, radmcdir):
        self.radmcdir = radmcdir

    def set_mu0(self, mu0):
        self.mu0 = mu0

    def calc_midplane_average(self):
        vnames = ["rhogas", "rhodust", "vr", "vt", "vp", "Tgas", "Tdust"]
        for _vn in vnames:
            if getattr(self, _vn, None) is None:
                continue
            _val = self.get_midplane_profile(_vn)
            setattr(self, _vn + "_mid", _val)

    def calc_column_density(self, colr=True, colz=True, colt=False):
        if colr:
            self.colr = column_density.calc_column_density(self, "r")
        if colz:
            self.colz = column_density.calc_column_density(self, "z")
        if colt:
            self.colt = column_density.calc_column_density(self, "theta")


class CassenMoosmanInnerEnvelope(ModelBase):
    """
    Cassen & Moosman (1981) inner envelope model
    """

    def __init__(self, grid, Mdot, CR, Ms, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.read_grid(grid)
        # self._cond_func = np.frompyfunc(self._cond, 2, 1)
        self.calc_kinematic_structure(Mdot, CR, Ms, cavangle)
        # self.set_cylindrical_velocity()

    def calc_kinematic_structure(self, Mdot, CR, Ms, cavangle):
        csol = np.frompyfunc(self._sol_with_cubic, 2, 1)
        zeta = CR / self.rr
        self.mu0 = csol(np.cos(self.tt), zeta).astype(float)
        sin0 = np.sqrt(1 - self.mu0**2)
        sin = np.sin(self.tt)
        mu_to_mu0 = 1 - zeta * sin0**2
        v0 = np.sqrt(G * Ms / self.rr)
        self.vr = -v0 * np.sqrt(1 + mu_to_mu0)
        self.vt = v0 * zeta * sin0**2 * self.mu0 / sin * np.sqrt(1 + mu_to_mu0)
        self.vp = v0 * sin0**2 / np.sin(self.tt) * np.sqrt(zeta)
        P2 = 1 - 3 / 2 * sin0**2
        rho = -Mdot / (4 * np.pi * self.rr**2 * self.vr * (1 + 2 * zeta * P2))
        cavmask = np.array(np.abs(self.mu0) <= np.cos(cavangle), dtype=float)
        self.rho = rho * cavmask
        if np.isnan(self.rho).any():
            logger.warning(
                "NaN in inner-envelope density "
                "(no cubic solution at some cells)"
            )

    def _sol_with_cubic(self, m, zeta):
        allsols = np.round(cubicsolver.solve(zeta, 0, 1 - zeta, -m).real, 8)
        sols = [sol for sol in allsols if self._cond(m, sol)]
        return sols[0] if len(sols) != 0 else np.nan

    @staticmethod
    def _cond(mu, sol):
        if (mu >= 0) and (mu <= sol <= 1):
            return True  # sol
        elif (mu < 0) and (-1 <= sol <= mu):
            return True  # sol
        else:
            return False  # None


class SimpleBallisticInnerEnvelope(ModelBase):
    def __init__(self, grid, Mdot, CR, M, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.read_grid(grid)
        self.calc_kinematic_structure(Mdot, CR, M, cavangle)
        # self.set_cylindrical_velocity()

    def calc_kinematic_structure(self, Mdot, CR, M, cavangle):
        vff = np.sqrt(2 * G * M / self.rr)
        CB = CR / 2
        rho_prof = Mdot / (4 * np.pi * self.rr**2 * vff)
        mu_cav = np.cos(cavangle)
        cavmask = np.array(np.abs(np.cos(self.tt)) <= mu_cav, dtype=float)
        cbmask = np.array(self.rr >= CB, dtype=float)
        self.rho = rho_prof * cbmask * cavmask
        self.vr = -vff * np.sqrt((1 - CB / self.rr).clip(0))
        self.vt = np.zeros_like(self.rr)
        self.vp = vff / np.sqrt(self.rr / CB)
        self.mu0 = np.cos(self.tt)


class TerebeyOuterEnvelope(ModelBase):
    def __init__(self, grid, t, cs, Omega, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.read_grid(grid)
        self.calc_kinematic_structure(t, cs, Omega, cavangle)
        self.rin_lim = cs * Omega**2 * t**3

    def calc_kinematic_structure(self, t, cs, Omega, cavangle):
        res = tsc.get_tsc(self.rc_ax, self.tc_ax, t, cs, Omega, mode="read")
        cavmask = np.array(self.tt >= cavangle, dtype=float)
        self.rho = res["rho"][:, :, np.newaxis] * cavmask
        self.vr = res["vr"][:, :, np.newaxis]
        self.vt = res["vt"][:, :, np.newaxis]
        self.vp = res["vp"][:, :, np.newaxis]
        self.Delta = res["Delta"]
        P2 = 1 - 1.5 * np.sin(self.tt) ** 2
        self.rin_lim = cs * Omega**2 * t**3 * 0.4 / (1 + self.Delta * P2)


class Disk(ModelBase):
    def calc_kinematic_structure_from_Sigma(self, Sigma, Ms, cs_disk):
        OmegaK = np.sqrt(G * Ms / self.R**3)
        H = cs_disk / OmegaK
        rho0 = Sigma / (np.sqrt(2 * np.pi) * H)
        self.rho = rho0 * np.exp(-0.5 * (self.z / H) ** 2)
        self.vr = np.zeros_like(self.rho)
        self.vt = np.zeros_like(self.rho)
        self.vp = OmegaK * self.R


class PowerlawDisk(Disk):
    def __init__(
        self,
        grid,
        Ms,
        Rd,
        ind_S=-1,
        Td10=40,
        ind_T=-0.5,
        fracMd=0.1,
        meanmolw=2.3,
        tail="exp",
        ind_tail=None,
        Tmid=None,
    ):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.read_grid(grid)
        self.tail = tail
        self.ind_tail = ind_tail
        Mdisk = fracMd * Ms
        self.Sigma = self.get_Sigma(Mdisk, Rd, ind_S)
        if Tmid is None:
            self.Td = Td10 * (self.R / 10 / au) ** ind_T
        else:
            self.Td = np.repeat(
                Tmid[:, None, None], len(self.tc_ax), axis=1
            )  # np.tile(Tmid.T, (1, len(self.tc_ax), 1 ) )
            if np.shape(self.Td) != np.shape(self.R):
                raise Exception
        cs_disk = np.sqrt(kB * self.Td / (meanmolw * amu))
        self.calc_kinematic_structure_from_Sigma(self.Sigma, Ms, cs_disk)
        # self.set_cylindrical_velocity()

    def get_Sigma(self, Mdisk, Rd, ind_S):
        _R = self.rc_ax / Rd
        power = _R**ind_S
        if self.tail == "exp":
            ind_tail = 2 + ind_S if self.ind_tail is None else self.ind_tail
            tailprof = np.exp(-(_R ** (ind_tail)))
        elif self.tail == "cut":
            tailprof = np.where(_R < 1, 1, 0)
        else:
            raise ValueError(
                f"Unknown tail type '{self.tail}'; expected 'exp' or 'cut'"
            )
        # Sigma0 = Mdisk * (ind_S + 2.) / (2*np.pi * Rd**2)  # assume ind_rho < -2
        Sigma0 = Mdisk / (
            2 * np.pi * Rd**2 * integrate.simpson(_R * power * tailprof, _R)
        )
        logger.info(
            f"Disk surface density profile: Sigma = {Sigma0} g/cm2 * (R/{Rd/au}au)**({ind_S})"
        )
        Mdisk_check = integrate.simpson(
            2 * np.pi * self.rc_ax * Sigma0 * power * tailprof, self.rc_ax
        )
        logger.info(
            f"Actual total disk mass = {Mdisk_check/Msun} Msun , excepted mass = {Mdisk/Msun} Msun "
        )
        Sigma_R = Sigma0 * power * tailprof
        return interpolate.interp1d(_R, Sigma_R, bounds_error=False, fill_value=0.0)(
            self.R / Rd
        )
