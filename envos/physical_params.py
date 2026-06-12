import numpy as np
from . import nconst as nc
from .log import logger


class PhysicalParameters:
    def __init__(
        self,
        *,
        Ms_Msun: float = None,
        T: float = None,
        Mdot_smpy: float = None,
        CR_au: float = None,
        t_yr: float = None,  # to be deleted
        Omega: float = None,
        jmid: float = None,
        rexp_au: float = None,
        meanmolw: float = 2.3,
        cavangle_deg: float = 0,
    ):
        self.meanmolw = meanmolw
        self.rexp = None

        self.calc_Mdot(self.meanmolw, T, Mdot_smpy)
        self.calc_Ms(self.Mdot, Ms_Msun, t_yr)
        self.calc_jmid(self.cs, self.Ms, self.t, CR_au, jmid, Omega, rexp_au)

        self.cavangle = np.radians(cavangle_deg)
        self.r_inlim_tsc = self.cs * self.Omega**2 * self.t**3
        self.log()

    def calc_Mdot(self, meanmolw, T=None, Mdot_smpy=None):
        m0 = 0.975
        if T is not None:
            cs = np.sqrt(nc.kB * T / (meanmolw * nc.amu))
            Mdot = cs**3 * m0 / nc.G
        elif Mdot_smpy is not None:
            Mdot = Mdot_smpy * nc.smpy
            cs = (Mdot * nc.G / m0) ** (1 / 3)
            T = cs**2 * meanmolw * nc.amu / nc.kB
        else:
            raise ValueError(
                "insufficient parameters: specify one of (T, Mdot_smpy)"
            )
        self.T = T
        self.cs = cs
        self.Mdot = Mdot

    def calc_Ms(self, Mdot, Ms_Msun=None, t_yr=None):
        if Ms_Msun is not None:
            Ms = Ms_Msun * nc.Msun
            t = Ms / Mdot
        elif t_yr is not None:
            t = t_yr * nc.yr
            Ms = Mdot * t
        else:
            raise ValueError(
                "insufficient parameters: specify one of (Ms_Msun, t_yr)"
            )
        self.Ms = Ms
        self.t = t

    def calc_jmid(self, cs, Ms, t, CR_au=None, jmid=None, Omega=None, rexp_au=None):
        m0 = 0.975
        if jmid is not None:
            CR = jmid**2 / (nc.G * Ms)
            Omega = jmid / (0.5 * cs * m0 * t) ** 2
        elif (rexp_au is not None) and (Omega is not None):
            rexp = rexp_au * nc.au
            jmid = (rexp * m0 / 2) ** 2 * Omega
            CR = jmid**2 / (nc.G * Ms)
            self.rexp = rexp
            self.t = rexp / self.cs
        elif CR_au is not None:
            CR = CR_au * nc.au
            jmid = np.sqrt(CR * nc.G * Ms)
            Omega = jmid / (0.5 * cs * m0 * t) ** 2
        elif Omega is not None:
            jmid = (0.5 * cs * m0 * t) ** 2 * Omega
            CR = jmid**2 / (nc.G * Ms)
        else:
            raise ValueError(
                "insufficient parameters: specify one of "
                "(CR_au, jmid, Omega, rexp_au+Omega)"
            )
        self.CR = CR
        self.jmid = jmid
        self.Omega = Omega

    def log(self):
        logger.info("Model Parameters:")
        self._logp("Ms", "Msun", self.Ms, nc.Msun)
        self._logp("Mdot", "Msun/yr", self.Mdot, nc.Msun / nc.yr)
        self._logp("Tenv", "K", self.T)
        self._logp("cs", "km/s", self.cs, nc.kms)
        self._logp("t", "yr", self.t, nc.yr)
        self._logp("Omega", "s^-1", self.Omega)
        self._logp("jmid", "au*km/s", self.jmid, nc.kms * nc.au)
        self._logp("jmid", "pc*km/s", self.jmid, nc.kms * nc.pc)
        self._logp("CR", "au", self.CR, nc.au)
        self._logp("CB", "au", self.CR / 2, nc.au)
        self._logp("meanmolw", "", self.meanmolw)
        self._logp("cavangle", "deg", np.rad2deg(self.cavangle))
        self._logp("rexp", "au", self.rexp, nc.au)
        self._logp("cs*t", "au", self.cs * self.t, nc.au)
        self._logp("Omega*t", "", self.Omega * self.t)
        self._logp("rinlim_tsc", "au", self.cs * self.Omega**2 * self.t**3, nc.au)
        self._logp("rinlim_tsc", "cs*t", self.Omega**2 * self.t**2)
        logger.info("")

    @staticmethod
    def _logp(name, unit, value, unitval=1):
        if value is not None:
            valstr = f"= {value/unitval:10.2g} " + unit.ljust(8)
        else:
            valstr = "= None"

        logger.info("    " + name.ljust(12) + valstr)
