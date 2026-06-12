import numpy as np
from .models import (
    CircumstellarModel,
    CassenMoosmanInnerEnvelope,
    SimpleBallisticInnerEnvelope,
    TerebeyOuterEnvelope,
    PowerlawDisk,
)
from .radmc3d import RadmcController
from .log import logger
from .grid import Grid
from .physical_params import PhysicalParameters
from . import tools


class ModelGenerator:
    def __init__(self, config=None, readfile=None):
        self.grid = None
        self.ppar = None
        self.models = {}
        self.inenv = None
        self.outenv = None
        self.disk = None
        self.model = CircumstellarModel()
        self.f_dg = None  # C-48: must be set before calc_kinematic_structure

        if config is not None:
            self.init_from_config(config)

        if readfile is not None:
            return self.init_from_file(readfile)

    def init_from_config(self, config):
        self.config = config

        grid = Grid(
            config.ri_ax,
            config.ti_ax,
            config.pi_ax,
            rau_lim=[config.rau_in, config.rau_out],
            theta_lim=[config.theta_in, config.theta_out],
            phi_lim=[config.phi_in, config.phi_out],
            nr=config.nr,
            ntheta=config.ntheta,
            nphi=config.nphi,
            dr_to_r=config.dr_to_r,
            aspect_ratio=config.aspect_ratio,
            logr=config.logr,
            ringhost=config.ringhost,
        )
        self.set_grid(grid=grid)

        self.set_physical_parameters(
            config.T,
            config.CR_au,
            config.Ms_Msun,
            config.t_yr,
            config.Omega,
            config.jmid,
            config.rexp_au,
            config.Mdot_smpy,
            config.meanmolw,
            config.cavangle_deg,
        )

        self.set_model(inenv=config.inenv, outenv=config.outenv, disk=config.disk)
        self.f_dg = config.f_dg

    def init_from_file(self, readfile):
        mg = tools.read_pickle(readfile)
        for k, v in mg.__dict__.items():
            setattr(self, k, v)

    def set_grid(self, ri=None, ti=None, pi=None, grid=None):
        if grid is not None:
            self.grid = grid
        elif ri is not None:
            self.grid = Grid(
                ri_ax=ri,
                ti_ax=ti,
                pi_ax=pi,
            )
        else:
            logger.error("Failed in making grid.")

        self.model.set_grid(self.grid)

    def get_meshgrid(self):
        return self.grid.rr, self.grid.tt, self.grid.pp

    def set_physical_parameters(
        self,
        T: float = None,
        CR_au: float = None,
        Ms_Msun: float = None,
        t_yr: float = None,
        Omega: float = None,
        jmid: float = None,
        rexp_au: float = None,
        Mdot_smpy: float = None,
        meanmolw: float = 2.3,
        cavangle_deg: float = 0,
    ):
        self.ppar = PhysicalParameters(
            T=T,
            CR_au=CR_au,
            Ms_Msun=Ms_Msun,
            t_yr=t_yr,
            Omega=Omega,
            jmid=jmid,
            rexp_au=rexp_au,
            Mdot_smpy=Mdot_smpy,
            meanmolw=meanmolw,
            cavangle_deg=cavangle_deg,
        )

    def set_model(self, inenv=None, outenv=None, disk=None):
        self.set_inenv(inenv)
        self.set_outenv(outenv)
        self.set_disk(disk)

    def set_gas_density(self, rho):
        self.model.set_gas_density(rho)

    def set_gas_velocity(self, vr, vt, vp):
        self.model.set_gas_velocity(vr, vt, vp)

    def calc_kinematic_structure(self, smoothing_TSC=True):
        """
        Compute the kinematic (density + velocity) structure of the model.

        Notes
        -----
        When ``smoothing_TSC=True`` the outer-envelope (TSC) contribution is
        blended smoothly into the inner region via an exponential factor.
        This blending is applied to the velocity field everywhere (including
        cavity cells where rho==0), but is *not* applied to the density in
        zero-density (cavity) cells.  The resulting density/velocity
        inconsistency in cavity cells is a known limitation; see ISSUES C-49
        and PLAN §9 D10.  Correcting it would be a ⚠ physics-change affecting
        all TSC models and is deferred to a later dedicated task.
        """
        ### Set grid
        if self.grid is None:
            raise Exception("grid is not set.")

        ### Set physical parameters
        if self.ppar is None:
            raise Exception("ppar is not set.")

        # C-48: f_dg fallback with warning
        if self.f_dg is None:
            logger.warning(
                "f_dg is not set; using default value 0.01. "
                "Set self.f_dg (or pass f_dg via Config) to suppress this warning."
            )
            self.f_dg = 0.01

        ### Set models
        logger.info("Calculating kinematic structure")
        logger.info("Models:")
        logger.info("    inner-envelope: %s", self.inenv if self.inenv else None)
        logger.info("    outer-envelope: %s", self.outenv if self.outenv else None)
        logger.info("    disk: %s\n", self.disk if self.disk else None)

        ### Make kmodel
        zeros = np.zeros_like(self.grid.rr)
        rho = np.copy(zeros)
        vr = np.copy(zeros)
        vt = np.copy(zeros)
        vp = np.copy(zeros)
        keys = ("rho", "vr", "vt", "vp")
        ismodel = lambda x: all(hasattr(x, k) for k in keys)

        if ismodel(self.inenv):
            logger.info("Setting inner envelop")
            cond = rho <= self.inenv.rho
            rho[cond] = self.inenv.rho[cond]
            vr[cond] = self.inenv.vr[cond]
            vt[cond] = self.inenv.vt[cond]
            vp[cond] = self.inenv.vp[cond]
            self.model.set_mu0(self.inenv.mu0)

        if ismodel(self.outenv):
            logger.info("Setting outer envelop")
            if smoothing_TSC:
                fac = np.exp(-((0.3 * self.outenv.rin_lim / self.grid.rr) ** 2))
                rho += (self.outenv.rho - rho) * fac * np.where(rho != 0, 1, 0)
                vr += (self.outenv.vr - vr) * fac
                vt += (self.outenv.vt - vt) * fac
                vp += (self.outenv.vp - vp) * fac

            else:
                cond1 = rho < self.outenv.rho
                cond2 = self.grid.rr > 1 / 5 * self.outenv.rin_lim
                cond = cond1 & cond2
                rho[cond] = self.outenv.rho[cond]
                vr[cond] = self.outenv.vr[cond]
                vt[cond] = self.outenv.vt[cond]
                vp[cond] = self.outenv.vp[cond]

        if ismodel(self.disk):
            logger.info("Setting disk")
            cond = rho < self.disk.rho
            self.disk_region = cond
            # D1: replace-mode — disk wins only where disk.rho > envelope rho.
            # (The old additive rho += disk.rho over-counted envelope density.)
            rho[cond] = self.disk.rho[cond]
            vr[cond] = self.disk.vr[cond]
            vt[cond] = self.disk.vt[cond]
            vp[cond] = self.disk.vp[cond]

        logger.info("Calculated kinematic structure")
        self.model.set_gas_density(rho)
        self.model.set_gas_velocity(vr, vt, vp)
        self.model.set_dust_density(f_dg=self.f_dg)
        self.model.set_physical_parameters(self.ppar)

    def set_inenv(self, inenv):
        if inenv is None:
            return

        if hasattr(inenv, "rho"):
            self.inenv = inenv
        elif inenv == "UCM":
            self.inenv = CassenMoosmanInnerEnvelope(
                self.grid,
                self.ppar.Mdot,
                self.ppar.CR,
                self.ppar.Ms,
                self.ppar.cavangle,
            )

        elif inenv == "Simple":
            self.inenv = SimpleBallisticInnerEnvelope(
                self.grid,
                self.ppar.Mdot,
                self.ppar.CR,
                self.ppar.Ms,
                self.ppar.cavangle,
            )
        else:
            raise Exception("Unknown inenv type")

    def set_outenv(self, outenv):
        if outenv is None:
            return

        if hasattr(outenv, "rho"):
            self.outenv = outenv
        elif outenv == "TSC":
            self.outenv = TerebeyOuterEnvelope(
                self.grid,
                self.ppar.t,
                self.ppar.cs,
                self.ppar.Omega,
                self.ppar.cavangle,
            )
        else:
            raise Exception("Unknown outenv type")

    def set_disk(self, disk):
        if disk is None:
            return

        if hasattr(disk, "rho"):
            self.disk = disk

        elif disk == "powerlaw":
            config = {
                "fracMd": 0.1,
                "ind_S": -1.0,
                "Td10": 40,
                "ind_T": -0.5,
                "meanmolw": self.ppar.meanmolw,
            }
            if self.config.disk_config is not None:
                config.update(self.config.disk_config)
            logger.info("disk config: %s", config)
            self.disk = PowerlawDisk(self.grid, self.ppar.Ms, self.ppar.CR, **config)
        else:
            raise Exception("Unknown disk type")

    def calc_thermal_structure(self):
        logger.info("Calculating thermal structure")
        # conf = self.radmc_config
        radmc = RadmcController(config=self.config)
        radmc.clean_radmc_dir()

        radmc.set_model(self.model)
        radmc.set_mctherm_inpfiles()
        radmc.run_mctherm()

        rho = radmc.get_gas_density()
        if not np.allclose(self.model.rhogas, rho, rtol=1e-07, atol=1e-20):
            logger.error(
                "Input value mismatches with that read by radmc3d: gas density\n"
            )
            # raise Exception

        Tgas = radmc.get_gas_temperature()
        self.model.set_gas_temperature(Tgas)

    def get_model(self):
        return self.model

    def save(self):
        tools.savefile(self, basename="mg")


def read_model(path):
    if ".pkl" in path:
        return tools.read_pickle(path)
    else:
        raise NotImplementedError("only .pkl is supported")
