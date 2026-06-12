#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import shutil
import numpy as np
import pandas as pd
import radmc3dPy.analyze as rmca
from . import tools
from . import nconst as nc
from . import gpath
from .log import logger

"""
    Classes

"""


class RadmcController:
    def __init__(
        self,
        config=None,
        run_dir: str = None,
        radmc_dir: str = None,
        storage_dir: str = None,
        #
    ):
        if config is not None:
            self.config = config
            run_dir = run_dir or config.run_dir
            radmc_dir = radmc_dir or config.radmc_dir
            storage_dir = storage_dir or config.storage_dir

            self.n_thread = config.n_thread
            self.nphot = config.nphot
            self.scattering_mode_max = config.scattering_mode_max
            self.mc_scat_maxtauabs = config.mc_scat_maxtauabs
            self.tgas_eq_tdust = config.tgas_eq_tdust
            self.modified_random_walk = config.modified_random_walk
            self.f_dg = config.f_dg
            self.opac = config.opac
            self.Lstar_Lsun = config.Lstar_Lsun
            self.Rstar_Rsun = config.Rstar_Rsun

            self.mfrac_H2 = config.mfrac_H2
            self.molname = config.molname
            self.molabun = config.molabun
            self.iline = config.iline
            self.nonlte = config.nonlte

            # D6: non-LTE mode is untested; raise early rather than produce
            # silently wrong output (lines.inp species-count bug, see ISSUES C-57).
            if self.nonlte:
                raise NotImplementedError(
                    "nonlte mode is untested (lines.inp species count bug); "
                    "see ISSUES C-57"
                )

        self.set_dirs(run_dir, radmc_dir, storage_dir)

    def set_dirs(self, run_dir=None, radmc_dir=None, storage_dir=None):
        self.run_dir = run_dir or gpath.run_dir  # global?
        gpath.make_dirs(run=self.run_dir)

        self.radmc_dir = radmc_dir or gpath.radmc_dir  # local
        gpath.make_dirs(radmc=self.radmc_dir)

        self.storage_dir = storage_dir or gpath.storage_dir
        if self.storage_dir is not gpath.storage_dir:
            gpath.make_dirs(storage=self.storage_dir)

    def set_input_params(
        self,
        n_thread: int = 1,
        nphot: int = 1e6,
        scattering_mode_max: int = 0,
        mc_scat_maxtauabs: float = 5.0,
        tgas_eq_tdust: bool = True,
        f_dg: float = 0.01,
        opac: str = "silicate",
        Lstar_Lsun: float = 1.0,
        Rstar_Rsun: float = 1.0,
        #
        mfrac_H2: float = 0.74,
        molname: str = None,
        molabun: float = None,
        iline: int = None,
    ):
        self.n_thread = n_thread
        self.nphot = nphot
        self.scattering_mode_max = scattering_mode_max
        self.mc_scat_maxtauabs = mc_scat_maxtauabs
        self.tgas_eq_tdust = tgas_eq_tdust
        self.f_dg = f_dg
        self.opac = opac
        self.Lstar_Lsun = Lstar_Lsun
        self.Rstar_Rsun = Rstar_Rsun

        self.mfrac_H2 = mfrac_H2
        self.molname = molname
        self.molabun = molabun
        self.iline = iline

    def set_model(self, model):
        """
        Attach a CircumstellarModel to this controller.

        Parameters
        ----------
        model : CircumstellarModel or str
            If a string, it must be the path to an existing pickle file.
            If None, a ValueError is raised.
        """
        if isinstance(model, str):
            if not os.path.isfile(model):
                raise FileNotFoundError(model)
            self.model = pd.read_pickle(model)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("model is None")

    def set_mctherm_inpfiles(self):
        logger.info("Setting input files used in radmc3d")
        md = self.model
        nrc = len(md.rc_ax)
        ntc = len(md.tc_ax)
        npc = len(md.pc_ax)
        self.ntot = nrc * ntc * npc
        rhog = md.rhogas
        self._rhog = rhog
        if np.max(rhog) == 0:
            raise Exception("Zero density")
        # B-15: hasattr always True for dataclass; check the value instead.
        if getattr(md, "rhodust", None) is not None:
            rhod = md.rhodust
        else:
            rhod = rhog * self.f_dg
            logger.info(
                "rhodust not set on model; computing as rhogas * f_dg (f_dg=%.4g)",
                self.f_dg,
            )
        lam = [
            *np.geomspace(0.1, 7, 20, endpoint=False),
            *np.geomspace(7, 25, 100, endpoint=False),
            *np.geomspace(25, 1e4, 30, endpoint=True),
        ]
        nlam = len(lam)
        Tstar = self.Rstar_Rsun ** (-0.5) * self.Lstar_Lsun**0.25 * nc.Tsun

        """
        Setting Radmc Parameters
        """
        # set grid
        coord_info = "1 1 " + ("1" if npc >= 2 else "0")
        self._save_input_file(
            "amr_grid.inp",
            "1",
            "0",
            "100",
            "0",
            coord_info,
            f"{nrc:d} {ntc:d} {npc:d}",
            *md.ri_ax,
            *md.ti_ax,
            *md.pi_ax,
        )

        # set wavelength
        self._save_input_file("wavelength_micron.inp", f"{nlam:d}", *lam)

        # set a star
        self._save_input_file(
            "stars.inp",
            "2",
            f"1 {nlam}",
            f"{self.Rstar_Rsun*nc.Rsun:13.8e} 0 0 0 0",
            *lam,
            f"{-Tstar:13.8e}",
        )

        # set dust density
        self.set_dust_density(rhod)

        # set_dust_opacity
        opacs = [self.opac]
        text_lines = ["2", f"{len(opacs)}", "==========="]
        for op in opacs:
            self._copy_from_storage(f"dustkappa_{op}.inp")
            text_lines += ["1", "0", f"{op}", "----------"]
        self._save_input_file("dustopac.inp", *text_lines)

        # set_input
        param_dict = {
            "nphot": int(self.nphot),
            "scattering_mode_max": self.scattering_mode_max,
            "iranfreqmode": 1,
            "mc_scat_maxtauabs": self.mc_scat_maxtauabs,
            "tgas_eq_tdust": int(self.tgas_eq_tdust),
            "modified_random_walk": int(self.modified_random_walk),
            #"camera_maxdphi": 0.001,
            #"camera_nrrefine": 100,
            #"camera_refine_criterion": 0.1,
            # "camera_min_drr":0.001,
            #"camera_min_dangle": 0.0001,
            #"camera_max_dangle": 0.001,
            # "optimized_motion":1,
            # "camera_spher_cavity_relres":0.01,
            # "camera_diagnostics_subpix": 1,
            "lines_mode": 3 if getattr(self, "nonlte", False) else 1,
            "istar_sphere": 1,
        }

        self._save_input_file(
            "radmc3d.inp", *[f"{k} = {v}" for k, v in param_dict.items()]
        )

        #    if self.temp_mode == "mctherm":
        # B-16: remove_file needs the full path, not just filename.
        remove_file(os.path.join(self.radmc_dir, "gas_temperature.inp"))
        remove_file(os.path.join(self.radmc_dir, "dust_temperature.dat"))

    def set_lineobs_inpfiles(
        self,
    ):
        if self.molabun is None:
            raise ValueError("molabun is not set")
        self.set_mctherm_inpfiles()
        vr, vt, vp = self.model.vr, self.model.vt, self.model.vp
        nh2 = self._rhog / (2 * nc.amu / self.mfrac_H2)
        nmol = nh2 * self.molabun
        # n_mol = np.where(rr < self.mol_rlim, n_mol, 0)

        # set molcular number density
        self.set_numberdens(nmol)

        # set_mol_lines
        self._copy_from_storage(f"molecule_{self.molname}.inp")
        # NOTE: the nonlte branch below is unreachable — RadmcController.__init__
        # raises NotImplementedError when config.nonlte is truthy (D6).
        # The code is kept as a scaffold for a future LTE-only implementation.
        if getattr(self, "nonlte", False):  # pragma: no cover
            speclines = [f"{self.molname} leiden 0 0 2", "o-h2", "p-h2"]
            self.set_numberdens_collpartners(nh2)
        else:
            speclines = [f"{self.molname} leiden 0 0 0"]

        self._save_input_file(
            "lines.inp",
            "2",
            "1",  # len(speclines),
            "\n".join(speclines),
        )

        # set_gas_velocity
        self.set_velocity(vr, vt, vp)

        if self.model.vturb is not None:
            self.set_turbulence(self.model.vturb)

        if self.model.heatrate is not None:
            self.set_heatrate(self.model.heatrate)

    def _save_input_file(self, filename, *text_lines):
        filepath = os.path.join(self.radmc_dir, filename)
        mapped_lines = map(self._strfunc, text_lines)
        text = "\n".join(mapped_lines)

        with open(filepath, "w+") as f:
            f.write(text)
        fname = os.path.basename(f.name)
        dname = os.path.dirname(f.name)
        logger.info(f"Saved {fname} in {dname}")

    def _strfunc(self, line):
        if isinstance(line, str):
            return line
        if isinstance(line, int):
            return str(line)
        elif isinstance(line, float):
            return f"{line:20.15e}"
        elif isinstance(line, (tuple, list, np.ndarray)):
            if np.ndim(line) == 1:
                return " ".join([self._strfunc(var) for var in line])
        raise Exception("Unknown line format: ", line)

    def _copy_from_storage(self, filename):
        src = os.path.join(self.storage_dir, filename)
        dst = os.path.join(self.radmc_dir, filename)
        tools.filecopy(src, dst)

    def remove_radmcdir(self):
        shutil.rmtree(self.radmc_dir)

    #    def _set_constant_temperature(self, T_const=None, vlocal_fwhm=None):
    #        if T_const is not None:
    #            v_fwhm = np.sqrt(T_const * (16 * np.log(2) * nc.kB) / self.mol_mass)
    #        elif vlocal_fwhm is not None:
    #            T_const = m_mol * vlocal_fwhm**2 /(16 * np.log(2) * nc.kB )
    #        logger.info(f"Constant temperature is  {T_const}")
    #        logger.info(f"v_FWHM_kms is  {vlocal_fwhm/nc.kms}")
    #        T = np.full_like(self.rr, T_const)
    #        self.set_temperature(T)
    #
    #    def _set_userdefined_temperature(self):
    #        T = self.tfunc(self)
    #        T = T.clip(min=0.1, max=10000)
    #        self.set_temperature(T)

    def set_dust_density(self, rhod):
        self._save_input_file(
            "dust_density.inp",
            "1",
            f"{rhod.size:d}",
            "1",
            *rhod.ravel(order="F"),
        )

    def set_temperature(self, temp):
        """
        Set gas & dust temperature
        """
        ntot = len(temp.ravel())  # temp.size  # len(temp.ravel())
        self._save_input_file(
            "gas_temperature.inp",
            "1",
            f"{ntot:d}",
            *temp.ravel(order="F"),
        )
        self._save_input_file(
            "dust_temperature.dat",
            "1",
            f"{ntot:d}",
            "1",
            *temp.ravel(order="F"),
        )

    def set_numberdens(self, nmol):
        self._save_input_file(
            f"numberdens_{self.molname}.inp",
            "1",
            f"{nmol.size:d}",
            *nmol.ravel(order="F"),
        )

    def set_numberdens_collpartners(self, nh2, opratio=0.75):
        # NOTE: unreachable in current code — nonlte raises NotImplementedError in
        # __init__ (D6).  Kept as scaffold for future non-LTE implementation.
        self._save_input_file(
            "numberdens_o-h2.inp",
            "1",
            f"{nh2.size:d}",
            *(nh2 * opratio).ravel(order="F"),
        )

        self._save_input_file(
            "numberdens_p-h2.inp",
            "1",
            f"{nh2.size:d}",
            *(nh2 * (1 - opratio)).ravel(order="F"),
        )

    def set_velocity(self, vr, vt, vp):
        _zipped_vel = zip(
            vr.ravel(order="F"),
            vt.ravel(order="F"),
            vp.ravel(order="F"),
        )
        self._save_input_file("gas_velocity.inp", "1", f"{vr.size:d}", *_zipped_vel)

    def set_turbulence(self, vturb):
        self._save_input_file(
            "microturbulence.inp",
            "1",
            f"{vturb.size:d}",
            *vturb.ravel(order="F"),
        )

    def set_heatrate(self, heatrate):
        self._save_input_file(
            "heatsource.inp",
            "1",
            f"{heatrate.size:d}",
            *heatrate.ravel(order="F"),
        )

    def clean_radmc_dir(self):
        logger.info(f"Cleaning {self.radmc_dir} which now contains:")
        files = glob.glob(f"{self.radmc_dir}/*")
        if len(files) == 0:
            logger.info("    Nothing")
        else:
            for f in files:
                logger.info("    " + f)
        shutil.rmtree(self.radmc_dir)
        os.mkdir(self.radmc_dir)
        logger.info("Done")

    def run_mctherm(self):
        logger.info("Executing RADMC3D with mctherm mode")

        if not os.path.isdir(self.radmc_dir):
            msg = "radmc working directory not found: {self.radmc_dir}"
            raise FileNotFoundError(msg)

        tools.shell(
            f"radmc3d mctherm setthreads {self.n_thread}",
            cwd=self.radmc_dir,
            error_keyword=["ERROR", "STOP"],
            log_prefix="    ",
        )

        # C-58: wrap chdir in try/finally so cwd is always restored even on error.
        cwd = os.getcwd()
        try:
            os.chdir(self.radmc_dir)
            # ddens=False, dtemp=False, gdens=False, gtemp=False, gvel=False,
            self.rmcdata = rmca.readData(ispec=self.molname)  # radmc3dData()
            """
            radmcdata keys
            {
            'grid': <radmc3dPy.reggrid.radmc3dGrid object at 0x7f759c4f9460>,
            'octree': False,
            'rhodust': array([], dtype=float64),
            'dusttemp': array([], dtype=float64),
            'rhogas': array([], dtype=float64),
            'ndens_mol': array([], dtype=float64),
            'ndens_cp': array([], dtype=float64),
            'gasvel': array([], dtype=float64),
            'gastemp': array([], dtype=float64),
            'vturb': array([], dtype=float64),
            'taux': array([], dtype=float64),
            'tauy': array([], dtype=float64),
            'tauz': array([], dtype=float64),
            'sigmadust': array([], dtype=float64),
            'sigmagas': array([], dtype=float64)
            }
            """
        finally:
            os.chdir(cwd)

    def get_dust_density(self):
        return self.get_value("rhodust")

    def get_gas_density(self):
        return self.get_value("rhodust") / self.f_dg

    def get_mol_number_density(self):
        return self.get_value("ndens_mol")

    def get_velocity(self):
        vr = self.get_value("gasvel", index=0)
        vt = self.get_value("gasvel", index=1)
        vp = self.get_value("gasvel", index=2)
        return vr, vt, vp

    def get_dust_temperature(self):
        return self.get_value("dusttemp")

    def get_gas_temperature(self):
        if self.tgas_eq_tdust:
            Tgas = self.get_dust_temperature()
        else:
            Tgas = self.get_value("gastemp")
        return Tgas

    def get_value(self, key, index=0):
        val = getattr(self.rmcdata, key)
        if len(val) != 0:
            logger.debug(f"Setting {key}")
            return val[:, :, :, index]
        else:
            logger.info(f"Tried to set {key} but not found.")
            return None


"""
 Functions
"""


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed {file_path}")


def del_mctherm_files(dpath):
    remove_file(f"{dpath}/radmc3d.out")

## moved from create_radmc3d_input.py
class kappa:
    def __init__(self, kappa0_micron, beta):
        self.N_lam = 100
        self.kappa0_micron = kappa0_micron
        self.beta = beta
        self.lam_micron = np.geomspace(0.1, 1e5, self.N_lam)
        self.kappa_abs = self.kappa_abs_func()
        self.kappa_sca = self.kappa_sca_func()
        self.table = np.array([self.lam_micron, self.kappa_abs, self.kappa_sca]).T

    def kappa_abs_func(self):
        return self.kappa0_micron * self.lam_micron**self.beta

    def kappa_sca_func(self):
        return np.full_like(self.lam_micron, 0.0)

    def save(self):
        # print(f"{radmc_dir}/dustkappa_kappa0{self.kappa0_micron:.0e}_beta{self.beta}.inp")
        np.savetxt(
            f"{gpath.radmc_dir}/dustkappa_kappa0{self.kappa0_micron:.0e}_beta{self.beta}.inp",
            self.table,
            header=f"2\n{self.N_lam}\n",
            comments="",
        )


if __name__ == "__main__":
    for k0 in [1e4, 1e3]:
        for b in [0, -1, -2]:
            kappa(k0, b).save()
