#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import numpy as np
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
import envos
import envos.nconst as nc


def main(conf):
    do_mori2023(conf)
    # do_pmap_survey_for_mass_estimate(conf)
    # do_pmap_survey_for_cube_correlation(conf)
    # do_2region_test(conf)
    # do_ps_survays()


#####################################################################
# Doing functions                                                   #
#####################################################################


def do_mori2023(conf):
    def _obs(conf, label, refcube=None, **kwargs):
        _conf = conf.replaced(
            run_dir=f"./run_m23/run_{label}",
            n_thread=4,
        )
        synobs(
            _conf,
            filename=f"lobs_{label}.pkl",
            refdata=refcube,
            **kwargs,
        )

    refcube = _obs(conf, "fid")
    refcube = envos.read_obsdata(
        Path(__file__).parent / "run_m23" / "run_fid/lobs_fid.pkl"
    )

    _obs(conf.replaced(Ms_Msun=0.45, CR_au=74), "Aso17", refcube)
    _obs(conf.replaced(inenv="Simple", cavangle_deg=80), "SB", refcube)

    for M in [0.1, 0.3]:
        _obs(conf.replaced(Ms_Msun=M), f"M{M}", refcube)

    for cr in [100, 300]:
        _obs(conf.replaced(CR_au=cr), f"cr{cr}", refcube)

    for incl in [30, 60]:
        _obs(conf.replaced(incl=incl), f"i{incl}", refcube)

    for ca in [0, 80]:
        _obs(conf.replaced(cavangle_deg=ca), f"cav{ca}", refcube)

    """
    for T0 in [10, 30, 100]:
        Tfunc = lambda m: T0 * (m.rr/10/envos.nc.au)
        _obs(conf, f"T{T0}-p0", refcube, Tfunc=Tfunc)

    for p in [-0.5, -1]:
        Tfunc = lambda m: 30 * (m.rr/10/envos.nc.au)**p
        _obs(conf, f"T30-p{p}", refcube, Tfunc=Tfunc)
    """


def do_pmap_survey_for_mass_estimate(
    conf,
):  # for mass estimate , not for cube correlation
    def task(_conf, M, CR, i, model="UCM"):
        label = f"{model}_M{M}_cr{CR}_i{i}"
        Vmax = np.sqrt(2 * nc.G * M * nc.Msun / (CR * nc.au)) / 1e5
        _conf = _conf.replaced(
            Ms_Msun=M,
            CR_au=CR,
            incl=i,
            inenv=model,
            run_dir=f"./pmap2/run_{label}",
            n_thread=1,
            # convmode="null",
            dr_to_r=0.01,
            size_au=4 * CR,
            vfw_kms=1.5 * Vmax * 2,
            #   nphot=1e7,
        )
        synobs(
            _conf,
            vkms_pvd=Vmax * 1.5,
            xlim_pvd=CR * 2,
            skip_if_exist=False,  # True
            skip_if_not_exist=True,
            calc_model=False,
            calc_obs=False,
            plot_model_structure=0,
        )

    Mlist = np.geomspace(0.03, 3, 10)
    crlist = np.geomspace(40, 400, 10)
    ilist = [90, 60]
    params = np.array(list(product(Mlist, crlist, ilist)))

    logging.info("Start")
    Parallel(n_jobs=-1)([delayed(task)(conf, *p) for p in params])
    logging.info("End")


def do_pmap_survey_for_cube_correlation(
    conf, single=False
):  # for mass estimate , not for cube correlation
    single = 0

    def task(_conf, M, CR, model):
        label = f"{model}_M{M}_cr{CR}"
        if model == "Simple":
            _conf = _conf.replaced(cavangle_deg=80)

        _conf = _conf.replaced(
            Ms_Msun=M,
            CR_au=CR,
            inenv=model,
            run_dir=f"./cubes5/run_{label}",
            n_thread=1,
            dr_to_r=0.02,
        )
        synobs(
            _conf,
            skip_if_exist=1,
            skip_if_not_exist=0,
            save_mg=0,
            calc_model=1,
            calc_obs=1,
            plot_model_structure=0,
        )

    model = ["UCM", "Simple"]
    Mlist = np.linspace(0, 0.6, 30)  # 0.025
    crlist = np.linspace(0, 600, 30)  # 15
    params = product(Mlist[1:], crlist[1:], model)

    if single:
        for p in params:
            task(conf, *p)
    else:
        logging.info("Start")
        Parallel(n_jobs=16)([delayed(task)(conf, *p) for p in params])
        logging.info("End")


#
# Two region test
#
def do_2region_test(conf):
    crau = conf.CR_au
    Rb_au = crau + 50
    refcube = synobs(
        conf.replaced(run_dir="run/run_ringall"),
        filename="lobs_ringall.pkl",
        calc_model=False,
        calc_obs=False,
    )

    def rhofunc_after_thermal1(model):
        return np.where(model.R / envos.nc.au <= Rb_au, model.rhogas, 0)

    def rhofunc_after_thermal2(model):
        return np.where(model.R / envos.nc.au > Rb_au, model.rhogas, 0)

    for tag, func in (("in", rhofunc_after_thermal1), ("out", rhofunc_after_thermal2)):
        synobs(
            conf.replaced(run_dir=f"run/run_Rb{Rb_au}{tag}"),
            rhofunc_after_thermal=func,
            filename=f"lobs_Rb{Rb_au}{tag}.pkl",
            refdata=refcube,
            #    calc_model=False, calc_obs=False
        )


#
# PS survays
#
def do_ps_survays(conf):
    def wrapfunc_model(M, CR, i, model="UCM"):
        label = f"{model}_M{M}_cr{CR}_i{i}"
        Vmax = np.sqrt(2 * nc.G * M * nc.Msun / (CR * nc.au)) / 1e5
        _conf = conf.replaced(
            Ms_Msun=M,
            CR_au=CR,
            incl=i,
            inenv=model,
            run_dir=f"psurvey9/run_{label}",
            n_thread=1,
            # convmode="",
            dr_to_r=0.01,
            size_au=4 * CR,
            vfw_kms=1.5 * Vmax * 2,
            nphot=1e7,
        )
        synobs(_conf, vkms_pvd=Vmax * 1.5, xlim_pvd=CR * 2, skip_if_exist=False)

    Mlist = np.geomspace(0.03, 3, 20)
    crlist = np.geomspace(40, 400, 20)
    ilist = [95]  # np.arange(90, 0, -20)
    res = [wrapfunc_model(*args) for args in product(Mlist, crlist, ilist)]
    return res


###########################################################

conf_default = envos.Config(
    run_dir="./run",
    n_thread=4,
    rau_in=10,
    rau_out=1000,
    theta_out=np.pi,
    dr_to_r=0.02,
    aspect_ratio=1,
    nphi=91,
    nphot=1e7,
    CR_au=200,  # 290, #124.623,  #157.9,    #157.36,
    Ms_Msun=0.2,  # 0.154, #0.192226, #0.1439, #0.17223,
    T=None,
    Mdot_smpy=4.5e-6,
    cavangle_deg=45,
    f_dg=0.01,
    opac="MRN20",
    Lstar_Lsun=2.75,
    molname="c3h2",
    molabun=1e-20,
    iline=69,
    size_au=2000,
    pixsize_au=17.5,
    vfw_kms=8,
    dv_kms=0.05,
    beam_maj_au=109.90,
    beam_min_au=94.771,
    vreso_kms=0.1469,
    beam_pa_deg=179.3,
    convmode="scipy",
    incl=95,
    posang=95,
    dpc=140,
)


def synobs(
    conf,
    filename="lineobs.pkl",
    *,
    rhofunc_before_thermal=None,
    rhofunc_after_thermal=None,
    Tfunc=None,
    p=None,
    skip_if_exist=False,
    skip_if_not_exist=False,
    pa_pv=0,
    vkms_pvd=2.0,
    xlim_pvd=700,
    obsdust=False,
    refdata=None,
    save_mg=True,
    calc_model=True,
    calc_obs=True,
    plot_model_structure=True,
    plot_pv=True,
    plot_pv_loglog=True,
    plot_pv_for_mass_estimate=True,
    plot_mom0_map=True,
):

    """
    This function performs a synthetic observation, which is a simulation of an observation that calculates radiative transfer for each pixel.

    Parameters:
    conf: Configuration object for the observation.
    filename: Name of the output file.
    rhofunc_before_thermal: Function to modify the density structure of the model before thermal calculations.
    rhofunc_after_thermal: Function to modify the density structure of the model after thermal calculations.
    Tfunc: Function to modify the temperature structure of the model.
    p: Additional parameters.
    skip_if_exist: If True, skip the observation if the output file already exists.
    skip_if_not_exist: If True, skip the observation if the output file does not exist.
    pa_pv: Position angle for the position-velocity diagram.
    vkms_pvd: Velocity for the position-velocity diagram in km/s.
    xlim_pvd: Limits for the x-axis of the position-velocity diagram.
    obsdust: If True, observe dust instead of gas.
    refdata: Reference data for comparison.
    save_mg: If True, save the model generator object.
    calc_model: If True, calculate the model.
    calc_obs: If True, calculate the observation.
    plot_model_structure: If True, plot the structure of the model.
    plot_pv: If True, plot the position-velocity diagram.
    plot_pv_loglog: If True, plot the position-velocity diagram in log-log scale.
    plot_pv_for_mass_estimate: If True, plot the position-velocity diagram for mass estimation.
    plot_mom0_map: If True, plot the zeroth moment map.
    """



    """
    check if file exists and if so skip synobs
    """
    savefile = conf.run_path / filename
    if savefile.exists() and skip_if_exist:
        print(
            f"Skip syonbs for {conf.run_path/filename} because it already exists"
        )
        return

    if (not savefile.exists()) and skip_if_not_exist:
        print(
            f"Skip syonbs for {conf.run_path/filename} because it does not exist"
        )
        return

    print(f"Start synobs to make {filename}")


    """
    set logging
    """
    envos.log.setup(logfile=conf.log_path)
    conf.log()


    """
    calc kinematic structure of model
    """
    if not calc_model:
        print("Try to read ", conf.run_path / "mg.pkl")

    if (not calc_model) and (conf.run_path / "mg.pkl").exists():
        mg = envos.ModelGenerator(readfile=conf.run_path / "mg.pkl")

    elif calc_model:
        mg = envos.ModelGenerator(conf)
        mg.calc_kinematic_structure()

        if rhofunc_before_thermal is not None:
            mg.model.rhogas = rhofunc_before_thermal(mg.model)

        if conf.inenv == "Simple":
            mg.model.Tgas = np.ones_like(mg.model.rhogas) * 30
        else:
            if Tfunc is None:
                mg.calc_thermal_structure()
            else:
                mg.model.Tgas = Tfunc(mg.model)

        if rhofunc_after_thermal is not None:
            mg.model.rhogas = rhofunc_after_thermal(mg.model)

        if save_mg:
            mg.save()
    else:
        raise FileNotFoundError(
            f"mg.pkl not found at {conf.run_path / 'mg.pkl'} and calc_model=False"
        )


    """
    plot model structure
    """
    if plot_model_structure:
        model = mg.get_model()
        envos.plot_tools.plot_rhogas_map(model, streams=True)
        envos.plot_tools.plot_Tgas_map(model, streams=True)
        envos.plot_tools.plot_velocity_midplane_profile(model)
        envos.plot_tools.plot_losvelocity_midplane_map(
            model, rlim=500, dvkms=0.2, streams=True
        )
        envos.plot_tools.plot_density_velocity_midplane_profile(model)


    """
    calc line emission from model 
    """
    if calc_obs:
        osim = envos.ObsSimulator(conf)
        model = mg.get_model()
        osim.set_model(model)
        cube = osim.observe_line(obsdust=obsdust)
        cube.save(filepath=savefile)
    else:
        cube = envos.read_obsdata(savefile)

    pvopt = {
        "figsize": None,
        "xlim": (-xlim_pvd, xlim_pvd),
        "ylim": (-vkms_pvd, vkms_pvd),
        "f_crit": 0.3,
        "incl": conf.incl,
        "smooth_contour": True,
    }


    """
    plot PV diagrams
    """
    if plot_pv or plot_pv_for_mass_estimate or plot_pv_loglog:
        pv = cube.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv)

    if plot_pv:
        envos.plot_tools.plot_pvdiagram(
            pv,
            mass_estimate=False,
            out="pvd.pdf",
            norm="max",
            Iunit=r"$I_{V,{\rm max}}$",
            **pvopt,
        )

    if plot_pv_for_mass_estimate:
        envos.plot_tools.plot_pvdiagram(
            pv, mass_estimate=True, out="pvd_mass.pdf", **pvopt
        )

    if plot_pv_loglog:
        envos.plot_tools.plot_pvdiagram(
            pv, mass_estimate=False, loglog=True, out="pvd_loglog.pdf", **pvopt
        )

    if refdata is not None:
        pv = cube.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv)
        refpv = refdata.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv)
        pv.norm_I() # normalize intensity in pv diagram by max intensity
        refpv.norm_I() # same for reference data
        envos.plot_tools.plot_pvdiagram(
            pv,
            refpv=refpv,
            out="pvd_comp.pdf",
            mass_estimate=False,
            norm=None,
            Iunit=r"$I^{\prime}_{V,{\rm max}}$",
            clim=(0, 1),
            **pvopt,
        )


    """
    plot mom0 map 
    """ 
    if plot_mom0_map:
        img = cube.get_mom0_map()
        img.trim(xlim=[-900, 900])
        envos.plot_tools.plot_mom0_map(
            img,
            arrow_angle_deg=conf.posang - 90 + pa_pv,
            norm="max",
        )


    envos.tools.clean_radmcdir(conf.radmc_path)
    return cube

if __name__ == "__main__":
    main(conf_default)
