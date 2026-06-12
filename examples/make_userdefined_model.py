#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import envos

# from . import envos


def main():
    config = envos.Config(
        run_dir="./run",
        n_thread=10,
        nphot=1e6,
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
        opac="MRN20",
        Lstar_Lsun=1.0,
        molname="c18o",
        molabun=1e-20,
        iline=3,
        size_au=2000,
        pixsize_au=25,
        vfw_kms=6,
        dv_kms=0.01,
        beam_maj_au=50,
        beam_min_au=50,
        vreso_kms=0.2,
        beam_pa_deg=0,
        convmode="scipy",
        incl=0,
        posang=0,
        dpc=100,
    )

    # Printing a Config class shows all arguments set in the instance,
    # including implicitly set arguments.
    print(config)

    # ModelGenerator generates a physical model with calculating
    # density, velocity, and temperature structure. The simplest way
    # to pass the argumets to ModelGenerator is just to pass a config class.
    mg = envos.ModelGenerator(config)

    # Put uesr-defined grid axes into the model generator
    import numpy as np

    ri = np.geomspace(10, 1000, 30) * envos.nc.au
    ti = np.linspace(0, np.pi, 31)
    pi = np.linspace(0, 2 * np.pi, 60)
    mg.set_grid(ri=ri, ti=ti, pi=pi)

    # Put user-defined physical values into the model generator
    rr, tt, pp = mg.get_meshgrid()
    rho = (
        1e-17 * (1 + 0.5 * np.sin(rr / envos.nc.au / 10) * np.sin(pp)) * np.sin(tt) ** 2
    )  # np.sin(tt)**2 * np.sin(pp)**2 * (rr/envos.nc.au)**(-0.5)
    vr = 0 * np.ones(rr.shape)
    vt = 0 * np.ones(rr.shape)
    vp = np.sqrt(envos.nc.G * config.Ms_Msun * envos.nc.Msun / rr)
    mg.set_gas_density(rho=rho)
    mg.set_gas_velocity(vr=vr, vt=vt, vp=vp)
    mg.model.set_dust_density(f_dg=config.f_dg)

    # Calculate temperature structure.
    mg.calc_thermal_structure()

    # Generate the physical model as a `Circumstellar Model` object
    # to check and visualize the model made by the model generator.
    model = mg.get_model()

    # Once you save a data file as:
    model.save_pickle("model.pkl")
    # you can read the model as
    #   model = envos.read_model("run/model.pkl")

    # "Circumstellar Model" object is printed as a readable format.
    # For n-dimension array, to be readable, only shape, min, and max of the array are printed.
    # Learn what variables are available, and check if the calculation is done collectly.
    print(model)

    # Draw useful figures...
    envos.plot_tools.plot_rhogas_midplane_profile(model)
    envos.plot_tools.plot_velocity_midplane_profile(model)
    envos.plot_tools.plot_Tgas_midplane_profile(model)
    envos.plot_tools.plot_losvelocity_midplane_map(model)
    envos.plot_tools.plot_rhogas_map(model, streams=True)
    envos.plot_tools.plot_Tgas_map(model, streams=True)

    # Synthetic observations by ObsSimulator
    osim = envos.ObsSimulator(config)
    osim.set_model(model)
    odat = osim.observe_line()

    envos.plot_tools.plot_mom0_map(odat.get_mom0_map())


if __name__ == "__main__":
    main()
