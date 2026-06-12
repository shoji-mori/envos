#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import envos


def main():
    config = envos.Config(
        run_dir="./run",
        n_thread=10,
        rau_in=10,
        rau_out=10000,
        dr_to_r=0.01,
        aspect_ratio=1,
        inenv="UCM",
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
        opac="MRN20",
        Lstar_Lsun=1.0,
        nphot=100,
    )

    # Printing a Config class shows all arguments set in the instance,
    # including implicitly set arguments.
    print(config)

    # ModelGenerator generates a physical model with calculating
    # density, velocity, and temperature structure. The simplest way
    # to pass the argumets to ModelGenerator is just to pass a config class.
    mg = envos.ModelGenerator(config)

    # Calculate density and velocity structure.
    mg.calc_kinematic_structure()

    # Calculate temperature structure.
    mg.calc_thermal_structure()

    # Generate the physical model as a `Circumstellar Model` object.
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


    # Calculate the column density along r-direction and z-direction
    # After that, model will have colr and colz in cgs units
    model.calc_column_density()

    # Before plotting, set the common potions for using trajectories
    # Note that streams option shows the streamlines based only on the velocity field
    # but the trajectories option shows the path from which we set the initial positions.
    theta0 = [80, 70, 60, 50, 40]
    option = {
        "r0_au": model.ppar.cs * model.ppar.t / envos.nc.au,
        "theta0_deg":theta0,
    }

    # calc_streamline and trajectories_option has the same names of the arguments.
    # Sorry for the confusiing names... I will change sooner or later.
    # Only when save option is turned on, trajectory data is output
    envos.streamline.calc_streamline(
        model, save=True, label="direct",
        names=["colr", "colz"], units=["g cm^-2","g cm^-2"], **option
    )

    envos.plot_tools.plot_rhogas_map(
        model, streams=True, trajectories=True,
        trajectories_option=option
    )
    envos.plot_tools.plot_Tgas_map(
        model, streams=True, trajectories=True,
        trajectories_option=option
    )
    # One can use plot_tools.plot_variable_meridional_map
    # for an arbitrary variable that is contained in model class
    # In this case, the str "colr" are passed to the function.

    envos.plot_tools.plot_variable_meridional_map(
        model, "colr",
        clabel=r"Column Desnity along $r$-Direction [g cm$^{-2}$]",
        streams=True, trajectories=True, trajectories_option=option
    )
    envos.plot_tools.plot_variable_meridional_map(
        model, "colz",
        clabel=r"Column Desnity along $z$-Direction [g cm$^{-2}$]",
        streams=True, trajectories=True, trajectories_option=option
    )

    # One can save the physical value with text data by using save_arrays(arrays, filename).
    # `arrays` is str of a variable name, or str list of variable names.
    # model.save_arrays("rhogas", "rhogas.txt")
    model.save_arrays(["rhogas","Tgas","colr","colz"], "data.txt")


if __name__ == "__main__":
    main()
