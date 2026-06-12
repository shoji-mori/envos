#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import envos


def main():
    config = envos.Config(
        run_dir="./run",
        n_thread=10,
        rau_in=10,
        rau_out=1000,
        dr_to_r=0.03,
        aspect_ratio=1,
        inenv="UCM",
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
        opac="MRN20",
        Lstar_Lsun=1.0,
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
    envos.plot_tools.plot_rhogas_map(model, streams=True)
    envos.plot_tools.plot_Tgas_map(model, streams=True)


if __name__ == "__main__":
    main()
