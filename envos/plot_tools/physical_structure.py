import numpy as np
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as mc
from scipy import interpolate
from cycler import cycler
from . import plot_funcs as pfun
from .. import nconst as nc
from .. import log
from .. import tools

figext = "pdf"
cycle_ls = cycler(ls=["-", "--", ":", "-."]).by_key()['ls']
logger = log.logger

def plot_variable_meridional_map(
    model,
    variable_name,
    save_name=None,
    xlim=500,
    rinlim=0,
    clabel="",
    clog=False,
    save=True,
    streams=False,
    streams_option={},
    trajectories=False,
    trajectories_option={},
    dlv=None,
    aspect="equal",
    extend="min",
    **kwargs,
):
    """
    Plot a variable in a meridional map.

    Args:
        model: Model object containing the data.
        variable_name: Name of the variable to plot.
        save_name: Name to use when saving the plot.
        xlim: Maximum distance from the star in au.
        rinlim: Minimum radial distance in au.
        clabel: Label for the colorbar.
        clog: Boolean indicating whether to use a logarithmic scale for the colorbar.
        save: Boolean indicating whether to save the plot.
        streams: Boolean indicating whether to add streamlines to the plot.
        streams_option: Dictionary of options for streamlines.
        trajectories: Boolean indicating whether to add trajectories to the plot.
        trajectories_option: Dictionary of options for trajectories.
        dlv: Delta level value for contour levels.
        aspect: String defining the aspect ratio of the plot.
        extend: String defining how to extend the colorbar.
        **kwargs: Additional keyword arguments for plot_colormap function.
    """


    if xlim > np.max(model.rr[...,0]) / nc.au:
        xlim = np.max(model.rr[...,0]) / nc.au

    var_toridal_average = np.average(getattr(model, variable_name), axis=2)

    pfun.plot_colormap(
        model.R[..., 0] / nc.au,
        model.z[..., 0] / nc.au,
        var_toridal_average,
        # zregion=(modepl.R[...,0] < xlim * nc.au) & (model.z[...,0] < xlim * nc.au),
        zregion=(model.R[..., 0] < xlim * nc.au)
        & (model.z[..., 0] < xlim * nc.au)
        & (model.rr[..., 0] > rinlim * nc.au),
        clabel=clabel,
        clog=clog,
        dlv=dlv,
        aspect=aspect,
        extend=extend,
        **kwargs,
    )

    if trajectories:
        _trj_option = {"r0_au": xlim, "theta0_deg": 30}
        _trj_option.update(trajectories_option)
        pfun.add_trajectories(model, **_trj_option)

    if streams:
        pfun.add_streams(
            model,
            xlim,
            r0=np.sqrt(2) * xlim,
            use_mu0=hasattr(model, "mu0"),
            cavity_region=(model.rhogas[..., 0] == 0),
        )

    pfun.set_plot_format(xlim=(0, xlim), ylim=(0, xlim), xlb=r"$R$ [au]", ylb=r"$z$ [au]")

    if save:
        name = variable_name if save_name is None else save_name
        pfun.savefig(name + "." + figext)
    return


def plot_midplane_radial_profile(
    model,
    variable_name,
    save_name=None,
    save=True,
    xlim=(None, None),
    ylim=(None, None),
    xlb=r"$R$ [au]",
    ylb=None,
    legend=False,
    xlog=False,
    ylog=False,
    loglog=False,
    midplane_average=True,
    lines=[],
    **kwargs
    ):
    if isinstance(variable_name, str):
        var = getattr(model, variable_name)
    elif isinstance(variable_name, np.ndarray):
        var = variable_name
        variable_name = "None"

    if midplane_average:
        var_mid = model.get_midplane_profile(var)
    else:
        var = getattr(model, variable_name)
        var_mid = var[:, model.get_argmid(), :]

    plt.plot(model.rc_ax/nc.au, var_mid, **kwargs)
    for line in lines:
        #plt.plot(model.rc_ax/nc.au, line["y"], **line["opt"])
        plt.plot(**line)

    pfun.set_plot_format(
        xlim=xlim,
        ylim=ylim,
        xlb=xlb,
        ylb=ylb,
        legend=legend,
        xlog=xlog,
        ylog=ylog,
        loglog=loglog,
    )

    if save:
        name = variable_name if save_name is None else save_name
        pfun.savefig(name + "." + figext)

    return




#################################################################
#################################################################

def plot_rhogas_map(
    model,
    **kwargs
):
    _kwargs = {
        "clabel": r"Gas Density [g cm$^{{-3}}$]",
        "clog": True,
        "cname": "viridis",
        "dlv": 0.2,
    }
    _kwargs.update(kwargs)
    plot_variable_meridional_map(model, "rhogas", **_kwargs)


def plot_Tgas_map(
    model,
    **kwargs
):
    _kwargs = {
        "clabel": r"Gas Temperature [K]",
        "rinlim": 20,
        "cname": "inferno",
        "dlv": 10,
        "extend": "neither",
    }
    _kwargs.update(kwargs)
    plot_variable_meridional_map(model,"Tgas",**_kwargs)

##########################################################

def plot_rhogas_midplane_profile(
    model,
    xlb="Distance from Star [au]",
    ylb="Gas Density [g cm$^{-3}$]",
    xlim=(50, None),
    ylim=None,
    midplane_average=True,
    **kwargs
):
    plot_midplane_radial_profile(
        model,
        "rhogas",
        xlb=xlb,
        ylb=ylb,
        xlim=xlim,
        ylim=ylim,
        midplane_average=midplane_average,
        **kwargs
    )
    return

def plot_Tgas_midplane_profile(
    model,
    xlb="Distance from Star [au]",
    ylb="Gas Temperature [K]",
    xlim=(50, None),
    ylim=None,
    midplane_average=True,
    **kwargs
):
    plot_midplane_radial_profile(
        model,
        "Tgas",
        xlb=xlb,
        ylb=ylb,
        xlim=xlim,
        ylim=ylim,
        midplane_average=midplane_average,
        **kwargs
    )

    return

def plot_velocity_midplane_profile(
    model,
    xlim=(0, 400),
    ylim=(-0.5, 4),
    xlb="Distance from Star [au]",
    ylb="Velocity [km s$^{-1}$]",
    midplane_average=True,
    icycle=0,
    legend=True,
    save=True,
    **kwargs,
):
    index_mid = np.argmin(np.abs(model.tc_ax - np.pi / 2))
    gm = model.get_gasmask()[:, index_mid, 0]
    if midplane_average:
        vrmid = model.get_midplane_profile("vr")/1e5
        vtmid = model.get_midplane_profile("vt", vabs=True)/1e5
        vpmid = model.get_midplane_profile("vp")/1e5
    else:
        vrmid = model.vr[:,index_mid,:]/1e5
        vtmid = model.vt[:,index_mid,:]/1e5
        vpmid = model.vp[:,index_mid,:]/1e5
    rau = model.rc_ax / nc.au
    plt.plot(rau, -vrmid * gm, label=r"$- v_r$", ls=cycle_ls[icycle])
    plt.plot(rau, np.abs(vtmid) * gm, label=r"$v_{\theta}$", ls=cycle_ls[icycle+1])
    plt.plot(rau, np.abs(vpmid) * gm, label=r"$v_{\phi}$", ls=cycle_ls[icycle+2])

    pfun.set_plot_format(
        xlim=xlim,
        ylim=ylim,
        xlb=xlb,
        ylb=ylb,
        legend=legend,
        **kwargs,
    )

    if save:
        name = save_name if save_name is not None else "velocity_profile"
        pfun.savefig(name + "." + figext)

    return

def plot_density_velocity_midplane_profile(
    model, xlim=(50, 1000), ylim_rho=(1e-18, 1e-15), ylim_vel=(-0.01, 2.5), midplane_average=True
):
    plot_rhogas_midplane_profile(
        model,
        midplane_average=True,
        save=False,
        c="dimgray",
        xlim=xlim,
        ylim=ylim_rho,
        loglog=True,
        xlb="Distance from Star [au]",
        ylb="Gas Density [g cm$^{-3}$]",
        legend=False
    )

    plt.twinx()
    plt.plot(np.nan, ls="-", c="dimgray", label=r"$\rho$")
    plot_velocity_midplane_profile(
        model,
        midplane_average=True,
        save=False,
        icycle=1,
        xlim=xlim,
        ylim=ylim_vel,
        xlog=True,
        xlb="Distance from Star [au]",
        ylb="Velocity [km s$^{-1}$]",
        legend=False
    )
    plt.legend(handlelength=2, fontsize=16)
    plt.minorticks_on()
    pfun.savefig("v_dens_prof.pdf")


    return

def plot_angular_velocity_midplane_profile(
    model,
    xlb="Distance from Star [au]",
    ylb=r"Angular velocity [$s^{-1}$]",
    xlim=(50, None),
    ylim=None,
    midplane_average=True,
    loglog=True,
    **kwargs
):
    Omega = model.vp / model.R
    plot_midplane_radial_profile(
        model,
        variable_name=Omega,
        save_name="Omega_prof",
        xlb=xlb,
        ylb=ylb,
        xlim=xlim,
        ylim=ylim,
        loglog=loglog,
        midplane_average=midplane_average,
        **kwargs
    )

    return

######################################
#           Not updated yet          #
######################################

def plot_losvelocity_midplane_map(model, rlim=400, dvkms=0.2, mode="average", streams=False):
    rax = model.rc_ax
    tax = model.tc_ax
    axsym = True if len(model.pc_ax) == 1 else False
    pax = model.pc_ax if not axsym else np.linspace(0, 2*np.pi, 91)

    rr, tt, pp = np.meshgrid(rax, tax, pax, indexing="ij")
    R, z = rr * [np.sin(tt), np.cos(tt)]
    x, y = R * np.array([np.cos(pp), np.sin(pp)])
    cav = np.where(model.rhogas != 0, 1, 0)
    #vls = (x / R * model.vp + y / R * model.vR)*cav
    #vx = (- y / R * model.vp + x / R * model.vR)*cav
    vx = (model.vR * np.cos(pp) - model.vp * np.sin(pp))*cav
    vls = (model.vR * np.sin(pp) + model.vp * np.cos(pp))*cav

    xau_mid = tools.take_midplane_average(model, x)/nc.au
    yau_mid = tools.take_midplane_average(model, y)/nc.au
    Vkms_mid = tools.take_midplane_average(model, vls)/1e5
    vxkms_mid = tools.take_midplane_average(model, vx)/1e5

    Vmax = np.max(Vkms_mid)
    lvs_half = np.arange(dvkms/2, Vmax + dvkms, dvkms)
    lvs = np.concatenate([-lvs_half[::-1], lvs_half[0:]])

    img = plt.tricontourf(
        xau_mid.ravel(),
        yau_mid.ravel(),
        Vkms_mid.ravel(),
        lvs,
        cmap=plt.get_cmap("RdBu_r"),
        extend="both",
    )

    """ stream lines """
    r0 = rlim
    xau = np.linspace(-r0, r0, 1000)
    yau = np.linspace(-r0, r0, 1000)
    xx, yy = np.meshgrid(xau, yau)
    linedens = (10, 10)
    #                                    R               φ from y=0
    newgrid = np.stack([np.sqrt(xx ** 2 + yy ** 2), tools.get_phi(xx, yy)], axis=-1)
    axes = (rax / nc.au, pax)
    vx = interpolate.interpn(axes, vxkms_mid, newgrid, bounds_error=False, fill_value=None)
    vy = interpolate.interpn(axes, Vkms_mid, newgrid, bounds_error=False, fill_value=None)
    phis = np.linspace(0, 2*np.pi, 13)
    start_points = r0/2 * np.array([np.cos( phis ), np.sin( phis )]).T
    opt = {"density": linedens, "linewidth": .8, "color": (1,1,1,0.4), "arrowsize": 0.6} # "broken_streamlines":True}
    plt.streamplot(xau, yau, vx, vy, start_points=start_points, **opt)
    #plt.contourf(xau, yau, vy)

    rhomid = tools.take_midplane_average(model, model.rhogas)
    Tmid = tools.take_midplane_average(model, model.Tgas)
    emis = rhomid * Tmid if not axsym else np.tile(rhomid * Tmid, 91)
    emis_av = np.array([np.average(emis, axis=1)] * x.shape[2]).T

    z = (100 * emis_av / np.max(emis_av)).ravel()
    cond = np.where(z > 0.1, True, False)
    lvs = (
        np.array(
            [2 ** (-6), 2 ** (-5), 2 ** (-4), 2 ** (-3), 2 ** (-2), 2 ** (-1), 0.99999]
        )
        * 100
    )
    lss = ["solid"] * (len(lvs) - 1) + ["dashed"]
    cont = plt.tricontour(
        xau_mid.ravel()[cond],
        yau_mid.ravel()[cond],
        z[cond],
        levels=lvs,
        cmap=plt.get_cmap("Greys"),
        extend="both",  # "neither",
        linewidths=[0.7] * (len(lvs) - 1) + [1.1],
        alpha=0.9,
        linestyles=lss,
        norm=mc.LogNorm(vmin=np.min(lvs)),
    )

    try:
        del cont.collections[-1]._paths[1]
    except:
        pass
    plt.gca().set_aspect("equal", adjustable="box")
    #plt.xlabel("Offset along Major Axis $x$ [au]", fontsize=14)
    #plt.ylabel("Offset along Line-of-Sight $y$ [au]", fontsize=14)
    plt.xlabel("$x$ [au]", fontsize=14)
    plt.ylabel("$y$ [au]", fontsize=14)
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim, rlim)

    ticks_half = np.arange(0, Vmax + 0.6, 0.6)
    ticks = np.concatenate([-ticks_half[::-1], ticks_half[1:]])
    cbar = plt.colorbar(img, ticks=ticks, pad=0.02)

    #cbar.set_label(r"Line-of-sight Velocity $V$ [km s$^{-1}$]")
    cbar.set_label(r"Velocity along $y$-axis [km s$^{-1}$]")
    cbar.ax.minorticks_off()
    cont.clabel(fmt="%1.0f%%", fontsize=7, inline_spacing=20)
    plt.draw()
    pfun.savefig("v_los.pdf")


def plot_opacity():
    wav, kabs, kscat = np.loadtxt(
        "./run/radmc/dustkappa_MRN20.inp", skiprows=3, unpack=True
    )
    plt.plot(wav, kabs, c="k", lw=3)
    ax = plt.gca()
    ax.minorticks_on()
    plt.xlim(0.1, 1e5)
    plt.ylim(1e-4, 1e6)
    plt.xscale("log")
    plt.yscale("log")
    ax.yaxis.set_minor_locator(mt.LogLocator(numticks=15))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())  # add the custom ticks

    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel(r"Absorption Opacity [cm$^{2}$ g$^{-1}$]")
    pfun.savefig("opac.pdf")
