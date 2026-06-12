# import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from scipy import interpolate, integrate
from .log import logger
from . import nconst as nc
from . import gpath


def calc_streamline(
    model,
    *,
    pos0=None,
    r0_au=None,
    theta0_deg=None,
    names=None,
    units=None,
    variables=None,
    t_eval=None,
    t0_yr=10,
    dt_yr=10,
    rtol=1e-4,
    method="RK23",
    save=False,
    filename="stream",
    dpath=None,
    label=None,
):
    if names is None:
        names = []
    if units is None:
        units = []
    if variables is None:
        variables = []

    if (r0_au is not None) and (theta0_deg is not None):
        if np.isscalar(r0_au) and np.isscalar(theta0_deg):
            pos0_list = [(r0_au, theta0_deg)]
        elif np.isscalar(r0_au) and (not np.isscalar(theta0_deg)):
            pos0_list = [(r0_au, _theta0) for _theta0 in theta0_deg]
        elif (not np.isscalar(r0_au)) and np.isscalar(theta0_deg):
            pos0_list = [(_r0, theta0_deg) for _r0 in r0_au]
        elif (not np.isscalar(r0_au) or np.isscalar(theta0_deg)) and (
            len(r0_au) == len(theta0_deg)
        ):
            pos0_list = [(_r0, _theta0) for _r0, _theta0 in zip(r0_au, theta0_deg)]
        else:
            raise ValueError(f"Wrong type of r0 and theta0, {r0_au} {theta0_deg}")
    elif (pos0 is not None) and (np.shape(pos0) == (2,)):
        pos0_list = [pos0]
    elif (pos0 is not None) and (np.shape(pos0)[1] == 2):
        pos0_list = pos0
    else:
        raise ValueError(f"Wrong shape of pos0, {np.shape(pos0)}")

    pos0_list = np.array(pos0_list) * np.array([nc.au, nc.deg2rad])

    if t_eval is None:
        t_eval = np.arange(t0_yr, 1e6, dt_yr) * nc.yr

    # Copy to avoid mutating the caller's list (B-24 fix)
    _variables = list(variables)
    _variables += [(n, getattr(model, n), u) for n, u in zip(names, units)]

    slc = StreamlineCalculator2(
        model,
        t_eval=t_eval,
        rtol=rtol,
        method=method,
        variables=_variables,
    )
    logger.debug(f"Starting positions: {pos0_list}")
    slc.calc_streamlines(pos0_list)

    if save:
        slc.save_data(filename=filename, dpath=dpath, label=label)

    return slc.streamlines


@dataclass
class Streamline:
    pos0: np.ndarray
    t: np.ndarray
    R: np.ndarray
    z: np.ndarray
    vR: np.ndarray
    vz: np.ndarray
    variables: list = field(default_factory=list)

    def add_variable(self, name, value, unit):
        self.variables.append([name, value, unit])

    def save_data(self, filename="stream", dpath=None, label=None):
        # B-23 fix: use gpath.run_dir lazily instead of global
        if dpath is None:
            dpath = gpath.run_dir
        Path(str(dpath)).mkdir(exist_ok=True)

        poslabel = f"r{self.pos0[0]/nc.au:.0f}_th{np.rad2deg(self.pos0[1]):.0f}"

        header_list = ["t [s]", "R [cm]", "z [cm]", "vr [cm/s]", "vt [cm/s]"]

        _var_list = []
        if len(self.variables) >= 1:
            for name, value, unitname in self.variables:
                header_list.append(f"{name} [{unitname}]")
                _var_list.append(value[..., 0])

        header = " ".join([hd.rjust(19) for hd in header_list])
        stream_data = np.stack(
            (self.t, self.R, self.z, self.vR, self.vz, *_var_list), axis=-1
        )
        fpath = (
            f"{dpath}/{filename}_{poslabel}" + (f"_{label}" if label else "") + ".txt"
        )
        np.savetxt(
            fpath,
            stream_data,
            header=header[2:],
            fmt="%19.12e",
        )
        logger.info(f"saved trajecory data: {fpath}")


class StreamlineCalculator2:
    """

    example
    ------------
    slc = StreamlineCalculator2(model, t_eval=np.arange(0, 1e12, 1e8), rtol=1e-4, method="RK23", save=True)
    slc.calc_streamline(r0=1000 au, theta0=80)
    slc.save_data()

    """

    def __init__(
        self,
        model,
        t_eval=np.geomspace(1, 1e30, 500),
        rtol=1e-8,
        method="RK45",
        variables=[],
    ):
        self.r_ax = model.rc_ax
        self.t_ax = model.tc_ax
        self.vr_field = interpolate.RegularGridInterpolator(
            (self.r_ax, self.t_ax),
            model.vr[..., 0],
            bounds_error=False,
            fill_value=None,
        )
        self.vt_field = interpolate.RegularGridInterpolator(
            (self.r_ax, self.t_ax),
            model.vt[..., 0],
            bounds_error=False,
            fill_value=None,
        )
        self.t_eval = t_eval
        self.rtol = rtol
        self.method = method
        self.streamlines = []
        self._hit_midplane.terminal = True

        self._var_list = []
        if hasattr(model, "rhogas") and model.rhogas is not None:
            self.add_variable("rhogas", model.rhogas, "g cm^-3")
        if hasattr(model, "Tgas") and model.Tgas is not None:
            self.add_variable("Tgas", model.Tgas, "K")
        self._var_list += variables

    def add_variable(self, name, value, unitname):
        self._var_list.append([name, value, unitname])

    def calc_streamlines(self, pos0_list):
        for r0, theta0 in pos0_list:
            self.calc_streamline(r0, theta0)

    def calc_streamline(self, r0, theta0):
        if r0 > self.r_ax[-1]:
            logger.info(
                f"Too large starting radius (r0 = {r0/nc.au:.2f} au). "
                + f"Use r0 = max(r_ax) = {self.r_ax[-1]/nc.au:.2f} au instead."
            )
            pos0 = (self.r_ax[-1], theta0)
        else:
            pos0 = (r0, theta0)

        if self.t_eval is None:
            t_range = (1, 1e30)
        else:
            t_range = (self.t_eval[0], self.t_eval[-1])

        logger.info(
            f"Calculate a streamline from (r0[au], theta0[deg]) = ({pos0[0]/nc.au}, {pos0[1]*nc.rad2deg}) "
        )
        res = integrate.solve_ivp(
            self._func,
            t_range,
            np.array(pos0),
            method=self.method,
            events=self._hit_midplane,
            t_eval=self.t_eval,
            rtol=self.rtol,
        )
        res.t = np.hstack((res.t, res.t_events[0].T))
        res.y = np.hstack((res.y, res.y_events[0].T))
        self.add_streamline(res)

    def _func(self, t, pos):
        vr = self.vr_field(pos)[0]
        vt = self.vt_field(pos)[0]
        if np.isnan(pos[0]):
            raise Exception
        return np.array([vr, vt / pos[0]])

    @staticmethod
    def _hit_midplane(t, pos):
        return 0.5 * np.pi - pos[1]

    def add_streamline(self, res):
        R = res.y[0] * np.sin(res.y[1])
        z = res.y[0] * np.cos(res.y[1])
        vr = self.vr_field(res.y.T)
        vt = self.vt_field(res.y.T)
        vR = np.sin(res.y[1]) * vr + np.cos(res.y[1]) * vt
        vz = np.cos(res.y[1]) * vr - np.sin(res.y[1]) * vt
        sl = Streamline(res.y[:, 0], res.t, R, z, vR, vz)
        for name, value, unitname in self._var_list:
            vint = self._interpolate_along_streamline(value, res.y)
            sl.add_variable(name, vint, unitname)
        self.streamlines.append(sl)

    def _interpolate_along_streamline(self, value, points):
        return interpolate.interpn(
            (self.r_ax, self.t_ax),
            value,
            points.T,
            bounds_error=False,
            fill_value=None,
        )

    def save_data(self, filename="stream", dpath=None, label=None):
        for sl in self.streamlines:
            sl.save_data(filename=filename, dpath=dpath, label=label)
