import re
import subprocess
import os
import sys
import shutil
import numpy as np
import pandas
from pathlib import Path
from dataclasses import asdict
from scipy import interpolate, integrate

from .log import logger
from . import nconst as nc

def read_pickle(filepath):
    logger.debug("Reading pickle data")
    cls = pandas.read_pickle(filepath)
    logger.debug(cls)
    return cls


# def savefile(basename="file", mode="pickle", dpc=None, filepath=None):
def savefile(target, basename="file", mode="pickle", dirpath=None, filepath=None):
    if filepath is None:
        if dirpath is None:
            raise ValueError("savefile requires either dirpath or filepath")
        # output_ext = {"joblib": "jb", "pickle": "pkl", "fits": "fits"}[mode]
        output_ext = {"joblib": "jb", "pickle": "pkl"}[mode]
        filename = basename + "." + output_ext
        filepath = Path(dirpath) / filename
        if filepath.exists():
            logger.info(f"remove old fits file: {filepath}")
            os.remove(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

    if mode == "joblib":
        import joblib

        joblib.dump(target, filepath, compress=3)
        logger.info(f"Saved joblib file: {filepath}")

    elif mode == "pickle":
        pandas.to_pickle(target, filepath)
        logger.info(f"Saved pickle file: {filepath}")

    #    elif mode == "fits":
    #        self.data.writeFits(fname=filepath, dpc=dpc)
    #        logger.info(f"Saved fits file: {filepath}")

    else:
        logger.error("Unknown mode")
        exit()


def save_array(arrays, fname, mode="ascii", header="", fmt="%.15e"):
    data = np.array([a.ravel() for a in arrays]).T
    with open(fname, "wb") as f:
        np.savetxt(f, data, header=header, fmt=fmt)


def clean_radmcdir(dirpath):
    import shutil

    logger.info(f"Cleaning {dirpath}")
    """
    files = glob.glob(f"{self.radmc_dir}/*")
    if len(files) == 0:
        logger.info("    Nothing")
    else:
        for f in files:
            logger.info(" V   " + f)
    """
    shutil.rmtree(dirpath)
    # os.mkdir(self.radmc_dir)


#    logger.info("Done")

# del log
# del pkl
# del fits etc...


#######

def get_phi(x,y):
    return np.arctan2(y, x) % (2*np.pi)

def freq_to_vkms_array(freq, freq0):
    return nc.c / 1e5 * (freq0 - freq) / freq0


def freq_to_vkms(freq0, dfreq):
    return nc.c / 1e5 * dfreq / freq0


def vkms_to_freq(vkms, freq0):
    return (1 - vkms / nc.c * 1e5) * freq0


def make_array_center(xi):
    return 0.5 * (xi[0:-1] + xi[1:])


def make_array_interface(xc):
    xi_s = xc[0] - 0.5 * (xc[1] - xc[0])
    xi_e = xc[-1] + 0.5 * (xc[-1] - xc[-2])
    return np.concatenate([[xi_s], 0.5 * (xc[0:-1] + xc[1:]), [xi_e]], axis=0)

def x_cross_zero(x1, x2, y1, y2):
    return x1 + y1 * (x2 - x1) / (y1 - y2)


def find_roots(x, y1, y2):
    dy = np.array(y1) - np.array(y2)
    n = len(x) - 1
    return np.array(
        [
            x_cross_zero(x[i], x[i + 1], dy[i], dy[i + 1])
            for i in range(n)
            if dy[i] * dy[i + 1] <= 0
        ]
    )


def position_line(pangle_deg, L, dx=None, dy=None, offset=0):
    pa = pangle_deg * nc.deg2rad
    if (dx is None) or (dy is None):
        dl = L * 0.01
    else:
        dl = ((np.sin(pa) / dx) ** 2 + (np.cos(pa) / dy) ** 2) ** (-0.5)
    posax = np.arange(-L / 2, L / 2 + dl, dl)
    pos_x = -posax * np.sin(pa) - offset * np.cos(pa)
    pos_y = posax * np.cos(pa) - offset * np.sin(pa)
    return np.stack([pos_x, pos_y], axis=-1)


def get_position_line(pangle_deg, L, dx=None, dy=None, offset=0):
    pa = pangle_deg * nc.deg2rad
    if (dx is None) or (dy is None):
        dl = L * 0.01
    else:
        dl = ((np.sin(pa) / dx) ** 2 + (np.cos(pa) / dy) ** 2) ** (-0.5)
    posax = np.arange(-L / 2, L / 2 + dl, dl)
    pos_x = -posax * np.sin(pa) - offset * np.cos(pa)
    pos_y = posax * np.cos(pa) - offset * np.sin(pa)
    return {"points": np.stack([pos_x, pos_y], axis=-1), "posax": posax}


def take_midplane_average(model, value, dtheta=0.03, ntheta=1000, vabs=False):
    # rho_mid(r) = 1/(2rΔθ) int_-Δθ^Δθ ρ r dθ
    #            = 1/(2Δθ) int_-Δθ^Δθ ρ dθ
    dth = np.pi / 2 - model.tc_ax
    if dth[-1] >= 0:
        return value[:, -1, :]

    val = np.abs(value) if vabs else value
    func = interpolate.interp1d(dth, val, axis=1)
    dth_new = np.linspace(-dtheta, dtheta, ntheta)
    midvalue = integrate.simpson(func(dth_new), dth_new, axis=1) / (2 * dtheta)
    return midvalue


def take_horizontal_average(value):
    return np.average(value, axis=-1)


def show_used_memory():
    import psutil

    mem = psutil.virtual_memory()
    logger.info("Used memory = %.3f GiB", mem.used / (1024**3))


def compute_object_size(o, handlers={}):
    from itertools import chain
    from collections import deque

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def dataclass_str(self, _w=""):
    space = "  "
    txt = self.__class__.__name__
    txt += "("
    for k, v in asdict(self).items():
        txt += "\n"
        var = str(k) + " = "
        if isinstance(v, list):
            # txt += "\n"
            # txt += space*2 + str(v).replace('\n', '\n'+space*2)
            space_var = " " * len(var)
            txt += space + var + str(v).replace("\n", "\n" + space + space_var)

        elif isinstance(v, np.ndarray):
            info = f"array(shape={np.shape(v)}, min={np.min(v):{_w}g}, max={np.max(v):{_w}g})"
            txt += space + var + info

        elif isinstance(v, str):
            txt += space + var + f'"{str(v)}"'
        elif isinstance(v, float):
            txt += space + var + f"{v:{_w}g}"
        elif isinstance(v, int):
            txt += space + var + f"{v:d}"
        else:
            txt += space + var + str(v)
        txt += ","
    else:
        txt = txt[:-1]
        txt += "\n" + ")"
    return txt


def shell(
    cmd,
    cwd=None,
    log=True,
    dryrun=False,
    skip_error=False,
    error_keyword=None,
    logger=logger,
    log_prefix="",
):
    if dryrun:
        logger.info(f"(dryrun) {cmd}")
        return

    if cwd is None:
        cwd = os.getcwd()

    logger.info(f'Running shell command at {cwd}:\n    "{cmd}"')

    if error_keyword is not None:
        if isinstance(error_keyword, (tuple, list)):
            pattern = re.compile("|".join(error_keyword))
        else:
            pattern = re.compile(error_keyword)
    else:
        pattern = None

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    error_flag = False
    while True:
        _line = proc.stdout.readline().rstrip()
        if _line:
            if log:
                logger.info(log_prefix + _line)
            if pattern is not None and pattern.search(_line):
                error_flag = True

        if (_line == "") and (proc.poll() is not None):
            break

    retcode = proc.wait()

    if (retcode != 0) or error_flag:
        e = subprocess.CalledProcessError(retcode, cmd)
        if skip_error:
            logger.warning("Skip error:")
            logger.warning("    %s" % e)
        else:
            logger.error(e)
            raise e


def filecopy(src, dst, error_already_exist=False):
    dpath_src = os.path.abspath(os.path.dirname(src))
    fname_src = os.path.basename(src)
    dpath_dst = os.path.abspath(os.path.dirname(dst))

    logger.info(f"Copying {fname_src} from {dpath_src} into {dpath_dst}")

    if os.path.exists(dst):
        msg = f"File already exists: {dst}"
        if error_already_exist:
            logger.error(msg)
            raise Exception(msg)
        else:
            logger.warning(msg)
            logger.warning("Do nothing.")
            return

    try:
        shutil.copy2(src, dst)

    except FileNotFoundError as err:
        msg = ""
        if not os.path.exists(dpath_src):
            msg = f"Source directory not found: {dpath_src}"
        elif not os.path.exists(src):
            fpath_src = os.path.abspath(src)
            msg = f"Source file not found: {fpath_src}"
        elif not os.path.exists(dpath_dst):
            msg = f"Destination directory not found: {dpath_dst}"
        logger.error(msg)
        logger.error(err)
        raise

    except Exception as e:
        logger.error(e)
        raise

    else:
        logger.debug("Sucsess copying")


# def _instance_pickle(filepath, base=None):
#     if ".pkl" in filepath:
#         dtype = "pickle"
#
#     if dtype == "pickle":
#         cls = pd.read_pickle(filepath)
#         if base is None:
#             return cls
#         else:
#             for k, v in cls.__dict__.items():
#                 setattr(base, k, v)


def setattr_from_pickle(cls, filepath):
    readcls = pandas.read_pickle(filepath)
    for k, v in readcls.__dict__.items():
        setattr(cls, k, v)
