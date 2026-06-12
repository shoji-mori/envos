"""
envos/obs/__init__.py — public API for the obs package.

Re-exports all symbols that were previously at envos.obs (flat module)
so that existing imports such as ``from envos.obs import ObsSimulator``
continue to work unchanged after the P2-B split.

Pickle compatibility: objects pickled with Cube.__module__ == "envos.obs"
can still be loaded because ``envos.obs.Cube`` resolves here via this
re-export.
"""

# simulator
from .simulator import gen_radmc_cmd, ObsSimulator, format_array, read_radmcdata

# convolve
from .convolve import convolve, Convolver

# data
from .data import (
    BaseObsData,
    Obreso,
    RefPos,
    Cube,
    Image,
    PVmap,
    minmaxargs,
)

# fits_io
from .fits_io import (
    save_fits,
    read_obsdata,
    read_image_fits,
    read_pv_fits,
    read_cube_fits,
    au2deg,
    deg2au,
    deg_to_ra,
    deg_to_dec,
    ra_to_deg,
    dec_to_deg,
)

__all__ = [
    # simulator
    "gen_radmc_cmd",
    "ObsSimulator",
    "format_array",
    "read_radmcdata",
    # convolve
    "convolve",
    "Convolver",
    # data
    "BaseObsData",
    "Obreso",
    "RefPos",
    "Cube",
    "Image",
    "PVmap",
    "minmaxargs",
    # fits_io
    "save_fits",
    "read_obsdata",
    "read_image_fits",
    "read_pv_fits",
    "read_cube_fits",
    "au2deg",
    "deg2au",
    "deg_to_ra",
    "deg_to_dec",
    "ra_to_deg",
    "dec_to_deg",
]
