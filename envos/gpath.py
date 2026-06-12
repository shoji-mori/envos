from pathlib import Path

"""
    This module contains global variables for the paths to the home directory,
    storage directory, run directory, radmc directory, figure directory, and
    log file.

"""


def set_homedir(path, update=True):
    """ Set the home directory to global variable, home_dir. """
    global home_dir
    home_dir = Path(path)
    if update:
        update_all_dirs_dependent_on_homedir(home_dir)


def set_rundir(path, update=True):
    """ Set the run directory to global variable, run_dir. """
    global run_dir
    run_dir = Path(path)
    make_dirs(run=path)
    if update:
        update_all_dirs_dependent_on_rundir(run_dir)


def set_storagedir(path):
    """ Set the storage directory to global variable, storage_dir. """
    global storage_dir
    storage_dir = Path(path)


def set_radmcdir(path):
    """ Set the radmc directory to global variable, radmc_dir. """
    global radmc_dir
    radmc_dir = Path(path)


def set_logfile(path):
    """ Set the logfile path to global variable, logfile. """
    global logfile
    logfile = Path(path)


def update_all_dirs_dependent_on_homedir(home_dir):
    """ Update all directories dependent on the home directory """
    global storage_dir, run_dir, radmc_dir, fig_dir, logfile
    storage_dir = home_dir / "radmc_storage"
    make_dirs(storage=storage_dir)
    run_dir = home_dir / "run"
    make_dirs(run=run_dir)
    update_all_dirs_dependent_on_rundir(run_dir)


def update_all_dirs_dependent_on_rundir(run_dir):
    """ Update all directories dependent on the run directory """
    global radmc_dir, fig_dir, logfile
    radmc_dir = run_dir / "radmc"
    make_dirs(radmc=radmc_dir)
    fig_dir = run_dir / "fig"
    make_dirs(fig=fig_dir)
    logfile = run_dir / "log.dat"

    from .log import change_rundir

    change_rundir(run_dir)


def make_dirs(radmc=None, run=None, storage=None, fig=None):
    """ Make directories for radmc, run, storage, and figure. """

    global storage_dir, run_dir, radmc_dir, fig_dir

    if storage is not None:
        storage_dir = Path(storage)
        storage_dir.mkdir(exist_ok=True, parents=True)

    if run is not None:
        run_dir = Path(run)
        run_dir.mkdir(exist_ok=True, parents=True)

    if radmc is not None:
        radmc_dir = Path(radmc)
        radmc_dir.mkdir(exist_ok=True, parents=True)

    if fig is not None:
        fig_dir = Path(fig)
        fig_dir.mkdir(exist_ok=True, parents=True)

def remove_radmcdir():
    """ Remove the radmc directory and all its contents. """
    import shutil
    shutil.rmtree(radmc_dir)


##############################################################################
# Set global variables when this module is imported for the first time.
this_file_path = Path(__file__)
working_dir = Path("./")
home_dir = this_file_path.parents[1]
storage_dir = home_dir / "storage"
run_dir = working_dir / "run"
radmc_dir = run_dir / "radmc"
fig_dir = run_dir / "fig"
logfile = run_dir / "log.dat"
