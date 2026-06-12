"""Backward-compatibility shim for the legacy ``envos.gpath`` global paths.

Historically this module held global path variables (``run_dir``,
``radmc_dir`` etc.) that were mutated as a side effect of constructing a
``Config``. Path management is now owned by ``Config`` (P2-A); this module
only survives for one release so that external scripts referencing
``envos.gpath.run_dir`` keep working, emitting a ``DeprecationWarning``.

Use the corresponding ``Config`` properties instead:
    run_dir   -> Config.run_path
    radmc_dir -> Config.radmc_path
    fig_dir   -> Config.fig_path
    storage_dir -> Config.storage_path
    logfile   -> Config.log_path
"""

import warnings
from pathlib import Path

# The most recently constructed Config registers itself here so that legacy
# attribute access reflects the active run.
_active_config = None


def _register(config):
    """Record the active Config so legacy attribute access can resolve."""
    global _active_config
    _active_config = config


# legacy attribute name -> Config property name (None = special-cased)
_LEGACY = {
    "run_dir": "run_path",
    "radmc_dir": "radmc_path",
    "fig_dir": "fig_path",
    "storage_dir": "storage_path",
    "logfile": "log_path",
    "home_dir": None,
}

# Defaults used when no Config has been constructed yet (mirror the historic
# module-level values).
_working_dir = Path("./")
_defaults = {
    "run_dir": _working_dir / "run",
    "radmc_dir": _working_dir / "run" / "radmc",
    "fig_dir": _working_dir / "run" / "fig",
    "storage_dir": Path(__file__).parents[1] / "storage",
    "logfile": _working_dir / "run" / "log.dat",
    "home_dir": Path(__file__).parents[1],
}


def _get(name):
    """Resolve a legacy path *without* emitting a DeprecationWarning.

    Intended for internal callers (e.g. log.py) that still rely on the
    legacy default until their own refactor lands (P2-C).
    """
    if name == "home_dir":
        return _defaults["home_dir"]
    if _active_config is not None and _LEGACY.get(name):
        return getattr(_active_config, _LEGACY[name])
    return _defaults[name]


def __getattr__(name):
    # PEP 562 module-level __getattr__: only invoked for names not found as
    # real module attributes.
    if name not in _LEGACY:
        raise AttributeError(f"module 'envos.gpath' has no attribute {name!r}")
    warnings.warn(
        f"envos.gpath.{name} is deprecated; use Config.{_LEGACY[name] or 'run_path'}",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get(name)
