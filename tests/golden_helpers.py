"""
Shared helpers for golden data generation and verification (P0-4).

Both make_golden.py and test_golden.py import from here to guarantee they
exercise exactly the same code paths — the only difference between generation
and verification is whether the results are written or compared.
"""

from pathlib import Path
import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"

# ---------------------------------------------------------------------------
# G1 spec: UCM model, radmc3d not required
# ---------------------------------------------------------------------------
G1_CONFIG = dict(
    rau_in=10,
    rau_out=1000,
    nr=30,
    ntheta=20,
    nphi=1,
    CR_au=100,
    Ms_Msun=0.3,
    T=10,
    cavangle_deg=45,
)
G1_ARRAYS = ("rhogas", "vr", "vt", "vp", "rc_ax", "tc_ax")

# ---------------------------------------------------------------------------
# G3 spec: PhysicalParameters scalar values (JSON)
# ---------------------------------------------------------------------------
G3_ATTRS = ("Mdot", "cs", "t", "CR", "jmid", "Omega")

G3A_KWARGS = dict(T=10, CR_au=100, Ms_Msun=0.3)
G3B_KWARGS = dict(Mdot_smpy=4.5e-6, Ms_Msun=0.2, CR_au=200)

# ---------------------------------------------------------------------------
# G4 spec: UCM + powerlaw disk (P1-9 physics-change: replace-mode disk synthesis)
# ---------------------------------------------------------------------------
G4_CONFIG = dict(
    rau_in=10,
    rau_out=1000,
    nr=30,
    ntheta=20,
    nphi=1,
    CR_au=100,
    Ms_Msun=0.3,
    T=10,
    cavangle_deg=45,
    disk="powerlaw",
)
# Same arrays as G1 — rhogas captures the replace-mode disk synthesis.
G4_ARRAYS = G1_ARRAYS


# ---------------------------------------------------------------------------
# Build functions (shared construction logic)
# ---------------------------------------------------------------------------

def build_g1_model(run_dir):
    """
    Construct the G1 UCM circumstellar model.

    Parameters
    ----------
    run_dir : str or Path
        A writable directory for Config's run_dir (used by gpath).

    Returns
    -------
    CircumstellarModel
    """
    from envos.config import Config
    from envos.model_generator import ModelGenerator

    config = Config(run_dir=str(run_dir), **G1_CONFIG)
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()


def build_g2_model(run_dir, storage_dir):
    """
    Construct the G2 UCM+TSC model.

    storage_dir is set as Config's storage_dir so that tsc.get_tsc()
    reads/writes tscsol.pkl from that directory rather than the global default.

    Parameters
    ----------
    run_dir : str or Path
    storage_dir : str or Path
        Directory where tscsol.pkl lives (or will be written).

    Returns
    -------
    CircumstellarModel
    """
    from envos.config import Config
    from envos.model_generator import ModelGenerator

    config = Config(
        run_dir=str(run_dir),
        storage_dir=str(storage_dir),
        outenv="TSC",
        **G1_CONFIG,
    )
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()


def build_g3(kwargs):
    """
    Construct a PhysicalParameters instance and return a dict of scalar attrs.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments for PhysicalParameters.

    Returns
    -------
    dict mapping attr name -> float
    """
    from envos.physical_params import PhysicalParameters

    pp = PhysicalParameters(**kwargs)
    return {attr: float(getattr(pp, attr)) for attr in G3_ATTRS}


def build_g4_model(run_dir):
    """
    Construct the G4 UCM + powerlaw disk model.

    This uses the replace-mode disk synthesis introduced in P1-9 (D1):
    ``rho[cond] = disk.rho[cond]`` where ``cond = rho < disk.rho``.

    Parameters
    ----------
    run_dir : str or Path
        A writable directory for Config's run_dir (used by gpath).

    Returns
    -------
    CircumstellarModel
    """
    from envos.config import Config
    from envos.model_generator import ModelGenerator

    config = Config(run_dir=str(run_dir), **G4_CONFIG)
    mg = ModelGenerator(config)
    mg.calc_kinematic_structure()
    return mg.get_model()


def extract_g1_arrays(model):
    """Extract the G1 array fields from a CircumstellarModel."""
    data = {}
    for name in G1_ARRAYS:
        arr = getattr(model, name)
        # rc_ax / tc_ax live directly on the model; rhogas/vr/vt/vp too
        data[name] = np.asarray(arr)
    return data


# G4 uses the same array fields as G1.
extract_g4_arrays = extract_g1_arrays
