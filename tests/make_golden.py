"""
Golden data generation script for envos regression tests (P0-4).

Usage
-----
Run from the repository root (worktree cwd):

    MPLBACKEND=Agg /home/user/envos/.venv/bin/python tests/make_golden.py

Or with an explicit storage override:

    MPLBACKEND=Agg /home/user/envos/.venv/bin/python tests/make_golden.py \\
        [--skip-g2]

Re-generation rules
-------------------
- Re-generate whenever physics code changes (⚠ "physics-change" PR).
- On a physics-change PR, regenerate golden data **in the same PR** and record
  the relative change in the PR description.  Tag the PR with `physics-change`.
- Always verify that the new golden data passes `pytest tests/test_golden.py`
  before merging.
- After re-generating, commit both the updated *.npz / *.json files **and**
  tscsol.pkl together so that G2 tests remain reproducible without re-solving.

⚠ Important: This script is the authoritative source of golden data.
  The test file (test_golden.py) imports shared helpers from golden_helpers.py
  so that make_golden.py and test_golden.py exercise identical code paths.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Ensure the worktree sources are importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.golden_helpers import (  # noqa: E402
    GOLDEN_DIR,
    G3A_KWARGS,
    G3B_KWARGS,
    build_g1_model,
    build_g2_model,
    build_g3,
    extract_g1_arrays,
)

import numpy as np


def generate_g1(run_dir):
    """Generate G1 golden data (UCM model arrays -> .npz)."""
    print("Generating G1 (UCM model)...")
    model = build_g1_model(run_dir)
    arrays = extract_g1_arrays(model)
    out_path = GOLDEN_DIR / "g1_ucm.npz"
    np.savez(out_path, **arrays)
    print(f"  Saved {out_path}  ({out_path.stat().st_size} bytes)")
    return out_path


def generate_g2(run_dir, storage_dir):
    """
    Generate G2 golden data (UCM+TSC model arrays -> .npz).

    tscsol.pkl is written to storage_dir (= tests/golden/) so subsequent
    test runs can load it without re-solving the ODE system.

    This may take several minutes the first time (ODE solve).
    """
    print("Generating G2 (UCM+TSC model, may take a few minutes)...")
    model = build_g2_model(run_dir, storage_dir)
    arrays = extract_g1_arrays(model)
    out_path = GOLDEN_DIR / "g2_ucm_tsc.npz"
    np.savez(out_path, **arrays)
    print(f"  Saved {out_path}  ({out_path.stat().st_size} bytes)")

    tscsol_path = GOLDEN_DIR / "tscsol.pkl"
    if tscsol_path.exists():
        print(f"  tscsol.pkl present  ({tscsol_path.stat().st_size} bytes)")
    else:
        print("  WARNING: tscsol.pkl was not written — check storage_dir plumbing")

    return out_path


def generate_g3():
    """Generate G3 golden data (PhysicalParameters scalars -> .json)."""
    print("Generating G3 (PhysicalParameters scalars)...")
    scalars_a = build_g3(G3A_KWARGS)
    scalars_b = build_g3(G3B_KWARGS)

    out_a = GOLDEN_DIR / "g3a_ppar.json"
    out_b = GOLDEN_DIR / "g3b_ppar.json"

    with open(out_a, "w") as f:
        json.dump(scalars_a, f, indent=2)
    with open(out_b, "w") as f:
        json.dump(scalars_b, f, indent=2)

    print(f"  Saved {out_a}  ({out_a.stat().st_size} bytes)")
    print(f"  Saved {out_b}  ({out_b.stat().st_size} bytes)")
    return out_a, out_b


def main():
    parser = argparse.ArgumentParser(description="Generate envos golden test data")
    parser.add_argument(
        "--skip-g2",
        action="store_true",
        help="Skip G2 (TSC model, slow ODE solve)",
    )
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as run_dir:
        generate_g1(run_dir)
        generate_g3()

        if args.skip_g2:
            print("Skipping G2 (--skip-g2 flag set)")
        else:
            try:
                generate_g2(run_dir, GOLDEN_DIR)
            except Exception as exc:
                print(f"\n*** G2 FAILED: {type(exc).__name__}: {exc} ***")
                print("G1 and G3 were generated successfully.")
                print("G2 is BLOCKED — see completion report.")
                sys.exit(1)

    print("\nDone. Golden files in", GOLDEN_DIR)
    for p in sorted(GOLDEN_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
