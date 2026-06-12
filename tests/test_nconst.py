"""
Unit tests for envos/nconst.py physical constants (P1-1).

Verifies that the Julian year correction (3.1454e7 -> 3.15576e7 s)
is correctly reflected in the derived constants.
"""

import pytest
import envos.nconst as nc


def test_year_is_julian_year():
    """nc.year must equal exactly 365.25 * 86400 seconds (Julian year, IAU)."""
    assert nc.year / 86400 == pytest.approx(365.25)


def test_yr_alias():
    """nc.yr must be identical to nc.year."""
    assert nc.yr is nc.year or nc.yr == nc.year


def test_myr_is_mega_year():
    """nc.Myr must equal 1e6 * nc.year."""
    assert nc.Myr == pytest.approx(nc.year * 1e6)


def test_smpy_is_msun_per_year():
    """nc.smpy (solar masses per year) must equal nc.Msun / nc.year."""
    assert nc.smpy == pytest.approx(nc.Msun / nc.year)
