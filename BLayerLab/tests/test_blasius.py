"""Tests for Module 1 -- the Blasius momentum boundary-layer solver."""

import numpy as np
import pytest

from blayerlab import constants
from blayerlab.blasius import HOWARTH_TABLE, BlasiusSolver


@pytest.fixture(scope="module")
def solution():
    return BlasiusSolver().solve()


def test_wall_curvature(solution):
    """f''(0) matches the accepted value 0.332057 to 5 decimals."""
    assert solution.fpp0 == pytest.approx(constants.BLASIUS_FPP0, abs=1e-5)


def test_integral_constants(solution):
    """Displacement, momentum thickness and shape factor match references."""
    assert solution.displacement_const == pytest.approx(1.720787, abs=1e-4)
    assert solution.momentum_const == pytest.approx(0.664116, abs=1e-4)
    assert solution.shape_factor == pytest.approx(2.591, abs=1e-3)
    assert solution.eta_99 == pytest.approx(4.910, abs=1e-2)


def test_boundary_conditions(solution):
    """f(0)=0, f'(0)=0 and f'(eta_max) -> 1."""
    assert solution.f[0] == pytest.approx(0.0, abs=1e-12)
    assert solution.fp[0] == pytest.approx(0.0, abs=1e-12)
    assert solution.fp[-1] == pytest.approx(1.0, abs=1e-6)


def test_profile_matches_howarth(solution):
    """The whole u/U_inf profile matches Howarth's tabulation (RMS < 1e-4)."""
    eta_ref = HOWARTH_TABLE[:, 0]
    fp_ref = HOWARTH_TABLE[:, 2]
    fp_num = np.interp(eta_ref, solution.eta, solution.fp)
    rms = np.sqrt(np.mean((fp_num - fp_ref) ** 2))
    assert rms < 1e-4


def test_skin_friction_scaling(solution):
    """C_f = 0.664/sqrt(Re_x) at a representative Reynolds number."""
    Re = 1.0e5
    assert solution.cf_local(Re) == pytest.approx(0.664 / np.sqrt(Re), rel=1e-3)


def test_wall_shear_stress_consistency(solution):
    """tau_w equals 0.5 rho U^2 C_f."""
    rho, u, nu, x = 1.2, 5.0, 1.5e-5, 0.3
    Re_x = u * x / nu
    tau = solution.wall_shear_stress(rho, u, nu, x)
    assert tau == pytest.approx(0.5 * rho * u**2 * solution.cf_local(Re_x), rel=1e-12)
