"""Tests for Module 2 -- the thermal boundary-layer solver."""

import numpy as np
import pytest

from blayerlab.blasius import BlasiusSolver
from blayerlab.thermal import ThermalSolver


@pytest.fixture(scope="module")
def solver():
    return ThermalSolver(BlasiusSolver().solve())


@pytest.mark.parametrize("Pr", [0.01, 0.1, 0.7, 1.0, 7.0, 100.0, 1000.0])
def test_numerical_matches_analytical(solver, Pr):
    """RK4/rescaling wall gradient matches the closed-form integral (rel<1e-3)."""
    sol = solver.solve(Pr)
    ana = solver.wall_gradient_analytical(Pr)
    assert sol.thetap0 == pytest.approx(ana, rel=1e-3)


def test_pr_one_equals_blasius(solver):
    """At Pr = 1 the energy equation equals the momentum equation: theta'(0)=f''(0)."""
    sol = solver.solve(1.0)
    assert sol.thetap0 == pytest.approx(solver.blasius.fpp0, rel=1e-4)


def test_pohlhausen_constant(solver):
    """For air (Pr=0.7) theta'(0) is close to 0.332 Pr^{1/3} (within a few %)."""
    sol = solver.solve(0.7)
    assert sol.thetap0 == pytest.approx(0.332 * 0.7 ** (1.0 / 3.0), rel=0.02)


def test_boundary_conditions(solver):
    """theta(0)=0 and theta(inf)=1 (after normalisation)."""
    sol = solver.solve(7.0)
    assert sol.theta[0] == pytest.approx(0.0, abs=1e-12)
    assert sol.theta[-1] == pytest.approx(1.0, abs=1e-6)


def test_thermal_thickness_ordering(solver):
    """delta_t/delta decreases as Pr increases (thin thermal layer at high Pr)."""
    ratios = [solver.solve(p).eta_t99 for p in (0.1, 1.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(ratios, ratios[1:]))


def test_high_pr_is_finite(solver):
    """The solver stays numerically stable (no overflow) at high Pr."""
    sol = solver.solve(1000.0)
    assert np.isfinite(sol.thetap0)
    assert sol.thetap0 == pytest.approx(0.3387 * 1000.0 ** (1.0 / 3.0), rel=0.02)


def test_rejects_nonpositive_pr(solver):
    with pytest.raises(ValueError):
        solver.solve(0.0)
