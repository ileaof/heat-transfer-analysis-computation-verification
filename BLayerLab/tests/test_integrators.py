"""Tests for the shared RK4 integrator and the verification error metrics."""

import numpy as np
import pytest

from blayerlab.integrators import rk4_integrate, rk4_step
from blayerlab.verification import absolute_error, relative_error, rms_error


def test_rk4_exponential():
    """y' = y integrates to e with 4th-order accuracy."""
    x, y = rk4_integrate(lambda t, s: s, np.array([1.0]), (0.0, 1.0), 100)
    assert y[-1, 0] == pytest.approx(np.e, abs=1e-8)


def test_rk4_convergence_order():
    """Global error scales like h^4 when the step is halved (~16x drop)."""
    def err(n):
        _, y = rk4_integrate(lambda t, s: s, np.array([1.0]), (0.0, 1.0), n)
        return abs(y[-1, 0] - np.e)

    ratio = err(20) / err(40)
    assert ratio > 10  # ideally 16 for perfect 4th order


def test_rk4_system_harmonic():
    """Harmonic oscillator y'' = -y stays on the unit circle."""
    def rhs(t, s):
        return np.array([s[1], -s[0]])

    x, y = rk4_integrate(rhs, np.array([1.0, 0.0]), (0.0, 2 * np.pi), 2000)
    assert y[-1, 0] == pytest.approx(1.0, abs=1e-4)
    assert y[-1, 1] == pytest.approx(0.0, abs=1e-4)


def test_single_step_matches_integrate():
    """One rk4_step equals a one-step rk4_integrate."""
    rhs = lambda t, s: 2.0 * s
    y1 = rk4_step(rhs, 0.0, np.array([1.0]), 0.1)
    _, y2 = rk4_integrate(rhs, np.array([1.0]), (0.0, 0.1), 1)
    assert y1 == pytest.approx(y2[-1])


def test_error_metrics():
    num = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.0, 2.0, 4.0])
    assert absolute_error(num, ref).tolist() == [0.0, 0.0, 1.0]
    assert relative_error(3.0, 4.0) == pytest.approx(0.25)
    assert rms_error(num, ref) == pytest.approx(np.sqrt(1.0 / 3.0))
