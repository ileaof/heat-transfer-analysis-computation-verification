"""Explicit Runge-Kutta integrators for systems of first-order ODEs.

The Boundary Layer Laboratory deliberately ships its *own* fourth-order
Runge-Kutta (RK4) integrator rather than delegating to :func:`scipy.integrate`.
The goal is pedagogical: a reader of *Heat Transfer: Analysis, Computation, and
Verification* should be able to open this file and see exactly how the Blasius
and thermal boundary-layer equations are marched in the similarity variable.

The integrator is written for a generic first-order system

.. math::

    \\frac{d\\mathbf{y}}{d\\eta} = \\mathbf{F}(\\eta, \\mathbf{y})

so that both the momentum equation (a third-order ODE written as three
first-order equations) and the energy equation (second-order -> two
first-order equations) share the same engine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

#: Type alias for a right-hand-side function ``f(x, y) -> dy/dx``.
RHS = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(rhs: RHS, x: float, y: np.ndarray, h: float) -> np.ndarray:
    """Advance one classical fourth-order Runge-Kutta step.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(x, y)`` returning ``dy/dx`` as a NumPy array with
        the same shape as ``y``.
    x : float
        Current value of the independent variable.
    y : numpy.ndarray
        Current state vector.
    h : float
        Step size in the independent variable.

    Returns
    -------
    numpy.ndarray
        The state vector advanced to ``x + h``.

    Notes
    -----
    The classical RK4 scheme has local truncation error :math:`O(h^5)` and
    global error :math:`O(h^4)`:

    .. math::

        k_1 &= F(x, y) \\\\
        k_2 &= F(x + h/2,\\; y + h k_1 / 2) \\\\
        k_3 &= F(x + h/2,\\; y + h k_2 / 2) \\\\
        k_4 &= F(x + h,\\; y + h k_3) \\\\
        y_{n+1} &= y_n + \\frac{h}{6}(k_1 + 2 k_2 + 2 k_3 + k_4)
    """
    k1 = np.asarray(rhs(x, y), dtype=float)
    k2 = np.asarray(rhs(x + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(rhs(x + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(rhs(x + h, y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    rhs: RHS,
    y0: np.ndarray,
    x_span: tuple[float, float],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a first-order ODE system with a uniform-grid RK4 march.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(x, y) -> dy/dx``.
    y0 : numpy.ndarray
        Initial state vector at ``x_span[0]``.
    x_span : tuple of float
        ``(x_start, x_end)`` integration interval.
    n_steps : int
        Number of uniform steps.  The returned arrays have ``n_steps + 1`` rows.

    Returns
    -------
    x : numpy.ndarray
        Grid of independent-variable values, shape ``(n_steps + 1,)``.
    y : numpy.ndarray
        Solution history, shape ``(n_steps + 1, len(y0))`` where row ``i`` is the
        state at ``x[i]``.

    Examples
    --------
    Integrate ``y' = y`` from 0 to 1 (exact solution ``e``):

    >>> import numpy as np
    >>> x, y = rk4_integrate(lambda t, s: s, np.array([1.0]), (0.0, 1.0), 100)
    >>> bool(abs(y[-1, 0] - np.e) < 1e-6)
    True
    """
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")

    x_start, x_end = x_span
    h = (x_end - x_start) / n_steps

    y0 = np.asarray(y0, dtype=float)
    x = np.linspace(x_start, x_end, n_steps + 1)
    y = np.empty((n_steps + 1, y0.size), dtype=float)
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = rk4_step(rhs, x[i], y[i], h)

    return x, y


__all__ = ["RHS", "rk4_step", "rk4_integrate"]
