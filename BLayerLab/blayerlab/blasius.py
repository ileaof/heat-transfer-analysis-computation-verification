"""Module 1 -- Blasius momentum boundary layer.

This module solves the Blasius equation for the laminar boundary layer over a
flat plate with zero pressure gradient,

.. math::

    f''' + \\tfrac{1}{2} f f'' = 0,
    \\qquad f(0) = 0,\\; f'(0) = 0,\\; f'(\\infty) = 1,

using a **shooting method** wrapped around a **fourth-order Runge-Kutta**
integrator (see :mod:`blayerlab.integrators`).  The similarity variable and
stream-function ansatz are

.. math::

    \\eta = y\\sqrt{\\frac{U_\\infty}{\\nu x}}, \\qquad
    \\psi = \\sqrt{\\nu x\\, U_\\infty}\\; f(\\eta), \\qquad
    \\frac{u}{U_\\infty} = f'(\\eta).

From the converged similarity solution the module computes every standard
integral parameter: boundary-layer thickness, displacement and momentum
thickness, shape factor, wall shear stress and skin-friction coefficient.

The unknown wall curvature :math:`f''(0)` is found by driving the residual
:math:`f'(\\eta_\\infty) - 1` to zero with a hand-coded **secant iteration**, so
the whole solution path -- integrator and root finder -- is transparent to the
reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants
from .integrators import rk4_integrate


# ---------------------------------------------------------------------------
# Howarth's classical tabulation (Howarth 1938) for verification.
# Columns: eta, f, f', f''.
# ---------------------------------------------------------------------------
HOWARTH_TABLE: np.ndarray = np.array(
    [
        [0.0, 0.00000, 0.00000, 0.33206],
        [0.4, 0.02656, 0.13277, 0.33147],
        [0.8, 0.10611, 0.26471, 0.32739],
        [1.2, 0.23795, 0.39378, 0.31659],
        [1.6, 0.42032, 0.51676, 0.29667],
        [2.0, 0.65003, 0.62977, 0.26675],
        [2.4, 0.92230, 0.72899, 0.22809],
        [2.8, 1.23099, 0.81152, 0.18401],
        [3.2, 1.56911, 0.87609, 0.13913],
        [3.6, 1.92954, 0.92333, 0.09809],
        [4.0, 2.30576, 0.95552, 0.06424],
        [4.4, 2.69238, 0.97587, 0.03897],
        [4.8, 3.08534, 0.98779, 0.02187],
        [5.0, 3.28329, 0.99155, 0.01591],
        [6.0, 4.27964, 0.99898, 0.00240],
        [7.0, 5.27926, 0.99992, 0.00022],
        [8.0, 6.27923, 1.00000, 0.00001],
    ]
)


@dataclass
class BlasiusSolution:
    """Container for a converged Blasius similarity solution.

    The object stores the dimensionless profile on the similarity grid together
    with the integral constants that are *independent of position*, and exposes
    methods that turn those constants into dimensional quantities once a local
    Reynolds number (and, where relevant, fluid density and free-stream speed)
    are supplied.

    Attributes
    ----------
    eta : numpy.ndarray
        Similarity coordinate grid.
    f, fp, fpp : numpy.ndarray
        The functions :math:`f`, :math:`f' = u/U_\\infty` and :math:`f''` on
        ``eta``.
    fpp0 : float
        Wall curvature :math:`f''(0)` (sets wall shear).  Reference: 0.33206.
    eta_99 : float
        Value of ``eta`` at which :math:`u/U_\\infty = 0.99`; the boundary-layer
        thickness constant, ``delta * sqrt(Re_x) / x``.
    displacement_const : float
        :math:`\\delta^* \\sqrt{Re_x}/x = \\int_0^\\infty (1 - f')\\,d\\eta`.
    momentum_const : float
        :math:`\\theta \\sqrt{Re_x}/x = \\int_0^\\infty f'(1 - f')\\,d\\eta`.
    shape_factor : float
        :math:`H = \\delta^*/\\theta`.
    """

    eta: np.ndarray
    f: np.ndarray
    fp: np.ndarray
    fpp: np.ndarray
    fpp0: float
    eta_99: float
    displacement_const: float
    momentum_const: float
    shape_factor: float

    # -- dimensional boundary-layer quantities --------------------------------

    def cf_local(self, Re_x):
        """Local skin-friction coefficient ``C_f = 2 f''(0) / sqrt(Re_x)``.

        Parameters
        ----------
        Re_x : float or array_like
            Local Reynolds number ``U_inf x / nu``.
        """
        return 2.0 * self.fpp0 / np.sqrt(Re_x)

    def wall_shear_stress(self, rho, u_inf, nu, x):
        """Wall shear stress ``tau_w`` (Pa) at position ``x``.

        .. math::

            \\tau_w = \\mu\\, U_\\infty
            \\sqrt{\\frac{U_\\infty}{\\nu x}}\\, f''(0)
            = 0.5\\,\\rho U_\\infty^2\\, C_f.

        Parameters
        ----------
        rho : float
            Density, kg/m^3.
        u_inf : float
            Free-stream velocity, m/s.
        nu : float
            Kinematic viscosity, m^2/s.
        x : float or array_like
            Streamwise position, m.
        """
        Re_x = u_inf * np.asarray(x, dtype=float) / nu
        return 0.5 * rho * u_inf**2 * self.cf_local(Re_x)

    def delta(self, Re_x, x):
        """99% velocity boundary-layer thickness ``delta`` (m)."""
        return self.eta_99 * np.asarray(x, dtype=float) / np.sqrt(Re_x)

    def displacement_thickness(self, Re_x, x):
        """Displacement thickness ``delta*`` (m)."""
        return self.displacement_const * np.asarray(x, dtype=float) / np.sqrt(Re_x)

    def momentum_thickness(self, Re_x, x):
        """Momentum thickness ``theta`` (m)."""
        return self.momentum_const * np.asarray(x, dtype=float) / np.sqrt(Re_x)

    def velocity_profile(self, u_inf: float = 1.0):
        """Return ``(eta, u/U_inf)`` -- or ``(eta, u)`` if ``u_inf`` is given.

        Parameters
        ----------
        u_inf : float, optional
            Free-stream velocity.  With the default ``1.0`` the second array is
            the dimensionless ``u/U_inf = f'``.
        """
        return self.eta, u_inf * self.fp

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return the profile as an ordered ``{column: array}`` mapping."""
        return {
            "eta": self.eta,
            "f": self.f,
            "df_deta (u/Uinf)": self.fp,
            "d2f_deta2": self.fpp,
        }

    def summary(self) -> str:
        """Human-readable summary comparing computed vs reference constants."""
        rows = [
            ("f''(0)", self.fpp0, constants.BLASIUS_FPP0),
            ("delta*sqrt(Re)/x (eta_99)", self.eta_99, constants.BLASIUS_DELTA99),
            ("delta*  const", self.displacement_const, constants.BLASIUS_DISPLACEMENT),
            ("theta   const", self.momentum_const, constants.BLASIUS_MOMENTUM),
            ("shape factor H", self.shape_factor, constants.BLASIUS_SHAPE_FACTOR),
        ]
        lines = ["Blasius solution -- computed vs reference:",
                 f"  {'quantity':30s} {'computed':>12s} {'reference':>12s} {'rel.err':>10s}"]
        for label, comp, ref in rows:
            rel = abs(comp - ref) / abs(ref)
            lines.append(f"  {label:30s} {comp:12.6f} {ref:12.6f} {rel:10.2e}")
        return "\n".join(lines)


class BlasiusSolver:
    """Solve the Blasius equation by shooting + fourth-order Runge-Kutta.

    Parameters
    ----------
    eta_max : float, optional
        Truncation of the semi-infinite domain used to approximate
        ``eta -> infinity``.  ``10`` is comfortably in the free stream.
    n_steps : int, optional
        Number of uniform RK4 steps across ``[0, eta_max]``.
    tol : float, optional
        Convergence tolerance on the free-stream residual ``f'(eta_max) - 1``.
    max_iter : int, optional
        Maximum number of secant iterations for the shooting parameter.

    Examples
    --------
    >>> sol = BlasiusSolver().solve()
    >>> round(sol.fpp0, 5)
    0.33206
    """

    def __init__(
        self,
        eta_max: float = 10.0,
        n_steps: int = 2000,
        tol: float = 1.0e-10,
        max_iter: int = 60,
    ) -> None:
        self.eta_max = float(eta_max)
        self.n_steps = int(n_steps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    # -- the governing system -------------------------------------------------

    @staticmethod
    def _rhs(eta: float, y: np.ndarray) -> np.ndarray:
        """Blasius equation as a first-order system.

        With ``y = [f, f', f'']`` the equation ``f''' = -0.5 f f''`` gives

        .. math:: y' = [\\,f',\\; f'',\\; -\\tfrac{1}{2} f f''\\,].
        """
        f, fp, fpp = y
        return np.array([fp, fpp, -0.5 * f * fpp])

    def _shoot(self, fpp0_guess: float) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the IVP for a trial wall curvature ``f''(0)``."""
        y0 = np.array([0.0, 0.0, fpp0_guess])
        return rk4_integrate(self._rhs, y0, (0.0, self.eta_max), self.n_steps)

    def _residual(self, fpp0_guess: float) -> float:
        """Free-stream boundary-condition residual ``f'(eta_max) - 1``."""
        _, y = self._shoot(fpp0_guess)
        return y[-1, 1] - 1.0

    def solve(self, fpp0_low: float = 0.1, fpp0_high: float = 0.6) -> BlasiusSolution:
        """Run the shooting method and return a :class:`BlasiusSolution`.

        Parameters
        ----------
        fpp0_low, fpp0_high : float
            Two initial guesses for the secant iteration on ``f''(0)``.  The
            true value ~0.332 lies between the defaults.

        Returns
        -------
        BlasiusSolution
            The converged similarity solution and its integral parameters.

        Raises
        ------
        RuntimeError
            If the secant iteration fails to converge within ``max_iter``.
        """
        fpp0 = self._secant(fpp0_low, fpp0_high)

        # Final integration on the converged shooting parameter.
        eta, y = self._shoot(fpp0)
        f, fp, fpp = y[:, 0], y[:, 1], y[:, 2]

        # Integral parameters (position-independent constants).
        eta_99 = float(np.interp(0.99, fp, eta))
        displacement_const = float(np.trapz(1.0 - fp, eta))
        momentum_const = float(np.trapz(fp * (1.0 - fp), eta))
        shape_factor = displacement_const / momentum_const

        return BlasiusSolution(
            eta=eta,
            f=f,
            fp=fp,
            fpp=fpp,
            fpp0=float(fpp[0]),
            eta_99=eta_99,
            displacement_const=displacement_const,
            momentum_const=momentum_const,
            shape_factor=shape_factor,
        )

    # -- root finder ----------------------------------------------------------

    def _secant(self, s0: float, s1: float) -> float:
        """Secant iteration to solve ``residual(f''(0)) = 0``.

        The secant method is used (rather than bisection) because the residual
        is smooth and monotone in ``f''(0)``, giving super-linear convergence in
        a handful of iterations.
        """
        r0 = self._residual(s0)
        r1 = self._residual(s1)
        for _ in range(self.max_iter):
            if abs(r1) < self.tol:
                return s1
            denom = r1 - r0
            if denom == 0.0:
                break
            s2 = s1 - r1 * (s1 - s0) / denom
            s0, r0 = s1, r1
            s1, r1 = s2, self._residual(s2)
        if abs(r1) < self.tol:
            return s1
        raise RuntimeError(
            f"Blasius shooting failed to converge: residual={r1:.3e} "
            f"after {self.max_iter} iterations."
        )

    # -- verification helper --------------------------------------------------

    def compare_to_howarth(
        self, solution: Optional[BlasiusSolution] = None
    ) -> dict[str, np.ndarray]:
        """Compare the computed profile against Howarth's tabulated values.

        Parameters
        ----------
        solution : BlasiusSolution, optional
            A previously computed solution; if omitted the solver is run.

        Returns
        -------
        dict
            ``{"eta", "f_num", "f_ref", "fp_num", "fp_ref", "fpp_num",
            "fpp_ref", "abs_err_fp"}`` sampled at Howarth's eta stations.
        """
        if solution is None:
            solution = self.solve()
        eta_ref = HOWARTH_TABLE[:, 0]
        f_num = np.interp(eta_ref, solution.eta, solution.f)
        fp_num = np.interp(eta_ref, solution.eta, solution.fp)
        fpp_num = np.interp(eta_ref, solution.eta, solution.fpp)
        return {
            "eta": eta_ref,
            "f_num": f_num,
            "f_ref": HOWARTH_TABLE[:, 1],
            "fp_num": fp_num,
            "fp_ref": HOWARTH_TABLE[:, 2],
            "fpp_num": fpp_num,
            "fpp_ref": HOWARTH_TABLE[:, 3],
            "abs_err_fp": np.abs(fp_num - HOWARTH_TABLE[:, 2]),
        }


__all__ = ["BlasiusSolution", "BlasiusSolver", "HOWARTH_TABLE"]
