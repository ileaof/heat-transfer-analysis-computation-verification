"""Module 2 -- Thermal boundary layer over an isothermal flat plate.

Given the Blasius velocity field (Module 1), the dimensionless temperature
:math:`\\theta = (T - T_w)/(T_\\infty - T_w)` obeys the self-similar energy
equation

.. math::

    \\theta'' + \\frac{\\mathrm{Pr}}{2}\\, f\\, \\theta' = 0,
    \\qquad \\theta(0) = 0,\\; \\theta(\\infty) = 1,

with the *same* similarity variable :math:`\\eta` as the momentum problem.  The
equation is **linear** in :math:`\\theta`, which the solver exploits: it
integrates once with a trial slope :math:`\\theta'(0) = 1` and then rescales, so
no iteration is needed.  A fully independent **analytical** slope,

.. math::

    \\theta'(0) = \\left[\\int_0^\\infty
        \\exp\\!\\left(-\\frac{\\mathrm{Pr}}{2}\\int_0^\\eta f\\,d\\eta'\\right)
        d\\eta\\right]^{-1},

is also provided as a cross-check.

The wall temperature gradient sets the heat transfer:

.. math::

    \\mathrm{Nu}_x = \\theta'(0)\\, \\mathrm{Re}_x^{1/2},
    \\qquad
    h = \\frac{k}{x}\\,\\mathrm{Nu}_x .

For ordinary fluids (:math:`\\mathrm{Pr} \\gtrsim 0.6`) this reproduces the
Pohlhausen correlation :math:`\\mathrm{Nu}_x = 0.332\\,\\mathrm{Re}_x^{1/2}
\\mathrm{Pr}^{1/3}`.

.. note::

    The thermal boundary layer thickens dramatically at low Prandtl number
    (:math:`\\delta_t/\\delta \\sim \\mathrm{Pr}^{-1/2}` for liquid metals), so the
    solver **automatically enlarges the similarity domain** for small ``Pr``.
    A momentum grid truncated at :math:`\\eta_\\infty = 10` badly under-resolves
    the ``Pr = 0.01`` thermal layer (which reaches :math:`\\eta \\approx 38`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .blasius import BlasiusSolution, BlasiusSolver
from .integrators import rk4_integrate

#: Canonical Prandtl numbers requested by the laboratory specification.
DEFAULT_PRANDTL_NUMBERS: tuple[float, ...] = (0.01, 0.1, 0.7, 1.0, 7.0, 100.0)


@dataclass
class ThermalSolution:
    """Converged thermal similarity solution for one Prandtl number.

    Attributes
    ----------
    Pr : float
        Prandtl number.
    eta : numpy.ndarray
        Similarity coordinate.  Its extent adapts to ``Pr`` (larger for low
        ``Pr``), so it is not necessarily identical to the momentum grid.
    theta : numpy.ndarray
        Dimensionless temperature ``(T - T_w)/(T_inf - T_w)``, 0 at the wall,
        1 in the free stream.
    theta_p : numpy.ndarray
        Derivative ``d(theta)/d(eta)``.
    thetap0 : float
        Wall temperature gradient ``theta'(0)``; equals ``Nu_x / sqrt(Re_x)``.
    eta_t99 : float
        Similarity coordinate at which ``theta = 0.99`` -- the thermal
        boundary-layer thickness constant ``delta_t sqrt(Re_x) / x``.
    """

    Pr: float
    eta: np.ndarray
    theta: np.ndarray
    theta_p: np.ndarray
    thetap0: float
    eta_t99: float

    # -- dimensionless heat-transfer parameters -------------------------------

    def nu_local(self, Re_x):
        """Local Nusselt number ``Nu_x = theta'(0) sqrt(Re_x)``."""
        return self.thetap0 * np.sqrt(Re_x)

    def nu_average(self, Re_L):
        """Average Nusselt number ``Nu_L = 2 theta'(0) sqrt(Re_L)``.

        Uses ``h_avg = 2 h(L)`` for the laminar flat plate (``Nu ~ x^{-1/2}``).
        """
        return 2.0 * self.thetap0 * np.sqrt(Re_L)

    # -- dimensional heat-transfer parameters ---------------------------------

    def h_local(self, k, Re_x, x):
        """Local convective coefficient ``h = Nu_x k / x`` (W/(m^2 K))."""
        return self.nu_local(Re_x) * k / np.asarray(x, dtype=float)

    def h_average(self, k, Re_L, L):
        """Average convective coefficient ``h_bar = Nu_L k / L`` (W/(m^2 K))."""
        return self.nu_average(Re_L) * k / L

    def thermal_thickness(self, Re_x, x):
        """99% thermal boundary-layer thickness ``delta_t`` (m)."""
        return self.eta_t99 * np.asarray(x, dtype=float) / np.sqrt(Re_x)

    def wall_heat_flux(self, h, t_wall, t_inf):
        """Wall heat flux ``q'' = h (T_w - T_inf)`` (W/m^2)."""
        return h * (t_wall - t_inf)

    def temperature_profile(self, t_wall: float, t_inf: float):
        """Return ``(eta, T)`` with the dimensional temperature (K).

        ``T(eta) = T_w + (T_inf - T_w) * theta(eta)``.
        """
        return self.eta, t_wall + (t_inf - t_wall) * self.theta

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return the profile as an ordered ``{column: array}`` mapping."""
        return {
            "eta": self.eta,
            "theta": self.theta,
            "dtheta_deta": self.theta_p,
        }


class ThermalSolver:
    """Solve the thermal boundary-layer equation for a given velocity field.

    Parameters
    ----------
    blasius : BlasiusSolution, optional
        A previously computed momentum solution.  Only its wall curvature
        ``f''(0)`` and minimum domain size are reused; the velocity field is
        re-integrated on a Prandtl-adapted grid so that the far-field boundary
        condition ``theta(inf) = 1`` is applied truly in the free stream.  If
        ``None`` a default :class:`~blayerlab.blasius.BlasiusSolver` is run.
    steps_per_unit_eta : int, optional
        RK4 resolution: number of steps per unit of ``eta``.  The total step
        count scales with the (Pr-dependent) domain size.

    Notes
    -----
    The combined state ``[f, f', f'', theta, theta']`` is marched with the
    shared RK4 integrator so ``f`` is available at every Runge-Kutta stage.
    Because ``f''(0)`` is already known from the momentum solution, a single
    integration (no root finding) reproduces the Blasius field exactly.
    """

    def __init__(
        self,
        blasius: BlasiusSolution | None = None,
        steps_per_unit_eta: int = 200,
    ) -> None:
        self.blasius = blasius if blasius is not None else BlasiusSolver().solve()
        self._fpp0 = self.blasius.fpp0
        self._min_eta_max = float(self.blasius.eta[-1])
        self._steps_per_unit = int(steps_per_unit_eta)

    # -- domain sizing --------------------------------------------------------

    def domain_for(self, Pr: float) -> tuple[float, int]:
        """Return ``(eta_max, n_steps)`` sized for the thermal layer at ``Pr``.

        Two competing requirements are balanced:

        * **Coverage.**  The thermal boundary layer is *thick* at low ``Pr``
          (:math:`\\delta_t/\\delta \\sim \\mathrm{Pr}^{-1/2}`, reaching
          :math:`\\eta \\approx 38` at ``Pr = 0.01``) and *thin* at high ``Pr``
          (:math:`\\sim \\mathrm{Pr}^{-1/3}`).  The domain grows like
          ``7/sqrt(Pr)`` for small ``Pr`` and shrinks like ``Pr^{-1/3}`` for
          large ``Pr`` so ``theta(inf) = 1`` is always applied outside the layer
          while never wastefully integrating far into the free stream.
        * **Stability.**  The energy term ``-(Pr/2) f theta'`` is stiff where
          ``f`` is large.  Keeping the momentum domain (``eta ~ 10``) for high
          ``Pr`` would put the explicit RK4 past its stability limit and blow
          up.  Shrinking ``eta_max`` avoids the large-``f`` region, and the step
          count is additionally raised to satisfy the RK4 real-axis stability
          bound ``h*(Pr/2)*f_max < 2.78`` (using ``f_max <= eta_max``).

        Returns
        -------
        (float, int)
            ``eta_max`` and the number of RK4 steps.
        """
        # Domain that just contains the thermal layer for this Pr.
        eta_thin = 8.0 * Pr ** (-1.0 / 3.0) + 2.0   # high-Pr (thin layer) branch
        eta_thick = 7.0 / math.sqrt(Pr)             # low-Pr (thick layer) branch
        eta_max = max(eta_thin, eta_thick)
        eta_max = min(eta_max, 200.0)  # cap to keep the problem well-conditioned

        # Base resolution, plus an explicit-RK4 stability floor for the stiff
        # high-Pr far field:  n >= eta_max^2 * Pr / (2 * 2.78).
        n_base = int(round(self._steps_per_unit * eta_max))
        n_stable = int(math.ceil(eta_max * eta_max * Pr / 5.0)) + 1
        n_steps = min(max(n_base, n_stable), 400_000)
        return eta_max, n_steps

    # -- governing systems ----------------------------------------------------

    def _rhs_coupled(self, eta: float, y: np.ndarray, Pr: float) -> np.ndarray:
        """Combined momentum + energy system.

        With ``y = [f, f', f'', theta, theta']``:

        .. math::

            y' = [\\,f',\\ f'',\\ -\\tfrac12 f f'',\\
                    \\theta',\\ -\\tfrac{Pr}{2} f \\theta'\\,].
        """
        f, fp, fpp, _theta, thetap = y
        return np.array(
            [fp, fpp, -0.5 * f * fpp, thetap, -0.5 * Pr * f * thetap]
        )

    def _velocity_field(self, eta_max: float, n_steps: int):
        """Re-integrate the Blasius field on an arbitrary domain.

        Uses the known ``f''(0)`` so a single RK4 pass reproduces ``f``.
        Returns ``(eta, f)``.
        """
        def rhs(e, y):
            f, fp, fpp = y
            return np.array([fp, fpp, -0.5 * f * fpp])

        y0 = np.array([0.0, 0.0, self._fpp0])
        eta, y = rk4_integrate(rhs, y0, (0.0, eta_max), n_steps)
        return eta, y[:, 0]

    # -- public API -----------------------------------------------------------

    def solve(self, Pr: float) -> ThermalSolution:
        """Solve for a single Prandtl number.

        Parameters
        ----------
        Pr : float
            Prandtl number (> 0).

        Returns
        -------
        ThermalSolution
        """
        if Pr <= 0.0:
            raise ValueError(f"Prandtl number must be positive, got {Pr!r}.")

        eta_max, n_steps = self.domain_for(Pr)

        # Trial integration with theta'(0) = 1; theta scales linearly, so we
        # renormalise afterwards to enforce theta(inf) = 1 exactly.
        y0 = np.array([0.0, 0.0, self._fpp0, 0.0, 1.0])
        eta, y = rk4_integrate(
            lambda e, s: self._rhs_coupled(e, s, Pr),
            y0,
            (0.0, eta_max),
            n_steps,
        )
        theta_raw = y[:, 3]
        thetap_raw = y[:, 4]

        theta_inf = theta_raw[-1]
        theta = theta_raw / theta_inf
        theta_p = thetap_raw / theta_inf
        thetap0 = float(theta_p[0])
        eta_t99 = float(np.interp(0.99, theta, eta))

        return ThermalSolution(
            Pr=float(Pr),
            eta=eta,
            theta=theta,
            theta_p=theta_p,
            thetap0=thetap0,
            eta_t99=eta_t99,
        )

    def solve_many(
        self, prandtl_numbers=DEFAULT_PRANDTL_NUMBERS
    ) -> dict[float, ThermalSolution]:
        """Solve for a list of Prandtl numbers.

        Parameters
        ----------
        prandtl_numbers : iterable of float, optional
            Defaults to :data:`DEFAULT_PRANDTL_NUMBERS`.

        Returns
        -------
        dict
            Mapping ``{Pr: ThermalSolution}`` preserving input order.
        """
        return {float(Pr): self.solve(float(Pr)) for Pr in prandtl_numbers}

    def wall_gradient_analytical(self, Pr: float) -> float:
        """Independent analytical value of ``theta'(0)``.

        Because the energy equation is a linear first-order ODE in
        ``g = theta'``, integrating factors give the closed form

        .. math::

            \\theta'(0) = \\left[\\int_0^\\infty
                e^{-\\frac{Pr}{2}\\int_0^\\eta f\\,d\\eta'}\\,d\\eta\\right]^{-1}.

        Evaluated by cumulative trapezoidal quadrature on the Blasius ``f``,
        using the *same* Pr-adapted domain as :meth:`solve` so the check is
        genuinely independent of truncation.
        """
        eta_max, n_steps = self.domain_for(Pr)
        eta, f = self._velocity_field(eta_max, n_steps)
        # Inner integral F(eta) = \int_0^eta f d eta' via cumulative trapezoid.
        dF = 0.5 * (f[1:] + f[:-1]) * np.diff(eta)
        F = np.concatenate(([0.0], np.cumsum(dF)))
        integrand = np.exp(-0.5 * Pr * F)
        denom = float(np.trapz(integrand, eta))
        return 1.0 / denom


def gradient_from_correlation(Pr: float) -> float:
    """Reference wall-gradient constant ``theta'(0)`` from simple correlations.

    Uses ``0.332 Pr^{1/3}`` for ``Pr >= 0.6`` and the liquid-metal limit
    ``0.564 Pr^{1/2}`` for smaller ``Pr``.  This is only a quick sanity value;
    the low-``Pr`` limit is approached slowly, so expect several-percent
    disagreement near ``Pr ~ 0.01``.  For an accurate all-``Pr`` reference use
    :func:`blayerlab.correlations.nu_local_churchill_ozoe` divided by
    ``sqrt(Re_x)``.
    """
    if Pr >= 0.6:
        return 0.332 * Pr ** (1.0 / 3.0)
    return 0.564 * Pr ** 0.5


__all__ = [
    "ThermalSolution",
    "ThermalSolver",
    "DEFAULT_PRANDTL_NUMBERS",
    "gradient_from_correlation",
]
