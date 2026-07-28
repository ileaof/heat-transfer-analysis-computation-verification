"""Module 3 -- Flat-plate heat-transfer calculator.

This module ties Modules 1 and 2 together into an engineering-facing tool.  The
user supplies physical inputs

* free-stream velocity ``U_inf``  (m/s),
* fluid properties (a :class:`~blayerlab.fluids.FluidProperties`),
* plate length ``L``  (m),
* wall temperature ``T_w``  (K),
* free-stream temperature ``T_inf``  (K),

and the calculator returns *every* momentum- and thermal-boundary-layer
quantity, both **locally** (at the trailing edge ``x = L``) and **averaged**
over the plate, together with a side-by-side comparison against the classical
engineering correlations.

All results are SI.  Fluid properties should be evaluated at the film
temperature :math:`T_f = (T_w + T_\\infty)/2`; the calculator does not itself
re-evaluate properties with temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import correlations as corr
from .blasius import BlasiusSolution, BlasiusSolver
from .fluids import FluidProperties
from .thermal import ThermalSolver

#: Nominal laminar-to-turbulent transition Reynolds number for a flat plate.
TRANSITION_RE: float = 5.0e5


@dataclass
class FlatPlateResult:
    """Complete set of computed flat-plate boundary-layer quantities.

    The record stores the inputs, the dimensionless groups, and the local
    (``x = L``) and plate-averaged results.  Streamwise distributions are kept
    in the ``x``/``*_x`` arrays for plotting and export.
    """

    # -- inputs --
    fluid: FluidProperties
    u_inf: float
    length: float
    t_wall: float
    t_inf: float

    # -- dimensionless groups --
    Re_L: float
    Pr: float

    # -- momentum layer at x = L --
    delta_L: float
    delta_star_L: float
    theta_mom_L: float
    shape_factor: float
    cf_local_L: float
    cf_avg: float
    tau_wall_L: float

    # -- thermal layer at x = L --
    delta_t_L: float
    nu_local_L: float
    nu_avg: float
    h_local_L: float
    h_avg: float
    q_local_L: float
    q_avg: float
    heat_rate_per_width: float

    # -- streamwise distributions --
    x: np.ndarray
    Re_x: np.ndarray
    delta_x: np.ndarray
    delta_t_x: np.ndarray
    cf_x: np.ndarray
    h_x: np.ndarray
    nu_x: np.ndarray
    q_x: np.ndarray

    # -- flags --
    is_laminar: bool = field(default=True)

    def to_distribution_dict(self) -> dict[str, np.ndarray]:
        """Return streamwise distributions as an ordered ``{column: array}``."""
        return {
            "x [m]": self.x,
            "Re_x": self.Re_x,
            "delta [m]": self.delta_x,
            "delta_t [m]": self.delta_t_x,
            "Cf_x": self.cf_x,
            "h [W/m2K]": self.h_x,
            "Nu_x": self.nu_x,
            "q'' [W/m2]": self.q_x,
        }

    def report(self) -> str:
        """Return a formatted engineering report as a multi-line string."""
        dT = self.t_wall - self.t_inf
        lines = [
            "=" * 66,
            " FLAT-PLATE BOUNDARY-LAYER / HEAT-TRANSFER REPORT",
            "=" * 66,
            f" Fluid                 : {self.fluid.name}",
            f" Free-stream velocity  : {self.u_inf:11.4g} m/s",
            f" Plate length L        : {self.length:11.4g} m",
            f" Wall temperature      : {self.t_wall:11.4g} K",
            f" Free-stream temp.     : {self.t_inf:11.4g} K",
            f" Temperature diff. dT  : {dT:11.4g} K",
            "-" * 66,
            " Dimensionless groups",
            f"   Reynolds number Re_L : {self.Re_L:11.4e}",
            f"   Prandtl number  Pr   : {self.Pr:11.4e}",
            f"   Flow regime          : "
            f"{'LAMINAR' if self.is_laminar else 'TRANSITIONAL/TURBULENT (Re_L > 5e5!)'}",
            "-" * 66,
            " Momentum boundary layer at x = L",
            f"   delta   (99%)        : {self.delta_L:11.4e} m",
            f"   delta*  (displacement): {self.delta_star_L:11.4e} m",
            f"   theta   (momentum)   : {self.theta_mom_L:11.4e} m",
            f"   shape factor H       : {self.shape_factor:11.4f}",
            f"   Cf,x  (local)        : {self.cf_local_L:11.4e}",
            f"   Cf    (average)      : {self.cf_avg:11.4e}",
            f"   tau_w (local)        : {self.tau_wall_L:11.4e} Pa",
            "-" * 66,
            " Thermal boundary layer at x = L",
            f"   delta_t (99%)        : {self.delta_t_L:11.4e} m",
            f"   Nu_x  (local)        : {self.nu_local_L:11.4e}",
            f"   Nu_L  (average)      : {self.nu_avg:11.4e}",
            f"   h_x   (local)        : {self.h_local_L:11.4e} W/(m^2 K)",
            f"   h_bar (average)      : {self.h_avg:11.4e} W/(m^2 K)",
            f"   q''_x (local)        : {self.q_local_L:11.4e} W/m^2",
            f"   q''   (average)      : {self.q_avg:11.4e} W/m^2",
            f"   Q'    (per unit width): {self.heat_rate_per_width:11.4e} W/m",
            "=" * 66,
        ]
        return "\n".join(lines)

    def correlation_comparison(self) -> str:
        """Return a numeric-vs-correlation comparison table as a string."""
        rows = [
            (
                "Cf,x",
                self.cf_local_L,
                corr.cf_local(self.Re_L),
            ),
            (
                "Cf,avg",
                self.cf_avg,
                corr.cf_average(self.Re_L),
            ),
            (
                "Nu_x",
                self.nu_local_L,
                corr.nu_local_churchill_ozoe(self.Re_L, self.Pr),
            ),
            (
                "Nu_L,avg",
                self.nu_avg,
                corr.nu_average_churchill_ozoe(self.Re_L, self.Pr),
            ),
        ]
        lines = [
            " Numerical (similarity) vs engineering correlation",
            f"   {'quantity':10s} {'numerical':>13s} {'correlation':>13s} {'rel.err':>10s}",
        ]
        for name, num, ref in rows:
            rel = abs(num - ref) / abs(ref) if ref != 0 else float("nan")
            lines.append(f"   {name:10s} {num:13.5e} {ref:13.5e} {rel:10.2e}")
        return "\n".join(lines)


class FlatPlateCalculator:
    """Compute all flat-plate boundary-layer quantities from physical inputs.

    Parameters
    ----------
    blasius : BlasiusSolution, optional
        A pre-computed momentum solution (re-used across calls for speed).
    thermal_solver : ThermalSolver, optional
        A pre-built thermal solver.  If omitted, one is created from ``blasius``.
    n_stations : int, optional
        Number of streamwise stations for the distribution arrays.

    Examples
    --------
    >>> from blayerlab.fluids import get_fluid
    >>> calc = FlatPlateCalculator()
    >>> res = calc.compute(u_inf=2.0, fluid=get_fluid("air"),
    ...                    length=0.5, t_wall=350.0, t_inf=300.0)
    >>> res.Re_L > 0
    True
    """

    def __init__(
        self,
        blasius: BlasiusSolution | None = None,
        thermal_solver: ThermalSolver | None = None,
        n_stations: int = 200,
    ) -> None:
        self.blasius = blasius if blasius is not None else BlasiusSolver().solve()
        self.thermal = (
            thermal_solver
            if thermal_solver is not None
            else ThermalSolver(self.blasius)
        )
        self.n_stations = int(n_stations)

    def compute(
        self,
        u_inf: float,
        fluid: FluidProperties,
        length: float,
        t_wall: float,
        t_inf: float,
    ) -> FlatPlateResult:
        """Evaluate the flat-plate boundary layer for the given operating point.

        Parameters
        ----------
        u_inf : float
            Free-stream velocity, m/s.
        fluid : FluidProperties
            Thermophysical properties (ideally at the film temperature).
        length : float
            Plate length, m.
        t_wall, t_inf : float
            Wall and free-stream temperatures, K.

        Returns
        -------
        FlatPlateResult
            All computed quantities plus streamwise distributions.
        """
        if u_inf <= 0 or length <= 0:
            raise ValueError("u_inf and length must be positive.")

        nu, k, rho = fluid.nu, fluid.k, fluid.rho
        Pr = fluid.Pr
        Re_L = u_inf * length / nu

        # Thermal similarity solution for this Prandtl number.
        tsol = self.thermal.solve(Pr)
        dT = t_wall - t_inf

        # --- momentum layer at x = L ---
        delta_L = float(self.blasius.delta(Re_L, length))
        delta_star_L = float(self.blasius.displacement_thickness(Re_L, length))
        theta_mom_L = float(self.blasius.momentum_thickness(Re_L, length))
        cf_local_L = float(self.blasius.cf_local(Re_L))
        cf_avg = 2.0 * cf_local_L  # (1/L) integral of 0.664/sqrt(Re_x)
        tau_wall_L = float(self.blasius.wall_shear_stress(rho, u_inf, nu, length))

        # --- thermal layer at x = L ---
        delta_t_L = float(tsol.thermal_thickness(Re_L, length))
        nu_local_L = float(tsol.nu_local(Re_L))
        nu_avg = float(tsol.nu_average(Re_L))
        h_local_L = nu_local_L * k / length
        h_avg = nu_avg * k / length
        q_local_L = h_local_L * dT
        q_avg = h_avg * dT
        heat_rate_per_width = q_avg * length  # W per metre of span

        # --- streamwise distributions (avoid x = 0 singularity) ---
        x = np.linspace(length / self.n_stations, length, self.n_stations)
        Re_x = u_inf * x / nu
        delta_x = self.blasius.delta(Re_x, x)
        delta_t_x = tsol.thermal_thickness(Re_x, x)
        cf_x = self.blasius.cf_local(Re_x)
        nu_x = tsol.nu_local(Re_x)
        h_x = nu_x * k / x
        q_x = h_x * dT

        return FlatPlateResult(
            fluid=fluid,
            u_inf=u_inf,
            length=length,
            t_wall=t_wall,
            t_inf=t_inf,
            Re_L=Re_L,
            Pr=Pr,
            delta_L=delta_L,
            delta_star_L=delta_star_L,
            theta_mom_L=theta_mom_L,
            shape_factor=self.blasius.shape_factor,
            cf_local_L=cf_local_L,
            cf_avg=cf_avg,
            tau_wall_L=tau_wall_L,
            delta_t_L=delta_t_L,
            nu_local_L=nu_local_L,
            nu_avg=nu_avg,
            h_local_L=h_local_L,
            h_avg=h_avg,
            q_local_L=q_local_L,
            q_avg=q_avg,
            heat_rate_per_width=heat_rate_per_width,
            x=x,
            Re_x=Re_x,
            delta_x=delta_x,
            delta_t_x=delta_t_x,
            cf_x=cf_x,
            h_x=h_x,
            nu_x=nu_x,
            q_x=q_x,
            is_laminar=bool(Re_L <= TRANSITION_RE),
        )


__all__ = ["FlatPlateResult", "FlatPlateCalculator", "TRANSITION_RE"]
