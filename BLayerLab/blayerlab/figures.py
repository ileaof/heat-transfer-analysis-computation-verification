"""Publication-quality figure generators for the Boundary Layer Laboratory.

Each function here takes solver output (plain arrays / solution objects) and
returns a Matplotlib :class:`~matplotlib.figure.Figure`.  Functions never call
``plt.show()`` or save to disk -- that is left to the caller (examples, the
interactive module, or :func:`blayerlab.plotting.savefig`) -- so the same
figure code serves batch scripts and interactive sessions alike.

The module applies :func:`blayerlab.plotting.use_publication_style` on import.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from . import correlations as corr
from .blasius import HOWARTH_TABLE, BlasiusSolution
from .calculator import FlatPlateResult
from .plotting import PALETTE, new_figure, use_publication_style
from .thermal import ThermalSolution

use_publication_style()


# ---------------------------------------------------------------------------
# Module 1 -- momentum boundary layer
# ---------------------------------------------------------------------------


def plot_blasius_profiles(solution: BlasiusSolution):
    """Plot ``f``, ``f' = u/U_inf`` and ``f''`` of the Blasius solution.

    Parameters
    ----------
    solution : BlasiusSolution

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = new_figure(
        xlabel=r"similarity variable $\eta = y\,\sqrt{U_\infty/(\nu x)}$",
        ylabel="dimensionless functions",
        title="Blasius momentum boundary layer",
    )
    ax.plot(solution.eta, solution.f, label=r"$f(\eta)$")
    ax.plot(solution.eta, solution.fp, label=r"$f'(\eta)=u/U_\infty$")
    ax.plot(solution.eta, solution.fpp, label=r"$f''(\eta)$")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.axvline(solution.eta_99, color="0.5", lw=0.8, ls="--")
    ax.text(solution.eta_99 + 0.1, 0.2, r"$\eta_{99}$", color="0.4")
    ax.set_xlim(0, solution.eta[-1] if solution.eta[-1] <= 10 else 10)
    ax.legend(loc="center right")
    return fig


def plot_blasius_verification(solution: BlasiusSolution):
    """Overlay the numerical velocity profile on Howarth's tabulated points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = new_figure(
        xlabel=r"$\eta$",
        ylabel=r"$u/U_\infty = f'(\eta)$",
        title="Blasius profile: numerical vs Howarth (1938)",
    )
    ax.plot(solution.eta, solution.fp, color=PALETTE[0], label="numerical (RK4 + shooting)")
    ax.plot(
        HOWARTH_TABLE[:, 0],
        HOWARTH_TABLE[:, 2],
        "o",
        color=PALETTE[1],
        label="Howarth tabulation",
        markersize=6,
        markerfacecolor="white",
    )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    return fig


# ---------------------------------------------------------------------------
# Module 2 -- thermal boundary layer
# ---------------------------------------------------------------------------


def plot_temperature_profiles(solutions: Mapping[float, ThermalSolution]):
    """Plot dimensionless temperature ``theta(eta)`` for several Prandtl numbers.

    Parameters
    ----------
    solutions : mapping of float to ThermalSolution
        ``{Pr: ThermalSolution}``, e.g. the output of
        :meth:`~blayerlab.thermal.ThermalSolver.solve_many`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = new_figure(
        xlabel=r"$\eta$",
        ylabel=r"$\theta = (T-T_w)/(T_\infty-T_w)$",
        title="Thermal boundary layer for varying Prandtl number",
    )
    for (Pr, sol), color in zip(solutions.items(), PALETTE):
        ax.plot(sol.eta, sol.theta, color=color, label=f"Pr = {Pr:g}")
    ax.axhline(0.99, color="0.6", lw=0.8, ls=":")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="thin layer = high Pr", loc="lower right")
    return fig


def plot_nusselt_vs_prandtl(solver, prandtl=None, re_x: float = 1.0e5):
    """Plot the local Nusselt constant ``Nu_x/Re_x^{1/2}`` versus Prandtl number.

    Compares the numerical wall gradient with the Pohlhausen ``0.332 Pr^{1/3}``
    and the all-``Pr`` Churchill-Ozoe correlation.

    Parameters
    ----------
    solver : ThermalSolver
        A thermal solver used to compute the exact wall gradients.
    prandtl : array_like, optional
        Prandtl numbers to sample (log-spaced default from 1e-2 to 1e3).
    re_x : float, optional
        Reynolds number at which the Churchill-Ozoe constant is evaluated.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if prandtl is None:
        prandtl = np.logspace(-2, 3, 40)
    prandtl = np.asarray(prandtl, dtype=float)

    numeric = np.array([solver.solve(Pr).thetap0 for Pr in prandtl])
    pohlhausen = 0.332 * np.cbrt(prandtl)
    churchill = corr.nu_local_churchill_ozoe(re_x, prandtl) / np.sqrt(re_x)

    fig, ax = new_figure(
        xlabel="Prandtl number, Pr",
        ylabel=r"$\mathrm{Nu}_x / \mathrm{Re}_x^{1/2} = \theta'(0)$",
        title="Local Nusselt constant vs Prandtl number",
    )
    ax.loglog(prandtl, numeric, "o", color=PALETTE[0], ms=5,
              markerfacecolor="white", label="numerical (similarity)")
    ax.loglog(prandtl, pohlhausen, "-", color=PALETTE[1],
              label=r"Pohlhausen $0.332\,\mathrm{Pr}^{1/3}$")
    ax.loglog(prandtl, churchill, "--", color=PALETTE[2],
              label="Churchill-Ozoe (all Pr)")
    ax.legend(loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# Module 3 -- dimensional flat-plate distributions
# ---------------------------------------------------------------------------


def plot_flatplate_distributions(result: FlatPlateResult):
    """Four-panel summary of streamwise distributions for one operating point.

    Panels: (1) boundary-layer thicknesses ``delta`` and ``delta_t``;
    (2) local skin-friction ``C_f,x``; (3) local heat-transfer coefficient
    ``h(x)``; (4) local heat flux ``q''(x)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Flat-plate boundary layer -- {result.fluid.name}, "
        f"$U_\\infty$={result.u_inf:g} m/s, $Re_L$={result.Re_L:.2e}",
        fontsize=14,
    )
    x_mm = result.x * 1e3

    ax = axes[0, 0]
    ax.plot(x_mm, result.delta_x * 1e3, color=PALETTE[0], label=r"$\delta$ (momentum)")
    ax.plot(x_mm, result.delta_t_x * 1e3, color=PALETTE[1], label=r"$\delta_t$ (thermal)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("thickness [mm]")
    ax.set_title("Boundary-layer growth")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x_mm, result.cf_x, color=PALETTE[2])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$C_{f,x}$")
    ax.set_title("Local skin-friction coefficient")

    ax = axes[1, 0]
    ax.plot(x_mm, result.h_x, color=PALETTE[3])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$h$ [W/(m$^2$K)]")
    ax.set_title("Local heat-transfer coefficient")

    ax = axes[1, 1]
    ax.plot(x_mm, result.q_x, color=PALETTE[4])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$q''$ [W/m$^2$]")
    ax.set_title("Local wall heat flux")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_cf_nu_vs_reynolds(re_x, cf, nu_x, pr: float):
    """Log-log plot of ``C_f,x`` and ``Nu_x`` versus ``Re_x``.

    Parameters
    ----------
    re_x : array_like
        Reynolds numbers.
    cf : array_like
        Local skin-friction coefficients.
    nu_x : array_like
        Local Nusselt numbers.
    pr : float
        Prandtl number (for the title/legend).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax1 = new_figure(
        xlabel=r"$\mathrm{Re}_x$",
        title=f"Skin friction and Nusselt number vs Reynolds (Pr = {pr:g})",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$C_{f,x}$", color=PALETTE[0])
    ax1.loglog(re_x, cf, color=PALETTE[0], label=r"$C_{f,x}$")
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\mathrm{Nu}_x$", color=PALETTE[1])
    ax2.loglog(re_x, nu_x, color=PALETTE[1], ls="--", label=r"$\mathrm{Nu}_x$")
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.grid(False)
    return fig


__all__ = [
    "plot_blasius_profiles",
    "plot_blasius_verification",
    "plot_temperature_profiles",
    "plot_nusselt_vs_prandtl",
    "plot_flatplate_distributions",
    "plot_cf_nu_vs_reynolds",
]
