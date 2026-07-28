"""Module 6 -- Parametric studies.

This module automates the "what happens if I change ..." experiments that a
student or lecturer wants from a virtual laboratory.  Each study sweeps one
governing quantity while holding the others fixed, returns the raw data as
``{name: array}`` mappings (ready for CSV/Tecplot export), and produces a
publication-quality figure.

Studies provided
----------------
* :meth:`ParametricStudy.sweep_reynolds` -- influence of Reynolds number.
* :meth:`ParametricStudy.sweep_prandtl`  -- influence of Prandtl number.
* :meth:`ParametricStudy.sweep_length`   -- influence of plate length.
* :meth:`ParametricStudy.sweep_fluids`   -- comparison across fluids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import correlations as corr
from .blasius import BlasiusSolver
from .calculator import FlatPlateCalculator
from .fluids import FluidProperties
from .plotting import PALETTE, new_figure
from .thermal import ThermalSolver


@dataclass
class StudyResult:
    """A parametric-study result: tabular data plus a rendered figure.

    Attributes
    ----------
    name : str
        Identifier of the study (used for file names).
    data : dict of str to numpy.ndarray
        Column data, suitable for :func:`blayerlab.io_utils.export_csv`.
    figure : matplotlib.figure.Figure
        The publication-quality figure for the study.
    """

    name: str
    data: dict[str, np.ndarray]
    figure: "plt.Figure"


class ParametricStudy:
    """Generate parametric studies of the laminar flat-plate boundary layer.

    Parameters
    ----------
    blasius, thermal_solver : optional
        Pre-built solvers, re-used across studies for efficiency.
    """

    def __init__(self, blasius=None, thermal_solver=None) -> None:
        self.blasius = blasius if blasius is not None else BlasiusSolver().solve()
        self.thermal = (
            thermal_solver
            if thermal_solver is not None
            else ThermalSolver(self.blasius)
        )
        self.calc = FlatPlateCalculator(self.blasius, self.thermal)

    # -- Reynolds sweep -------------------------------------------------------

    def sweep_reynolds(
        self,
        re_range: tuple[float, float] = (1.0e3, 5.0e5),
        pr: float = 0.71,
        n: int = 60,
    ) -> StudyResult:
        """Influence of Reynolds number on ``C_f``, ``Nu_x`` and thickness ratio.

        Parameters
        ----------
        re_range : (float, float)
            Logarithmic range of local Reynolds number.
        pr : float
            Prandtl number held fixed.
        n : int
            Number of samples.

        Returns
        -------
        StudyResult
        """
        Re = np.logspace(np.log10(re_range[0]), np.log10(re_range[1]), n)
        tsol = self.thermal.solve(pr)

        cf = self.blasius.cf_local(Re)
        nu_x = tsol.nu_local(Re)
        # delta/x and delta_t/x are constants * Re^{-1/2}; report delta/x.
        delta_over_x = self.blasius.eta_99 / np.sqrt(Re)

        data = {
            "Re_x": Re,
            "Cf_x": cf,
            "Nu_x": nu_x,
            "delta_over_x": delta_over_x,
        }

        fig, ax1 = new_figure(
            xlabel=r"$\mathrm{Re}_x$",
            title=f"Reynolds-number influence (Pr = {pr:g})",
        )
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.loglog(Re, cf, color=PALETTE[0], label=r"$C_{f,x}\propto Re_x^{-1/2}$")
        ax1.set_ylabel(r"$C_{f,x}$", color=PALETTE[0])
        ax1.tick_params(axis="y", labelcolor=PALETTE[0])
        ax2 = ax1.twinx()
        ax2.set_yscale("log")
        ax2.loglog(Re, nu_x, color=PALETTE[1], ls="--",
                   label=r"$\mathrm{Nu}_x\propto Re_x^{1/2}$")
        ax2.set_ylabel(r"$\mathrm{Nu}_x$", color=PALETTE[1])
        ax2.tick_params(axis="y", labelcolor=PALETTE[1])
        ax2.grid(False)
        return StudyResult("sweep_reynolds", data, fig)

    # -- Prandtl sweep --------------------------------------------------------

    def sweep_prandtl(
        self,
        pr_range: tuple[float, float] = (1.0e-2, 1.0e3),
        n: int = 40,
    ) -> StudyResult:
        """Influence of Prandtl number on the wall gradient and thickness ratio.

        Reports ``theta'(0) = Nu_x/Re_x^{1/2}`` and the thermal-to-momentum
        thickness ratio ``delta_t/delta`` (~ ``Pr^{-1/3}`` for gases/liquids).

        Returns
        -------
        StudyResult
        """
        Pr = np.logspace(np.log10(pr_range[0]), np.log10(pr_range[1]), n)
        thetap0 = np.empty_like(Pr)
        ratio = np.empty_like(Pr)
        for i, p in enumerate(Pr):
            sol = self.thermal.solve(p)
            thetap0[i] = sol.thetap0
            ratio[i] = sol.eta_t99 / self.blasius.eta_99

        pohlhausen = 0.332 * np.cbrt(Pr)

        data = {
            "Pr": Pr,
            "theta_prime_0": thetap0,
            "pohlhausen_0.332Pr^(1/3)": pohlhausen,
            "delta_t_over_delta": ratio,
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.loglog(Pr, thetap0, "o", ms=4, color=PALETTE[0],
                   markerfacecolor="white", label="numerical")
        ax1.loglog(Pr, pohlhausen, "-", color=PALETTE[1],
                   label=r"$0.332\,\mathrm{Pr}^{1/3}$")
        ax1.set_xlabel("Pr")
        ax1.set_ylabel(r"$\theta'(0) = \mathrm{Nu}_x/\mathrm{Re}_x^{1/2}$")
        ax1.set_title("Wall temperature gradient")
        ax1.legend()

        ax2.loglog(Pr, ratio, "s", ms=4, color=PALETTE[2],
                   markerfacecolor="white", label=r"numerical $\delta_t/\delta$")
        ax2.loglog(Pr, Pr ** (-1.0 / 3.0), "-", color=PALETTE[3],
                   label=r"$\mathrm{Pr}^{-1/3}$")
        ax2.set_xlabel("Pr")
        ax2.set_ylabel(r"$\delta_t/\delta$")
        ax2.set_title("Thermal / momentum thickness ratio")
        ax2.legend()
        fig.tight_layout()
        return StudyResult("sweep_prandtl", data, fig)

    # -- Length sweep ---------------------------------------------------------

    def sweep_length(
        self,
        fluid: FluidProperties,
        u_inf: float,
        t_wall: float,
        t_inf: float,
        length_range: tuple[float, float] = (0.05, 2.0),
        n: int = 50,
    ) -> StudyResult:
        """Influence of plate length on trailing-edge quantities.

        Sweeps ``L`` and records the trailing-edge boundary-layer thickness,
        average heat-transfer coefficient and total heat rate per unit width.

        Returns
        -------
        StudyResult
        """
        lengths = np.linspace(length_range[0], length_range[1], n)
        delta_L = np.empty_like(lengths)
        h_avg = np.empty_like(lengths)
        q_prime = np.empty_like(lengths)
        for i, L in enumerate(lengths):
            res = self.calc.compute(u_inf, fluid, float(L), t_wall, t_inf)
            delta_L[i] = res.delta_L
            h_avg[i] = res.h_avg
            q_prime[i] = res.heat_rate_per_width

        data = {
            "L [m]": lengths,
            "delta_L [m]": delta_L,
            "h_avg [W/m2K]": h_avg,
            "Q_per_width [W/m]": q_prime,
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Plate-length influence -- {fluid.name}, "
            f"$U_\\infty$={u_inf:g} m/s, $\\Delta T$={t_wall - t_inf:g} K"
        )
        ax1.plot(lengths, delta_L * 1e3, color=PALETTE[0], label=r"$\delta(L)$")
        ax1.set_xlabel("L [m]")
        ax1.set_ylabel(r"$\delta$ at trailing edge [mm]")
        ax1.set_title("Boundary-layer thickness")

        ax2.plot(lengths, h_avg, color=PALETTE[1], label=r"$\bar h$")
        ax2.plot(lengths, q_prime, color=PALETTE[3], ls="--", label=r"$Q'$ [W/m]")
        ax2.set_xlabel("L [m]")
        ax2.set_ylabel(r"$\bar h$ [W/(m$^2$K)]   /   $Q'$ [W/m]")
        ax2.set_title("Average coefficient and heat rate")
        ax2.legend()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return StudyResult("sweep_length", data, fig)

    # -- Fluid comparison -----------------------------------------------------

    def sweep_fluids(
        self,
        fluids: Sequence[FluidProperties],
        u_inf: float,
        length: float,
        t_wall: float,
        t_inf: float,
    ) -> StudyResult:
        """Compare heat transfer across several fluids at one operating point.

        Returns
        -------
        StudyResult
            Bar-chart figure and per-fluid data (Re, Pr, Nu_L, h_avg).
        """
        names, Re, Pr, Nu, h = [], [], [], [], []
        for fl in fluids:
            res = self.calc.compute(u_inf, fl, length, t_wall, t_inf)
            names.append(fl.name)
            Re.append(res.Re_L)
            Pr.append(res.Pr)
            Nu.append(res.nu_avg)
            h.append(res.h_avg)

        data = {
            "fluid_index": np.arange(len(names), dtype=float),
            "Re_L": np.array(Re),
            "Pr": np.array(Pr),
            "Nu_L_avg": np.array(Nu),
            "h_avg [W/m2K]": np.array(h),
        }

        fig, ax = new_figure(
            ylabel=r"$\overline{\mathrm{Nu}}_L$",
            title=f"Average Nusselt number across fluids "
            f"($U_\\infty$={u_inf:g} m/s, L={length:g} m)",
        )
        positions = np.arange(len(names))
        bars = ax.bar(positions, Nu, color=PALETTE[: len(names)])
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        for rect, p in zip(bars, Pr):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"Pr={p:.2g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        return StudyResult("sweep_fluids", data, fig)


__all__ = ["StudyResult", "ParametricStudy"]
