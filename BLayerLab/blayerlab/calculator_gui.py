"""Module 3 (GUI) -- graphical flat-plate heat-transfer calculator.

Where Module 4 (:mod:`blayerlab.interactive`) is an *exploratory* laboratory
driven by sliders, this module is a *precise engineering calculator*: the user
types exact operating values into entry fields, selects a fluid, presses
**Calculate**, and reads the complete boundary-layer / heat-transfer report, a
numerical-vs-correlation comparison, and the streamwise distribution plots.
Results can be exported to CSV or Tecplot with one click.

Like the rest of the laboratory it depends only on Matplotlib
(``matplotlib.widgets``: :class:`~matplotlib.widgets.TextBox`,
:class:`~matplotlib.widgets.RadioButtons`, :class:`~matplotlib.widgets.Button`),
so no additional GUI toolkit is required.

Inputs
------
* Free-stream velocity ``U_inf`` [m/s]
* Plate length ``L`` [m]
* Wall temperature ``T_wall`` [K]
* Free-stream temperature ``T_inf`` [K]
* Fluid (radio selector -> density, viscosity, conductivity, Prandtl number)

Outputs (computed on **Calculate**)
-----------------------------------
Reynolds and Prandtl numbers; momentum and thermal boundary-layer thicknesses;
displacement and momentum thickness; local and average skin-friction
coefficient; wall shear stress; local and average Nusselt number; local and
average heat-transfer coefficient; local and average wall heat flux -- each next
to the classical correlation value.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from . import correlations as corr
from .blasius import BlasiusSolver
from .calculator import FlatPlateCalculator, FlatPlateResult
from .fluids import FLUID_DATABASE, FluidProperties
from .io_utils import export_csv, export_tecplot
from .plotting import PALETTE, use_publication_style
from .thermal import ThermalSolver


class CalculatorGUI:
    """Interactive, entry-field driven flat-plate calculator window.

    Parameters
    ----------
    fluids : dict of str to FluidProperties, optional
        Fluids offered in the selector; defaults to
        :data:`blayerlab.fluids.FLUID_DATABASE`.
    output_dir : str, optional
        Directory used by the export buttons.  Defaults to the current working
        directory.

    Examples
    --------
    >>> gui = CalculatorGUI()      # doctest: +SKIP
    >>> gui.launch()               # opens the window   # doctest: +SKIP
    """

    #: Default operating point shown on first launch.
    DEFAULTS = dict(u_inf="2.0", length="0.5", t_wall="350.0", t_inf="300.0")

    def __init__(
        self,
        fluids: dict[str, FluidProperties] | None = None,
        output_dir: str | None = None,
    ) -> None:
        use_publication_style()
        self.fluids = dict(fluids) if fluids is not None else dict(FLUID_DATABASE)
        self.output_dir = os.path.abspath(output_dir or os.getcwd())

        # Shared solvers (computed once, re-used for every calculation).
        self.blasius = BlasiusSolver().solve()
        self.calc = FlatPlateCalculator(self.blasius, ThermalSolver(self.blasius))

        self._fluid_keys = list(self.fluids.keys())
        self._current_key = "air" if "air" in self.fluids else self._fluid_keys[0]
        self._last: FlatPlateResult | None = None

        self._build_figure()
        self._connect()
        self.calculate()  # populate with the default operating point

    # -- figure construction --------------------------------------------------

    def _build_figure(self) -> None:
        """Create axes, entry fields, buttons, text panels and result plots."""
        self.fig = plt.figure(figsize=(14.0, 8.6))
        self.fig.suptitle(
            "Flat-Plate Heat-Transfer Calculator", fontsize=15, fontweight="bold"
        )

        # ---- input column ----
        self.ax_radio = self.fig.add_axes([0.015, 0.60, 0.15, 0.31])
        self.ax_radio.set_title("Fluid", fontsize=11)
        labels = [self.fluids[k].name for k in self._fluid_keys]
        active = self._fluid_keys.index(self._current_key)
        self.radio = RadioButtons(self.ax_radio, labels, active=active)

        # Entry fields (label drawn to the left by the TextBox widget).
        self.tb_u = TextBox(self.fig.add_axes([0.095, 0.53, 0.06, 0.035]),
                            r"$U_\infty$ [m/s]  ", initial=self.DEFAULTS["u_inf"])
        self.tb_L = TextBox(self.fig.add_axes([0.095, 0.475, 0.06, 0.035]),
                            r"$L$ [m]  ", initial=self.DEFAULTS["length"])
        self.tb_tw = TextBox(self.fig.add_axes([0.095, 0.42, 0.06, 0.035]),
                             r"$T_{wall}$ [K]  ", initial=self.DEFAULTS["t_wall"])
        self.tb_ti = TextBox(self.fig.add_axes([0.095, 0.365, 0.06, 0.035]),
                             r"$T_\infty$ [K]  ", initial=self.DEFAULTS["t_inf"])

        # Buttons.
        self.btn_calc = Button(self.fig.add_axes([0.02, 0.29, 0.145, 0.05]),
                               "Calculate", color="#0072B2", hovercolor="#3792d0")
        self.btn_calc.label.set_color("white")
        self.btn_calc.label.set_fontweight("bold")
        self.btn_csv = Button(self.fig.add_axes([0.02, 0.225, 0.068, 0.045]),
                              "Export CSV")
        self.btn_tec = Button(self.fig.add_axes([0.097, 0.225, 0.068, 0.045]),
                              "Export .dat")
        self.btn_csv.label.set_fontsize(8.5)
        self.btn_tec.label.set_fontsize(8.5)

        # Status line.
        self.ax_status = self.fig.add_axes([0.015, 0.10, 0.16, 0.11])
        self.ax_status.axis("off")
        self._status = self.ax_status.text(
            0.0, 1.0, "", va="top", ha="left", fontsize=7.0, wrap=True,
            transform=self.ax_status.transAxes,
        )

        # ---- results text panels ----
        self.ax_report = self.fig.add_axes([0.185, 0.06, 0.20, 0.86])
        self.ax_report.axis("off")
        self._report = self.ax_report.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9.0,
            transform=self.ax_report.transAxes,
        )

        # ---- distribution plots (2x2) ----
        self.ax_delta = self.fig.add_axes([0.46, 0.58, 0.22, 0.33])
        self.ax_cf = self.fig.add_axes([0.75, 0.58, 0.22, 0.33])
        self.ax_h = self.fig.add_axes([0.46, 0.12, 0.22, 0.33])
        self.ax_q = self.fig.add_axes([0.75, 0.12, 0.22, 0.33])
        self.plot_axes = (self.ax_delta, self.ax_cf, self.ax_h, self.ax_q)

    def _connect(self) -> None:
        """Wire widget callbacks."""
        self.btn_calc.on_clicked(lambda _e: self.calculate())
        self.btn_csv.on_clicked(self._on_export_csv)
        self.btn_tec.on_clicked(self._on_export_tecplot)
        self.radio.on_clicked(self._on_fluid)
        # Pressing Enter in any field also triggers a calculation.
        for tb in (self.tb_u, self.tb_L, self.tb_tw, self.tb_ti):
            tb.on_submit(lambda _text: self.calculate())

    # -- callbacks ------------------------------------------------------------

    def _on_fluid(self, label: str) -> None:
        """Radio callback: switch active fluid then recompute."""
        for key, fl in self.fluids.items():
            if fl.name == label:
                self._current_key = key
                break
        self.calculate()

    def _on_export_csv(self, _event) -> None:
        if self._last is None:
            self._set_status("Nothing to export -- press Calculate first.", ok=False)
            return
        path = os.path.join(self.output_dir, self._file_stem() + ".csv")
        export_csv(
            path,
            self._last.to_distribution_dict(),
            header_comment=f"Flat-plate distributions -- {self._last.fluid.name}, "
            f"U={self._last.u_inf} m/s, L={self._last.length} m",
        )
        self._set_status(f"Saved CSV:\n{path}", ok=True)

    def _on_export_tecplot(self, _event) -> None:
        if self._last is None:
            self._set_status("Nothing to export -- press Calculate first.", ok=False)
            return
        path = os.path.join(self.output_dir, self._file_stem() + ".dat")
        export_tecplot(
            path,
            self._last.to_distribution_dict(),
            title=f"Flat-plate distributions ({self._last.fluid.name})",
            zone_name=self._last.fluid.name.upper(),
        )
        self._set_status(f"Saved Tecplot:\n{path}", ok=True)

    # -- the core action ------------------------------------------------------

    def calculate(self) -> None:
        """Read the entry fields, run the calculator and refresh the window."""
        try:
            u_inf = float(self.tb_u.text)
            length = float(self.tb_L.text)
            t_wall = float(self.tb_tw.text)
            t_inf = float(self.tb_ti.text)
        except ValueError:
            self._set_status("Invalid input: enter numeric values.", ok=False)
            return
        if u_inf <= 0 or length <= 0:
            self._set_status("U_inf and L must be positive.", ok=False)
            return

        fluid = self.fluids[self._current_key]
        result = self.calc.compute(u_inf, fluid, length, t_wall, t_inf)
        self._last = result

        self._report.set_text(self._format_report(result))
        self._draw_plots(result)
        regime = "LAMINAR" if result.is_laminar else "TURBULENT (Re_L > 5e5)"
        # Regime on its own line, under "Computed OK.", so it never overlaps
        # the neighbouring report panel.
        self._set_status(f"Computed OK.\nRegime: {regime}",
                         ok=result.is_laminar)
        self.fig.canvas.draw_idle()

    # -- rendering helpers ----------------------------------------------------

    def _draw_plots(self, r: FlatPlateResult) -> None:
        """Redraw the four streamwise-distribution panels for result ``r``."""
        for ax in self.plot_axes:
            ax.clear()
        x_mm = r.x * 1e3

        self.ax_delta.plot(x_mm, r.delta_x * 1e3, color=PALETTE[0], label=r"$\delta$")
        self.ax_delta.plot(x_mm, r.delta_t_x * 1e3, color=PALETTE[1],
                           label=r"$\delta_t$")
        self.ax_delta.set_title("Boundary-layer thickness", fontsize=11)
        self.ax_delta.set_ylabel("[mm]")
        self.ax_delta.legend(fontsize=9, loc="upper left")

        self.ax_cf.plot(x_mm, r.cf_x, color=PALETTE[2])
        self.ax_cf.set_title(r"Local skin friction $C_{f,x}$", fontsize=11)

        self.ax_h.plot(x_mm, r.h_x, color=PALETTE[3])
        self.ax_h.set_title(r"Heat-transfer coeff. $h$ [W/m$^2$K]", fontsize=11)
        self.ax_h.set_xlabel("x [mm]")

        self.ax_q.plot(x_mm, r.q_x, color=PALETTE[4])
        self.ax_q.set_title(r"Wall heat flux $q''$ [W/m$^2$]", fontsize=11)
        self.ax_q.set_xlabel("x [mm]")

    def _format_report(self, r: FlatPlateResult) -> str:
        """Return a compact monospace report with correlation comparison."""
        sep = "-" * 30
        dT = r.t_wall - r.t_inf
        # Correlation references.
        cf_c = corr.cf_local(r.Re_L)
        nu_c = corr.nu_local_churchill_ozoe(r.Re_L, r.Pr)

        def rel(a, b):
            return abs(a - b) / abs(b) if b else float("nan")

        lines = [
            "INPUTS",
            sep,
            f" Fluid    : {r.fluid.name}",
            f" U_inf    : {r.u_inf:.4g} m/s",
            f" L        : {r.length:.4g} m",
            f" T_wall   : {r.t_wall:.4g} K",
            f" T_inf    : {r.t_inf:.4g} K  (dT={dT:.4g})",
            "",
            "DIMENSIONLESS",
            sep,
            f" Re_L     : {r.Re_L:.4e}",
            f" Pr       : {r.Pr:.4e}",
            "",
            "MOMENTUM LAYER (x = L)",
            sep,
            f" delta    : {r.delta_L*1e3:9.4f} mm",
            f" delta*   : {r.delta_star_L*1e3:9.4f} mm",
            f" theta    : {r.theta_mom_L*1e3:9.4f} mm",
            f" H factor : {r.shape_factor:9.4f}",
            f" Cf,x     : {r.cf_local_L:.4e}",
            f" Cf,avg   : {r.cf_avg:.4e}",
            f" tau_w    : {r.tau_wall_L:.4e} Pa",
            "",
            "THERMAL LAYER (x = L)",
            sep,
            f" delta_t  : {r.delta_t_L*1e3:9.4f} mm",
            f" Nu_x     : {r.nu_local_L:.4e}",
            f" Nu_avg   : {r.nu_avg:.4e}",
            f" h_x      : {r.h_local_L:.4e} W/m2K",
            f" h_bar    : {r.h_avg:.4e} W/m2K",
            f" q''_x    : {r.q_local_L:.4e} W/m2",
            f" q''_avg  : {r.q_avg:.4e} W/m2",
            f" Q' (span): {r.heat_rate_per_width:.4e} W/m",
            "",
            "NUMERICAL vs CORRELATION",
            sep,
            f" Cf,x  {r.cf_local_L:.3e} | {cf_c:.3e}  {rel(r.cf_local_L, cf_c):.1e}",
            f" Nu_x  {r.nu_local_L:.3e} | {nu_c:.3e}  {rel(r.nu_local_L, nu_c):.1e}",
        ]
        return "\n".join(lines)

    def _file_stem(self) -> str:
        """Filename stem for exports, based on the current fluid."""
        assert self._last is not None
        name = self._last.fluid.name.lower().replace(" ", "_")
        return f"flatplate_{name}"

    def _set_status(self, message: str, ok: bool = True) -> None:
        """Update the status line (green if ok, red on error)."""
        self._status.set_color("#1a7a3a" if ok else "#b00020")
        self._status.set_text(message)
        self.fig.canvas.draw_idle()

    # -- entry point ----------------------------------------------------------

    def launch(self) -> None:
        """Show the calculator window (blocks until it is closed)."""
        plt.show()


def launch() -> None:
    """Convenience: build and immediately show a :class:`CalculatorGUI`."""
    CalculatorGUI().launch()


__all__ = ["CalculatorGUI", "launch"]
