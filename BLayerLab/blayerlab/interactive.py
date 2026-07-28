"""Module 4 -- Interactive Boundary Layer Laboratory.

A single Matplotlib window that lets the user vary the operating point and see
*every* momentum- and thermal-boundary-layer quantity update in real time.  It
uses only :mod:`matplotlib.widgets` (sliders, radio buttons, a button) so that
no dependency beyond Matplotlib is required.

Controls
--------
* **Fluid** (radio buttons) -- selects density, viscosity, conductivity and
  hence the Prandtl number from the built-in database.
* **U_inf** slider -- free-stream velocity (drives the Reynolds number).
* **L** slider -- plate length (also drives the Reynolds number).
* **T_wall** / **T_inf** sliders -- wall and free-stream temperatures.
* **Reset** button.

Live displays
-------------
* Velocity profile ``u(y)`` and temperature profile ``T(y)`` at the trailing
  edge ``x = L``, with ``delta`` and ``delta_t`` marked.
* Boundary-layer growth ``delta(x)``, ``delta_t(x)`` along the plate.
* A text panel with ``Re_L``, ``Pr``, ``delta``, ``delta_t``, ``C_f``,
  ``tau_w``, ``Nu_x``, ``Nu_L``, ``h``, ``h_bar`` and ``q''``.

Design note
-----------
Reynolds and Prandtl numbers are *derived* from the physical inputs rather than
set independently: a virtual laboratory should stay thermodynamically
consistent (you cannot pick ``Re`` and ``Pr`` in contradiction with the chosen
fluid).  Because the Blasius solution is universal and the thermal solutions are
cached per fluid, slider motion triggers only cheap array rescaling -- so the
figure updates smoothly without re-solving any ODE.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from .blasius import BlasiusSolver
from .fluids import FLUID_DATABASE, FluidProperties
from .plotting import PALETTE, use_publication_style
from .thermal import ThermalSolver


class InteractiveLab:
    """Build and run the interactive flat-plate boundary-layer laboratory.

    Parameters
    ----------
    fluids : dict of str to FluidProperties, optional
        Fluids offered in the radio selector.  Defaults to
        :data:`blayerlab.fluids.FLUID_DATABASE`.

    Examples
    --------
    >>> lab = InteractiveLab()          # doctest: +SKIP
    >>> lab.launch()                    # opens the window   # doctest: +SKIP
    """

    def __init__(self, fluids: dict[str, FluidProperties] | None = None) -> None:
        use_publication_style()
        self.fluids = dict(fluids) if fluids is not None else dict(FLUID_DATABASE)

        # Solve the universal momentum field once and cache thermal fields.
        self.blasius = BlasiusSolver().solve()
        self.thermal_solver = ThermalSolver(self.blasius)
        self._thermal_cache: dict[str, object] = {}

        # Default operating point.
        self._fluid_keys = list(self.fluids.keys())
        self._current_key = "air" if "air" in self.fluids else self._fluid_keys[0]
        self._defaults = dict(u_inf=2.0, length=0.5, t_wall=350.0, t_inf=300.0)

        self._build_figure()
        self._connect()
        self.update()

    # -- thermal caching ------------------------------------------------------

    def _thermal(self, fluid: FluidProperties):
        """Return (cached) thermal solution for a fluid's Prandtl number."""
        key = fluid.name
        if key not in self._thermal_cache:
            self._thermal_cache[key] = self.thermal_solver.solve(fluid.Pr)
        return self._thermal_cache[key]

    # -- figure construction --------------------------------------------------

    def _build_figure(self) -> None:
        """Lay out axes, plot lines, sliders, radio buttons and text panel."""
        self.fig = plt.figure(figsize=(13.5, 8.0))
        self.fig.suptitle(
            "Interactive Boundary Layer Laboratory", fontsize=15, fontweight="bold"
        )

        # Plot axes.
        self.ax_vel = self.fig.add_axes([0.31, 0.55, 0.30, 0.36])
        self.ax_temp = self.fig.add_axes([0.67, 0.55, 0.30, 0.36])
        self.ax_growth = self.fig.add_axes([0.31, 0.30, 0.66, 0.17])

        self.ax_vel.set_title("Velocity profile at x = L")
        self.ax_vel.set_xlabel(r"$u$ [m/s]")
        self.ax_vel.set_ylabel(r"$y$ [mm]")
        self.ax_temp.set_title("Temperature profile at x = L")
        self.ax_temp.set_xlabel(r"$T$ [K]")
        self.ax_temp.set_ylabel(r"$y$ [mm]")
        self.ax_growth.set_title("Boundary-layer growth")
        self.ax_growth.set_xlabel(r"$x$ [mm]")
        self.ax_growth.set_ylabel("thickness [mm]")

        # Plot lines (updated in place).
        (self._ln_u,) = self.ax_vel.plot([], [], color=PALETTE[0], lw=2)
        self._ln_delta = self.ax_vel.axhline(0.0, color="0.5", ls="--", lw=1)
        (self._ln_T,) = self.ax_temp.plot([], [], color=PALETTE[1], lw=2)
        self._ln_delta_t = self.ax_temp.axhline(0.0, color="0.5", ls="--", lw=1)
        (self._ln_gd,) = self.ax_growth.plot([], [], color=PALETTE[0],
                                             label=r"$\delta$")
        (self._ln_gdt,) = self.ax_growth.plot([], [], color=PALETTE[1], ls="--",
                                              label=r"$\delta_t$")
        self.ax_growth.legend(loc="upper left", ncol=2)

        # Text panel (results).
        self.ax_text = self.fig.add_axes([0.02, 0.30, 0.24, 0.30])
        self.ax_text.axis("off")
        self._txt = self.ax_text.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10,
            transform=self.ax_text.transAxes,
        )

        # Fluid radio selector.
        self.ax_radio = self.fig.add_axes([0.02, 0.63, 0.16, 0.28])
        self.ax_radio.set_title("Fluid", fontsize=11)
        labels = [self.fluids[k].name for k in self._fluid_keys]
        active = self._fluid_keys.index(self._current_key)
        self.radio = RadioButtons(self.ax_radio, labels, active=active)

        # Sliders.
        axcolor = "0.92"
        self.ax_u = self.fig.add_axes([0.33, 0.19, 0.52, 0.025], facecolor=axcolor)
        self.ax_L = self.fig.add_axes([0.33, 0.15, 0.52, 0.025], facecolor=axcolor)
        self.ax_tw = self.fig.add_axes([0.33, 0.11, 0.52, 0.025], facecolor=axcolor)
        self.ax_ti = self.fig.add_axes([0.33, 0.07, 0.52, 0.025], facecolor=axcolor)

        d = self._defaults
        self.s_u = Slider(self.ax_u, r"$U_\infty$ [m/s]", 0.1, 20.0, valinit=d["u_inf"])
        self.s_L = Slider(self.ax_L, r"$L$ [m]", 0.02, 3.0, valinit=d["length"])
        self.s_tw = Slider(self.ax_tw, r"$T_{wall}$ [K]", 250.0, 600.0,
                           valinit=d["t_wall"])
        self.s_ti = Slider(self.ax_ti, r"$T_\infty$ [K]", 250.0, 600.0,
                           valinit=d["t_inf"])

        # Reset button.
        self.ax_reset = self.fig.add_axes([0.90, 0.11, 0.07, 0.045])
        self.btn_reset = Button(self.ax_reset, "Reset")

    def _connect(self) -> None:
        """Wire the widget callbacks to :meth:`update`."""
        for s in (self.s_u, self.s_L, self.s_tw, self.s_ti):
            s.on_changed(lambda _val: self.update())
        self.radio.on_clicked(self._on_fluid)
        self.btn_reset.on_clicked(self._on_reset)

    # -- callbacks ------------------------------------------------------------

    def _on_fluid(self, label: str) -> None:
        """Radio callback: switch the active fluid by its display name."""
        for key, fl in self.fluids.items():
            if fl.name == label:
                self._current_key = key
                break
        self.update()

    def _on_reset(self, _event) -> None:
        """Reset button callback: restore default operating point."""
        for s in (self.s_u, self.s_L, self.s_tw, self.s_ti):
            s.reset()
        self.update()

    # -- the core update ------------------------------------------------------

    def update(self, *_args) -> None:
        """Recompute all quantities and refresh every artist in the window."""
        fluid = self.fluids[self._current_key]
        nu, k, rho, Pr = fluid.nu, fluid.k, fluid.rho, fluid.Pr
        u_inf = self.s_u.val
        L = self.s_L.val
        t_wall = self.s_tw.val
        t_inf = self.s_ti.val
        dT = t_wall - t_inf

        Re_L = u_inf * L / nu
        tsol = self._thermal(fluid)

        # Similarity -> physical length scale at x = L:  y = eta*sqrt(nu*L/U).
        y_scale = np.sqrt(nu * L / u_inf)

        # Velocity profile.
        y_u_mm = self.blasius.eta * y_scale * 1e3
        u_prof = u_inf * self.blasius.fp
        self._ln_u.set_data(u_prof, y_u_mm)

        # Temperature profile (own eta grid).
        y_t_mm = tsol.eta * y_scale * 1e3
        T_prof = t_wall + (t_inf - t_wall) * tsol.theta
        self._ln_T.set_data(T_prof, y_t_mm)

        # Scalar quantities at x = L.
        delta = float(self.blasius.delta(Re_L, L))
        delta_star = float(self.blasius.displacement_thickness(Re_L, L))
        theta_mom = float(self.blasius.momentum_thickness(Re_L, L))
        cf = float(self.blasius.cf_local(Re_L))
        tau_w = 0.5 * rho * u_inf**2 * cf
        delta_t = float(tsol.thermal_thickness(Re_L, L))
        nu_x = float(tsol.nu_local(Re_L))
        nu_avg = float(tsol.nu_average(Re_L))
        h_x = nu_x * k / L
        h_avg = nu_avg * k / L
        q_x = h_x * dT

        # Marker lines for delta / delta_t.
        self._ln_delta.set_ydata([delta * 1e3, delta * 1e3])
        self._ln_delta_t.set_ydata([delta_t * 1e3, delta_t * 1e3])

        # Boundary-layer growth along the plate.
        x = np.linspace(L / 200.0, L, 200)
        Re_x = u_inf * x / nu
        self._ln_gd.set_data(x * 1e3, self.blasius.delta(Re_x, x) * 1e3)
        self._ln_gdt.set_data(x * 1e3, tsol.thermal_thickness(Re_x, x) * 1e3)

        # Rescale axes.
        y_top = max(delta, delta_t) * 1e3 * 1.6
        self.ax_vel.set_xlim(0, u_inf * 1.05)
        self.ax_vel.set_ylim(0, y_top)
        self.ax_temp.set_xlim(min(t_wall, t_inf) - 5, max(t_wall, t_inf) + 5)
        self.ax_temp.set_ylim(0, y_top)
        self.ax_growth.set_xlim(0, L * 1e3)
        self.ax_growth.relim()
        self.ax_growth.autoscale_view(scalex=False)

        # Text panel.
        regime = "LAMINAR" if Re_L <= 5e5 else "Re_L>5e5 (transition!)"
        self._txt.set_text(
            "RESULTS  (at x = L)\n"
            "--------------------------\n"
            f"Fluid    : {fluid.name}\n"
            f"Re_L     : {Re_L:10.3e}\n"
            f"Pr       : {Pr:10.3e}\n"
            f"regime   : {regime}\n"
            "--------------------------\n"
            f"delta    : {delta*1e3:8.3f} mm\n"
            f"delta*   : {delta_star*1e3:8.3f} mm\n"
            f"theta    : {theta_mom*1e3:8.3f} mm\n"
            f"delta_t  : {delta_t*1e3:8.3f} mm\n"
            f"Cf,x     : {cf:10.3e}\n"
            f"tau_w    : {tau_w:10.3e} Pa\n"
            "--------------------------\n"
            f"Nu_x     : {nu_x:10.3e}\n"
            f"Nu_L,avg : {nu_avg:10.3e}\n"
            f"h_x      : {h_x:10.3e} W/m2K\n"
            f"h_bar    : {h_avg:10.3e} W/m2K\n"
            f"q''_x    : {q_x:10.3e} W/m2"
        )

        self.fig.canvas.draw_idle()

    # -- entry point ----------------------------------------------------------

    def launch(self) -> None:
        """Show the interactive window (blocks until the window is closed)."""
        plt.show()


def launch() -> None:
    """Convenience function: build and immediately show an :class:`InteractiveLab`."""
    InteractiveLab().launch()


__all__ = ["InteractiveLab", "launch"]
