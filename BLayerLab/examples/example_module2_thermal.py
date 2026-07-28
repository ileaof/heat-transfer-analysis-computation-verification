"""Example -- Module 2: Thermal boundary layer.

Solves the energy equation for the canonical set of Prandtl numbers, tabulates
the wall temperature gradient ``theta'(0)`` (= ``Nu_x/Re_x^{1/2}``) against its
analytical closed form, and writes comparative temperature-profile figures and
data files.

Run::

    python examples/example_module2_thermal.py
"""

from __future__ import annotations

from _common import out

from blayerlab import export_csv
from blayerlab.blasius import BlasiusSolver
from blayerlab.thermal import DEFAULT_PRANDTL_NUMBERS, ThermalSolver
from blayerlab import figures
from blayerlab.plotting import savefig


def main() -> None:
    blasius = BlasiusSolver().solve()
    solver = ThermalSolver(blasius)

    # --- solve for every Prandtl number ---
    solutions = solver.solve_many(DEFAULT_PRANDTL_NUMBERS)

    print(f"{'Pr':>8} {'theta_prime(0)':>15} {'analytical':>12} "
          f"{'Nu_x/Re^0.5':>12} {'eta_t99':>9}")
    pr_col, num_col, ana_col, tt_col = [], [], [], []
    for Pr, sol in solutions.items():
        ana = solver.wall_gradient_analytical(Pr)
        print(f"{Pr:8.2f} {sol.thetap0:15.6f} {ana:12.6f} "
              f"{sol.thetap0:12.6f} {sol.eta_t99:9.4f}")
        pr_col.append(Pr)
        num_col.append(sol.thetap0)
        ana_col.append(ana)
        tt_col.append(sol.eta_t99)

    # --- figures ---
    fig1 = figures.plot_temperature_profiles(solutions)
    fig2 = figures.plot_nusselt_vs_prandtl(solver)
    for name in savefig(fig1, out("module2", "temperature_profiles"), ("png", "pdf")):
        print("wrote", name)
    for name in savefig(fig2, out("module2", "nusselt_vs_prandtl"), ("png",)):
        print("wrote", name)

    # --- export the summary table ---
    csv_path = export_csv(
        out("module2", "thermal_summary.csv"),
        {
            "Pr": pr_col,
            "theta_prime_0_numerical": num_col,
            "theta_prime_0_analytical": ana_col,
            "eta_t99": tt_col,
        },
        header_comment="Thermal boundary layer: wall gradient theta'(0) vs Pr",
    )
    print("wrote", csv_path)

    # Per-Pr temperature profiles.
    for Pr, sol in solutions.items():
        p = export_csv(out("module2", f"theta_profile_Pr{Pr:g}.csv"), sol.to_dict())
        print("wrote", p)


if __name__ == "__main__":
    main()
