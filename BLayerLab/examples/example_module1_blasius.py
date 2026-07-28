"""Example -- Module 1: Blasius momentum boundary layer.

Solves the Blasius equation, prints the integral parameters next to their
accepted reference values, verifies the profile against Howarth's table, and
writes publication-quality figures plus CSV / Tecplot data.

Run::

    python examples/example_module1_blasius.py
"""

from __future__ import annotations

from _common import out

from blayerlab import export_csv, export_tecplot
from blayerlab.blasius import BlasiusSolver
from blayerlab import figures
from blayerlab.plotting import savefig


def main() -> None:
    # --- solve ---
    solver = BlasiusSolver(eta_max=10.0, n_steps=2000)
    solution = solver.solve()

    print(solution.summary())
    print()

    comp = solver.compare_to_howarth(solution)
    print("Verification vs Howarth (1938):")
    print(f"  max |f'_num - f'_ref| = {comp['abs_err_fp'].max():.3e}")
    print()

    # --- figures ---
    fig1 = figures.plot_blasius_profiles(solution)
    fig2 = figures.plot_blasius_verification(solution)
    for name in savefig(fig1, out("module1", "blasius_profiles"), ("png", "pdf")):
        print("wrote", name)
    for name in savefig(fig2, out("module1", "blasius_vs_howarth"), ("png",)):
        print("wrote", name)

    # --- data export (CSV + Tecplot) ---
    csv_path = export_csv(
        out("module1", "blasius_profile.csv"),
        solution.to_dict(),
        header_comment="Blasius similarity solution  f''' + 0.5 f f'' = 0",
    )
    tec_path = export_tecplot(
        out("module1", "blasius_profile.dat"),
        solution.to_dict(),
        title="Blasius momentum boundary layer",
        zone_name="BLASIUS",
    )
    print("wrote", csv_path)
    print("wrote", tec_path)


if __name__ == "__main__":
    main()
