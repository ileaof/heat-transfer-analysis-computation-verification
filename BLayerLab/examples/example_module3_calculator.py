"""Example -- Module 3: Flat-plate heat-transfer calculator.

Evaluates the complete flat-plate boundary layer for a couple of operating
points (air and water), prints the engineering report and the
numerical-vs-correlation comparison, and writes the streamwise distribution
figure and CSV.

Run::

    python examples/example_module3_calculator.py
"""

from __future__ import annotations

from _common import out

from blayerlab import export_csv, export_tecplot, get_fluid
from blayerlab.calculator import FlatPlateCalculator
from blayerlab import figures
from blayerlab.plotting import savefig


CASES = [
    # (label, fluid, U_inf [m/s], L [m], T_wall [K], T_inf [K])
    ("air", get_fluid("air"), 2.0, 0.5, 350.0, 300.0),
    ("water", get_fluid("water"), 0.5, 0.5, 320.0, 300.0),
]


def main() -> None:
    calc = FlatPlateCalculator()

    for label, fluid, u_inf, length, t_wall, t_inf in CASES:
        result = calc.compute(u_inf, fluid, length, t_wall, t_inf)

        print(result.report())
        print()
        print(result.correlation_comparison())
        print()

        fig = figures.plot_flatplate_distributions(result)
        for name in savefig(fig, out("module3", f"distributions_{label}"), ("png",)):
            print("wrote", name)

        csv_path = export_csv(
            out("module3", f"distributions_{label}.csv"),
            result.to_distribution_dict(),
            header_comment=f"Flat-plate distributions -- {fluid.name}, "
            f"U={u_inf} m/s, L={length} m",
        )
        tec_path = export_tecplot(
            out("module3", f"distributions_{label}.dat"),
            result.to_distribution_dict(),
            title=f"Flat-plate distributions ({fluid.name})",
            zone_name=label.upper(),
        )
        print("wrote", csv_path)
        print("wrote", tec_path)
        print("=" * 66)


if __name__ == "__main__":
    main()
