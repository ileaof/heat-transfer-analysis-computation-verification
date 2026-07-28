"""Example -- Module 6: Parametric studies.

Generates the four standard parametric studies (Reynolds, Prandtl, plate
length, fluid comparison), saving each figure and its underlying data.

Run::

    python examples/example_module6_parametric.py
"""

from __future__ import annotations

from _common import out

from blayerlab import export_csv, get_fluid
from blayerlab.parametric import ParametricStudy
from blayerlab.plotting import savefig


def main() -> None:
    study = ParametricStudy()

    results = [
        study.sweep_reynolds(re_range=(1e3, 5e5), pr=0.71),
        study.sweep_prandtl(pr_range=(1e-2, 1e3), n=40),
        study.sweep_length(get_fluid("air"), u_inf=3.0, t_wall=350.0, t_inf=300.0),
        study.sweep_fluids(
            [get_fluid(f) for f in ("air", "water", "engine_oil", "mercury", "glycerin")],
            u_inf=1.0, length=0.5, t_wall=350.0, t_inf=300.0,
        ),
    ]

    for r in results:
        for name in savefig(r.figure, out("module6", r.name), ("png",)):
            print("wrote", name)
        csv_path = export_csv(out("module6", f"{r.name}.csv"), r.data)
        print("wrote", csv_path)


if __name__ == "__main__":
    main()
