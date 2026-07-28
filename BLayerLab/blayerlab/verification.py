"""Module 5 -- Engineering correlation verification.

The purpose of this module is *quantitative trust*: it places the numerical
similarity solutions (Modules 1 and 2) next to their analytical values and to
the classical engineering correlations, and reports absolute, relative and
root-mean-square (RMS) errors.

Three families of checks are provided:

1. **Momentum layer** -- computed Blasius constants (``f''(0)``, ``delta*``,
   ``theta``, shape factor, ``C_f``) versus accepted reference values, plus an
   RMS comparison of the whole velocity profile against Howarth's table.
2. **Thermal layer** -- the numerical wall gradient ``theta'(0)`` versus its
   analytical closed form and versus the Pohlhausen / Churchill-Ozoe
   correlations, across a range of Prandtl numbers.
3. **Analogies** -- the Reynolds analogy (``St = C_f/2`` at ``Pr = 1``) and the
   Chilton-Colburn analogy (``St Pr^{2/3} = C_f/2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants
from . import correlations as corr
from .blasius import HOWARTH_TABLE, BlasiusSolver
from .thermal import DEFAULT_PRANDTL_NUMBERS, ThermalSolver


def absolute_error(numerical, reference):
    """Absolute error ``|numerical - reference|`` (scalar or elementwise)."""
    return np.abs(np.asarray(numerical) - np.asarray(reference))


def relative_error(numerical, reference):
    """Relative error ``|numerical - reference| / |reference|``.

    Zero references yield ``inf`` (guarded to avoid a divide-by-zero warning).
    """
    numerical = np.asarray(numerical, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denom = np.where(reference == 0.0, np.nan, np.abs(reference))
    return np.abs(numerical - reference) / denom


def rms_error(numerical, reference):
    """Root-mean-square error ``sqrt(mean((numerical - reference)^2))``."""
    diff = np.asarray(numerical, dtype=float) - np.asarray(reference, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


@dataclass
class VerificationItem:
    """A single scalar comparison between a computed and a reference value.

    Attributes
    ----------
    quantity : str
        Label of the quantity being checked.
    numerical : float
        Value produced by the numerical solver.
    reference : float
        The value it is compared against.
    reference_kind : str
        ``"analytical"`` or ``"correlation"`` -- the nature of ``reference``.
    """

    quantity: str
    numerical: float
    reference: float
    reference_kind: str

    @property
    def abs_error(self) -> float:
        """Absolute error of this item."""
        return float(absolute_error(self.numerical, self.reference))

    @property
    def rel_error(self) -> float:
        """Relative error of this item."""
        return float(relative_error(self.numerical, self.reference))


class VerificationSuite:
    """Run and report the full set of verification checks.

    Parameters
    ----------
    blasius : BlasiusSolution, optional
        Pre-computed momentum solution (otherwise solved on construction).
    prandtl_numbers : iterable of float, optional
        Prandtl numbers used in the thermal and analogy checks.
    re_x : float, optional
        Reference Reynolds number for the analogy checks.
    """

    def __init__(
        self,
        blasius=None,
        prandtl_numbers=DEFAULT_PRANDTL_NUMBERS,
        re_x: float = 1.0e5,
    ) -> None:
        self.blasius = blasius if blasius is not None else BlasiusSolver().solve()
        self.thermal = ThermalSolver(self.blasius)
        self.prandtl_numbers = tuple(float(p) for p in prandtl_numbers)
        self.re_x = float(re_x)

    # -- momentum layer -------------------------------------------------------

    def verify_momentum(self) -> list[VerificationItem]:
        """Compare Blasius integral constants against reference values."""
        b = self.blasius
        checks = [
            ("f''(0)", b.fpp0, constants.BLASIUS_FPP0),
            ("delta99 const", b.eta_99, constants.BLASIUS_DELTA99),
            ("delta* const", b.displacement_const, constants.BLASIUS_DISPLACEMENT),
            ("theta const", b.momentum_const, constants.BLASIUS_MOMENTUM),
            ("shape factor H", b.shape_factor, constants.BLASIUS_SHAPE_FACTOR),
            ("Cf*sqrt(Re)", 2.0 * b.fpp0, constants.BLASIUS_CF_CONST),
        ]
        return [
            VerificationItem(name, num, ref, "analytical")
            for name, num, ref in checks
        ]

    def blasius_profile_rms(self) -> float:
        """RMS error of the numerical ``u/U_inf`` vs Howarth's tabulation."""
        eta_ref = HOWARTH_TABLE[:, 0]
        fp_ref = HOWARTH_TABLE[:, 2]
        fp_num = np.interp(eta_ref, self.blasius.eta, self.blasius.fp)
        return rms_error(fp_num, fp_ref)

    # -- thermal layer --------------------------------------------------------

    def verify_thermal(self) -> list[VerificationItem]:
        """Compare numerical ``theta'(0)`` vs analytical and correlations."""
        items: list[VerificationItem] = []
        for Pr in self.prandtl_numbers:
            sol = self.thermal.solve(Pr)
            num = sol.thetap0
            analytical = self.thermal.wall_gradient_analytical(Pr)
            # All-Pr correlation constant = Nu_x / sqrt(Re_x).
            corr_const = corr.nu_local_churchill_ozoe(self.re_x, Pr) / np.sqrt(
                self.re_x
            )
            items.append(
                VerificationItem(
                    f"theta'(0) [Pr={Pr:g}] vs analytical", num, analytical,
                    "analytical",
                )
            )
            items.append(
                VerificationItem(
                    f"theta'(0) [Pr={Pr:g}] vs Churchill", num, corr_const,
                    "correlation",
                )
            )
        return items

    def thermal_gradient_rms(self) -> float:
        """RMS error of numerical vs analytical ``theta'(0)`` over all ``Pr``."""
        num = [self.thermal.solve(Pr).thetap0 for Pr in self.prandtl_numbers]
        ana = [self.thermal.wall_gradient_analytical(Pr) for Pr in self.prandtl_numbers]
        return rms_error(num, ana)

    # -- analogies ------------------------------------------------------------

    def verify_analogies(self) -> list[VerificationItem]:
        """Check the Reynolds and Chilton-Colburn analogies at ``re_x``.

        * Reynolds (``Pr = 1``): ``St = C_f / 2``.  The numerical Stanton number
          is ``St = Nu_x / (Re_x Pr)`` from the similarity solution.
        * Chilton-Colburn: ``St Pr^{2/3} = C_f / 2`` for ``0.6 < Pr < 60``.
        """
        items: list[VerificationItem] = []
        cf_over_2 = corr.cf_local(self.re_x) / 2.0

        # Reynolds analogy at Pr = 1.
        sol1 = self.thermal.solve(1.0)
        nu1 = sol1.nu_local(self.re_x)
        st1 = nu1 / (self.re_x * 1.0)
        items.append(
            VerificationItem("Reynolds analogy St (Pr=1)", st1, cf_over_2, "analytical")
        )

        # Chilton-Colburn for mid-range Pr.
        for Pr in [p for p in self.prandtl_numbers if 0.6 <= p <= 60.0]:
            sol = self.thermal.solve(Pr)
            nu = sol.nu_local(self.re_x)
            st = nu / (self.re_x * Pr)
            j = st * Pr ** (2.0 / 3.0)
            items.append(
                VerificationItem(
                    f"Chilton-Colburn j_H [Pr={Pr:g}]", j, cf_over_2, "correlation"
                )
            )
        return items

    # -- aggregation ----------------------------------------------------------

    def run_all(self) -> dict[str, list[VerificationItem]]:
        """Run every check and return a grouped dictionary of items."""
        return {
            "momentum": self.verify_momentum(),
            "thermal": self.verify_thermal(),
            "analogies": self.verify_analogies(),
        }

    def report(self) -> str:
        """Return a formatted verification report as a multi-line string."""
        groups = self.run_all()
        lines = [
            "=" * 74,
            " ENGINEERING CORRELATION VERIFICATION  (Re_x = "
            f"{self.re_x:.2e})",
            "=" * 74,
        ]
        for title, items in groups.items():
            lines.append(f"\n [{title.upper()}]")
            lines.append(
                f"   {'quantity':34s} {'numerical':>12s} {'reference':>12s} "
                f"{'rel.err':>10s}"
            )
            for it in items:
                lines.append(
                    f"   {it.quantity:34s} {it.numerical:12.5e} "
                    f"{it.reference:12.5e} {it.rel_error:10.2e}"
                )
        lines.append("\n [AGGREGATE RMS ERRORS]")
        lines.append(f"   Blasius u/Uinf vs Howarth  : {self.blasius_profile_rms():.3e}")
        lines.append(f"   theta'(0) num vs analytical: {self.thermal_gradient_rms():.3e}")
        lines.append("=" * 74)
        return "\n".join(lines)


__all__ = [
    "absolute_error",
    "relative_error",
    "rms_error",
    "VerificationItem",
    "VerificationSuite",
]
