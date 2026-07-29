"""Boundary Layer Laboratory -- a virtual laboratory for the flat-plate boundary layer.

Companion software for *Heat Transfer: Analysis, Computation, and Verification*.

The package computes the laminar momentum and thermal boundary layers over a
flat plate from first principles and connects them to the engineering
correlations used in practice.  It is organised as six modules:

======  ====================================  =====================================
Module  Topic                                 Key class / entry point
======  ====================================  =====================================
1       Blasius momentum boundary layer       :class:`~blayerlab.blasius.BlasiusSolver`
2       Thermal boundary layer                :class:`~blayerlab.thermal.ThermalSolver`
3       Flat-plate heat-transfer calculator   :class:`~blayerlab.calculator.FlatPlateCalculator`
4       Interactive laboratory (GUI)          :class:`~blayerlab.interactive.InteractiveLab`
5       Correlation verification              :class:`~blayerlab.verification.VerificationSuite`
6       Parametric studies                    :class:`~blayerlab.parametric.ParametricStudy`
======  ====================================  =====================================

Quick start
-----------
>>> from blayerlab import BlasiusSolver, ThermalSolver, FlatPlateCalculator, get_fluid
>>> blasius = BlasiusSolver().solve()
>>> round(blasius.fpp0, 5)
0.33206
>>> result = FlatPlateCalculator(blasius).compute(
...     u_inf=2.0, fluid=get_fluid("air"), length=0.5, t_wall=350.0, t_inf=300.0)
>>> result.Re_L > 0 and result.nu_local_L > 0
True

All quantities are in SI units.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core infrastructure.
from . import constants, correlations
from .fluids import FLUID_DATABASE, FluidProperties, get_fluid
from .integrators import rk4_integrate, rk4_step
from .io_utils import export_csv, export_tecplot

# Module 1 -- Blasius momentum boundary layer.
from .blasius import HOWARTH_TABLE, BlasiusSolution, BlasiusSolver

# Module 2 -- Thermal boundary layer.
from .thermal import DEFAULT_PRANDTL_NUMBERS, ThermalSolution, ThermalSolver

# Module 3 -- Flat-plate calculator.
from .calculator import FlatPlateCalculator, FlatPlateResult

# Module 5 -- Verification.
from .verification import VerificationSuite, absolute_error, relative_error, rms_error

# Module 6 -- Parametric studies.
from .parametric import ParametricStudy, StudyResult

# Plotting helpers (Modules 1-3, 6).  Figure/interactive modules import
# Matplotlib lazily via the functions below to keep a bare import lightweight.
from .plotting import savefig, use_publication_style

__all__ = [
    "__version__",
    # core
    "constants",
    "correlations",
    "FluidProperties",
    "FLUID_DATABASE",
    "get_fluid",
    "rk4_integrate",
    "rk4_step",
    "export_csv",
    "export_tecplot",
    # module 1
    "BlasiusSolver",
    "BlasiusSolution",
    "HOWARTH_TABLE",
    # module 2
    "ThermalSolver",
    "ThermalSolution",
    "DEFAULT_PRANDTL_NUMBERS",
    # module 3
    "FlatPlateCalculator",
    "FlatPlateResult",
    # module 5
    "VerificationSuite",
    "absolute_error",
    "relative_error",
    "rms_error",
    # module 6
    "ParametricStudy",
    "StudyResult",
    # plotting
    "use_publication_style",
    "savefig",
]


def launch_interactive() -> None:
    """Launch the Module 4 interactive laboratory (imports Matplotlib widgets).

    Kept as a function (rather than a top-level import) so that importing
    :mod:`blayerlab` never forces a GUI backend.
    """
    from .interactive import InteractiveLab

    InteractiveLab().launch()


def launch_calculator() -> None:
    """Launch the Module 3 graphical flat-plate calculator (entry-field GUI).

    Like :func:`launch_interactive`, the Matplotlib-widgets import is deferred
    so a bare ``import blayerlab`` never forces a GUI backend.
    """
    from .calculator_gui import CalculatorGUI

    CalculatorGUI().launch()
