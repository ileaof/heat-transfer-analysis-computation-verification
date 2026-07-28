"""Physical constants and canonical reference values (SI units).

This module centralises the small set of numeric constants that appear
throughout the Boundary Layer Laboratory.  Two categories are stored:

1. *Physical* constants (e.g. the acceleration of gravity).
2. *Canonical* dimensionless boundary-layer constants obtained from the
   classical self-similar solutions (Blasius, Pohlhausen).  These are used
   as reference values against which the numerical solvers are verified.

All quantities are expressed in SI units unless explicitly stated.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

#: Standard acceleration of gravity, m/s^2.
GRAVITY: float = 9.80665

#: Standard atmospheric pressure, Pa.
ATM_PRESSURE: float = 101_325.0

#: Universal gas constant, J/(mol*K).
R_UNIVERSAL: float = 8.314_462_618

#: Specific gas constant for dry air, J/(kg*K).
R_AIR: float = 287.058


# ---------------------------------------------------------------------------
# Canonical Blasius (momentum boundary layer) constants
# ---------------------------------------------------------------------------
# These follow from the self-similar solution of  f''' + 0.5 f f'' = 0  with
# f(0) = f'(0) = 0 and f'(inf) = 1.  Values are the accepted reference numbers
# (Howarth 1938; Schlichting & Gersten, "Boundary-Layer Theory", 8th ed.).

#: Wall curvature of the Blasius profile, f''(0).  Sets the wall shear stress.
BLASIUS_FPP0: float = 0.332_057_336

#: Velocity boundary-layer thickness constant:  delta_99 * sqrt(Re_x) / x.
#: Defined at u/U = 0.99 (eta approx 4.910).
BLASIUS_DELTA99: float = 4.910_0

#: Displacement-thickness constant:  delta_star * sqrt(Re_x) / x  =  1.720787.
BLASIUS_DISPLACEMENT: float = 1.720_787

#: Momentum-thickness constant:  theta * sqrt(Re_x) / x  =  0.664116.
BLASIUS_MOMENTUM: float = 0.664_116

#: Shape factor  H = delta_star / theta  for the Blasius profile.
BLASIUS_SHAPE_FACTOR: float = BLASIUS_DISPLACEMENT / BLASIUS_MOMENTUM  # ~2.591

#: Local skin-friction constant:  C_f * sqrt(Re_x)  =  2 f''(0)  =  0.664.
BLASIUS_CF_CONST: float = 2.0 * BLASIUS_FPP0  # ~0.664


# ---------------------------------------------------------------------------
# Canonical thermal (Pohlhausen) constants
# ---------------------------------------------------------------------------

#: Local Nusselt constant for the classical correlation
#: Nu_x = C * Re_x^(1/2) * Pr^(1/3), valid for Pr >~ 0.6.
POHLHAUSEN_NU_CONST: float = 0.332

#: Average Nusselt constant:  Nu_L = 0.664 Re_L^(1/2) Pr^(1/3).
POHLHAUSEN_NU_AVG_CONST: float = 0.664

__all__ = [
    "GRAVITY",
    "ATM_PRESSURE",
    "R_UNIVERSAL",
    "R_AIR",
    "BLASIUS_FPP0",
    "BLASIUS_DELTA99",
    "BLASIUS_DISPLACEMENT",
    "BLASIUS_MOMENTUM",
    "BLASIUS_SHAPE_FACTOR",
    "BLASIUS_CF_CONST",
    "POHLHAUSEN_NU_CONST",
    "POHLHAUSEN_NU_AVG_CONST",
]
