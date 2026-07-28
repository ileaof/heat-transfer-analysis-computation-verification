"""Classical engineering correlations for the laminar flat-plate boundary layer.

These closed-form expressions are the textbook results that the numerical
solvers (Modules 1 and 2) are meant to reproduce.  Collecting them in one place
lets Module 5 (verification) compare *numerical*, *analytical* and *correlation*
predictions side by side.

Unless noted otherwise the correlations assume:

* steady, incompressible, laminar flow;
* zero pressure gradient (flat plate aligned with the free stream);
* constant properties;
* an isothermal wall for the thermal results.

Every function is dimensionless-in / dimensionless-out and vectorises over
NumPy arrays.

References
----------
Schlichting & Gersten, *Boundary-Layer Theory*, 8th ed.
Incropera et al., *Fundamentals of Heat and Mass Transfer*, 7th ed.
Kays, Crawford & Weigand, *Convective Heat and Mass Transfer*, 4th ed.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Momentum boundary layer (Blasius)
# ---------------------------------------------------------------------------


def blasius_delta(Re_x, x):
    """99% velocity boundary-layer thickness, ``delta = 5.0 x / sqrt(Re_x)``.

    Parameters
    ----------
    Re_x : float or array_like
        Local Reynolds number ``U_inf x / nu``.
    x : float or array_like
        Streamwise coordinate, m.

    Returns
    -------
    float or numpy.ndarray
        Boundary-layer thickness, m.
    """
    return 5.0 * x / np.sqrt(Re_x)


def blasius_displacement_thickness(Re_x, x):
    """Displacement thickness ``delta* = 1.721 x / sqrt(Re_x)`` (m)."""
    return 1.720787 * x / np.sqrt(Re_x)


def blasius_momentum_thickness(Re_x, x):
    """Momentum thickness ``theta = 0.664 x / sqrt(Re_x)`` (m)."""
    return 0.664116 * x / np.sqrt(Re_x)


def blasius_shape_factor():
    """Shape factor ``H = delta*/theta`` of the Blasius profile (~2.591)."""
    return 1.720787 / 0.664116


def cf_local(Re_x):
    """Local skin-friction coefficient ``C_f = 0.664 / sqrt(Re_x)``.

    Follows from the wall shear stress ``tau_w = 0.332 rho U^2 / sqrt(Re_x)``.
    """
    return 0.664 / np.sqrt(Re_x)


def cf_average(Re_L):
    """Plate-averaged skin-friction coefficient ``Cf = 1.328 / sqrt(Re_L)``."""
    return 1.328 / np.sqrt(Re_L)


# ---------------------------------------------------------------------------
# Thermal boundary layer (Pohlhausen and extensions)
# ---------------------------------------------------------------------------


def nu_local_pohlhausen(Re_x, Pr):
    """Local Nusselt number, classical Pohlhausen result.

    .. math:: \\mathrm{Nu}_x = 0.332\\, \\mathrm{Re}_x^{1/2}\\, \\mathrm{Pr}^{1/3}

    Valid for ``Pr >~ 0.6``.  For liquid metals use :func:`nu_local_low_pr`;
    for an all-``Pr`` fit use :func:`nu_local_churchill_ozoe`.
    """
    return 0.332 * np.sqrt(Re_x) * np.cbrt(Pr)


def nu_average_pohlhausen(Re_L, Pr):
    """Average Nusselt number ``Nu_L = 0.664 Re_L^{1/2} Pr^{1/3}`` (``Pr >~ 0.6``).

    For laminar flow ``h_avg = 2 h(L)``, hence the leading constant doubles
    relative to the local value.
    """
    return 0.664 * np.sqrt(Re_L) * np.cbrt(Pr)


def nu_local_low_pr(Re_x, Pr):
    """Local Nusselt number for liquid metals (``Pr -> 0``).

    .. math:: \\mathrm{Nu}_x = 0.565\\,(\\mathrm{Re}_x \\mathrm{Pr})^{1/2}
              = 0.565\\, \\mathrm{Pe}_x^{1/2}

    where ``Pe_x = Re_x Pr`` is the local Peclet number.  Valid for
    ``Pr <~ 0.05``.
    """
    return 0.565 * np.sqrt(Re_x * Pr)


def nu_local_churchill_ozoe(Re_x, Pr):
    """All-``Pr`` local Nusselt correlation (Churchill & Ozoe, 1973).

    .. math::

        \\mathrm{Nu}_x = \\frac{0.3387\\, \\mathrm{Re}_x^{1/2}\\,
        \\mathrm{Pr}^{1/3}}{\\left[1 + (0.0468/\\mathrm{Pr})^{2/3}\\right]^{1/4}}

    Valid for the entire Prandtl-number range provided the local Peclet number
    ``Re_x Pr > 100``.  Reproduces both the ``0.332 Pr^{1/3}`` and the
    ``0.565 Pr^{1/2}`` limits.
    """
    num = 0.3387 * np.sqrt(Re_x) * np.cbrt(Pr)
    den = (1.0 + (0.0468 / Pr) ** (2.0 / 3.0)) ** 0.25
    return num / den


def nu_average_churchill_ozoe(Re_L, Pr):
    """Average form of :func:`nu_local_churchill_ozoe` (``Nu_L = 2 Nu_x(L)``)."""
    return 2.0 * nu_local_churchill_ozoe(Re_L, Pr)


# ---------------------------------------------------------------------------
# Analogies between momentum and heat transfer
# ---------------------------------------------------------------------------


def stanton_from_nu(Nu, Re, Pr):
    """Stanton number ``St = Nu / (Re Pr)``."""
    return Nu / (Re * Pr)


def reynolds_analogy_st(Re_x):
    """Stanton number from the Reynolds analogy (``Pr = 1``).

    .. math:: \\mathrm{St} = C_f / 2

    Strictly valid only at ``Pr = 1`` (equal momentum and thermal diffusion).
    """
    return cf_local(Re_x) / 2.0


def chilton_colburn_st(Re_x, Pr):
    """Stanton number from the Chilton-Colburn (modified Reynolds) analogy.

    .. math:: \\mathrm{St}\\, \\mathrm{Pr}^{2/3} = C_f / 2

    i.e. ``St = (C_f/2) Pr^{-2/3}``.  Valid for ``0.6 < Pr < 60``.  The group
    ``j_H = St Pr^{2/3}`` is the Colburn ``j``-factor.
    """
    return (cf_local(Re_x) / 2.0) * Pr ** (-2.0 / 3.0)


def colburn_j_factor(Re_x):
    """Colburn ``j``-factor ``j_H = St Pr^{2/3} = C_f/2``."""
    return cf_local(Re_x) / 2.0


__all__ = [
    "blasius_delta",
    "blasius_displacement_thickness",
    "blasius_momentum_thickness",
    "blasius_shape_factor",
    "cf_local",
    "cf_average",
    "nu_local_pohlhausen",
    "nu_average_pohlhausen",
    "nu_local_low_pr",
    "nu_local_churchill_ozoe",
    "nu_average_churchill_ozoe",
    "stanton_from_nu",
    "reynolds_analogy_st",
    "chilton_colburn_st",
    "colburn_j_factor",
]
