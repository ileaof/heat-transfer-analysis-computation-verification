"""Fluid property model and a small reference database (SI units).

A boundary-layer calculation needs a consistent set of thermophysical
properties.  This module provides:

* :class:`FluidProperties` -- an immutable container that stores density,
  viscosity, conductivity, specific heat, etc., and derives whatever is
  missing (e.g. kinematic viscosity, thermal diffusivity, Prandtl number)
  so that the object is always internally consistent.
* :data:`FLUID_DATABASE` -- a dictionary of commonly used fluids evaluated at
  a representative temperature, spanning six decades of Prandtl number
  (mercury, air, water, engine oil, ...).  This lets students immediately see
  how the thermal boundary layer changes with ``Pr``.

All properties are in SI units:

===================  =========================  ====================
Symbol               Meaning                    Unit
===================  =========================  ====================
``rho``              density                    kg/m^3
``mu``               dynamic viscosity          Pa*s  (kg/(m*s))
``nu``               kinematic viscosity        m^2/s
``k``                thermal conductivity       W/(m*K)
``cp``               specific heat (const. p)   J/(kg*K)
``alpha``            thermal diffusivity        m^2/s
``Pr``               Prandtl number             -
===================  =========================  ====================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FluidProperties:
    """Self-consistent thermophysical properties of a fluid at one state.

    Only a subset of the properties needs to be supplied; the constructor
    fills in the rest using the defining relations

    .. math::

        \\nu = \\mu / \\rho, \\qquad
        \\alpha = k / (\\rho c_p), \\qquad
        \\mathrm{Pr} = \\nu / \\alpha = \\mu c_p / k .

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"Air"``).
    rho : float
        Density, kg/m^3.
    mu : float
        Dynamic viscosity, Pa*s.
    k : float
        Thermal conductivity, W/(m*K).
    cp : float
        Specific heat at constant pressure, J/(kg*K).
    temperature : float, optional
        State temperature, K (metadata only; not used in the correlations).
    nu, alpha, Pr : float, optional
        Optional overrides.  If omitted they are derived from the primary
        properties above.

    Notes
    -----
    The object is *frozen* (immutable) so that a set of properties passed to a
    solver cannot be mutated behind the caller's back.  Derived fields are
    written once with :func:`object.__setattr__` inside :meth:`__post_init__`.
    """

    name: str
    rho: float
    mu: float
    k: float
    cp: float
    temperature: Optional[float] = None
    nu: float = field(default=0.0)
    alpha: float = field(default=0.0)
    Pr: float = field(default=0.0)

    def __post_init__(self) -> None:
        # Validate the primary properties.
        for attr in ("rho", "mu", "k", "cp"):
            value = getattr(self, attr)
            if value <= 0.0:
                raise ValueError(
                    f"FluidProperties.{attr} must be positive, got {value!r}."
                )

        # Derive secondary properties from first principles.  ``frozen=True``
        # blocks normal assignment, so we bypass it with object.__setattr__.
        object.__setattr__(self, "nu", self.mu / self.rho)
        object.__setattr__(self, "alpha", self.k / (self.rho * self.cp))
        object.__setattr__(self, "Pr", self.nu / self.alpha)

    def summary(self) -> str:
        """Return a formatted multi-line summary of the properties."""
        lines = [
            f"Fluid: {self.name}"
            + (f"  (T = {self.temperature:.1f} K)" if self.temperature else ""),
            f"  rho   = {self.rho:12.5e} kg/m^3",
            f"  mu    = {self.mu:12.5e} Pa*s",
            f"  nu    = {self.nu:12.5e} m^2/s",
            f"  k     = {self.k:12.5e} W/(m*K)",
            f"  cp    = {self.cp:12.5e} J/(kg*K)",
            f"  alpha = {self.alpha:12.5e} m^2/s",
            f"  Pr    = {self.Pr:12.5e} -",
        ]
        return "\n".join(lines)


def _from_nu(name, rho, nu, k, cp, temperature=None):
    """Helper: build :class:`FluidProperties` from kinematic viscosity.

    Many handbooks tabulate ``nu`` rather than ``mu``; this converts using
    ``mu = rho * nu`` so database entries can mirror common references.
    """
    return FluidProperties(
        name=name, rho=rho, mu=rho * nu, k=k, cp=cp, temperature=temperature
    )


# ---------------------------------------------------------------------------
# Reference database (properties from Incropera, "Fundamentals of Heat and Mass
# Transfer", 7th ed., property tables, at the stated temperature).
# Spanning Pr ~ 0.025 (mercury) to Pr ~ 6400 (engine oil).
# ---------------------------------------------------------------------------
FLUID_DATABASE: dict[str, FluidProperties] = {
    "mercury": _from_nu(
        "Mercury", rho=13529.0, nu=1.125e-7, k=8.540, cp=139.3, temperature=300.0
    ),
    "air": _from_nu(
        "Air", rho=1.1614, nu=15.89e-6, k=26.3e-3, cp=1007.0, temperature=300.0
    ),
    "water": _from_nu(
        "Water", rho=997.0, nu=8.576e-7, k=0.613, cp=4179.0, temperature=300.0
    ),
    "engine_oil": _from_nu(
        "Engine oil", rho=884.1, nu=5.5e-4, k=0.145, cp=1909.0, temperature=300.0
    ),
    "glycerin": _from_nu(
        "Glycerin", rho=1259.9, nu=1.12e-3, k=0.286, cp=2427.0, temperature=300.0
    ),
    "ammonia": _from_nu(
        "Ammonia", rho=599.8, nu=2.15e-7, k=0.4927, cp=4798.0, temperature=300.0
    ),
    "co2": _from_nu(
        "Carbon dioxide", rho=1.7730, nu=8.321e-6, k=16.55e-3, cp=851.0,
        temperature=300.0,
    ),
}


def get_fluid(name: str) -> FluidProperties:
    """Look up a fluid in :data:`FLUID_DATABASE` (case/space insensitive).

    Parameters
    ----------
    name : str
        Fluid key, e.g. ``"air"``, ``"Engine Oil"``, ``"water"``.

    Returns
    -------
    FluidProperties
        The stored, self-consistent property set.

    Raises
    ------
    KeyError
        If the fluid is not present, with the list of valid keys.
    """
    key = name.strip().lower().replace(" ", "_")
    if key not in FLUID_DATABASE:
        valid = ", ".join(sorted(FLUID_DATABASE))
        raise KeyError(f"Unknown fluid {name!r}. Available: {valid}.")
    return FLUID_DATABASE[key]


__all__ = ["FluidProperties", "FLUID_DATABASE", "get_fluid"]
