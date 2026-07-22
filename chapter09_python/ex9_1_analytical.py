"""
================================================================================
 EXAMPLE 9.1 -- BLACKBODY RADIATION FROM FIRST PRINCIPLES
 Planck's law, and the constants that follow from it
================================================================================

 OBJECTIVE
 ---------
 Chapters 1 to 8 quoted the Stefan-Boltzmann constant sigma and the Wien
 displacement constant b as data.  Neither is data.  Both follow from Planck's
 distribution by integration and by differentiation respectively, and this
 example computes them.

 THIS EXAMPLE IS UNUSUALLY WELL POSED FOR VERIFICATION.  Since the 2019
 redefinition of the SI, the Planck constant h, the speed of light c and the
 Boltzmann constant k are EXACT defined values, not measurements.  Therefore

        sigma = 2 pi^5 k^4 / (15 c^2 h^3)

 is an exact rational multiple of exact constants, and so is Wien's b.  There
 is no experimental uncertainty to hide behind: a computed sigma either matches
 to machine precision or the code is wrong.  Few problems in this book offer
 that.

 WHAT IS COMPUTED
 ----------------
   1. Planck's spectral emissive power, and its integral over all wavelengths,
      done numerically and compared with 2 pi^5 k^4 / (15 c^2 h^3).
   2. The underlying mathematical identity int_0^inf x^3/(e^x - 1) dx = pi^4/15,
      verified to machine precision.
   3. Wien's displacement law, obtained by solving the transcendental equation
      x = 5(1 - e^-x) rather than by quoting 2898 um K.
   4. The band emission fraction F(0 -> lambda T), computed two independent
      ways -- a rapidly convergent series and direct quadrature -- and checked
      against the closure condition F(0 -> inf) = 1.
   5. An engineering application: the fraction of solar and of room-temperature
      radiation falling in the visible band, which is why a "cold" object and a
      "hot" one look so different.

 GOVERNING RELATIONS
 -------------------
 Planck's spectral emissive power of a blackbody, per unit wavelength:

        E_b,lambda(lambda, T) = C1 / ( lambda^5 [ exp(C2/(lambda T)) - 1 ] )

 with C1 = 2 pi h c^2 and C2 = h c / k.  Integrating over all lambda gives the
 Stefan-Boltzmann law E_b = sigma T^4; differentiating and setting the result
 to zero gives Wien's law lambda_max T = b.

 SYMBOLS
 -------
   h       [J s]          Planck constant (exact, SI 2019)
   c       [m/s]          speed of light (exact, SI 2019)
   k       [J/K]          Boltzmann constant (exact, SI 2019)
   C1      [W m^4/m^2]    first radiation constant, 2 pi h c^2
   C2      [m K]          second radiation constant, h c / k
   lambda  [m]            wavelength
   T       [K]            absolute temperature
   E_b     [W/m^2]        total blackbody emissive power
   sigma   [W/m^2 K^4]    Stefan-Boltzmann constant
   b       [m K]          Wien displacement constant
   F       [-]            band emission fraction

 OUTPUTS
 -------
   fig_9_1a_planck.png     Planck curves, Wien locus, visible band
   fig_9_1b_bands.png      band fraction, and the two routes compared

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

# --- NumPy compatibility -----------------------------------------------------
# `np.trapz` was renamed `np.trapezoid` in NumPy 2.0 and the old name now warns;
# the new name does not exist before 2.0.  Bind once so the script runs on both.
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

T_START = time.perf_counter()

# ==============================================================================
# 1. THE DEFINING CONSTANTS (exact since the 2019 SI redefinition)
# ==============================================================================
H_PLANCK = 6.62607015e-34       # J s   -- exact by definition
C_LIGHT = 2.99792458e8          # m/s   -- exact by definition
K_BOLTZ = 1.380649e-23          # J/K   -- exact by definition

C1 = 2.0 * np.pi * H_PLANCK * C_LIGHT ** 2      # W m^4 / m^2
C2 = H_PLANCK * C_LIGHT / K_BOLTZ               # m K

# The closed forms these must reproduce
SIGMA_CLOSED = (2.0 * np.pi ** 5 * K_BOLTZ ** 4 /
                (15.0 * C_LIGHT ** 2 * H_PLANCK ** 3))


def planck(lam, T):
    """Spectral emissive power E_b,lambda [W/m^2 per m of wavelength].

    Written with expm1 rather than exp(...) - 1.  At long wavelengths the
    exponent C2/(lambda T) becomes small and exp(x) - 1 loses all its
    significant figures to cancellation, while expm1(x) is accurate there by
    construction.  The Rayleigh-Jeans tail is exactly the region where a
    careless implementation quietly returns noise.
    """
    lam = np.asarray(lam, dtype=float)
    return C1 / (lam ** 5 * np.expm1(C2 / (lam * T)))


# ==============================================================================
# 2. THE STEFAN-BOLTZMANN CONSTANT BY INTEGRATION
# ==============================================================================
def sigma_by_quadrature():
    """Integrate Planck's law over all wavelengths at T = 1 K.

    Substituting x = C2/(lambda T) turns the integral into

        E_b = (2 pi k^4 T^4 / (c^2 h^3)) int_0^inf x^3/(e^x - 1) dx

    so the whole of the physics reduces to one dimensionless integral whose
    exact value is pi^4/15.  Both forms are evaluated below.
    """
    f = lambda x: x ** 3 / np.expm1(x)
    val, err = quad(f, 0.0, np.inf, limit=400)
    return val, err


print("=" * 78)
print("EXAMPLE 9.1 -- BLACKBODY RADIATION FROM FIRST PRINCIPLES")
print("=" * 78)
print(f"  h = {H_PLANCK:.9e} J s      (exact, SI 2019)")
print(f"  c = {C_LIGHT:.9e} m/s      (exact, SI 2019)")
print(f"  k = {K_BOLTZ:.9e} J/K      (exact, SI 2019)")
print(f"  C1 = 2 pi h c^2 = {C1:.9e} W m^2")
print(f"  C2 = h c / k    = {C2:.9e} m K")

print("\n" + "-" * 78)
print("  THE DIMENSIONLESS INTEGRAL")
I_num, I_err = sigma_by_quadrature()
I_exact = np.pi ** 4 / 15.0
print(f"    int_0^inf x^3/(e^x - 1) dx = {I_num:.14f}")
print(f"    exact pi^4/15              = {I_exact:.14f}")
print(f"    relative difference        = {abs(I_num/I_exact - 1):.3e}")
print(f"    quadrature's own error estimate = {I_err:.3e}")

sigma_num = (2.0 * np.pi * K_BOLTZ ** 4 / (C_LIGHT ** 2 * H_PLANCK ** 3)) * I_num
print("\n  THE STEFAN-BOLTZMANN CONSTANT")
print(f"    from the computed integral = {sigma_num:.12e} W/m^2 K^4")
print(f"    closed form 2 pi^5 k^4/15 c^2 h^3 = {SIGMA_CLOSED:.12e}")
print(f"    relative difference        = {abs(sigma_num/SIGMA_CLOSED - 1):.3e}")
print("    Because h, c and k are exact defined values, sigma is an exact")
print("    number and this comparison has no experimental slack in it.")

# A second, independent route: integrate Planck's law directly in wavelength
# at a physical temperature, with no change of variable.
T_TEST = 1000.0
lam_lo, lam_hi = 1e-9, 1e-2
E_direct, E_err = quad(lambda L: planck(L, T_TEST), lam_lo, lam_hi, limit=400)
E_exact = SIGMA_CLOSED * T_TEST ** 4
print("\n  DIRECT INTEGRATION IN WAVELENGTH (no substitution), T = 1000 K")
print(f"    integral over 1 nm to 10 mm = {E_direct:.9f} W/m^2")
print(f"    sigma T^4                   = {E_exact:.9f} W/m^2")
print(f"    relative difference         = {abs(E_direct/E_exact - 1):.3e}")
print("    The residual here is the TRUNCATION of the integration range, not")
print("    the quadrature: the tails beyond 1 nm and 10 mm carry real energy.")

# ==============================================================================
# 3. WIEN'S DISPLACEMENT LAW
# ==============================================================================
print("\n" + "-" * 78)
print("  WIEN'S DISPLACEMENT LAW")
print("""    Setting d(E_b,lambda)/d(lambda) = 0 and writing x = C2/(lambda T)
    gives the transcendental equation

        x = 5 (1 - e^-x)

    whose nonzero root is found below by bisection.  The displacement
    constant is then b = C2 / x.""")
wien_res = lambda x: x - 5.0 * (1.0 - np.exp(-x))
x_wien = brentq(wien_res, 1.0, 10.0, xtol=1e-14, rtol=8.9e-16)
b_wien = C2 / x_wien
print(f"\n    root x                = {x_wien:.12f}")
print(f"    residual at the root  = {wien_res(x_wien):.3e}")
print(f"    b = C2/x              = {b_wien:.12e} m K")
print(f"                          = {b_wien*1e6:.6f} um K")
print("    The value usually quoted is 2898 um K; the computed value is")
print(f"    {b_wien*1e6:.3f} um K, and the difference is rounding in the quotation.")

# independent check: locate the peak numerically for several temperatures
print("\n    Cross-check by locating the peak of the curve directly:")
print(f"    {'T [K]':>8} {'lambda_max [um]':>18} {'lambda_max T [um K]':>22}")
for T in (300.0, 1000.0, 5772.0):
    lam_grid = np.logspace(np.log10(b_wien / T) - 1.0,
                           np.log10(b_wien / T) + 1.0, 2000001)
    lam_pk = lam_grid[np.argmax(planck(lam_grid, T))]
    print(f"    {T:>8.0f} {lam_pk*1e6:>18.6f} {lam_pk*T*1e6:>22.4f}")
print("    The product is the same number at every temperature -- which is")
print("    the content of the displacement law, and is not assumed anywhere")
print("    in the calculation that produced the three rows.")

# ==============================================================================
# 4. BAND EMISSION FRACTIONS, TWO WAYS
# ==============================================================================
def band_fraction_series(lamT, n_terms=200):
    """F(0 -> lambda T) by the classical rapidly convergent series.

        F = (15/pi^4) sum_{n>=1} e^{-n z}/n * (z^3 + 3z^2/n + 6z/n^2 + 6/n^3)

    with z = C2/(lambda T).  Each term carries a factor e^{-n z}, so for
    moderate z a handful of terms suffices; for very small z (long wavelength,
    hot body) convergence slows and the quadrature route below is preferable.
    """
    lamT = np.atleast_1d(np.asarray(lamT, dtype=float))
    z = C2 / lamT
    out = np.zeros_like(z)
    for n in range(1, n_terms + 1):
        out += (np.exp(-n * z) / n *
                (z ** 3 + 3.0 * z ** 2 / n + 6.0 * z / n ** 2 + 6.0 / n ** 3))
    return 15.0 / np.pi ** 4 * out


def band_fraction_quad(lamT):
    """F(0 -> lambda T) by direct quadrature of the dimensionless integrand.

    F = (15/pi^4) int_z^inf x^3/(e^x - 1) dx, with z = C2/(lambda T).  Shares
    no algebra with the series above, so agreement between the two is a real
    check.
    """
    lamT = np.atleast_1d(np.asarray(lamT, dtype=float))
    f = lambda x: x ** 3 / np.expm1(x)
    out = np.empty_like(lamT)
    for i, LT in enumerate(lamT):
        z = C2 / LT
        val, _ = quad(f, z, np.inf, limit=400)
        out[i] = 15.0 / np.pi ** 4 * val
    return out


print("\n" + "-" * 78)
print("  BAND EMISSION FRACTION F(0 -> lambda T), TWO INDEPENDENT ROUTES")
print(f"  {'lambda T [um K]':>16} {'series':>12} {'quadrature':>12} {'diff':>11}")
lamT_tab = np.array([1000, 2000, 2898, 4000, 6000, 8000, 12000, 20000, 50000]) * 1e-6
Fs = band_fraction_series(lamT_tab)
Fq = band_fraction_quad(lamT_tab)
for LT, a, bq in zip(lamT_tab, Fs, Fq):
    print(f"  {LT*1e6:>16.0f} {a:>12.8f} {bq:>12.8f} {abs(a-bq):>11.2e}")
print(f"\n    maximum disagreement over the table = {np.max(np.abs(Fs-Fq)):.2e}")

# Closure.  This test was WRONG in a first draft, in an instructive way.
# The draft evaluated F at lambda T = 1 m K, found 0.99999985 rather than 1,
# and attributed the shortfall to truncation of the series.  It is not a
# numerical error at all: at that lambda T the argument z = C2/(lambda T) is
# 0.0144, small but finite, so a real sliver of energy lies beyond and F is
# genuinely less than one.  Both methods agreed on the shortfall, which should
# have been the clue -- two independent routes do not make the same
# truncation error.
#
# The honest test is against the ASYMPTOTE.  As z -> 0 the integrand
# x^3/(e^x - 1) -> x^2, so
#
#       1 - F = (15/pi^4) int_0^z x^3/(e^x - 1) dx  ->  5 z^3 / pi^4
#
# which is a prediction, not a restatement, and it can be checked over
# several decades of z.
print("\n    CLOSURE.  F -> 1 as lambda T -> infinity, and the RATE is known:")
print("    1 - F -> 5 z^3 / pi^4 with z = C2/(lambda T).")
print(f"    {'lambda T [m K]':>16} {'z':>11} {'1 - F':>14} {'5 z^3/pi^4':>14} {'ratio':>9}")
for LT in (0.1, 0.3, 1.0, 3.0, 10.0):
    z = C2 / LT
    Fq = band_fraction_quad(np.array([LT]))[0]
    asym = 5.0 * z ** 3 / np.pi ** 4
    print(f"    {LT:>16.1f} {z:>11.3e} {1.0-Fq:>14.6e} {asym:>14.6e} "
          f"{(1.0-Fq)/asym:>9.5f}")
print("    The ratio tends to one as z falls, confirming both the value and")
print("    the cube law.  F does reach unity, but only in the limit.")

# The series, by contrast, DOES have a truncation error here, and it is
# separable from the physics above by simply adding terms.
print("\n    Series truncation at lambda T = 1 m K (z = %.4f), by term count:"
      % (C2 / 1.0))
F_ref = band_fraction_quad(np.array([1.0]))[0]
for nt in (50, 200, 2000, 20000):
    Fv = band_fraction_series(np.array([1.0]), n_terms=nt)[0]
    print(f"      {nt:>6d} terms: {Fv:.12f}   error vs quadrature "
          f"{abs(Fv-F_ref):.2e}")
print("      Every term carries e^{-n z}; when z is small the decay is slow")
print("      and the series is the wrong tool.  Where the main table agrees")
print("      to machine precision, both methods are converged -- agreement")
print("      there says nothing about either outside that range.")

# the peak of the Planck curve sits at a fixed fraction of the total
F_at_peak = band_fraction_series(np.array([b_wien]))[0]
print(f"\n    F(0 -> lambda_max T) = {F_at_peak:.8f}")
print("    A quarter of a blackbody's energy lies below the peak wavelength")
print("    and three quarters above it.  The distribution is strongly skewed,")
print("    which is why the peak is a poor summary of where the energy is.")

# ==============================================================================
# 5. ENGINEERING CONSEQUENCE: WHY HOT THINGS GLOW AND WARM THINGS DO NOT
# ==============================================================================
print("\n" + "-" * 78)
print("  THE VISIBLE BAND (0.40 - 0.70 um)")
print(f"  {'source':>22} {'T [K]':>8} {'visible fraction':>18} {'E_b [W/m^2]':>16}")
VIS_LO, VIS_HI = 0.40e-6, 0.70e-6
vis = {}
for name, T in (("room-temperature wall", 300.0),
                ("hot plate", 700.0),
                ("incandescent filament", 2800.0),
                ("the Sun's photosphere", 5772.0)):
    # The visible band sits far out on the short-wavelength tail at 300 K,
    # where the two band fractions are equal to many figures and their
    # DIFFERENCE is destroyed by cancellation.  Integrating the band directly
    # avoids subtracting two nearly equal numbers.
    zl, zh = C2 / (VIS_HI * T), C2 / (VIS_LO * T)
    fint, _ = quad(lambda x: x ** 3 / np.expm1(x), zl, zh, limit=400)
    f_vis = 15.0 / np.pi ** 4 * fint
    vis[T] = f_vis
    print(f"  {name:>22} {T:>8.0f} {f_vis:>18.4e} {SIGMA_CLOSED*T**4:>16.4g}")
print("""
    This single column explains an everyday observation.  A wall at 300 K
    radiates about 460 W/m^2, which is not a small number -- but essentially
    none of it is visible, so the wall is seen only by reflected light.  Raise
    the temperature to 2800 K and the emissive power rises by a factor of
    {:.0f}, while the VISIBLE fraction rises by a factor of {:.1e}.  The
    filament does not merely radiate more; it radiates somewhere else.""".format(
        (2800.0 / 300.0) ** 4, vis[2800.0] / vis[300.0]))

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

lam = np.logspace(-7.3, -3.7, 1200)
temps = [300.0, 700.0, 1500.0, 2800.0, 5772.0]
cols = ["#2166ac", "#1b7837", "#e08214", "#b2182b", "#762a83"]
for T, c in zip(temps, cols):
    ax[0].loglog(lam * 1e6, planck(lam, T), "-", lw=1.9, color=c,
                 label=rf"$T = {T:g}$ K")
# Wien locus
Tl = np.logspace(np.log10(200), np.log10(8000), 200)
ax[0].loglog(b_wien / Tl * 1e6, planck(b_wien / Tl, Tl), "k--", lw=1.5,
             label=r"Wien locus $\lambda_{max}T = b$")
ax[0].axvspan(VIS_LO * 1e6, VIS_HI * 1e6, color="0.5", alpha=0.25)
ax[0].annotate("visible", xy=(0.52, 0.04), xycoords=("data", "axes fraction"),
               fontsize=8.5, color="0.25", ha="center", rotation=90)
ax[0].set_xlabel(r"Wavelength $\lambda$  [$\mu$m]")
ax[0].set_ylabel(r"$E_{b,\lambda}$  [W m$^{-2}$ $\mu$m$^{-1}$ $\times 10^{6}$]")
ax[0].set_title("(a) Planck's distribution")
ax[0].set_ylim(1e2, 1e15)
ax[0].legend(fontsize=7.5, loc="upper right")

LT = np.logspace(np.log10(400), np.log10(1e5), 400) * 1e-6
ax[1].semilogx(LT * 1e6, band_fraction_series(LT), "-", lw=2.2,
               color="#b2182b", label="series")
LTq = np.logspace(np.log10(400), np.log10(1e5), 25) * 1e-6
ax[1].semilogx(LTq * 1e6, band_fraction_quad(LTq), "o", ms=5.5, mfc="none",
               mew=1.5, color="#2166ac", label="quadrature")
ax[1].axvline(b_wien * 1e6, color="#1b7837", ls=":", lw=1.6)
ax[1].annotate(rf"$\lambda_{{max}}T = {b_wien*1e6:.0f}\ \mu$m K" "\n"
               rf"$F = {F_at_peak:.3f}$",
               xy=(0.30, 0.62), xycoords="axes fraction", fontsize=8.5,
               color="#1b7837",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.28"))
ax[1].set_xlabel(r"$\lambda T$  [$\mu$m K]")
ax[1].set_ylabel(r"$F(0 \rightarrow \lambda T)$")
ax[1].set_title("(b) Band fraction, two independent routes")
ax[1].set_ylim(0, 1.02)
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 9.1 -- Blackbody radiation from first principles",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_1a_planck.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

# convergence of the series in the number of terms
for LTv, c in ((2000e-6, "#2166ac"), (6000e-6, "#b2182b"),
               (50000e-6, "#1b7837")):
    ref = band_fraction_quad(np.array([LTv]))[0]
    ns = np.arange(1, 41)
    errs = [abs(band_fraction_series(np.array([LTv]), n_terms=n)[0] - ref)
            for n in ns]
    errs = np.maximum(np.array(errs), 1e-17)
    ax[0].semilogy(ns, errs, "-", lw=1.8, color=c,
                   label=rf"$\lambda T = {LTv*1e6:.0f}\ \mu$m K")
ax[0].set_xlabel("terms retained in the series")
ax[0].set_ylabel("error against quadrature")
ax[0].set_title("(a) The series converges faster when $\\lambda T$ is small")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].set_ylim(1e-17, 1)

# visible fraction against temperature
Tv = np.logspace(np.log10(250), np.log10(10000), 300)
fv = (band_fraction_series(VIS_HI * Tv) - band_fraction_series(VIS_LO * Tv))
ax[1].loglog(Tv, np.maximum(fv, 1e-20), "-", lw=2.2, color="#b2182b",
             label="fraction in 0.40-0.70 $\\mu$m")
for name, T, dy in (("300 K wall", 300.0, 8.0), ("filament", 2800.0, 0.06),
                    ("Sun", 5772.0, 0.06)):
    f_here = (band_fraction_series(np.array([VIS_HI * T]))[0] -
              band_fraction_series(np.array([VIS_LO * T]))[0])
    ax[1].plot([T], [max(f_here, 1e-20)], "o", ms=7, mfc="none", mew=1.8,
               color="#2166ac")
    ax[1].annotate(name, xy=(T, max(f_here, 1e-20)),
                   xytext=(T * 0.42, max(f_here, 1e-20) * dy),
                   fontsize=8.2, color="#2166ac",
                   arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.9))
ax[1].set_xlabel(r"$T$  [K]")
ax[1].set_ylabel("fraction of emission in the visible")
ax[1].set_title("(b) Why warm objects do not glow")
ax[1].set_ylim(1e-20, 1)
ax[1].legend(fontsize=8.5, loc="lower right")

fig.suptitle("Example 9.1 -- Band fractions and their consequences",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_1b_bands.png")
plt.close(fig)

print("Figures written: fig_9_1a_planck.png, fig_9_1b_bands.png")
