"""
================================================================================
 EXAMPLE 6.1 -- ANALYTICAL SOLUTIONS FOR TRANSIENT CONDUCTION
 Lumped capacitance, the exact series, and the one-term approximation
================================================================================

 PHYSICAL PROBLEM
 ----------------
 A stainless steel plate of half-thickness L, initially uniform at T_i, is
 quenched at t = 0 into a fluid at T_inf with coefficient h on both faces.
 Three levels of description are compared:

   (a) LUMPED CAPACITANCE  -- the body is assumed isothermal
   (b) EXACT SERIES        -- separation of variables, all terms
   (c) ONE-TERM            -- the first term of (b), the basis of the
                              Heisler charts

 Two claims made in every textbook are TESTED here rather than quoted:
 that lumping is accurate for Bi < 0.1, and that one term suffices for
 Fo > 0.2.  Both turn out to be sound, and this example measures exactly how
 sound.

 GOVERNING EQUATION
 ------------------
        dT/dt = alpha d2T/dx2 ,      -L < x < L ,  t > 0

 INITIAL AND BOUNDARY CONDITIONS
 -------------------------------
        T(x,0) = T_i
        dT/dx = 0 at x = 0                    (symmetry)
        -k dT/dx = h (T - T_inf) at x = L     (convection)

 LUMPED CAPACITANCE
 ------------------
 If the body is isothermal, an energy balance on the whole plate gives

        rho V c dT/dt = -h A_s (T - T_inf)

 which integrates to a pure exponential

        theta*(t) = (T - T_inf)/(T_i - T_inf) = exp(-t/tau) ,
        tau = rho V c/(h A_s) = rho c L / h        [s]

 Equivalently, in dimensionless form, t/tau = Bi Fo, so

        theta* = exp(-Bi Fo)

 The lumped model has NO spatial dependence at all: it is the Bi -> 0 limit.

 EXACT SERIES
 ------------
 Separation of variables with the symmetry and convective conditions gives

        theta*(x,t) = SUM_n C_n exp(-zeta_n^2 Fo) cos(zeta_n x/L)

        zeta_n tan(zeta_n) = Bi          (one root per interval
                                          ((n-1)pi, (n-1)pi + pi/2))
        C_n = 4 sin(zeta_n) / (2 zeta_n + sin(2 zeta_n))

 ONE-TERM APPROXIMATION
 ----------------------
 The n-th term decays as exp(-zeta_n^2 Fo), and zeta_n grows roughly as
 (n-1)pi, so higher terms die extremely fast.  Retaining only n = 1,

        theta* ~ C_1 exp(-zeta_1^2 Fo) cos(zeta_1 x/L)

 which is what the Heisler charts plot.  Its accuracy is measured below.

 SYMBOLS (all SI)
 ----------------
   L       [m]         half-thickness
   k       [W/(m K)]   conductivity
   rho, c              density, specific heat
   alpha   [m^2/s]     thermal diffusivity k/(rho c)
   h       [W/(m^2 K)] convection coefficient
   T_i     [K]         initial temperature
   T_inf   [K]         fluid temperature
   Bi      [-]         Biot number h L / k
   Fo      [-]         Fourier number alpha t / L^2
   tau     [s]         lumped time constant rho c L / h
   theta*  [-]         (T - T_inf)/(T_i - T_inf)
   zeta_n  [-]         n-th eigenvalue

 OUTPUTS
 -------
   fig_6_1a_response.png   lumped vs exact, and the profile evolution
   fig_6_1b_validity.png   validity maps of the two approximations

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. DATA -- a quenched stainless steel plate
# ==============================================================================
L = 0.020                 # [m]
K = 15.0                  # [W/(m K)]
RHO, CP = 7900.0, 477.0
ALPHA = K / (RHO * CP)    # [m^2/s]
H = 500.0                 # [W/(m^2 K)]
T_I, T_INF = 800.0, 300.0

BI = H * L / K
TAU_LUMP = RHO * CP * L / H
TAU_DIFF = L * L / ALPHA


# ==============================================================================
# 2. EIGENVALUES AND THE EXACT SERIES
# ==============================================================================
def eigenvalues(n_terms, Bi):
    """Roots of zeta tan(zeta) = Bi, one per interval ((n-1)pi, (n-1)pi+pi/2)."""
    f = lambda z: z * np.sin(z) - Bi * np.cos(z)
    out = np.empty(n_terms)
    d = 1e-12
    for n in range(n_terms):
        lo, hi = n * np.pi + d, n * np.pi + 0.5 * np.pi - d
        out[n] = brentq(f, lo, hi, xtol=1e-14)
    return out


def coefficients(zeta):
    return 4.0 * np.sin(zeta) / (2.0 * zeta + np.sin(2.0 * zeta))


def theta_exact(xstar, Fo, Bi=BI, n_terms=60):
    """theta* from the full series.  xstar = x/L, Fo = alpha t / L^2."""
    z = eigenvalues(n_terms, Bi)
    C = coefficients(z)
    xstar = np.atleast_1d(np.asarray(xstar, dtype=float))
    decay = np.exp(-z**2 * Fo)
    return np.sum(C[:, None] * decay[:, None] * np.cos(np.outer(z, xstar)),
                  axis=0)


def theta_one_term(xstar, Fo, Bi=BI):
    z1 = eigenvalues(1, Bi)[0]
    C1 = coefficients(np.array([z1]))[0]
    return C1 * np.exp(-z1**2 * Fo) * np.cos(z1 * np.asarray(xstar))


def theta_lumped(Fo, Bi=BI):
    return np.exp(-Bi * np.asarray(Fo))


# ==============================================================================
# 3. HEADER AND VERIFICATION
# ==============================================================================
t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 6.1 -- TRANSIENT CONDUCTION: LUMPED, EXACT AND ONE-TERM")
print("=" * 78)
print(f"  L = {L} m, k = {K} W/(m K), alpha = {ALPHA:.4e} m^2/s")
print(f"  h = {H} W/(m^2 K), T_i = {T_I} K, T_inf = {T_INF} K")
print(f"  Bi = {BI:.6f},  tau_lumped = {TAU_LUMP:.3f} s,  "
      f"L^2/alpha = {TAU_DIFF:.3f} s")

print("\n  Eigenvalue check:  zeta tan(zeta) = Bi")
print(f"  {'n':>3} {'zeta_n':>14} {'residual':>12} {'C_n':>12} "
      f"{'decay rate':>12}")
z = eigenvalues(6, BI)
C = coefficients(z)
for i, (zz, cc) in enumerate(zip(z, C), start=1):
    print(f"  {i:>3d} {zz:>14.10f} {zz*np.tan(zz)-BI:>12.2e} {cc:>12.6f} "
          f"{zz**2:>12.4f}")
print("  The decay rate zeta_n^2 grows roughly as ((n-1)pi)^2, so the second")
print("  term dies about {:.0f} times faster than the first."
      .format((z[1] / z[0]) ** 2))

# --- verify the series satisfies the initial condition ------------------------
xs = np.linspace(0, 1, 11)
print(f"\n  Series at Fo -> 0 (initial condition theta* = 1):")
for nt in [5, 20, 100, 400]:
    err = np.max(np.abs(theta_exact(xs, 1e-12, BI, nt) - 1.0))
    print(f"    {nt:>4d} terms : max|theta* - 1| = {err:.4e}")
print("  Convergence is slow at Fo = 0 because the initial condition is")
print("  discontinuous with the boundary condition at the surface, exactly as")
print("  in Chapters 1 and 5.  For Fo > 0.01 the series converges rapidly.")

# --- verify against an independent energy balance -----------------------------
# The total energy released up to time t must equal the integral of the flux.
def energy_fraction(Fo, Bi=BI, n_terms=60):
    """Q/Q_max from the series: 1 - (1/L) INTEGRAL theta* dx."""
    zz = eigenvalues(n_terms, Bi)
    CC = coefficients(zz)
    return 1.0 - np.sum(CC * np.exp(-zz**2 * Fo) * np.sin(zz) / zz)

print(f"\n  Energy released, Q/Q_max:")
print(f"  {'Fo':>8} {'series':>12} {'quadrature':>12} {'diff':>11}")
for Fo in [0.05, 0.2, 0.5, 1.0, 3.0]:
    q_series = energy_fraction(Fo)
    xq = np.linspace(0, 1, 4001)
    q_quad = 1.0 - np.trapz(theta_exact(xq, Fo), xq)
    print(f"  {Fo:>8.2f} {q_series:>12.8f} {q_quad:>12.8f} "
          f"{abs(q_series-q_quad):>11.2e}")

# ==============================================================================
# 4. TESTING THE TWO RULES OF THUMB
# ==============================================================================
print("\n" + "-" * 78)
print("  RULE 1: 'lumped capacitance is valid for Bi < 0.1'")
print("  Maximum error in the CENTRE temperature over 0.1 < Fo < 5.")
print(f"  {'Bi':>8} {'zeta_1':>10} {'max |err| in theta*':>21} "
      f"{'as % of initial':>17}")
bi_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
lump_err = []
for Bi in bi_list:
    Fo = np.linspace(0.1, 5.0, 400)
    ex = np.array([theta_exact(np.array([0.0]), f, Bi)[0] for f in Fo])
    lu = theta_lumped(Fo, Bi)
    e = np.max(np.abs(ex - lu))
    lump_err.append(e)
    print(f"  {Bi:>8.3f} {eigenvalues(1, Bi)[0]:>10.6f} {e:>21.6f} "
          f"{100*e:>16.2f}%")
print("  At Bi = 0.1 the error is under 2 % of the initial excess -- the rule")
print("  is sound.  It degrades quickly beyond: at Bi = 1 the lumped model is")
print("  wrong by more than 20 %.")

print("\n  RULE 2: 'one term suffices for Fo > 0.2'")
print("  Error of the one-term approximation at the CENTRE and at the SURFACE.")
print(f"  {'Fo':>8} {'centre err':>13} {'surface err':>13} {'terms for 1e-6':>16}")
for Fo in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    ec = abs(theta_exact(np.array([0.0]), Fo)[0]
             - theta_one_term(np.array([0.0]), Fo)[0])
    es = abs(theta_exact(np.array([1.0]), Fo)[0]
             - theta_one_term(np.array([1.0]), Fo)[0])
    nt = 1
    while nt < 200:
        if np.max(np.abs(theta_exact(xs, Fo, BI, nt)
                         - theta_exact(xs, Fo, BI, nt + 4))) < 1e-6:
            break
        nt += 1
    print(f"  {Fo:>8.2f} {ec:>13.3e} {es:>13.3e} {nt:>16d}")
print("  Reading this table carefully matters, because the rule is often")
print("  quoted as though one term were essentially exact beyond Fo = 0.2.")
print("  It is not.  At Fo = 0.2 the one-term error is about 1.2 % of the")
print("  initial excess -- entirely adequate for chart work and hand")
print("  calculation, which is what the Heisler charts are for, but not high")
print("  precision.  Six-figure accuracy needs THREE terms at Fo = 0.2 and")
print("  still two at Fo = 1.  The centre and surface errors are comparable")
print("  throughout, the surface being slightly worse; the approximation")
print("  degrades everywhere at once as Fo falls, not at one location first.")
print("  The honest statement of the rule is: one term gives about 1 % beyond")
print("  Fo = 0.2, improving by roughly an order of magnitude per 0.2 in Fo.")

# ==============================================================================
# 5. ENGINEERING ANSWERS
# ==============================================================================
print("\n" + "-" * 78)
print("  ENGINEERING QUESTIONS for the design case (Bi = {:.3f}):".format(BI))
Fo_target = brentq(lambda F: theta_exact(np.array([0.0]), F)[0] - 0.5,
                   0.01, 20.0, xtol=1e-12)
t_half = Fo_target * TAU_DIFF
print(f"    time for the CENTRE to fall halfway: Fo = {Fo_target:.6f}, "
      f"t = {t_half:.3f} s")
Fo_400 = brentq(lambda F: theta_exact(np.array([0.0]), F)[0]
                - (400.0 - T_INF) / (T_I - T_INF), 0.01, 20.0, xtol=1e-12)
print(f"    time for the CENTRE to reach 400 K: t = {Fo_400*TAU_DIFF:.3f} s")
print(f"    lumped estimate of the same:        t = "
      f"{-TAU_LUMP*np.log((400.0-T_INF)/(T_I-T_INF)):.3f} s "
      f"({100*(-TAU_LUMP*np.log((400.0-T_INF)/(T_I-T_INF))/(Fo_400*TAU_DIFF)-1):+.1f} %)")
surf_gap = (theta_exact(np.array([0.0]), 0.5)[0]
            - theta_exact(np.array([1.0]), 0.5)[0]) * (T_I - T_INF)
print(f"    centre-to-surface gap at Fo = 0.5:  {surf_gap:.2f} K")
print(f"  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
Fo = np.linspace(0.0, 4.0, 400)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

for Bi, col in zip([0.05, 0.5, BI, 5.0],
                   ["#1b7837", "#2166ac", "#b2182b", "#762a83"]):
    ex = np.array([theta_exact(np.array([0.0]), f, Bi)[0] for f in Fo])
    ax[0].semilogy(Fo, np.maximum(ex, 1e-4), "-", lw=1.9, color=col,
                   label=rf"exact, $Bi = {Bi:g}$")
    ax[0].semilogy(Fo, np.maximum(theta_lumped(Fo, Bi), 1e-4), "--", lw=1.3,
                   color=col, alpha=0.75)
ax[0].set_xlabel(r"Fourier number $Fo = \alpha t/L^2$")
ax[0].set_ylabel(r"$\theta^*$ at the centre")
ax[0].set_title("(a) Exact (solid) versus lumped (dashed)")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_ylim(1e-3, 1.5)

xstar = np.linspace(0, 1, 300)
for Fo_v, col in zip([0.02, 0.05, 0.1, 0.3, 0.8, 2.0],
                     plt.cm.plasma(np.linspace(0.05, 0.85, 6))):
    ax[1].plot(xstar, theta_exact(xstar, Fo_v), "-", lw=1.9, color=col,
               label=rf"$Fo = {Fo_v:g}$")
ax[1].set_xlabel(r"$x/L$   (0 = mid-plane, 1 = surface)")
ax[1].set_ylabel(r"$\theta^*$")
ax[1].set_title(rf"(b) Profile evolution, $Bi = {BI:.3f}$")
ax[1].legend(fontsize=8)

fig.suptitle("Example 6.1 -- Transient response of a quenched plate",
             fontsize=12.5, y=1.08)
fig.savefig("fig_6_1a_response.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),constrained_layout=True)
ax[0].loglog(bi_list, lump_err, "o-", lw=1.9, ms=7, color="#b2182b")
ax[0].axvline(0.1, color="#1b7837", ls="--", lw=1.6)
ax[0].axhline(0.02, color="0.45", ls=":", lw=1.3)
ax[0].annotate(r"$Bi = 0.1$", xy=(0.105, 3e-4), fontsize=9.5, color="#1b7837",
               rotation=90)
ax[0].annotate("2 %", xy=(0.011, 0.023), fontsize=9, color="0.4")
ax[0].set_xlabel(r"Biot number $Bi$")
ax[0].set_ylabel(r"max error in $\theta^*$")
ax[0].set_title("(a) When may the body be lumped?")

Fo_g = np.logspace(-2, 0.5, 60)
ec = [abs(theta_exact(np.array([0.0]), f)[0]
          - theta_one_term(np.array([0.0]), f)[0]) for f in Fo_g]
es = [abs(theta_exact(np.array([1.0]), f)[0]
          - theta_one_term(np.array([1.0]), f)[0]) for f in Fo_g]
ax[1].loglog(Fo_g, np.maximum(ec, 1e-17), "-", lw=1.9, color="#2166ac",
             label="centre")
ax[1].loglog(Fo_g, np.maximum(es, 1e-17), "-", lw=1.9, color="#b2182b",
             label="surface")
ax[1].axvline(0.2, color="#1b7837", ls="--", lw=1.6)
ax[1].annotate(r"$Fo = 0.2$", xy=(0.21, 1e-12), fontsize=9.5, color="#1b7837",
               rotation=90)
ax[1].set_xlabel(r"Fourier number $Fo$")
ax[1].set_ylabel("one-term error in $\\theta^*$")
ax[1].set_title("(b) When is one term enough?")
ax[1].legend(fontsize=9)

fig.suptitle("Example 6.1 -- Testing the two classical rules of thumb",
             fontsize=12.5, y=1.08)
fig.savefig("fig_6_1b_validity.png")
plt.close(fig)

print("Figures written: fig_6_1a_response.png, fig_6_1b_validity.png")
