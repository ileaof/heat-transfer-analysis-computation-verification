"""
================================================================================
 EXAMPLE 1.1 -- ANALYTICAL SOLUTION
 Composite-mode heat transfer through a furnace wall
================================================================================

 PHYSICAL PROBLEM
 ----------------
 A plane wall of thickness L separates a hot process gas from the surroundings.
 The inner face (x = 0) is held at a known temperature T0 by the process.  The
 outer face (x = L) loses heat simultaneously by convection to ambient air at
 T_inf and by thermal radiation to surroundings at T_sur.  The wall conductivity
 k is constant, there is no internal generation, and the process is steady.

 This single problem exercises all three transport mechanisms introduced in
 Chapter 1 -- Fourier conduction, Newton convection, Stefan-Boltzmann radiation
 -- and is therefore used as the reference problem for Examples 1.1, 1.2, 1.3.

 GOVERNING EQUATION  (steady, 1-D, constant k, no generation)
 -----------------------------------------------------------
        d  /     dT \
       --- | k * -- |  = 0        for   0 < x < L
        dx \     dx /

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :  T(0) = T0                                        (Dirichlet)
   x = L : -k dT/dx|_L = h ( T_L - T_inf )
                         + eps * sigma * ( T_L^4 - T_sur^4 ) (nonlinear Robin)

 INITIAL CONDITIONS
 ------------------
   None -- the problem is steady.  (Example 1.3 restores the transient term.)

 ANALYTICAL SOLUTION
 -------------------
 With k constant and no source term, the governing equation integrates twice to
 give a strictly LINEAR temperature profile

        T(x) = T0 - G * x ,      G = -dT/dx = q'' / k = const.

 The gradient G is fixed by the surface energy balance at x = L.  Substituting
 T_L = T0 - G*L into the Robin condition gives the scalar nonlinear equation

        F(G) = k*G - h*(T0 - G*L - T_inf)
                   - eps*sigma*((T0 - G*L)^4 - T_sur^4) = 0

 F is strictly monotone increasing in G on the physical range, so it has exactly
 one root, obtained here to machine precision with Brent's method.  The solution
 is therefore CLOSED FORM in x and requires only a scalar root-find for G --
 this is what is meant by "analytical" for a nonlinear-boundary problem.

 SYMBOLS  (all SI)
 -----------------
   L      [m]        wall thickness
   k      [W/(m K)]  thermal conductivity
   h      [W/(m^2 K)] convection coefficient
   eps    [-]        total hemispherical emissivity of the outer face
   sigma  [W/(m^2 K^4)] Stefan-Boltzmann constant
   T0     [K]        prescribed inner-face temperature
   T_inf  [K]        ambient air temperature
   T_sur  [K]        effective surroundings temperature
   T_L    [K]        outer-face temperature (unknown)
   G      [K/m]      temperature gradient magnitude
   q''    [W/m^2]    heat flux through the wall

 OUTPUTS
 -------
   fig_1_1a_profile.png   temperature profile + flux decomposition
   fig_1_1b_study.png     linearised-radiation error and evaluation convergence
   printed console tables (reproducible; no random numbers used anywhere)

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ------------------------------------------------------------------------------
# Publication-quality plot defaults (applied to every figure in this book)
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ==============================================================================
# 1. PROBLEM DATA  -- a fire-clay furnace wall, SI units throughout
# ==============================================================================
L = 0.05            # [m]          wall thickness
k = 1.4             # [W/(m K)]    thermal conductivity (fire-clay brick)
h = 25.0            # [W/(m^2 K)]  outer-face convection coefficient
eps = 0.85          # [-]          outer-face emissivity
SIGMA = 5.670374419e-8   # [W/(m^2 K^4)] Stefan-Boltzmann constant (CODATA 2018)

T0 = 600.0          # [K]  inner-face temperature (prescribed)
T_inf = 300.0       # [K]  ambient air temperature
T_sur = 300.0       # [K]  surroundings temperature for radiation exchange


# ==============================================================================
# 2. ANALYTICAL MACHINERY
# ==============================================================================
def surface_balance(G, emissivity=eps):
    """Residual F(G) of the outer-face energy balance.

    Conduction arriving at the face (k*G) must equal the sum of the convective
    and radiative fluxes leaving it.  Root F(G) = 0 gives the true gradient.
    Passing emissivity = 0 switches radiation off.
    """
    T_L = T0 - G * L                                    # outer face [K]
    q_cond = k * G                                      # Fourier    [W/m^2]
    q_conv = h * (T_L - T_inf)                          # Newton     [W/m^2]
    q_rad = emissivity * SIGMA * (T_L**4 - T_sur**4)    # Stefan-Boltzmann
    return q_cond - q_conv - q_rad


def solve_gradient(emissivity=eps):
    """Bracket and solve F(G) = 0 with Brent's method.

    Physical bracket: G = 0 (perfectly insulated face, F < 0) up to the value
    that would drive T_L down to T_inf (F > 0).  F is monotone between them.
    """
    G_lo = 0.0
    G_hi = (T0 - T_inf) / L                # T_L = T_inf : upper physical bound
    return brentq(surface_balance, G_lo, G_hi, args=(emissivity,), xtol=1e-14)


def temperature(x, G):
    """Exact temperature field T(x) = T0 - G*x  [K]."""
    return T0 - G * x


def h_radiation(T_L):
    """Linearised radiation coefficient h_r [W/(m^2 K)].

    Factorising T_L^4 - T_sur^4 = (T_L + T_sur)(T_L^2 + T_sur^2)(T_L - T_sur)
    lets the radiative flux be written in Newton form q_r = h_r (T_L - T_sur).
    This is EXACT at the point of evaluation and is the standard engineering
    device for folding radiation into a thermal-resistance network.
    """
    return eps * SIGMA * (T_L + T_sur) * (T_L**2 + T_sur**2)


# ==============================================================================
# 3. SOLVE
# ==============================================================================
t_start = time.perf_counter()

G_full = solve_gradient(emissivity=eps)   # with radiation
G_conv = solve_gradient(emissivity=0.0)   # convection only (for comparison)

T_L_full = temperature(L, G_full)
T_L_conv = temperature(L, G_conv)

q_total = k * G_full                        # [W/m^2] total flux through wall
q_conv_part = h * (T_L_full - T_inf)
q_rad_part = eps * SIGMA * (T_L_full**4 - T_sur**4)

h_r = h_radiation(T_L_full)

# Dimensionless groups (Chapter 1 dimensional analysis)
Bi = h * L / k                              # convective Biot number       [-]
Bi_r = h_r * L / k                          # radiative Biot number        [-]
Bi_tot = (h + h_r) * L / k                  # combined Biot number         [-]
N_rc = h_r / h                              # radiation-to-convection ratio[-]

# Thermal resistances per unit area [m^2 K/W]
R_cond = L / k
R_conv = 1.0 / h
R_rad = 1.0 / h_r
R_surf = 1.0 / (h + h_r)                    # parallel combination
R_tot = R_cond + R_surf

print("=" * 78)
print("EXAMPLE 1.1 -- ANALYTICAL SOLUTION, COMPOSITE-MODE FURNACE WALL")
print("=" * 78)
print(f"  Wall thickness            L      = {L:.4f} m")
print(f"  Thermal conductivity      k      = {k:.4f} W/(m K)")
print(f"  Convection coefficient    h      = {h:.4f} W/(m^2 K)")
print(f"  Emissivity                eps    = {eps:.4f} -")
print("-" * 78)
print(f"  Gradient (with radiation) G      = {G_full:.10f} K/m")
print(f"  Outer-face temperature    T_L    = {T_L_full:.10f} K")
print(f"  Total flux                q''    = {q_total:.10f} W/m^2")
print(f"    ...convective share            = {q_conv_part:10.4f} W/m^2 "
      f"({100*q_conv_part/q_total:5.2f} %)")
print(f"    ...radiative  share            = {q_rad_part:10.4f} W/m^2 "
      f"({100*q_rad_part/q_total:5.2f} %)")
print("-" * 78)
print(f"  Neglecting radiation:     T_L    = {T_L_conv:.6f} K "
      f"(error {T_L_conv - T_L_full:+.3f} K)")
print(f"                            q''    = {k*G_conv:.6f} W/m^2 "
      f"(error {100*(k*G_conv - q_total)/q_total:+.2f} %)")
print("-" * 78)
print(f"  Linearised radiation      h_r    = {h_r:.6f} W/(m^2 K)")
print(f"  Biot (convective)         Bi     = {Bi:.6f}")
print(f"  Biot (radiative)          Bi_r   = {Bi_r:.6f}")
print(f"  Biot (combined)           Bi_tot = {Bi_tot:.6f}")
print(f"  Radiation/convection      N_rc   = {N_rc:.6f}")
print("-" * 78)
print(f"  R_cond = {R_cond:.6f}   R_conv = {R_conv:.6f}   R_rad = {R_rad:.6f}")
print(f"  R_surf = {R_surf:.6f}   R_tot  = {R_tot:.6f}   [m^2 K/W]")
print(f"  Check:  (T0 - T_inf)/R_tot = {(T0 - T_inf)/R_tot:.10f} W/m^2")
print(f"          residual vs q''    = {abs((T0 - T_inf)/R_tot - q_total):.3e}")
print("=" * 78)

# --- Independent verification of the closed-form solution ---------------------
# The analytical field must satisfy (i) the ODE, (ii) both boundary conditions.
x_chk = np.linspace(0.0, L, 1001)
T_chk = temperature(x_chk, G_full)
d2T = np.gradient(np.gradient(T_chk, x_chk), x_chk)[1:-1]   # interior only
print(f"  Verification  max|d2T/dx2|            = {np.max(np.abs(d2T)):.3e} K/m^2")
print(f"  Verification  |T(0) - T0|             = {abs(T_chk[0] - T0):.3e} K")
print(f"  Verification  |surface balance at L|  = "
      f"{abs(surface_balance(G_full)):.3e} W/m^2")

# ==============================================================================
# 4. CONVERGENCE STUDY
# ==============================================================================
# The field is exact, so there is no discretisation error to refine away.  What
# CAN be studied is (a) how fast the root-finder converges on G, and (b) how the
# error of a piecewise-linear reconstruction of T(x) behaves under refinement --
# the latter is identically zero here because T is itself linear, which is a
# useful sanity result to carry into Example 1.2.
grids = np.array([5, 10, 20, 40, 80, 160, 320])
recon_err = []
for N in grids:
    xs = np.linspace(0.0, L, N + 1)
    Ts = temperature(xs, G_full)
    xf = np.linspace(0.0, L, 2001)
    err = np.max(np.abs(np.interp(xf, xs, Ts) - temperature(xf, G_full)))
    recon_err.append(err)
recon_err = np.array(recon_err)

print("-" * 78)
print("  Piecewise-linear reconstruction error (exact profile is linear)")
print(f"  {'N':>6} {'max|err| [K]':>16}")
for N, e in zip(grids, recon_err):
    print(f"  {N:>6d} {e:>16.3e}")

# --- Sensitivity of T_L and q'' to the emissivity -----------------------------
eps_scan = np.linspace(0.0, 1.0, 51)
TL_scan, q_scan = [], []
for e_val in eps_scan:
    G_s = solve_gradient(emissivity=e_val)
    TL_scan.append(temperature(L, G_s))
    q_scan.append(k * G_s)
TL_scan, q_scan = np.array(TL_scan), np.array(q_scan)

cpu = time.perf_counter() - t_start
print("-" * 78)
print(f"  CPU time for Example 1.1 = {cpu:.4f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
x = np.linspace(0.0, L, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

# (a) temperature profiles
ax[0].plot(x * 1e3, temperature(x, G_full), "-", lw=2.2, color="#b2182b",
           label="Conduction + convection + radiation")
ax[0].plot(x * 1e3, temperature(x, G_conv), "--", lw=1.8, color="#2166ac",
           label="Conduction + convection only")
ax[0].axhline(T_inf, color="0.45", lw=1.0, ls=":")
ax[0].annotate(r"$T_\infty = T_{sur}$", xy=(L * 1e3 * 0.05, T_inf + 6),
               color="0.35", fontsize=10)
ax[0].plot([L * 1e3], [T_L_full], "o", ms=6, color="#b2182b", zorder=5)
ax[0].annotate(rf"$T_L = {T_L_full:.1f}$ K",
               xy=(L * 1e3, T_L_full), xytext=(-96, 26),
               textcoords="offset points", fontsize=10, color="#b2182b",
               arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.0))
ax[0].set_xlabel(r"Position $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Exact temperature field")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_xlim(0, L * 1e3)

# (b) flux split as a stacked bar + resistance network values
labels = ["Convection\n$q''_{conv}$", "Radiation\n$q''_{rad}$", "Total\n$q''$"]
vals = [q_conv_part, q_rad_part, q_total]
colors = ["#2166ac", "#b2182b", "#4d4d4d"]
bars = ax[1].bar(labels, vals, color=colors, width=0.6, edgecolor="k", lw=0.6)
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v + 60, f"{v:.0f}",
               ha="center", fontsize=10)
ax[1].set_ylabel(r"Heat flux [W m$^{-2}$]")
ax[1].set_title("(b) Outer-face flux decomposition")
ax[1].set_ylim(0, q_total * 1.22)

fig.suptitle("Example 1.1 -- Analytical solution, composite-mode plane wall",
             fontsize=12.5, y=1.02)
fig.savefig("fig_1_1a_profile.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

# (a) emissivity sensitivity
ax[0].plot(eps_scan, TL_scan, "-", lw=2.0, color="#b2182b")
ax[0].axvline(eps, color="0.4", ls="--", lw=1.0)
ax[0].annotate(rf"design $\varepsilon = {eps}$", xy=(eps, TL_scan[-1] + 10),
               fontsize=9, color="0.3", ha="right")
ax[0].set_xlabel(r"Emissivity $\varepsilon$ [-]")
ax[0].set_ylabel(r"Outer-face temperature $T_L$ [K]")
ax[0].set_title("(a) Sensitivity of $T_L$ to surface emissivity")

axb = ax[0].twinx()
axb.plot(eps_scan, q_scan, "--", lw=1.6, color="#2166ac")
axb.set_ylabel(r"Heat flux $q''$ [W m$^{-2}$]", color="#2166ac")
axb.tick_params(axis="y", colors="#2166ac")
axb.grid(False)

# (b) reconstruction error (machine-precision floor)
ax[1].loglog(grids, np.maximum(recon_err, 1e-18), "o-", lw=1.8, ms=6,
             color="#4d4d4d", label="Piecewise-linear reconstruction")
ax[1].axhline(np.finfo(float).eps * T0, color="#b2182b", ls=":", lw=1.4,
              label=r"machine precision $\sim \epsilon_{mach} T_0$")
ax[1].set_xlabel(r"Number of intervals $N$")
ax[1].set_ylabel(r"$\max|T_{interp} - T_{exact}|$ [K]")
ax[1].set_title("(b) Reconstruction error of the exact field")
ax[1].legend(loc="upper right", fontsize=9 )

fig.suptitle("Example 1.1 -- Sensitivity and exactness checks",
             fontsize=12.5, y=1.02)
fig.savefig("fig_1_1b_study.png")

plt.close(fig)
print("Figures written: fig_1_1a_profile.png, fig_1_1b_study.png")

