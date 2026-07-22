"""
================================================================================
 EXAMPLE 2.1 -- ANALYTICAL SOLUTION
 Plane wall with uniform internal heat generation
================================================================================

 PHYSICAL PROBLEM
 ----------------
 A flat plate of half-thickness L generates heat uniformly throughout its
 volume at the rate q''' -- a resistively heated element, a nuclear fuel plate,
 or a slab curing exothermically.  Both faces are cooled identically by a fluid
 at T_inf with coefficient h.  By symmetry only half the plate need be solved:
 the mid-plane x = 0 is adiabatic, and x = L is the cooled surface.

 This is the reference problem for Examples 2.1, 2.2 and 2.3.

 GOVERNING EQUATION  (steady, 1-D, constant k, uniform generation)
 -----------------------------------------------------------------
        d  /     dT \
       --- | k * -- |  + q''' = 0        for   0 < x < L
        dx \     dx /

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :   dT/dx = 0                       (symmetry plane, adiabatic)
   x = L :  -k dT/dx|_L = h (T_L - T_inf)    (convective, third kind)

 ANALYTICAL SOLUTION
 -------------------
 Integrating once,  k dT/dx = -q''' x + C1.  The symmetry condition forces
 C1 = 0, so the local conduction flux grows linearly from the mid-plane:

        q''(x) = -k dT/dx = q''' x                                     (*)

 which is simply the statement that the heat crossing any plane equals the heat
 generated inboard of it.  Integrating again gives the PARABOLIC profile

        T(x) = T_max - q''' x^2 / (2k)

 The surface temperature follows from (*) evaluated at x = L combined with the
 convective condition, and requires no iteration because the problem is linear:

        q''' L = h (T_L - T_inf)   =>   T_L = T_inf + q''' L / h

 and therefore

        T_max = T(0) = T_L + q''' L^2 / (2k)
                     = T_inf + q''' L / h + q''' L^2 / (2k)

 The two terms have a clean physical reading: the first is the temperature rise
 needed to push the generated heat through the surface film, the second is the
 rise needed to conduct it from the mid-plane to the surface.  They are
 independent resistances in series and are compared below.

 SYMBOLS (all SI)
 ----------------
   L       [m]         half-thickness of the plate
   k       [W/(m K)]   thermal conductivity
   q'''    [W/m^3]     volumetric generation rate
   h       [W/(m^2 K)] convection coefficient
   T_inf   [K]         coolant temperature
   T_L     [K]         surface temperature
   T_max   [K]         mid-plane (maximum) temperature
   Bi      [-]         Biot number, hL/k

 OUTPUTS
 -------
   fig_2_1a_profile.png   temperature profile and flux distribution
   fig_2_1b_study.png     resistance split and dimensionless master curve

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. PROBLEM DATA -- a water-cooled resistively heated stainless plate
# ==============================================================================
L = 0.02            # [m]         half-thickness
K = 15.0            # [W/(m K)]   conductivity (AISI 304 at temperature)
QGEN = 5.0e6        # [W/m^3]     uniform volumetric generation
H = 500.0           # [W/(m^2 K)] convection coefficient (forced water)
T_INF = 350.0       # [K]         coolant bulk temperature

BI = H * L / K      # [-] Biot number


# ==============================================================================
# 2. ANALYTICAL SOLUTION
# ==============================================================================
def surface_temperature(q=QGEN, h=H, Tinf=T_INF):
    """T_L [K] from the surface energy balance q''' L = h (T_L - T_inf)."""
    return Tinf + q * L / h


def max_temperature(q=QGEN, k=K, h=H, Tinf=T_INF):
    """Mid-plane temperature [K]: film rise plus conduction rise."""
    return surface_temperature(q, h, Tinf) + q * L * L / (2.0 * k)


def temperature(x, q=QGEN, k=K, h=H, Tinf=T_INF):
    """Exact parabolic field T(x) [K]."""
    return max_temperature(q, k, h, Tinf) - q * np.asarray(x) ** 2 / (2.0 * k)


def flux(x, q=QGEN):
    """Exact conduction flux q''(x) = q''' x  [W/m^2]."""
    return q * np.asarray(x)


# ==============================================================================
# 3. SOLVE AND VERIFY
# ==============================================================================
t0 = time.perf_counter()

T_L = surface_temperature()
T_max = max_temperature()
q_surface = QGEN * L                      # [W/m^2] flux leaving the surface

dT_film = QGEN * L / H                    # [K] rise across the film
dT_cond = QGEN * L * L / (2.0 * K)        # [K] rise across the solid

R_cond_eff = L / (2.0 * K)                # [m^2 K/W] effective conduction R
R_conv = 1.0 / H                          # [m^2 K/W]

print("=" * 78)
print("EXAMPLE 2.1 -- PLANE WALL WITH UNIFORM INTERNAL GENERATION")
print("=" * 78)
print(f"  Half-thickness      L      = {L:.4f} m")
print(f"  Conductivity        k      = {K:.4f} W/(m K)")
print(f"  Generation          q'''   = {QGEN:.4e} W/m^3")
print(f"  Convection          h      = {H:.4f} W/(m^2 K)")
print(f"  Coolant             T_inf  = {T_INF:.4f} K")
print(f"  Biot number         Bi     = {BI:.6f}")
print("-" * 78)
print(f"  Surface flux        q''    = {q_surface:.6f} W/m^2")
print(f"  Surface temperature T_L    = {T_L:.10f} K")
print(f"  Mid-plane temperature T_max= {T_max:.10f} K")
print(f"  Total rise above coolant   = {T_max - T_INF:.6f} K")
print("-" * 78)
print("  Decomposition of the temperature rise:")
print(f"    across the surface film  = {dT_film:9.4f} K "
      f"({100*dT_film/(T_max-T_INF):5.2f} %)   R = {R_conv:.6e} m^2K/W")
print(f"    across the solid         = {dT_cond:9.4f} K "
      f"({100*dT_cond/(T_max-T_INF):5.2f} %)   R = {R_cond_eff:.6e} m^2K/W")
print(f"    ratio  dT_cond/dT_film   = {dT_cond/dT_film:.6f}  (= Bi/2)")

# --- Verification -------------------------------------------------------------
# (i) the field must satisfy k T'' + q''' = 0
# (ii) the mid-plane gradient must vanish
# (iii) the surface balance must close
# (iv) global conservation: generated = removed
# edge_order=2 matters here.  NumPy's default first-order edge stencil
# evaluates the derivative half a cell inside the domain, which for a parabola
# yields a spurious surface "residual" of exactly QGEN*dx/2 -- an artefact of
# the check itself, not an error in the solution.  A second-order stencil is
# exact for a quadratic and reduces the residual to round-off.
xs = np.linspace(0.0, L, 2001)
Ts = temperature(xs)
dTdx = np.gradient(Ts, xs, edge_order=2)
d2T = np.gradient(dTdx, xs, edge_order=2)[2:-2]
res_ode = np.max(np.abs(K * d2T + QGEN)) / QGEN
grad0 = abs(dTdx[0])
bal = abs(-K * dTdx[-1] - H * (Ts[-1] - T_INF))
generated = quad(lambda s: QGEN, 0.0, L)[0]      # [W/m^2] per unit area
removed = H * (T_L - T_INF)

print("-" * 78)
print(f"  Verification  max|k T'' + q'''| / q'''  = {res_ode:.3e}")
print(f"  Verification  |dT/dx| at x = 0          = {grad0:.3e} K/m")
print(f"  Verification  surface balance residual  = {bal:.3e} W/m^2")
print(f"  Verification  generated {generated:.6f} vs removed {removed:.6f} "
      f"W/m^2  (diff {abs(generated-removed):.3e})")

# --- Where does the maximum sit, and what if cooling is asymmetric? -----------
# For the symmetric plate the maximum is at x = 0 by construction.  The
# dimensionless form below shows that the SHAPE never changes: only the two
# scales (film rise and conduction rise) do.
print("-" * 78)
print("  Dimensionless form:  (T - T_L)/(T_max - T_L) = 1 - (x/L)^2")
xstar = np.linspace(0.0, 1.0, 6)
theta = 1.0 - xstar**2
theta_num = (temperature(xstar * L) - T_L) / (T_max - T_L)
print(f"  {'x/L':>6} {'theta exact':>13} {'theta computed':>16} {'diff':>11}")
for a, b, c in zip(xstar, theta, theta_num):
    print(f"  {a:>6.2f} {b:>13.6f} {c:>16.6f} {abs(b-c):>11.2e}")

# --- Parametric study ---------------------------------------------------------
q_scan = np.linspace(0.0, 2.0e7, 200)
TL_scan = surface_temperature(q_scan)
Tmax_scan = max_temperature(q_scan)

h_scan = np.logspace(1.7, 4.5, 200)
Tmax_h = np.array([max_temperature(h=hh) for hh in h_scan])

# design limit: suppose the material must stay below 800 K
T_LIMIT = 800.0
q_allow = (T_LIMIT - T_INF) / (L / H + L * L / (2.0 * K))
print("-" * 78)
print(f"  Design question: largest q''' keeping T_max below {T_LIMIT:.0f} K")
print(f"    q'''_max = {q_allow:.6e} W/m^3  "
      f"({q_allow/QGEN:.3f} times the design value)")
print(f"    check: T_max(q'''_max) = {max_temperature(q_allow):.6f} K")
print(f"  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
x = np.linspace(0.0, L, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax[0].plot(x * 1e3, temperature(x), "-", lw=2.2, color="#b2182b",
           label=r"$T(x)$, exact")
ax[0].axhline(T_INF, color="0.45", lw=1.0, ls=":")
ax[0].annotate(r"$T_\infty$", xy=(0.5, T_INF + 4), color="0.35", fontsize=10)
ax[0].plot([0], [T_max], "o", ms=6, color="#b2182b")
ax[0].plot([L * 1e3], [T_L], "s", ms=6, color="#2166ac")
ax[0].annotate(rf"$T_{{max}} = {T_max:.1f}$ K", xy=(0, T_max),
               xytext=(18, -6), textcoords="offset points", fontsize=9.5,
               color="#b2182b")
ax[0].annotate(rf"$T_L = {T_L:.1f}$ K", xy=(L * 1e3, T_L),
               xytext=(-92, 18), textcoords="offset points", fontsize=9.5,
               color="#2166ac",
               arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.0))
ax[0].annotate("", xy=(L * 1e3, T_L), xytext=(L * 1e3, T_INF),
               arrowprops=dict(arrowstyle="<->", color="#1b7837", lw=1.3))
ax[0].annotate(rf"film $\Delta T = {dT_film:.0f}$ K",
               xy=(L * 1e3 * 0.62, 0.5 * (T_L + T_INF)), fontsize=9,
               color="#1b7837")
ax[0].set_xlabel(r"Position $x$ [mm]   ($x=0$ is the mid-plane)")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Parabolic temperature field")
ax[0].legend(loc="lower left", fontsize=9)
ax[0].set_xlim(0, L * 1e3)

ax[1].plot(x * 1e3, flux(x) * 1e-3, "-", lw=2.2, color="#2166ac",
           label=r"$q''(x) = q''' x$")
ax[1].plot([L * 1e3], [q_surface * 1e-3], "s", ms=6, color="#2166ac")
ax[1].annotate(rf"$q''(L) = {q_surface*1e-3:.0f}$ kW m$^{{-2}}$",
               xy=(L * 1e3, q_surface * 1e-3), xytext=(-130, -14),
               textcoords="offset points", fontsize=9.5, color="#2166ac")
ax[1].set_xlabel(r"Position $x$ [mm]")
ax[1].set_ylabel(r"Conduction flux $q''$ [kW m$^{-2}$]")
ax[1].set_title("(b) Flux grows linearly from the mid-plane")
ax[1].legend(loc="upper left", fontsize=9)
ax[1].set_xlim(0, L * 1e3)

fig.suptitle("Example 2.1 -- Analytical solution, uniformly heated plate",
             fontsize=12.5, y=1.02)
fig.savefig("fig_2_1a_profile.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax[0].plot(q_scan * 1e-6, Tmax_scan, "-", lw=2.0, color="#b2182b",
           label=r"$T_{max}$")
ax[0].plot(q_scan * 1e-6, TL_scan, "--", lw=1.8, color="#2166ac",
           label=r"$T_L$")
ax[0].axhline(T_LIMIT, color="0.3", ls="-.", lw=1.3)
ax[0].annotate(rf"material limit {T_LIMIT:.0f} K", xy=(0.4, T_LIMIT + 12),
               fontsize=9, color="0.3")
ax[0].axvline(q_allow * 1e-6, color="#1b7837", ls=":", lw=1.6)
ax[0].annotate(rf"$q'''_{{max}} = {q_allow*1e-6:.2f}$ MW m$^{{-3}}$",
               xy=(q_allow * 1e-6 + 0.4, 420), fontsize=9, color="#1b7837")
ax[0].plot([QGEN * 1e-6], [T_max], "o", ms=6, color="#b2182b")
ax[0].set_xlabel(r"Generation $q'''$ [MW m$^{-3}$]")
ax[0].set_ylabel("Temperature [K]")
ax[0].set_title("(a) Design envelope")
ax[0].legend(loc="upper left", fontsize=9)

ax[1].semilogx(h_scan, Tmax_h, "-", lw=2.0, color="#762a83")
ax[1].axhline(T_INF + dT_cond, color="#b2182b", ls="--", lw=1.4)
ax[1].annotate(r"conduction floor $T_\infty + q'''L^2/2k$",
               xy=(60, T_INF + dT_cond + 22), fontsize=9, color="#b2182b")
ax[1].axvline(H, color="0.45", ls=":", lw=1.4)
ax[1].annotate(rf"design $h = {H:.0f}$", xy=(H * 1.15, 900), fontsize=9,
               color="0.35", rotation=90)
ax[1].set_xlabel(r"Convection coefficient $h$ [W m$^{-2}$ K$^{-1}$]")
ax[1].set_ylabel(r"Mid-plane temperature $T_{max}$ [K]")
ax[1].set_title("(b) Diminishing returns from better cooling")

fig.suptitle("Example 2.1 -- Design sensitivity", fontsize=12.5, y=1.02)
fig.savefig("fig_2_1b_study.png")
plt.close(fig)

print("Figures written: fig_2_1a_profile.png, fig_2_1b_study.png")
