"""
================================================================================
 EXAMPLE 3.1 -- ANALYTICAL SOLUTION
 Radial conduction in hollow and composite cylinders; critical insulation radius
================================================================================

 PHYSICAL PROBLEM
 ----------------
 An insulated steam pipe carries process fluid. The inner surface of the
 insulation is held at T1; the outer surface loses heat by convection to
 ambient air. All quantities are per unit length of pipe, so the heat rate is
 q' [W/m] rather than a flux.

 The same section also treats the CRITICAL INSULATION RADIUS -- the result,
 counter-intuitive on first meeting, that adding insulation to a thin cable can
 INCREASE its heat loss.

 GOVERNING EQUATION  (steady, radial, no generation)
 ---------------------------------------------------
        1  d  /       dT \
       --- -- | r k * -- |  = 0        for   r1 < r < r2
        r  dr \       dr /

 BOUNDARY CONDITIONS
 -------------------
   r = r1 :  T = T1                                     (Dirichlet)
   r = r2 : -k dT/dr|_r2 = h (T_2 - T_inf)              (convective)

 ANALYTICAL SOLUTION
 -------------------
 With k constant the equation integrates once to  r dT/dr = const, so

        T(r) = T1 - (q' / (2 pi k)) ln(r / r1)

 The profile is LOGARITHMIC, not linear: the conducting area 2 pi r grows with
 radius, so the same heat rate needs an ever smaller gradient as r increases.

 The conduction resistance per unit length follows immediately:

        R'_cond = ln(r2/r1) / (2 pi k)        [m K / W]

 and the convective resistance at the outer surface is

        R'_conv = 1 / (2 pi r2 h)

 so that, for prescribed T1 and T_inf,

        q' = (T1 - T_inf) / (R'_cond + R'_conv)

 CRITICAL RADIUS
 ---------------
 Adding insulation raises R'_cond (good) but also enlarges the outer surface,
 which LOWERS R'_conv (bad).  Differentiating the total resistance with respect
 to the outer radius and setting it to zero,

        d/dr2 [ ln(r2/r1)/(2 pi k) + 1/(2 pi r2 h) ] = 1/(2 pi k r2)
                                                      - 1/(2 pi h r2^2) = 0

 gives the critical radius

        r_cr = k / h

 For r2 < r_cr the total resistance DECREASES with added insulation, so heat
 loss rises; only beyond r_cr does insulation begin to insulate.  The second
 derivative is positive there, confirming a minimum of resistance (a maximum of
 heat loss).

 SYMBOLS (all SI)
 ----------------
   r1, r2   [m]         inner and outer radii
   k        [W/(m K)]   thermal conductivity
   h        [W/(m^2 K)] convection coefficient
   T1       [K]         inner surface temperature
   T_inf    [K]         ambient temperature
   q'       [W/m]       heat rate per unit length
   R'       [m K/W]     thermal resistance per unit length
   r_cr     [m]         critical insulation radius, k/h

 OUTPUTS
 -------
   fig_3_1a_profile.png   logarithmic profile and resistance split
   fig_3_1b_critical.png  critical radius behaviour

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
# 1. PROBLEM DATA -- insulated steam pipe
# ==============================================================================
R1 = 0.050          # [m]          inner radius of the insulation
R2 = 0.100          # [m]          outer radius of the insulation
K_INS = 0.050       # [W/(m K)]    calcium silicate insulation
H = 10.0            # [W/(m^2 K)]  natural convection to still air
T1 = 450.0          # [K]          inner surface temperature
T_INF = 300.0       # [K]          ambient


# ==============================================================================
# 2. ANALYTICAL MACHINERY
# ==============================================================================
def R_cond(r_in, r_out, k):
    """Conduction resistance per unit length of a cylindrical shell [m K/W]."""
    return np.log(r_out / r_in) / (2.0 * np.pi * k)


def R_conv(r, h):
    """Convective resistance per unit length at radius r [m K/W]."""
    return 1.0 / (2.0 * np.pi * r * h)


def heat_rate(r_out=R2, k=K_INS, h=H, r_in=R1, T_hot=T1, T_amb=T_INF):
    """Heat rate per unit length q' [W/m] through shell plus outer film."""
    return (T_hot - T_amb) / (R_cond(r_in, r_out, k) + R_conv(r_out, h))


def temperature(r, q, k=K_INS, r_in=R1, T_hot=T1):
    """Exact logarithmic profile T(r) [K]."""
    return T_hot - q / (2.0 * np.pi * k) * np.log(np.asarray(r) / r_in)


# ==============================================================================
# 3. SOLVE AND VERIFY -- the pipe
# ==============================================================================
t0 = time.perf_counter()

Rc = R_cond(R1, R2, K_INS)
Rv = R_conv(R2, H)
q_prime = heat_rate()
T2 = temperature(R2, q_prime)

print("=" * 78)
print("EXAMPLE 3.1 -- RADIAL CONDUCTION IN CYLINDERS")
print("=" * 78)
print(f"  r1 = {R1:.4f} m, r2 = {R2:.4f} m, k = {K_INS:.4f} W/(m K), "
      f"h = {H:.2f} W/(m^2 K)")
print(f"  T1 = {T1:.2f} K, T_inf = {T_INF:.2f} K")
print("-" * 78)
print(f"  R'_cond = {Rc:.10f} m K/W   ({100*Rc/(Rc+Rv):5.2f} %)")
print(f"  R'_conv = {Rv:.10f} m K/W   ({100*Rv/(Rc+Rv):5.2f} %)")
print(f"  R'_tot  = {Rc+Rv:.10f} m K/W")
print(f"  q'      = {q_prime:.10f} W/m")
print(f"  T2      = {T2:.10f} K")

# --- verification -------------------------------------------------------------
# (i) the field must satisfy  d/dr ( r dT/dr ) = 0
# (ii) the outer surface balance must close
# (iii) the heat rate must be the SAME through every radius (conservation)
rr = np.linspace(R1, R2, 2001)
Tr = temperature(rr, q_prime)
dTdr = np.gradient(Tr, rr, edge_order=2)
resid = np.gradient(rr * dTdr, rr, edge_order=2)[2:-2]
bal = abs(-K_INS * dTdr[-1] * 2 * np.pi * R2 - 2 * np.pi * R2 * H * (Tr[-1] - T_INF))
q_at_r = -2.0 * np.pi * rr * K_INS * dTdr        # [W/m] at each radius
print("-" * 78)
print(f"  Verification  max|d(r dT/dr)/dr|      = {np.max(np.abs(resid)):.3e}")
print(f"  Verification  surface balance         = {bal:.3e} W/m")
print(f"  Verification  spread of q' over r     = "
      f"{np.max(q_at_r[2:-2]) - np.min(q_at_r[2:-2]):.3e} W/m")
print(f"  Verification  |q'(r) - q'| mean       = "
      f"{np.mean(np.abs(q_at_r[2:-2] - q_prime)):.3e} W/m")

# --- how wrong is a plane-wall approximation? ---------------------------------
# A common shortcut treats the shell as a plane wall of thickness (r2-r1) and
# area 2*pi*r_mean.  Compare against the exact logarithmic resistance.
r_arith = 0.5 * (R1 + R2)
r_log = (R2 - R1) / np.log(R2 / R1)              # log-mean radius
R_plane_arith = (R2 - R1) / (2.0 * np.pi * r_arith * K_INS)
R_plane_log = (R2 - R1) / (2.0 * np.pi * r_log * K_INS)
print("-" * 78)
print("  Plane-wall approximations of the shell resistance:")
print(f"    exact (logarithmic)        R' = {Rc:.10f} m K/W")
print(f"    plane wall, arithmetic r   R' = {R_plane_arith:.10f} "
      f"({100*(R_plane_arith/Rc-1):+.4f} %)")
print(f"    plane wall, LOG-MEAN r     R' = {R_plane_log:.10f} "
      f"({100*(R_plane_log/Rc-1):+.4e} %)")
print("    The log-mean radius makes the plane-wall form EXACT: this is the")
print("    identity that Example 3.2 exploits to build an exact FVM scheme.")

# --- composite cylinder -------------------------------------------------------
# steel pipe wall + insulation, with an inner convective film
R_I, R_M, R_O = 0.040, 0.050, 0.100      # [m]
K_STEEL, K_INSUL = 15.0, 0.050           # [W/(m K)]
H_IN, H_OUT = 500.0, 10.0                # [W/(m^2 K)]
T_STEAM, T_AMB = 480.0, 300.0

layers = [
    ("inner film", R_conv(R_I, H_IN)),
    ("steel pipe", R_cond(R_I, R_M, K_STEEL)),
    ("insulation", R_cond(R_M, R_O, K_INSUL)),
    ("outer film", R_conv(R_O, H_OUT)),
]
R_total = sum(R for _, R in layers)
q_comp = (T_STEAM - T_AMB) / R_total

print("-" * 78)
print("  Composite cylinder: steam -> steel pipe -> insulation -> air")
print(f"  {'layer':>12} {'R prime [m K/W]':>18} {'share':>9} {'dT [K]':>10}")
T_node = T_STEAM
for name, R in layers:
    dT = q_comp * R
    print(f"  {name:>12} {R:>18.8f} {100*R/R_total:>8.2f}% {dT:>10.4f}")
    T_node -= dT
print(f"  {'TOTAL':>12} {R_total:>18.8f} {100.0:>8.2f}% "
      f"{T_STEAM-T_AMB:>10.4f}")
print(f"  q' = {q_comp:.10f} W/m")
print(f"  closure check: T after all drops = {T_node:.10f} K "
      f"(should be {T_AMB:.4f}, diff {abs(T_node-T_AMB):.3e})")
print("  The insulation carries the overwhelming majority of the resistance;")
print("  the steel pipe wall is thermally almost invisible.")

# ==============================================================================
# 4. CRITICAL INSULATION RADIUS
# ==============================================================================
# A thin electrical cable: bare radius 1 mm, plastic insulation, still air.
R_WIRE = 0.0010          # [m]
K_PLASTIC = 0.15         # [W/(m K)]
H_AIR = 20.0             # [W/(m^2 K)]
T_WIRE, T_A = 350.0, 300.0
r_cr = K_PLASTIC / H_AIR

print("\n" + "=" * 78)
print("CRITICAL INSULATION RADIUS")
print("-" * 78)
print(f"  Bare wire radius r1 = {R_WIRE*1e3:.3f} mm, k = {K_PLASTIC:.3f} "
      f"W/(m K), h = {H_AIR:.1f} W/(m^2 K)")
print(f"  Critical radius r_cr = k/h = {r_cr*1e3:.4f} mm "
      f"({r_cr/R_WIRE:.1f} times the bare radius)")

q_bare = 2.0 * np.pi * R_WIRE * H_AIR * (T_WIRE - T_A)
print(f"  Bare wire heat loss              q' = {q_bare:.6f} W/m")
for r_out in [0.0015, 0.0030, r_cr, 0.0150, 0.0300]:
    q_i = heat_rate(r_out=r_out, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE,
                    T_hot=T_WIRE, T_amb=T_A)
    tag = "  <-- critical radius (maximum loss)" if abs(r_out - r_cr) < 1e-12 else ""
    print(f"    insulated to r2 = {r_out*1e3:7.3f} mm : q' = {q_i:9.6f} W/m "
          f"({100*(q_i/q_bare-1):+7.2f} % vs bare){tag}")

# Verify that r_cr is a genuine maximum of q' (minimum of resistance)
def R_tot_of(r_out):
    return R_cond(R_WIRE, r_out, K_PLASTIC) + R_conv(r_out, H_AIR)

# The first derivative is available in closed form, so use it: a central
# finite difference here carries a truncation error eps^2 R'''/6 which, with
# R''' ~ -1e7, swamps the very quantity being tested.  This is the same trap
# as the edge-stencil artefact of Example 2.1 -- the CHECK can be less
# accurate than the thing checked.
def dR_dr(r):
    return 1.0 / (2 * np.pi * K_PLASTIC * r) - 1.0 / (2 * np.pi * H_AIR * r * r)

eps = 1e-6
d2 = (R_tot_of(r_cr + eps) - 2 * R_tot_of(r_cr) + R_tot_of(r_cr - eps)) / eps**2
d1_fd = (R_tot_of(r_cr + eps) - R_tot_of(r_cr - eps)) / (2 * eps)
print(f"\n  Verification at r = r_cr:")
print(f"    analytic   dR'/dr2   = {dR_dr(r_cr):.3e}  (exactly 0 by construction)")
print(f"    central FD dR'/dr2   = {d1_fd:.3e}  <- its own truncation error,")
print(f"                                             eps^2 R'''/6 with R''' ~ -1e7")
print(f"    d2R'/dr2^2           = {d2:.4f} > 0  -> minimum of R', so q' is a MAXIMUM")

# The radius at which insulation finally pays: q'(r_pay) = q'_bare, r_pay > r_cr
r_pay = brentq(lambda r: heat_rate(r_out=r, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE,
                                   T_hot=T_WIRE, T_amb=T_A) - q_bare,
               r_cr, 100.0, xtol=1e-14)
print(f"\n  Insulation only becomes beneficial beyond r2 = {r_pay*1e3:.1f} mm = {r_pay:.4f} m")
print("    -- an absurd amount for a 1 mm wire, which is the practical point.")
print(f"    check: q'(r_pay) = "
      f"{heat_rate(r_out=r_pay, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE, T_hot=T_WIRE, T_amb=T_A):.8f}"
      f" vs bare {q_bare:.8f} W/m")
print(f"\n  For the STEAM PIPE, r_cr = k/h = {K_INS/H*1e3:.3f} mm, far below")
print(f"  its actual radius of {R2*1e3:.1f} mm, so insulation always helps there.")
print(f"\n  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
r = np.linspace(R1, R2, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)

ax[0].plot(r * 1e3, temperature(r, q_prime), "-", lw=2.2, color="#b2182b",
           label="Exact logarithmic profile")
lin = T1 + (T2 - T1) * (r - R1) / (R2 - R1)
ax[0].plot(r * 1e3, lin, "--", lw=1.6, color="#2166ac",
           label="Straight line (for comparison)")
ax[0].axhline(T_INF, color="0.45", ls=":", lw=1.0)
ax[0].annotate(r"$T_\infty$", xy=(R1 * 1e3 + 1, T_INF + 3), color="0.35",
               fontsize=10)
ax[0].plot([R2 * 1e3], [T2], "o", ms=6, color="#b2182b")
ax[0].annotate(rf"$T_2 = {T2:.1f}$ K", xy=(R2 * 1e3, T2), xytext=(-96, 24),
               textcoords="offset points", fontsize=9.5, color="#b2182b",
               arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.0))
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) The profile curves: area grows with $r$")
ax[0].legend(fontsize=9, loc="upper right")
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

names = [n for n, _ in layers]
vals = [100 * R / R_total for _, R in layers]
colors = ["#92c5de", "#4d4d4d", "#b2182b", "#2166ac"]
bars = ax[1].barh(names, vals, color=colors, edgecolor="k", lw=0.6)
for b, v in zip(bars, vals):
    ax[1].text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.2f} %",
               va="center", fontsize=9.5)
ax[1].set_xlabel("Share of total thermal resistance [%]")
ax[1].set_title("(b) Composite pipe: the insulation controls")
ax[1].set_xlim(0, 108)

fig.suptitle("Example 3.1 -- Hollow and composite cylinders", fontsize=12.5,
             y=1.08)
fig.savefig("fig_3_1a_profile.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)

r_out = np.linspace(R_WIRE, 0.040, 500)
q_curve = np.array([heat_rate(r_out=rr_, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE,
                              T_hot=T_WIRE, T_amb=T_A) for rr_ in r_out])
ax[0].plot(r_out * 1e3, q_curve, "-", lw=2.2, color="#b2182b",
           label=r"$q'(r_2)$ with insulation")
ax[0].axhline(q_bare, color="#2166ac", ls="--", lw=1.6,
              label="bare wire")
ax[0].axvline(r_cr * 1e3, color="#1b7837", ls=":", lw=1.8)
ax[0].plot([r_cr * 1e3], [q_curve[np.argmin(np.abs(r_out - r_cr))]], "o",
           ms=7, color="#1b7837", zorder=5)
ax[0].annotate(rf"$r_{{cr}} = k/h = {r_cr*1e3:.1f}$ mm",
               xy=(r_cr * 1e3, q_curve.max()), xytext=(14, -6),
               textcoords="offset points", fontsize=9.5, color="#1b7837")
ax[0].annotate(rf"break-even at {r_pay:.2f} m (off scale)",
               xy=(12, q_bare * 1.06), fontsize=9, color="0.35")
ax[0].set_xlabel(r"Outer insulation radius $r_2$ [mm]")
ax[0].set_ylabel(r"Heat loss $q'$ [W m$^{-1}$]")
ax[0].set_title("(a) Thin wire: insulation first makes things worse")
ax[0].legend(fontsize=9, loc="lower right")

Rc_c = np.array([R_cond(R_WIRE, rr_, K_PLASTIC) for rr_ in r_out])
Rv_c = np.array([R_conv(rr_, H_AIR) for rr_ in r_out])
ax[1].plot(r_out * 1e3, Rc_c, "-", lw=1.9, color="#b2182b",
           label=r"$R'_{cond}$ (rises)")
ax[1].plot(r_out * 1e3, Rv_c, "-", lw=1.9, color="#2166ac",
           label=r"$R'_{conv}$ (falls)")
ax[1].plot(r_out * 1e3, Rc_c + Rv_c, "-", lw=2.4, color="#4d4d4d",
           label=r"$R'_{tot}$")
ax[1].axvline(r_cr * 1e3, color="#1b7837", ls=":", lw=1.8)
ax[1].set_xlabel(r"Outer insulation radius $r_2$ [mm]")
ax[1].set_ylabel(r"Resistance per unit length [m K W$^{-1}$]")
ax[1].set_title("(b) The competition that creates $r_{cr}$")
ax[1].legend(fontsize=9)
ax[1].set_ylim(0, (Rc_c + Rv_c).max() * 1.1)

fig.suptitle("Example 3.1 -- Critical insulation radius", fontsize=12.5, y=1.02)
fig.savefig("fig_3_1b_critical.png")
plt.close(fig)

print("Figures written: fig_3_1a_profile.png, fig_3_1b_critical.png")
