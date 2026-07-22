"""
================================================================================
 EXAMPLE 4.1 -- ANALYTICAL SOLUTION
 Radial conduction in hollow, composite and solid spheres
================================================================================

 PHYSICAL PROBLEM
 ----------------
 Three configurations are treated, all with the same machinery:

   (a) a hollow sphere (a spherical storage vessel with insulation),
       deliberately using the SAME radii, conductivity and boundary data as the
       cylinder of Example 3.1 so the two geometries can be compared directly;
   (b) a composite sphere;
   (c) a SOLID sphere with uniform internal generation, using the same k, q''',
       h and T_inf as the plate of Example 2.1, again for direct comparison.

 GOVERNING EQUATION  (steady, radial, spherical)
 -----------------------------------------------
        1   d  /  2      dT \
       --- -- | r  k  *  -- |  + q''' = 0
        r^2 dr \         dr /

 The conducting area is now 4 pi r^2 -- it grows with the SQUARE of the radius,
 not linearly as in a cylinder.  Everything in this chapter follows from that.

 ANALYTICAL SOLUTION, HOLLOW SPHERE (no generation, k constant)
 --------------------------------------------------------------
 The equation integrates once to  r^2 dT/dr = const, so with q = -4 pi r^2 k dT/dr
 the heat rate is the same through every radius and

        T(r) = T1 - (q / (4 pi k)) (1/r1 - 1/r)

 The profile varies as 1/r.  Evaluating at r2 gives the shell resistance

        R_cond = (1/r1 - 1/r2) / (4 pi k)        [K/W]

 which, unlike the cylinder, remains FINITE as r2 -> infinity:

        R_cond(r2 -> inf) = 1 / (4 pi k r1)

 An infinite thickness of insulation around a sphere has finite resistance.  No
 such limit exists for a cylinder, whose logarithm diverges.

 CRITICAL RADIUS
 ---------------
 With convection at the outer surface,

        R_tot = (1/r1 - 1/r2)/(4 pi k) + 1/(4 pi r2^2 h)

        dR_tot/dr2 = 1/(4 pi k r2^2) - 2/(4 pi h r2^3) = 0   =>   r_cr = 2k/h

 TWICE the cylindrical value 2.k/h vs k/h.  The factor of two comes from the
 area exponent: the convective resistance falls as r^-2 for a sphere but only
 as r^-1 for a cylinder, so the competition tips later.

 SOLID SPHERE WITH UNIFORM GENERATION
 ------------------------------------
 Symmetry gives dT/dr = 0 at r = 0, and an energy balance on a sphere of radius
 r requires  4 pi r^2 q''(r) = (4/3) pi r^3 q''',  so

        q''(r) = q''' r / 3

 Integrating -k dT/dr = q''' r/3,

        T(r) = T_max - q''' r^2 / (6 k)

 and the surface balance q''' r0 / 3 = h (T_s - T_inf) gives

        T_s   = T_inf + q''' r0 / (3 h)
        T_max = T_inf + q''' r0/(3h) + q''' r0^2/(6k)

 Compare the plate of Example 2.1, T_max = T_inf + q'''L/h + q'''L^2/(2k): the
 sphere carries the same generation with a factor 3 less film rise and a factor
 3 less conduction rise, because a sphere has three times the surface area per
 unit volume of a slab of the same half-thickness.

 SYMBOLS (all SI)
 ----------------
   r1, r2  [m]         inner and outer radii of a shell
   r0      [m]         radius of a solid sphere
   k       [W/(m K)]   thermal conductivity
   h       [W/(m^2 K)] convection coefficient
   q       [W]         heat rate (total, not per unit length)
   q'''    [W/m^3]     volumetric generation
   R       [K/W]       thermal resistance
   r_cr    [m]         critical insulation radius, 2k/h

 OUTPUTS
 -------
   fig_4_1a_profile.png   1/r profile, sphere vs cylinder vs plane
   fig_4_1b_critical.png  critical radius and the generating sphere

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
# 1. DATA -- (a) insulated spherical vessel, identical to the Example 3.1 pipe
# ==============================================================================
R1, R2 = 0.050, 0.100
K_INS = 0.050
H = 10.0
T1 = 450.0
T_INF = 300.0


# ==============================================================================
# 2. ANALYTICAL MACHINERY
# ==============================================================================
def R_cond_sph(r_in, r_out, k):
    """Conduction resistance of a spherical shell [K/W]."""
    return (1.0 / r_in - 1.0 / r_out) / (4.0 * np.pi * k)


def R_conv_sph(r, h):
    """Convective resistance at a spherical surface [K/W]."""
    return 1.0 / (4.0 * np.pi * r * r * h)


def heat_rate(r_out=R2, k=K_INS, h=H, r_in=R1, T_hot=T1, T_amb=T_INF):
    """Total heat rate q [W]."""
    return (T_hot - T_amb) / (R_cond_sph(r_in, r_out, k) + R_conv_sph(r_out, h))


def temperature(r, q, k=K_INS, r_in=R1, T_hot=T1):
    """Exact 1/r profile [K]."""
    r = np.asarray(r, dtype=float)
    return T_hot - q / (4.0 * np.pi * k) * (1.0 / r_in - 1.0 / r)


t0 = time.perf_counter()
Rc = R_cond_sph(R1, R2, K_INS)
Rv = R_conv_sph(R2, H)
q_tot = heat_rate()
T2 = float(temperature(np.array([R2]), q_tot)[0])

print("=" * 78)
print("EXAMPLE 4.1 -- RADIAL CONDUCTION IN SPHERES")
print("=" * 78)
print(f"  (a) HOLLOW SPHERE  r1 = {R1} m, r2 = {R2} m, k = {K_INS} W/(m K), "
      f"h = {H} W/(m^2 K)")
print(f"      R_cond = {Rc:.10f} K/W  ({100*Rc/(Rc+Rv):5.2f} %)")
print(f"      R_conv = {Rv:.10f} K/W  ({100*Rv/(Rc+Rv):5.2f} %)")
print(f"      q      = {q_tot:.10f} W")
print(f"      T2     = {T2:.10f} K")

# --- verification -------------------------------------------------------------
rr = np.linspace(R1, R2, 4001)
Tr = temperature(rr, q_tot)
dTdr = np.gradient(Tr, rr, edge_order=2)
resid = np.gradient(rr**2 * dTdr, rr, edge_order=2)[3:-3]
q_at_r = -4.0 * np.pi * rr**2 * K_INS * dTdr
bal = abs(q_at_r[-1] - 4 * np.pi * R2**2 * H * (Tr[-1] - T_INF))
print(f"      Verify  max|d(r^2 dT/dr)/dr|  = {np.max(np.abs(resid)):.3e}")
print(f"      Verify  surface balance       = {bal:.3e} W")
print(f"      Verify  spread of q over r    = "
      f"{np.max(q_at_r[3:-3])-np.min(q_at_r[3:-3]):.3e} W")

# --- the finite-resistance limit ---------------------------------------------
print("\n      Resistance of an INFINITELY thick shell:")
print(f"        R_cond(r2 -> inf) = 1/(4 pi k r1) = "
      f"{1.0/(4*np.pi*K_INS*R1):.10f} K/W")
for r_out in [0.2, 0.5, 2.0, 100.0, 1e6]:
    frac = R_cond_sph(R1, r_out, K_INS) / (1.0 / (4 * np.pi * K_INS * R1))
    print(f"        r2 = {r_out:>9.1f} m : {100*frac:7.3f} % of the limit")
print("      A cylinder has no such limit: ln(r2/r1) grows without bound.")

# --- geometric comparison at identical radii ---------------------------------
R_cyl = np.log(R2 / R1) / (2.0 * np.pi * K_INS)      # [m K/W], per unit length
print("\n      Shell resistance for the SAME radii and k:")
print(f"        sphere    R = {Rc:.6f} K/W")
print(f"        cylinder  R'= {R_cyl:.6f} m K/W   (per unit length)")
print("      The two are not directly comparable in units; what IS comparable")
print("      is the shape of the profile, plotted in the figure.")

# ==============================================================================
# 3. COMPOSITE SPHERE
# ==============================================================================
RA, RB, RC_ = 0.040, 0.050, 0.100
K_STEEL, K_INSUL = 15.0, 0.050
H_IN, H_OUT = 500.0, 10.0
T_HOT, T_AMB = 480.0, 300.0

layers = [
    ("inner film", R_conv_sph(RA, H_IN)),
    ("steel shell", R_cond_sph(RA, RB, K_STEEL)),
    ("insulation", R_cond_sph(RB, RC_, K_INSUL)),
    ("outer film", R_conv_sph(RC_, H_OUT)),
]
R_tot = sum(R for _, R in layers)
q_comp = (T_HOT - T_AMB) / R_tot

print("\n  (b) COMPOSITE SPHERE: fluid -> steel -> insulation -> air")
print(f"      {'layer':>12} {'R [K/W]':>14} {'share':>9} {'dT [K]':>10}")
T_node = T_HOT
for name, R in layers:
    dT = q_comp * R
    print(f"      {name:>12} {R:>14.8f} {100*R/R_tot:>8.2f}% {dT:>10.4f}")
    T_node -= dT
print(f"      {'TOTAL':>12} {R_tot:>14.8f} {100.0:>8.2f}% {T_HOT-T_AMB:>10.4f}")
print(f"      q = {q_comp:.10f} W,  closure |T_end - T_amb| = "
      f"{abs(T_node-T_AMB):.3e} K")

# ==============================================================================
# 4. CRITICAL RADIUS -- and why it is twice the cylindrical value
# ==============================================================================
R_WIRE, K_PLASTIC, H_AIR = 0.0010, 0.15, 20.0
T_W, T_A = 350.0, 300.0
r_cr_sph = 2.0 * K_PLASTIC / H_AIR
r_cr_cyl = K_PLASTIC / H_AIR

print("\n  (c) CRITICAL RADIUS")
print(f"      sphere    r_cr = 2k/h = {r_cr_sph*1e3:.3f} mm")
print(f"      cylinder  r_cr =  k/h = {r_cr_cyl*1e3:.3f} mm   (Example 3.1)")
print("      The sphere's convective resistance falls as r^-2 rather than")
print("      r^-1, so the competition tips at twice the radius.")

q_bare = 4 * np.pi * R_WIRE**2 * H_AIR * (T_W - T_A)
print(f"\n      Bare sphere (r = {R_WIRE*1e3:.1f} mm): q = {q_bare*1e3:.6f} mW")
for r_out in [0.002, 0.005, r_cr_sph, 0.030, 0.060]:
    qi = heat_rate(r_out=r_out, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE,
                   T_hot=T_W, T_amb=T_A)
    tag = "   <-- r_cr, maximum loss" if abs(r_out - r_cr_sph) < 1e-12 else ""
    print(f"        coated to {r_out*1e3:7.3f} mm : q = {qi*1e3:9.6f} mW "
          f"({100*(qi/q_bare-1):+8.2f} %){tag}")

def dR_dr(r):
    return 1.0 / (4 * np.pi * K_PLASTIC * r * r) - 2.0 / (
        4 * np.pi * H_AIR * r**3)

print(f"\n      Verify  analytic dR/dr2 at r_cr = {dR_dr(r_cr_sph):.3e} "
      "(exactly 0)")
eps = 1e-6
def R_of(r):
    return R_cond_sph(R_WIRE, r, K_PLASTIC) + R_conv_sph(r, H_AIR)
d2 = (R_of(r_cr_sph + eps) - 2 * R_of(r_cr_sph) + R_of(r_cr_sph - eps)) / eps**2
print(f"      Verify  d2R/dr2^2 at r_cr      = {d2:.4f} > 0 -> minimum of R")

# Does a break-even radius exist at all?  For a CYLINDER it always does,
# because ln(r2/r1) grows without bound.  For a SPHERE the conduction
# resistance is bounded by 1/(4 pi k r1), so if that ceiling is lower than the
# bare-sphere resistance, NO thickness of insulation can ever help.
R_bare = R_conv_sph(R_WIRE, H_AIR)
R_ceiling = 1.0 / (4 * np.pi * K_PLASTIC * R_WIRE)
q_limit = (T_W - T_A) / R_ceiling
print(f"\n      Bare-sphere resistance            R = {R_bare:.4f} K/W")
print(f"      Ceiling of coated resistance      R = {R_ceiling:.4f} K/W "
      f"(r2 -> inf)")
if R_ceiling > R_bare:
    r_pay = brentq(lambda r: heat_rate(r_out=r, k=K_PLASTIC, h=H_AIR,
                                       r_in=R_WIRE, T_hot=T_W, T_amb=T_A)
                   - q_bare, r_cr_sph, 100.0, xtol=1e-14)
    print(f"      Coating becomes beneficial beyond r2 = {r_pay*1e3:.2f} mm")
else:
    print("      The ceiling is BELOW the bare value, so no thickness of")
    print("      coating can ever reduce the heat loss.  Even an infinitely")
    print(f"      thick coating gives q = {q_limit*1e3:.4f} mW against "
          f"{q_bare*1e3:.4f} mW bare")
    print(f"      -- a permanent penalty of "
          f"{100*(q_limit/q_bare-1):+.1f} %.")
    print("      A break-even exists only when r1 > k/h; here r1 = "
          f"{R_WIRE*1e3:.1f} mm < k/h = {K_PLASTIC/H_AIR*1e3:.1f} mm.")
    r_pay = None

# ==============================================================================
# 5. SOLID SPHERE WITH UNIFORM GENERATION
# ==============================================================================
R0 = 0.020
K_S = 15.0
QGEN = 5.0e6
H_S = 500.0
TINF_S = 350.0

Ts_sph = TINF_S + QGEN * R0 / (3.0 * H_S)
Tmax_sph = Ts_sph + QGEN * R0 * R0 / (6.0 * K_S)
# the Example 2.1 plate, same q''', k, h, T_inf and half-thickness
Ts_pl = TINF_S + QGEN * R0 / H_S
Tmax_pl = Ts_pl + QGEN * R0 * R0 / (2.0 * K_S)
# and the equivalent long cylinder, for completeness
Ts_cyl = TINF_S + QGEN * R0 / (2.0 * H_S)
Tmax_cyl = Ts_cyl + QGEN * R0 * R0 / (4.0 * K_S)

print("\n  (d) SOLID SPHERE WITH UNIFORM GENERATION")
print(f"      r0 = {R0} m, k = {K_S} W/(m K), q''' = {QGEN:.2e} W/m^3, "
      f"h = {H_S} W/(m^2 K)")
print(f"      Surface flux q''(r0) = q''' r0/3 = {QGEN*R0/3:.4f} W/m^2")
print(f"      T_s   = {Ts_sph:.6f} K")
print(f"      T_max = {Tmax_sph:.6f} K   (rise above coolant "
      f"{Tmax_sph-TINF_S:.4f} K)")
print("\n      The same generation in three geometries with the same "
      "characteristic length:")
print(f"      {'geometry':>10} {'film rise':>11} {'cond. rise':>12} "
      f"{'T_max [K]':>11}")
print(f"      {'plate':>10} {Ts_pl-TINF_S:>11.4f} {Tmax_pl-Ts_pl:>12.4f} "
      f"{Tmax_pl:>11.4f}")
print(f"      {'cylinder':>10} {Ts_cyl-TINF_S:>11.4f} {Tmax_cyl-Ts_cyl:>12.4f} "
      f"{Tmax_cyl:>11.4f}")
print(f"      {'sphere':>10} {Ts_sph-TINF_S:>11.4f} {Tmax_sph-Ts_sph:>12.4f} "
      f"{Tmax_sph:>11.4f}")
print("      Both rises scale as 1/(m+1) with m = 0, 1, 2 the geometry index:")
print("      the sphere sheds heat through three times the area per unit")
print("      volume of the plate, so it runs far cooler on identical duty.")

# verification of the generating sphere
# The finite-difference Laplacian is unreliable as r -> 0, where the factor
# 1/r^2 amplifies round-off without bound; the exact field is perfectly
# regular there.  Evaluate the check away from the origin.
rs = np.linspace(0.0, R0, 4001)
Tg = Tmax_sph - QGEN * rs**2 / (6.0 * K_S)
dTg = np.gradient(Tg, rs, edge_order=2)
lap_all = np.gradient(rs**2 * dTg, rs, edge_order=2)
inner = rs > 0.05 * R0
lap = lap_all[inner] / rs[inner] ** 2
print(f"\n      Verify  max|k*lap + q'''|/q''' (r > 0.05 r0) = "
      f"{np.max(np.abs(K_S*lap + QGEN))/QGEN:.3e}")
print(f"      Verify  generated {QGEN*4/3*np.pi*R0**3:.6f} W vs removed "
      f"{4*np.pi*R0**2*H_S*(Ts_sph-TINF_S):.6f} W")
print(f"\n  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
r = np.linspace(R1, R2, 400)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

# normalised profiles for the three geometries, same radii and same end temps
th_sph = (1 / R1 - 1 / r) / (1 / R1 - 1 / R2)
th_cyl = np.log(r / R1) / np.log(R2 / R1)
th_pla = (r - R1) / (R2 - R1)
ax[0].plot(r * 1e3, 1 - th_sph, "-", lw=2.2, color="#b2182b",
           label=r"sphere, $\propto 1/r$")
ax[0].plot(r * 1e3, 1 - th_cyl, "-", lw=2.0, color="#2166ac",
           label=r"cylinder, $\propto \ln r$")
ax[0].plot(r * 1e3, 1 - th_pla, "--", lw=1.7, color="#4d4d4d",
           label="plane wall, linear")
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"$(T - T_2)/(T_1 - T_2)$ [-]")
ax[0].set_title("(a) The same shell in three geometries")
ax[0].legend(fontsize=9)
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

r_out = np.linspace(R_WIRE, 0.060, 500)
q_c = np.array([heat_rate(r_out=x, k=K_PLASTIC, h=H_AIR, r_in=R_WIRE,
                          T_hot=T_W, T_amb=T_A) for x in r_out])
ax[1].plot(r_out * 1e3, q_c * 1e3, "-", lw=2.2, color="#b2182b",
           label="coated sphere")
ax[1].axhline(q_bare * 1e3, color="#2166ac", ls="--", lw=1.6, label="bare")
ax[1].axvline(r_cr_sph * 1e3, color="#1b7837", ls=":", lw=1.8)
ax[1].annotate(rf"$r_{{cr}} = 2k/h = {r_cr_sph*1e3:.0f}$ mm",
               xy=(r_cr_sph * 1e3 + 1.5, q_c.max() * 1e3 * 0.97), fontsize=9.5,
               color="#1b7837")
ax[1].axvline(r_cr_cyl * 1e3, color="0.45", ls="-.", lw=1.3)
ax[1].annotate(rf"cylinder $k/h$", xy=(r_cr_cyl * 1e3 + 1.0, q_bare * 1e3 * 1.5),
               fontsize=8.5, color="0.4", rotation=90)
ax[1].set_xlabel(r"Outer coating radius $r_2$ [mm]")
ax[1].set_ylabel(r"Heat loss $q$ [mW]")
ax[1].set_title("(b) Critical radius, twice the cylindrical value")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 4.1 -- Hollow and composite spheres", fontsize=12.5,
             y=1.02)
fig.savefig("fig_4_1a_profile.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
rg = np.linspace(0.0, R0, 400)
for lbl, Tm, Ts, expo, col in [
        ("plate ($m=0$)", Tmax_pl, Ts_pl, 2.0 * K_S, "#4d4d4d"),
        ("cylinder ($m=1$)", Tmax_cyl, Ts_cyl, 4.0 * K_S, "#2166ac"),
        ("sphere ($m=2$)", Tmax_sph, Ts_sph, 6.0 * K_S, "#b2182b")]:
    ax[0].plot(rg * 1e3, Tm - QGEN * rg**2 / expo, "-", lw=2.0, color=col,
               label=lbl)
ax[0].axhline(TINF_S, color="0.5", ls=":", lw=1.1)
ax[0].annotate(r"$T_\infty$", xy=(0.5, TINF_S + 6), fontsize=10, color="0.4")
ax[0].set_xlabel(r"Radial position [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Same $q'''$, same $L$: geometry decides")
ax[0].legend(fontsize=9)
ax[0].set_xlim(0, R0 * 1e3)

Rc_c = np.array([R_cond_sph(R_WIRE, x, K_PLASTIC) for x in r_out])
Rv_c = np.array([R_conv_sph(x, H_AIR) for x in r_out])
ax[1].plot(r_out * 1e3, Rc_c, "-", lw=1.9, color="#b2182b",
           label=r"$R_{cond}$ (rises, bounded)")
ax[1].plot(r_out * 1e3, Rv_c, "-", lw=1.9, color="#2166ac",
           label=r"$R_{conv} \propto r^{-2}$")
ax[1].plot(r_out * 1e3, Rc_c + Rv_c, "-", lw=2.4, color="#4d4d4d",
           label=r"$R_{tot}$")
ax[1].axhline(1.0 / (4 * np.pi * K_PLASTIC * R_WIRE), color="#b2182b",
              ls=":", lw=1.4)
ax[1].annotate(r"$1/(4\pi k r_1)$ ceiling", xy=(20, 100 + 1.0 / (4 * np.pi * K_PLASTIC * R_WIRE) * 0.9),
               fontsize=10.0, color="#b2182b")
ax[1].axvline(r_cr_sph * 1e3, color="#1b7837", ls=":", lw=1.8)
ax[1].set_xlabel(r"Outer coating radius $r_2$ [mm]")
ax[1].set_ylabel(r"Resistance [K W$^{-1}$]")
ax[1].set_title("(b) Bounded conduction resistance")
ax[1].legend(fontsize=8.5)
ax[1].set_ylim(0, (Rc_c + Rv_c).max() * 1.08)

fig.suptitle("Example 4.1 -- Generation and the critical radius",
             fontsize=12.5, y=1.02)
fig.savefig("fig_4_1b_critical.png")
plt.close(fig)

print("Figures written: fig_4_1a_profile.png, fig_4_1b_critical.png")
