"""
================================================================================
 EXAMPLE 3.3 -- ADVANCED VERIFICATION BY THE METHOD OF MANUFACTURED SOLUTIONS
 Transient radial conduction in a cylindrical shell
================================================================================

 WHY A MANUFACTURED SOLUTION
 ---------------------------
 Chapters 1 and 2 verified transient codes against eigenfunction series, which
 exist because those geometries have simple eigenfunctions.  In cylindrical
 geometry the eigenfunctions are Bessel functions, and for many realistic
 problems -- variable properties, mixed boundary conditions, several materials
 -- no analytical solution exists at all.

 The Method of Manufactured Solutions (MMS) removes that dependence entirely.
 Instead of solving a problem and hunting for its exact answer, one CHOOSES an
 analytic field T_mms(r,t), substitutes it into the governing equation, and
 defines the source term S(r,t) as whatever is left over.  By construction
 T_mms is then the exact solution of the modified problem, and the code must
 reproduce it to within its own discretisation error.  The manufactured field
 need not be physically realistic -- it needs only to be smooth and to exercise
 every term in the equation.

 MMS is the most powerful verification tool in computational engineering
 (Roache; Salari and Knupp) because it applies to ANY code, however complicated,
 and it tests every term including the boundary conditions.  It verifies
 correctness of the SOLVER, never the appropriateness of the model.

 GOVERNING EQUATION
 ------------------
                        1  d  /       dT \
   rho c dT/dt  =  k * --- -- | r  *  -- |  +  S(r,t)
                        r  dr \       dr /

 THE MANUFACTURED FIELD
 ----------------------
   T_mms(r,t) = Ta + G ln(r/r1) + P (r^2 - r1^2)
                   + E exp(-t/tau) (r - r1)(r2 - r)

 Each term is deliberate.  The logarithm is the natural steady solution and is
 annihilated by the operator, so it tests nothing but must be reproduced
 exactly.  The quadratic exercises the curvature term.  The final term carries
 all the time dependence and vanishes at BOTH boundaries, which keeps the
 boundary data simple while still forcing the transient term to work.

 REQUIRED SOURCE
 ---------------
 Applying the operator term by term (with k constant):

   ln term      : r dT/dr = G  -> d/dr = 0      -> contributes 0
   quadratic    : r dT/dr = 2P r^2 -> (1/r) d/dr = 4P
   transient    : (r-r1)(r2-r) = -r^2 + (r1+r2) r - r1 r2
                  r dT/dr = -2r^2 + (r1+r2) r
                  (1/r) d/dr = -4 + (r1+r2)/r

 and dT/dt = -(E/tau) exp(-t/tau) (r-r1)(r2-r), so

   S(r,t) = rho c dT/dt - k [ 4P + E exp(-t/tau) ( -4 + (r1+r2)/r ) ]

 BOUNDARY AND INITIAL CONDITIONS
 -------------------------------
   r = r1 : Dirichlet, T = T_mms(r1,t) = Ta         (the transient term
                                                     vanishes there, so this
                                                     is a constant)
   r = r2 : Robin, -k dT/dr = h (T - T_inf_eff(t)), with the effective sink
            temperature chosen so the condition is satisfied identically:
              T_inf_eff(t) = T_mms(r2,t) + (k/h) dT_mms/dr|_r2
   t = 0  : T = T_mms(r,0)

 The Robin condition is driven by a TIME-VARYING sink temperature; this is what
 makes the manufactured problem exercise the convective boundary implementation
 rather than merely a Dirichlet one.

 SYMBOLS -- see Examples 3.1 and 3.2; additionally
   Ta, G, P, E, tau   manufactured-solution parameters [K, K, K/m^2, K/m^2, s]
   S       [W/m^3]    manufactured volumetric source
   alpha   [m^2/s]    thermal diffusivity

 OUTPUTS
 -------
   fig_3_3a_mms.png       manufactured field and its evolution
   fig_3_3b_orders.png    spatial and temporal convergence
   fig_3_3c_physical.png  physical cross-check and sensitivity

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.optimize import brentq

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. DATA
# ==============================================================================
R1, R2 = 0.050, 0.100          # [m]
K = 0.050                      # [W/(m K)]
RHO, CP = 200.0, 900.0         # [kg/m^3], [J/(kg K)]
ALPHA = K / (RHO * CP)         # [m^2/s]
H = 10.0                       # [W/(m^2 K)]

# manufactured-solution parameters
TA = 450.0                     # [K]
G = -50.0                      # [K]
P = -10000.0                   # [K/m^2]
E = 40000.0                    # [K/m^2]
TAU = 3000.0                   # [s]

TDIFF = (R2 - R1) ** 2 / ALPHA


# ==============================================================================
# 2. MANUFACTURED SOLUTION AND ITS SOURCE
# ==============================================================================
def T_mms(r, t):
    r = np.asarray(r, dtype=float)
    return (TA + G * np.log(r / R1) + P * (r * r - R1 * R1)
            + E * np.exp(-t / TAU) * (r - R1) * (R2 - r))


def dTdr_mms(r, t):
    r = np.asarray(r, dtype=float)
    return (G / r + 2.0 * P * r
            + E * np.exp(-t / TAU) * ((R2 - r) - (r - R1)))


def source(r, t):
    """S(r,t) [W/m^3] that makes T_mms an exact solution."""
    r = np.asarray(r, dtype=float)
    dTdt = -(E / TAU) * np.exp(-t / TAU) * (r - R1) * (R2 - r)
    lap = 4.0 * P + E * np.exp(-t / TAU) * (-4.0 + (R1 + R2) / r)
    return RHO * CP * dTdt - K * lap


def T_inf_eff(t):
    """Sink temperature making the Robin condition exact at r = R2."""
    return (float(T_mms(np.array([R2]), t)[0])
            + (K / H) * float(dTdr_mms(np.array([R2]), t)[0]))


# ==============================================================================
# 3. MESH
# ==============================================================================
def make_mesh(N, ratio=1.0):
    w = np.ones(N) if abs(ratio - 1.0) < 1e-12 else ratio ** np.arange(N)
    w = w / w.sum() * (R2 - R1)
    rf = np.concatenate(([R1], R1 + np.cumsum(w)))
    rf[-1] = R2
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc, rf


# ==============================================================================
# 4. THETA-SCHEME STEP (cylindrical, log-mean conductance)
# ==============================================================================
def step(T_old, dt, rc, rf, t_new, t_old, theta=1.0, mms=True, T_in=None,
         Tinf_phys=None):
    """Advance one step.  Volumes and areas carry the radius."""
    N = len(T_old)
    # shell volume per unit length, divided by 2 pi
    vol = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)

    aW = np.zeros(N)
    aE = np.zeros(N)
    Sp = np.zeros(N)

    g = K / np.log(rc[1:] / rc[:-1])          # log-mean conductance (Scheme B)
    aW[1:] = g
    aE[:-1] = g

    aPs = aW + aE
    Su_new = np.zeros(N)
    Su_old = np.zeros(N)

    # inner boundary
    a_bw = K / np.log(rc[0] / R1)
    Ti_new = float(T_mms(np.array([R1]), t_new)[0]) if mms else T_in
    Ti_old = float(T_mms(np.array([R1]), t_old)[0]) if mms else T_in
    Su_new[0] += a_bw * Ti_new
    Su_old[0] += a_bw * Ti_old
    Sp[0] -= a_bw

    # outer boundary: half shell in series with the film (area R2)
    a_be = K / np.log(R2 / rc[-1])
    U = 1.0 / (1.0 / a_be + 1.0 / (R2 * H))
    Tinf_new = T_inf_eff(t_new) if mms else Tinf_phys
    Tinf_old = T_inf_eff(t_old) if mms else Tinf_phys
    Su_new[-1] += U * Tinf_new
    Su_old[-1] += U * Tinf_old
    Sp[-1] -= U

    aPs = aPs - Sp

    if mms:
        Su_new += source(rc, t_new) * vol
        Su_old += source(rc, t_old) * vol

    aP0 = RHO * CP * vol / dt
    TW = np.concatenate(([0.0], T_old[:-1]))
    TE = np.concatenate((T_old[1:], [0.0]))
    F_old = aW * TW + aE * TE - aPs * T_old + Su_old

    aP = aP0 + theta * aPs
    b = aP0 * T_old + theta * Su_new + (1.0 - theta) * F_old

    ab = np.zeros((3, N))
    ab[0, 1:] = -theta * aE[:-1]
    ab[1, :] = aP
    ab[2, :-1] = -theta * aW[1:]
    return solve_banded((1, 1), ab, b)


def integrate(N, ratio, dt, t_end, theta=1.0, mms=True, T0=None,
              T_in=None, Tinf_phys=None, adaptive=False, rtol=1e-4,
              dt_min=1e-2, dt_max=2000.0):
    rc, rf = make_mesh(N, ratio)
    T = T_mms(rc, 0.0) if mms else np.full(N, T0)
    t, steps = 0.0, 0
    ht, hdt = [], []

    while t < t_end - 1e-9:
        trial = min(dt, t_end - t)
        if not adaptive:
            T = step(T, trial, rc, rf, t + trial, t, theta, mms, T_in,
                     Tinf_phys)
            t += trial
            steps += 1
            continue
        while True:
            trial = min(trial, t_end - t)
            Tb = step(T, trial, rc, rf, t + trial, t, 1.0, mms, T_in,
                      Tinf_phys)
            Th = step(T, 0.5 * trial, rc, rf, t + 0.5 * trial, t, 1.0, mms,
                      T_in, Tinf_phys)
            Ts = step(Th, 0.5 * trial, rc, rf, t + trial, t + 0.5 * trial,
                      1.0, mms, T_in, Tinf_phys)
            err = np.max(np.abs(Ts - Tb))
            scale = rtol * max(1.0, np.max(np.abs(Ts)))
            if err <= scale or trial <= dt_min * 1.000001:
                T = 2.0 * Ts - Tb
                t += trial
                steps += 1
                ht.append(t)
                hdt.append(trial)
                fac = 0.9 * np.sqrt(scale / max(err, 1e-300))
                dt = float(np.clip(trial * np.clip(fac, 0.3, 2.0), dt_min,
                                   dt_max))
                break
            trial = max(dt_min, 0.5 * trial)

    return rc, rf, T, {"steps": steps, "hist_t": np.array(ht),
                       "hist_dt": np.array(hdt)}


def norms(T, rc, rf, t):
    e = T - T_mms(rc, t)
    vol = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)
    return np.sqrt(np.sum(e**2 * vol) / np.sum(vol)), np.max(np.abs(e))


# ==============================================================================
# 5. HEADER AND SANITY CHECKS ON THE MANUFACTURED SOLUTION
# ==============================================================================
print("=" * 78)
print("EXAMPLE 3.3 -- METHOD OF MANUFACTURED SOLUTIONS, CYLINDRICAL SHELL")
print("=" * 78)
print(f"  r1 = {R1} m, r2 = {R2} m, k = {K} W/(m K), alpha = {ALPHA:.4e} m^2/s")
print(f"  diffusion time (r2-r1)^2/alpha = {TDIFF:.1f} s,  tau = {TAU:.1f} s")
print(f"  T_mms(r1,t) = {TA:.4f} K (constant), "
      f"T_mms(r2,inf) = {float(T_mms(np.array([R2]), 1e9)[0]):.4f} K")

# Verify the manufactured source really does satisfy the PDE, by evaluating
# every term numerically and checking the residual.
rr = np.linspace(R1, R2, 4001)
for t_chk in [0.0, 1500.0, 6000.0]:
    Tn = T_mms(rr, t_chk)
    dT = np.gradient(Tn, rr, edge_order=2)
    lap = np.gradient(rr * dT, rr, edge_order=2) / rr
    dTdt = -(E / TAU) * np.exp(-t_chk / TAU) * (rr - R1) * (R2 - rr)
    resid = RHO * CP * dTdt - K * lap - source(rr, t_chk)
    scale = np.max(np.abs(source(rr, t_chk)))
    print(f"  PDE residual at t = {t_chk:7.1f} s : "
          f"max|res|/max|S| = {np.max(np.abs(resid[3:-3]))/scale:.3e}")

_T2 = float(T_mms(np.array([R2]), 0.0)[0])
_dT2 = float(dTdr_mms(np.array([R2]), 0.0)[0])
print(f"  Robin condition residual at r2, t=0 : "
      f"{abs(-K * _dT2 - H * (_T2 - T_inf_eff(0.0))):.3e}")

# ==============================================================================
# 6. SPATIAL ORDER (Crank-Nicolson, small dt)
# ==============================================================================
T_EVAL = 1500.0
print("\n" + "=" * 78)
print(f"SPATIAL CONVERGENCE at t = {T_EVAL:.0f} s, Crank-Nicolson, dt = 0.5 s")
print("-" * 78)
print(f"  {'N':>5} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} {'p_inf':>7} "
      f"{'CPU [s]':>9}")
sp = []
for N in [10, 20, 40, 80, 160]:
    t0 = time.perf_counter()
    rc, rf, T, dg = integrate(N, 1.0, 0.5, T_EVAL, theta=0.5)
    cpu = time.perf_counter() - t0
    l2, li = norms(T, rc, rf, T_EVAL)
    p2 = np.log(sp[-1][1] / l2) / np.log(2.0) if sp else float("nan")
    pi_ = np.log(sp[-1][2] / li) / np.log(2.0) if sp else float("nan")
    sp.append((N, l2, li, p2, pi_, cpu))
    print(f"  {N:>5d} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} {pi_:>7.3f} "
          f"{cpu:>9.3f}")

mids = []
for N in [40, 80, 160]:
    rc, rf, T, _ = integrate(N, 1.0, 0.5, T_EVAL, theta=0.5)
    mids.append(float(np.interp(0.5 * (R1 + R2), rc, T)))
m3, m2, m1 = mids
p_sp = np.log(abs((m3 - m2) / (m2 - m1))) / np.log(2.0)
m_rich = m1 + (m1 - m2) / (2.0**p_sp - 1.0)
m_ex = float(T_mms(np.array([0.5 * (R1 + R2)]), T_EVAL)[0])
GCI = 1.25 * abs((m1 - m2) / m1) / (2.0**p_sp - 1.0) * 100.0
print("\n  Richardson extrapolation at the mid-radius:")
print(f"    N=40/80/160 : {m3:.10f} / {m2:.10f} / {m1:.10f} K")
print(f"    observed order p       = {p_sp:.4f}")
print(f"    extrapolated           = {m_rich:.10f} K")
print(f"    manufactured (exact)   = {m_ex:.10f} K")
print(f"    |extrapolated - exact| = {abs(m_rich-m_ex):.3e} K")
print(f"    |finest      - exact|  = {abs(m1-m_ex):.3e} K")
print(f"    GCI_fine               = {GCI:.6f} %")

# ==============================================================================
# 7. TEMPORAL ORDER
# ==============================================================================
print("\n" + "=" * 78)
print("TEMPORAL CONVERGENCE at t = 1500 s, N = 400")
print("  Errors measured against the same mesh at dt = 0.5 s (theta = 1/2).")
print("-" * 78)
_, _, T_ref, _ = integrate(400, 1.0, 0.5, T_EVAL, theta=0.5)
print(f"  {'dt [s]':>9} {'Linf [K]':>14} {'p_t':>8} {'steps':>8}")
tm = []
for dt in [100.0, 50.0, 25.0, 12.5, 6.25]:
    _, _, Tb, dgb = integrate(400, 1.0, dt, T_EVAL, theta=1.0)
    e = np.max(np.abs(Tb - T_ref))
    p = np.log(tm[-1][1] / e) / np.log(2.0) if tm else float("nan")
    tm.append((dt, e, p, dgb["steps"]))
    print(f"  {dt:>9.3f} {e:>14.4e} {p:>8.3f} {dgb['steps']:>8d}")

# ==============================================================================
# 8. NON-UNIFORM MESH AND ADAPTIVE STEPPING
# ==============================================================================
print("\n" + "=" * 78)
print("NON-UNIFORM MESH AND ADAPTIVE TIME STEPPING")
print("-" * 78)
print(f"  {'mesh':>24} {'L2 [K]':>13} {'Linf [K]':>13}")
for lbl, r_ in [("uniform (r = 1.00)", 1.00), ("graded  (r = 1.05)", 1.05),
                ("graded  (r = 0.95)", 0.95)]:
    rc, rf, T, _ = integrate(40, r_, 0.5, T_EVAL, theta=0.5)
    l2, li = norms(T, rc, rf, T_EVAL)
    print(f"  {lbl:>24} {l2:>13.4e} {li:>13.4e}")

print(f"\n  Adaptive stepping, N = 80:")
print(f"  {'rtol':>10} {'Linf [K]':>13} {'steps':>8} {'dt range [s]':>26} "
      f"{'CPU [s]':>9}")
ad = {}
for rtol in [1e-3, 1e-4, 1e-5]:
    t0 = time.perf_counter()
    rc, rf, T, dg = integrate(80, 1.0, 5.0, T_EVAL, adaptive=True, rtol=rtol)
    cpu = time.perf_counter() - t0
    _, li = norms(T, rc, rf, T_EVAL)
    ad[rtol] = dg
    print(f"  {rtol:>10.0e} {li:>13.4e} {dg['steps']:>8d} "
          f"{dg['hist_dt'].min():>11.3f} - {dg['hist_dt'].max():<12.3f} "
          f"{cpu:>9.3f}")
t0 = time.perf_counter()
rcf, rff, Tf, dgf = integrate(80, 1.0, 0.5, T_EVAL, theta=1.0)
cpuf = time.perf_counter() - t0
_, lif = norms(Tf, rcf, rff, T_EVAL)
print(f"\n  Fixed dt = 0.5 s: Linf = {lif:.4e} K, {dgf['steps']} steps, "
      f"{cpuf:.3f} s")
print(f"  Adaptive rtol=1e-5: {ad[1e-5]['steps']} steps "
      f"({dgf['steps']/ad[1e-5]['steps']:.1f}x fewer)")

# ==============================================================================
# 9. PHYSICAL CROSS-CHECK AGAINST EXAMPLE 3.1
# ==============================================================================
print("\n" + "=" * 78)
print("PHYSICAL CROSS-CHECK: switch the source off and march to steady state")
print("-" * 78)
T1_PHYS, TINF_PHYS = 450.0, 300.0
rc, rf, T_ss, dg = integrate(320, 1.0, 5000.0, 400.0 * TDIFF, theta=1.0,
                             mms=False, T0=TINF_PHYS, T_in=T1_PHYS,
                             Tinf_phys=TINF_PHYS)
q_exact = 2 * np.pi * (T1_PHYS - TINF_PHYS) / (
    np.log(R2 / R1) / K + 1.0 / (R2 * H))
T_an = T1_PHYS - q_exact / (2 * np.pi * K) * np.log(rc / R1)
a_be = K / np.log(R2 / rc[-1])
T2_num = brentq(lambda Ts: a_be * (T_ss[-1] - Ts) - R2 * H * (Ts - TINF_PHYS),
                200.0, 600.0, xtol=1e-13)
q_num = 2 * np.pi * R2 * H * (T2_num - TINF_PHYS)
print(f"  Example 3.1 analytical q' = {q_exact:.10f} W/m")
print(f"  Transient solver  steady q' = {q_num:.10f} W/m")
print(f"  |difference|                = {abs(q_num-q_exact):.3e} W/m")
print(f"  max|T_numeric - T_analytic| = {np.max(np.abs(T_ss - T_an)):.3e} K")
print("  The log-mean conductance of Example 3.2 is exact for this problem,")
print("  so the only residual is the time-marching tolerance.")

# ==============================================================================
# 10. SENSITIVITY
# ==============================================================================
print("\n" + "-" * 78)
print("SENSITIVITY: radius ratio r2/r1 at fixed shell thickness")
print(f"  {'r2/r1':>8} {'R_cond [m K/W]':>16} {'vs plane wall':>15}")
for ratio in [1.05, 1.25, 2.0, 4.0, 10.0]:
    r1_ = (R2 - R1) / (ratio - 1.0)
    r2_ = ratio * r1_
    Rc_ = np.log(r2_ / r1_) / (2 * np.pi * K)
    R_plane = (r2_ - r1_) / (2 * np.pi * 0.5 * (r1_ + r2_) * K)
    print(f"  {ratio:>8.2f} {Rc_:>16.8f} {100*(R_plane/Rc_-1):>14.4f} %")
print("  As r2/r1 -> 1 the shell behaves like a plane wall and the arithmetic")
print("  mean becomes exact; the error of the plane-wall shortcut grows with")
print("  curvature, reaching several percent for thick-walled pipes.")
print("=" * 78)

# ==============================================================================
# 11. FIGURES
# ==============================================================================
rp = np.linspace(R1, R2, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for tt, col in zip([0.0, 750.0, 1500.0, 3000.0, 9000.0],
                   plt.cm.viridis(np.linspace(0.05, 0.85, 5))):
    ax[0].plot(rp * 1e3, T_mms(rp, tt), "-", lw=1.9, color=col,
               label=rf"$t = {tt:.0f}$ s")
    rcn, rfn, Tn, _ = integrate(40, 1.0, 5.0, max(tt, 1e-9), theta=0.5)
    ax[0].plot(rcn * 1e3, Tn, "o", ms=3.6, mfc="none", mew=1.0, color=col)
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Manufactured field\n(lines exact, symbols FVM $N=40$)")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

for tt, col in zip([0.0, 1500.0, 9000.0], ["#2166ac", "#b2182b", "#1b7837"]):
    ax[1].plot(rp * 1e3, source(rp, tt), "-", lw=1.9, color=col,
               label=rf"$t = {tt:.0f}$ s")
ax[1].axhline(0, color="k", lw=0.9)
ax[1].set_xlabel(r"Radius $r$ [mm]")
ax[1].set_ylabel(r"Manufactured source $S$ [W m$^{-3}$]")
ax[1].set_title("(b) The source that makes $T_{mms}$ exact")
ax[1].legend(fontsize=9)

fig.suptitle("Example 3.3 -- Method of manufactured solutions", fontsize=12.5,
             y=1.02)
fig.savefig("fig_3_3a_mms.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
Ns = np.array([r[0] for r in sp])
drs = (R2 - R1) / Ns
ax[0].loglog(drs * 1e3, [r[1] for r in sp], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(drs * 1e3, [r[2] for r in sp], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$")
ax[0].loglog(drs * 1e3, sp[0][1] * (drs / drs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta r$ [mm]")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title(f"(a) Spatial convergence ($p = {p_sp:.2f}$)")
ax[0].legend(fontsize=9, loc="lower right")

dts = np.array([r[0] for r in tm])
ax[1].loglog(dts, [r[1] for r in tm], "o-", lw=1.8, ms=6, color="#762a83",
             label=r"backward Euler ($\theta = 1$)")
ax[1].loglog(dts, tm[0][1] * (dts / dts[0]), "k--", lw=1.3, label="slope 1")
ax[1].set_xlabel(r"$\Delta t$ [s]")
ax[1].set_ylabel("Error norm [K]")
ax[1].set_title("(b) Temporal convergence")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 3.3 -- Order-of-accuracy verification by MMS",
             fontsize=12.5, y=1.02)
fig.savefig("fig_3_3b_orders.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(rp * 1e3, T1_PHYS - q_exact / (2 * np.pi * K) * np.log(rp / R1),
           "-", lw=2.0, color="#b2182b", label="Example 3.1 analytical")
ax[0].plot(rc[::16] * 1e3, T_ss[::16], "o", ms=6, mfc="none", mew=1.4,
           color="#2166ac", label="Example 3.3 transient $\\to$ steady")
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Physical cross-check")
ax[0].legend(fontsize=9)
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

for rtol, sty in [(1e-3, "-"), (1e-4, "--"), (1e-5, ":")]:
    d = ad[rtol]
    ax[1].semilogy(d["hist_t"], d["hist_dt"], sty, lw=1.7,
                   label=rf"$rtol = 10^{{{int(np.log10(rtol))}}}$")
ax[1].set_xlabel(r"Time $t$ [s]")
ax[1].set_ylabel(r"Adaptive time step $\Delta t$ [s]")
ax[1].set_title("(b) Adaptive step-size history")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 3.3 -- Physical cross-check and adaptive stepping",
             fontsize=12.5, y=1.02)
fig.savefig("fig_3_3c_physical.png")
plt.close(fig)

print("\nFigures written: fig_3_3a_mms.png, fig_3_3b_orders.png, "
      "fig_3_3c_physical.png")
