"""
================================================================================
 EXAMPLE 4.3 -- ADVANCED VERIFICATION
 Transient startup of a solid sphere with uniform internal generation
================================================================================

 OBJECTIVE
 ---------
 The sphere of Example 4.1(d) begins in equilibrium with its coolant when the
 generation is switched on at t = 0.  Unlike the cylinder of Chapter 3, this
 problem HAS a closed-form eigenfunction solution in elementary functions, so
 the manufactured solution of Chapter 3 is not needed here and the two
 verification strategies can be compared.

 GOVERNING EQUATION
 ------------------
                       1   d  /  2   dT \
   rho c dT/dt  =  k  --- -- | r   * -- |  +  q'''       0 < r < r0
                       r^2 dr \      dr /

 INITIAL AND BOUNDARY CONDITIONS
 -------------------------------
   T(r,0)  = T_inf                       (in equilibrium, generation off)
   dT/dr   = 0 at r = 0                  (symmetry)
   -k dT/dr = h (T - T_inf) at r = r0    (convection)

 EXACT SOLUTION
 --------------
 The substitution u = r T turns the spherical operator into the plane one:

        (1/r^2) d/dr ( r^2 dT/dr )  =  (1/r) d2u/dr2

 so the sphere inherits the mathematics of the slab.  Splitting off the steady
 state of Example 4.1(d),

        T(r,t) = T_ss(r) + theta(r,t),
        T_ss(r) = T_inf + q''' r0/(3h) + q''' (r0^2 - r^2)/(6k)

 the excess theta obeys the homogeneous problem, and separation of variables
 gives eigenfunctions sin(lambda r)/r -- finite at the origin, where the
 companion solution cos(lambda r)/r is not.  The Robin condition at r0 yields

        1 - mu cot(mu) = Bi ,      mu = lambda r0 ,  Bi = h r0 / k

 with exactly one root in each interval ((n-1)pi, n pi).  Hence

        theta(r,t) = SUM_n A_n [ sin(lambda_n r) / r ] exp(-alpha lambda_n^2 t)

 Orthogonality holds for u = r theta on [0, r0] with unit weight, so with
 f(r) = r [T_inf - T_ss(r)] = -C r + D r^3, where

        C = q''' r0/(3h) + q''' r0^2/(6k) ,    D = q'''/(6k)

 the coefficients are A_n = [ -C I1 + D I3 ] / I2 with the elementary integrals

        I1 = int_0^r0 r sin(lam r) dr   = [ sin(mu) - mu cos(mu) ] / lam^2
        I3 = int_0^r0 r^3 sin(lam r) dr = [ (3mu^2 - 6) sin(mu)
                                            + (6mu - mu^3) cos(mu) ] / lam^4
        I2 = int_0^r0 sin^2(lam r) dr   = r0/2 - sin(2 mu)/(4 lam)

 All three are verified against numerical quadrature in the program.

 At r = 0 the eigenfunction is evaluated by its limit sin(lam r)/r -> lam.

 SYMBOLS -- see Example 4.1; additionally
   rho, c   [kg/m^3], [J/(kg K)]
   alpha    [m^2/s]    thermal diffusivity
   Bi       [-]        Biot number h r0 / k
   Fo       [-]        Fourier number alpha t / r0^2
   mu_n     [-]        dimensionless eigenvalue lambda_n r0

 OUTPUTS
 -------
   fig_4_3a_transient.png    evolution and adaptive stepping
   fig_4_3b_orders.png       spatial and temporal convergence
   fig_4_3c_sensitivity.png  Biot sweep and geometry comparison

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import solve_banded

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. DATA (the Example 4.1(d) sphere)
# ==============================================================================
R0 = 0.020
K = 15.0
RHO, CP = 7900.0, 477.0
ALPHA = K / (RHO * CP)
QGEN = 5.0e6
H = 500.0
T_INF = 350.0

BI = H * R0 / K
TAU = R0 * R0 / ALPHA


# ==============================================================================
# 2. EXACT SOLUTION
# ==============================================================================
@lru_cache(maxsize=None)
def _eigs(n_terms, Bi):
    """Roots of 1 - mu cot(mu) = Bi, one in each ((n-1)pi, n pi)."""
    f = lambda mu: mu * np.cos(mu) - (1.0 - Bi) * np.sin(mu)
    out = np.empty(n_terms)
    d = 1e-10
    for n in range(n_terms):
        lo, hi = n * np.pi + d, (n + 1) * np.pi - d
        out[n] = brentq(f, lo, hi, xtol=1e-14)
    out.flags.writeable = False
    return out


def eigenvalues(n_terms, Bi=BI):
    return _eigs(int(n_terms), float(Bi))


def steady(r, Bi=BI, q=QGEN):
    r = np.asarray(r, dtype=float)
    h_eff = Bi * K / R0
    return T_INF + q * R0 / (3.0 * h_eff) + q * (R0 * R0 - r * r) / (6.0 * K)


def _coeffs(n_terms, Bi, q):
    mus = eigenvalues(n_terms, Bi)
    lam = mus / R0
    h_eff = Bi * K / R0
    C = q * R0 / (3.0 * h_eff) + q * R0 * R0 / (6.0 * K)
    D = q / (6.0 * K)
    I1 = (np.sin(mus) - mus * np.cos(mus)) / lam**2
    I3 = ((3.0 * mus**2 - 6.0) * np.sin(mus)
          + (6.0 * mus - mus**3) * np.cos(mus)) / lam**4
    I2 = R0 / 2.0 - np.sin(2.0 * mus) / (4.0 * lam)
    return lam, (-C * I1 + D * I3) / I2


def exact(r, t, n_terms=80, Bi=BI, q=QGEN):
    """Exact T(r,t) [K].  Handles r = 0 by the limit sin(lam r)/r -> lam."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    lam, A = _coeffs(n_terms, Bi, q)
    arg = np.outer(lam, r)
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = np.where(r > 0.0, np.sin(arg) / np.where(r > 0, r, 1.0),
                         lam[:, None])
    decay = np.exp(-ALPHA * lam**2 * t)
    theta = np.sum(A[:, None] * decay[:, None] * shape, axis=0)
    return steady(r, Bi, q) + theta


# ==============================================================================
# 3. MESH AND SOLVER (geometric-mean conductance from Example 4.2)
# ==============================================================================
def make_mesh(N, ratio=1.0):
    w = np.ones(N) if abs(ratio - 1.0) < 1e-12 else ratio ** np.arange(N)
    w = w / w.sum() * R0
    rf = np.concatenate(([0.0], np.cumsum(w)))
    rf[-1] = R0
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc, rf


def step(T_old, dt, rc, rf, theta=1.0, Bi=BI, q=QGEN):
    N = len(T_old)
    vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0        # volume/(4 pi)
    h_eff = Bi * K / R0

    aW = np.zeros(N)
    aE = np.zeros(N)
    Sp = np.zeros(N)

    # exact spherical shell conductance between node radii (Example 4.2)
    g = K * rc[:-1] * rc[1:] / (rc[1:] - rc[:-1])
    aW[1:] = g
    aE[:-1] = g
    # r = 0 is a symmetry point: the area vanishes there, so there is simply
    # no link to add.  Spherical symmetry needs no special treatment at all.

    a_be = K * rc[-1] * R0 / (R0 - rc[-1])
    U = 1.0 / (1.0 / a_be + 1.0 / (R0 * R0 * h_eff))
    Su = q * vol
    Su[-1] += U * T_INF
    Sp[-1] -= U

    aPs = aW + aE - Sp
    aP0 = RHO * CP * vol / dt
    TW = np.concatenate(([0.0], T_old[:-1]))
    TE = np.concatenate((T_old[1:], [0.0]))
    F_old = aW * TW + aE * TE - aPs * T_old + Su

    aP = aP0 + theta * aPs
    b = aP0 * T_old + theta * Su + (1.0 - theta) * F_old

    ab = np.zeros((3, N))
    ab[0, 1:] = -theta * aE[:-1]
    ab[1, :] = aP
    ab[2, :-1] = -theta * aW[1:]
    return solve_banded((1, 1), ab, b)


def integrate(N, ratio, dt, t_end, theta=1.0, Bi=BI, q=QGEN, adaptive=False,
              rtol=1e-4, dt_min=1e-3, dt_max=500.0):
    rc, rf = make_mesh(N, ratio)
    T = np.full(N, T_INF)
    t, steps = 0.0, 0
    ht, hdt = [], []
    while t < t_end - 1e-12:
        trial = min(dt, t_end - t)
        if not adaptive:
            T = step(T, trial, rc, rf, theta, Bi, q)
            t += trial
            steps += 1
            continue
        while True:
            trial = min(trial, t_end - t)
            Tb = step(T, trial, rc, rf, 1.0, Bi, q)
            Th = step(T, 0.5 * trial, rc, rf, 1.0, Bi, q)
            Ts = step(Th, 0.5 * trial, rc, rf, 1.0, Bi, q)
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


def norms(T, rc, rf, t, Bi=BI):
    """Volume-weighted L2, Linf, and Linf excluding the origin cell.

    An observation, reported as measured rather than explained.  On this
    problem the L2 norm attains its asymptotic second-order range immediately
    (p = 2.000 on every refinement), while Linf approaches it only slowly:
    1.79, 1.81, 1.84, 1.85 over a sixteen-fold refinement, rising monotonically
    towards 2 without reaching it.  The maximum error always sits in the
    control volume touching r = 0.

    Two candidate explanations were tested and BOTH were refuted:

      * that the single origin cell is responsible -- excluding it (the third
        value returned here) makes the measured order WORSE, not better;
      * that the node should sit at the volume centroid (3/4)dr rather than
        at the arithmetic centre -- moving it leaves Linf essentially
        unchanged at p ~ 1.8.

    What the evidence supports is that the scheme is second order and that
    Linf simply has not reached its asymptotic range: the order rises steadily
    towards 2 rather than settling at some lower value, which is what a
    genuine first-order defect would do.  The conservation statement, the heat
    balance and the L2 norm are all clean.  A definitive account is left open;
    Exercise 4.C5 pursues it.
    """
    e = T - exact(rc, t, Bi=Bi)
    vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
    l2 = np.sqrt(np.sum(e**2 * vol) / np.sum(vol))
    return l2, np.max(np.abs(e)), np.max(np.abs(e[1:]))


# ==============================================================================
# 4. VERIFY THE EXACT SOLUTION
# ==============================================================================
print("=" * 78)
print("EXAMPLE 4.3 -- TRANSIENT SOLID SPHERE WITH GENERATION")
print("=" * 78)
print(f"  r0 = {R0} m, k = {K} W/(m K), alpha = {ALPHA:.4e} m^2/s")
print(f"  Bi = {BI:.6f},  tau = r0^2/alpha = {TAU:.2f} s")
print(f"  Steady: T_s = {T_INF + QGEN*R0/(3*H):.6f} K, "
      f"T_max = {float(steady(np.array([0.0]))[0]):.6f} K")
print("-" * 78)
print("  Eigenvalue check:  1 - mu cot(mu) = Bi")
print(f"  {'n':>3} {'mu_n':>14} {'residual':>13} {'bracket':>20}")
for i, mu in enumerate(eigenvalues(6), start=1):
    res = 1.0 - mu / np.tan(mu) - BI
    print(f"  {i:>3d} {mu:>14.10f} {res:>13.2e}   "
          f"({(i-1)*np.pi:7.4f}, {i*np.pi:7.4f})")

print("\n  Expansion coefficients: closed form vs quadrature")
lam, A = _coeffs(5, BI, QGEN)
f = lambda s: s * (T_INF - float(steady(np.array([s]))[0]))
print(f"  {'mu_n':>13} {'A_n closed':>16} {'A_n quad':>16} {'diff':>11}")
for lm, a in zip(lam, A):
    num = quad(lambda s: f(s) * np.sin(lm * s), 0.0, R0)[0]
    den = quad(lambda s: np.sin(lm * s) ** 2, 0.0, R0)[0]
    print(f"  {lm*R0:>13.9f} {a:>16.8f} {num/den:>16.8f} "
          f"{abs(a - num/den):>11.2e}")

xp = np.linspace(0.0, R0, 11)
print(f"\n  Series at t -> 0   : max|T - T_inf| = "
      f"{np.max(np.abs(exact(xp, 1e-9) - T_INF)):.3e} K (compatible data)")
print(f"  Series at t -> inf : max|T - T_ss|  = "
      f"{np.max(np.abs(exact(xp, 80.0*TAU) - steady(xp))):.3e} K")

# ==============================================================================
# 5. ORDERS OF ACCURACY
# ==============================================================================
T_EVAL = 60.0
print("\n" + "=" * 78)
print(f"SPATIAL CONVERGENCE at t = {T_EVAL:.0f} s "
      f"(Fo = {ALPHA*T_EVAL/R0**2:.4f}), Crank-Nicolson, dt = 0.01 s")
print("-" * 78)
print("  Linf* excludes the single control volume touching r = 0; see the")
print("  note in norms() for why that cell behaves differently.")
print(f"  {'N':>5} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} {'p_inf':>7} "
      f"{'Linf* [K]':>13} {'p*':>7}")
sp = []
for N in [10, 20, 40, 80, 160]:
    t0 = time.perf_counter()
    rc, rf, T, _ = integrate(N, 1.0, 0.01, T_EVAL, theta=0.5)
    cpu = time.perf_counter() - t0
    l2, li, li1 = norms(T, rc, rf, T_EVAL)
    p2 = np.log(sp[-1][1] / l2) / np.log(2.0) if sp else float("nan")
    pi_ = np.log(sp[-1][2] / li) / np.log(2.0) if sp else float("nan")
    pi1 = np.log(sp[-1][5] / li1) / np.log(2.0) if sp else float("nan")
    sp.append((N, l2, li, p2, pi_, li1, pi1, cpu))
    print(f"  {N:>5d} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} {pi_:>7.3f} "
          f"{li1:>13.4e} {pi1:>7.3f}")

mids = []
for N in [40, 80, 160]:
    rc, rf, T, _ = integrate(N, 1.0, 0.01, T_EVAL, theta=0.5)
    mids.append(float(np.interp(R0 / 2.0, rc, T)))
m3, m2, m1 = mids
p_sp = np.log(abs((m3 - m2) / (m2 - m1))) / np.log(2.0)
m_rich = m1 + (m1 - m2) / (2.0**p_sp - 1.0)
m_ex = float(exact(np.array([R0 / 2.0]), T_EVAL)[0])
GCI = 1.25 * abs((m1 - m2) / m1) / (2.0**p_sp - 1.0) * 100.0
print(f"\n  Richardson at r = r0/2:")
print(f"    N=40/80/160 : {m3:.10f} / {m2:.10f} / {m1:.10f} K")
print(f"    observed order p       = {p_sp:.4f}")
print(f"    extrapolated           = {m_rich:.10f} K")
print(f"    exact                  = {m_ex:.10f} K")
print(f"    |extrapolated - exact| = {abs(m_rich-m_ex):.3e} K")
print(f"    |finest      - exact|  = {abs(m1-m_ex):.3e} K")
print(f"    GCI_fine               = {GCI:.6f} %")

print("\n" + "=" * 78)
print("TEMPORAL CONVERGENCE at t = 60 s, N = 300")
print("-" * 78)
_, _, T_ref, _ = integrate(300, 1.0, 0.01, T_EVAL, theta=0.5)
print(f"  {'dt [s]':>9} {'Linf [K]':>14} {'p_t':>8} {'steps':>8}")
tm = []
for dt in [4.0, 2.0, 1.0, 0.5, 0.25]:
    _, _, Tb, dgb = integrate(300, 1.0, dt, T_EVAL, theta=1.0)
    e = np.max(np.abs(Tb - T_ref))
    p = np.log(tm[-1][1] / e) / np.log(2.0) if tm else float("nan")
    tm.append((dt, e, p, dgb["steps"]))
    print(f"  {dt:>9.3f} {e:>14.4e} {p:>8.3f} {dgb['steps']:>8d}")

# ==============================================================================
# 6. MESH, ADAPTIVE STEPPING, CROSS-CHECK
# ==============================================================================
print("\n" + "=" * 78)
print("MESH GRADING AND ADAPTIVE TIME STEPPING")
print("-" * 78)
print(f"  {'mesh':>24} {'L2 [K]':>13} {'Linf [K]':>13}")
for lbl, r_ in [("uniform (r = 1.00)", 1.00), ("graded  (r = 1.05)", 1.05),
                ("graded  (r = 0.95)", 0.95)]:
    rc, rf, T, _ = integrate(40, r_, 0.01, T_EVAL, theta=0.5)
    l2, li, _ = norms(T, rc, rf, T_EVAL)
    print(f"  {lbl:>24} {l2:>13.4e} {li:>13.4e}")

print(f"\n  Adaptive stepping, N = 80:")
print(f"  {'rtol':>10} {'Linf [K]':>13} {'steps':>8} {'dt range [s]':>24} "
      f"{'CPU [s]':>9}")
ad = {}
for rtol in [1e-3, 1e-4, 1e-5]:
    t0 = time.perf_counter()
    rc, rf, T, dg = integrate(80, 1.0, 0.1, T_EVAL, adaptive=True, rtol=rtol)
    cpu = time.perf_counter() - t0
    _, li, _ = norms(T, rc, rf, T_EVAL)
    ad[rtol] = dg
    print(f"  {rtol:>10.0e} {li:>13.4e} {dg['steps']:>8d} "
          f"{dg['hist_dt'].min():>10.4f} - {dg['hist_dt'].max():<11.4f} "
          f"{cpu:>9.3f}")
t0 = time.perf_counter()
rcf, rff, Tf, dgf = integrate(80, 1.0, 0.01, T_EVAL, theta=1.0)
cpuf = time.perf_counter() - t0
_, lif, _ = norms(Tf, rcf, rff, T_EVAL)
print(f"\n  Fixed dt = 0.01 s: Linf = {lif:.4e} K, {dgf['steps']} steps, "
      f"{cpuf:.3f} s")
print(f"  Adaptive rtol=1e-5: {ad[1e-5]['steps']} steps "
      f"({dgf['steps']/ad[1e-5]['steps']:.1f}x fewer)")

print("\n" + "=" * 78)
print("CROSS-CHECK AGAINST EXAMPLE 4.1(d)")
print("-" * 78)
rc, rf, T_ss_num, _ = integrate(320, 1.0, 20.0, 80.0 * TAU, theta=1.0)
ss = steady(rc)
a_be = K * rc[-1] * R0 / (R0 - rc[-1])
T_s_num = brentq(lambda Ts: a_be * (T_ss_num[-1] - Ts)
                 - R0 * R0 * H * (Ts - T_INF), 300.0, 900.0, xtol=1e-13)
q_out = 4 * np.pi * R0 * R0 * H * (T_s_num - T_INF)
q_gen = QGEN * 4.0 / 3.0 * np.pi * R0**3
print(f"  max|T_transient - T_ss(Example 4.1)| = "
      f"{np.max(np.abs(T_ss_num - ss)):.3e} K")
print(f"  T_max: numeric {T_ss_num[0]:.8f} vs analytic "
      f"{float(steady(np.array([rc[0]]))[0]):.8f} K")
print(f"  generated {q_gen:.8f} W vs removed {q_out:.8f} W "
      f"(imbalance {abs(q_gen-q_out):.3e})")
print("  The geometric-mean conductance of Example 4.2 is exact for the")
print("  constant-k steady state, so the residual is the marching tolerance.")

# ==============================================================================
# 7. SENSITIVITY
# ==============================================================================
print("\n" + "-" * 78)
print("SENSITIVITY")
print("  (a) Biot number")
print(f"  {'Bi':>8} {'mu_1':>10} {'T_s [K]':>11} {'T_max [K]':>11} "
      f"{'Fo_99':>9}")
bi_rows = []
for Bi in [0.1, 0.3, BI, 2.0, 10.0, 50.0]:
    mu1 = eigenvalues(1, Bi)[0]
    h_eff = Bi * K / R0
    T_s = T_INF + QGEN * R0 / (3.0 * h_eff)
    T_mx = T_s + QGEN * R0 * R0 / (6.0 * K)
    ts = np.linspace(0.5, 60.0 * TAU, 600)
    Tm = np.array([exact(np.array([0.0]), tt, 60, Bi)[0] for tt in ts])
    frac = (Tm - T_INF) / (T_mx - T_INF)
    hit = np.flatnonzero(frac >= 0.99)
    if hit.size == 0:
        raise RuntimeError(f"99 % never reached for Bi = {Bi}")
    bi_rows.append((Bi, mu1, T_s, T_mx, ALPHA * ts[hit[0]] / R0**2))
    print(f"  {Bi:>8.4f} {mu1:>10.6f} {T_s:>11.3f} {T_mx:>11.3f} "
          f"{ALPHA*ts[hit[0]]/R0**2:>9.4f}")

print("\n  (b) Geometry, same q''', k, h and characteristic length")
print(f"  {'geometry':>10} {'m':>3} {'film rise':>11} {'cond. rise':>12} "
      f"{'T_max [K]':>11}")
for name, m in [("plate", 0), ("cylinder", 1), ("sphere", 2)]:
    film = QGEN * R0 / ((m + 1) * H)
    cond = QGEN * R0 * R0 / (2 * (m + 1) * K)
    print(f"  {name:>10} {m:>3} {film:>11.4f} {cond:>12.4f} "
          f"{T_INF+film+cond:>11.4f}")
print()
print("  (c) On the two error norms.  L2 converges at exactly 2.000 on every")
print("  mesh.  Linf converges more slowly -- about 1.79 to 1.85 -- with the")
print("  maximum always in the cell touching r = 0.  Excluding that cell makes")
print("  the measured order worse, and placing the node at the volume centroid")
print("  changes nothing, so neither of the obvious explanations survives.")
print("  Because the order rises steadily towards 2 under refinement rather")
print("  than settling below it, the evidence points to Linf not having")
print("  reached its asymptotic range rather than to a first-order defect.")
print("  This is recorded as an open observation, not as a resolved question.")
print("  Both contributions scale as 1/(m+1): the geometry index alone")
print("  determines the peak temperature for a given duty.")
print("=" * 78)

# ==============================================================================
# 8. FIGURES
# ==============================================================================
rp = np.linspace(0.0, R0, 400)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for tt, col in zip([2.0, 8.0, 25.0, 60.0, 300.0],
                   plt.cm.inferno(np.linspace(0.15, 0.8, 5))):
    ax[0].plot(rp * 1e3, exact(rp, tt), "-", lw=1.9, color=col,
               label=rf"$t = {tt:.0f}$ s")
    rcn, rfn, Tn, _ = integrate(40, 1.0, 0.05, tt, theta=0.5)
    ax[0].plot(rcn * 1e3, Tn, "o", ms=3.6, mfc="none", mew=1.0, color=col)
ax[0].plot(rp * 1e3, steady(rp), "k--", lw=1.4, label="steady state")
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Startup\n(lines exact, symbols FVM $N=40$)")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_xlim(0, R0 * 1e3)

for rtol, sty in [(1e-3, "-"), (1e-4, "--"), (1e-5, ":")]:
    d = ad[rtol]
    ax[1].semilogy(d["hist_t"], d["hist_dt"], sty, lw=1.7,
                   label=rf"$rtol = 10^{{{int(np.log10(rtol))}}}$")
ax[1].set_xlabel(r"Time $t$ [s]")
ax[1].set_ylabel(r"Adaptive time step $\Delta t$ [s]")
ax[1].set_title("(b) Adaptive step-size history")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 4.3 -- Transient sphere with generation", fontsize=12.5,
             y=1.02)
fig.savefig("fig_4_3a_transient.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
Ns = np.array([r[0] for r in sp])
drs = R0 / Ns
ax[0].loglog(drs * 1e3, [r[1] for r in sp], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(drs * 1e3, [r[2] for r in sp], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$ (all cells)")
ax[0].loglog(drs * 1e3, [r[5] for r in sp], "^-", lw=1.8, ms=6,
             color="#1b7837", label=r"$\|e\|_\infty$ excl. origin cell")
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

fig.suptitle("Example 4.3 -- Order-of-accuracy verification", fontsize=12.5,
             y=1.02)
fig.savefig("fig_4_3b_orders.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
bis = np.array([r[0] for r in bi_rows])
ax[0].loglog(bis, [r[3] - T_INF for r in bi_rows], "o-", lw=1.9, ms=6,
             color="#b2182b", label=r"$T_{max}-T_\infty$")
ax[0].loglog(bis, [r[2] - T_INF for r in bi_rows], "s--", lw=1.7, ms=6,
             color="#2166ac", label=r"$T_s-T_\infty$")
ax[0].axvline(BI, color="0.45", ls=":", lw=1.3)
ax[0].set_xlabel(r"Biot number $Bi = h r_0/k$")
ax[0].set_ylabel("Temperature rise [K]")
ax[0].set_title("(a) Cooling effectiveness")
ax[0].legend(fontsize=9)

geo = np.array([0, 1, 2])
films = QGEN * R0 / ((geo + 1) * H)
conds = QGEN * R0 * R0 / (2 * (geo + 1) * K)
w = 0.55
ax[1].bar(geo, films, w, label="film rise", color="#2166ac",
          edgecolor="k", lw=0.6)
ax[1].bar(geo, conds, w, bottom=films, label="conduction rise",
          color="#b2182b", edgecolor="k", lw=0.6)
for g, f_, c_ in zip(geo, films, conds):
    ax[1].text(g, f_ + c_ + 6, f"{T_INF+f_+c_:.0f} K", ha="center",
               fontsize=9.5)
ax[1].set_xticks(geo)
ax[1].set_xticklabels(["plate\n$m=0$", "cylinder\n$m=1$", "sphere\n$m=2$"])
ax[1].set_ylabel(r"Temperature rise above $T_\infty$ [K]")
ax[1].set_title(r"(b) Both rises scale as $1/(m+1)$")
ax[1].legend(fontsize=9)
ax[1].set_ylim(0, (films + conds).max() * 1.18)

fig.suptitle("Example 4.3 -- Sensitivity and geometry comparison",
             fontsize=12.5, y=1.02)
fig.savefig("fig_4_3c_sensitivity.png")
plt.close(fig)

print("\nFigures written: fig_4_3a_transient.png, fig_4_3b_orders.png, "
      "fig_4_3c_sensitivity.png")
