"""
================================================================================
 EXAMPLE 2.3 -- ADVANCED VERIFICATION: STARTUP OF THE HEATED PLATE
================================================================================

 OBJECTIVE
 ---------
 The plate of Examples 2.1 and 2.2 is initially in equilibrium with its coolant
 when the volumetric heating is switched on at t = 0 -- reactor startup, or a
 heating element energised.  The full verification apparatus is applied:

   * exact eigenfunction solution for the transient with a steady source
   * theta-family time integration (backward Euler and Crank-Nicolson)
   * geometrically graded meshes
   * adaptive time stepping by step doubling with Richardson correction
   * SEPARATE spatial and temporal order-of-accuracy measurements
   * Richardson extrapolation and the Grid Convergence Index
   * L2 and Linf error norms, automated error tables, CPU timings
   * closed-form prediction of the Example 2.2 boundary error, verified
   * cross-checks against Examples 2.1 and 2.2
   * sensitivity analysis

 GOVERNING EQUATION
 ------------------
   rho c dT/dt = d/dx ( k dT/dx ) + q'''        0 < x < L ,  t > 0

 INITIAL CONDITION
 -----------------
   T(x, 0) = T_inf                     (plate in equilibrium, heating off)

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :   dT/dx = 0                            (symmetry)
   x = L :  -k dT/dx|_L = h (T_L - T_inf)         (convective)

 EXACT SOLUTION
 --------------
 Because the source is time independent, split off the steady state of
 Example 2.1,

        T(x,t) = T_ss(x) + theta(x,t) ,   T_ss(x) = T_max - q''' x^2/(2k)

 Substituting, the source cancels identically and theta obeys the homogeneous
 problem  d(theta)/dt = alpha d2(theta)/dx2  with theta_x(0,t) = 0 and
 -k theta_x(L,t) = h theta(L,t).  Separation of variables gives eigenfunctions
 X_n(x) = cos(lambda_n x), and the Robin condition yields

        mu tan(mu) = Bi ,        mu = lambda L ,  Bi = hL/k

 which has exactly one root in each interval ((n-1)pi, (n-1)pi + pi/2).  The
 system is a regular Sturm-Liouville problem, so

        theta(x,t) = SUM_n A_n cos(lambda_n x) exp(-alpha lambda_n^2 t)

        A_n = [ int_0^L f(x) cos(lambda_n x) dx ] / [ int_0^L cos^2(lambda_n x) dx ]

 with f(x) = T_inf - T_ss(x).  The required integrals are elementary:

        int_0^L cos(lam x) dx    = sin(lam L)/lam
        int_0^L x^2 cos(lam x) dx = [(lam^2 L^2 - 2) sin(lam L)
                                     + 2 lam L cos(lam L)] / lam^3
        int_0^L cos^2(lam x) dx  = L/2 + sin(2 lam L)/(4 lam)

 Note that the initial data are COMPATIBLE here (the initial field satisfies
 both boundary conditions), unlike the step-change problem of Chapter 1.  The
 series therefore converges rapidly at all times and shows no Gibbs behaviour.

 SYMBOLS -- see Examples 2.1 and 2.2; additionally
   rho   [kg/m^3]    density
   c     [J/(kg K)]  specific heat
   alpha [m^2/s]     thermal diffusivity
   Fo    [-]         Fourier number
   theta [K]         excess temperature above the steady state
   mu_n  [-]         dimensionless eigenvalue

 OUTPUTS
 -------
   fig_2_3a_transient.png    evolution, mesh, adaptive step history
   fig_2_3b_orders.png       spatial and temporal convergence, boundary error
   fig_2_3c_sensitivity.png  parametric study

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
# 1. PROBLEM DATA
# ==============================================================================
L = 0.02
K = 15.0
RHO = 7900.0
CP = 477.0
ALPHA = K / (RHO * CP)
QGEN = 5.0e6
H = 500.0
T_INF = 350.0

BI = H * L / K
TAU = L * L / ALPHA                    # [s] diffusion time scale


# ==============================================================================
# 2. EXACT SOLUTION
# ==============================================================================
@lru_cache(maxsize=None)
def _eigs(n_terms, Bi):
    """Roots of mu tan(mu) = Bi, one per interval ((n-1)pi, (n-1)pi + pi/2)."""
    f = lambda mu: mu * np.sin(mu) - Bi * np.cos(mu)
    out = np.empty(n_terms)
    d = 1e-12
    for n in range(n_terms):
        lo, hi = n * np.pi + d, n * np.pi + 0.5 * np.pi - d
        out[n] = brentq(f, lo, hi, xtol=1e-14)
    out.flags.writeable = False
    return out


def eigenvalues(n_terms, Bi=BI):
    return _eigs(int(n_terms), float(Bi))


def steady(x, Bi=BI, q=QGEN):
    """Steady profile T_ss(x) [K] (Example 2.1)."""
    h_eff = Bi * K / L
    T_L = T_INF + q * L / h_eff
    return T_L + q * (L * L - np.asarray(x) ** 2) / (2.0 * K)


def exact(x, t, n_terms=120, Bi=BI, q=QGEN):
    """Exact T(x,t) [K] by eigenfunction expansion."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    mus = eigenvalues(n_terms, Bi)
    lam = mus / L

    # f(x) = T_inf - T_ss(x) = -(q/2k)(L^2 - x^2) - q L/h_eff
    h_eff = Bi * K / L
    c0 = -(q * L / h_eff) - q * L * L / (2.0 * K)      # constant part
    c2 = q / (2.0 * K)                                  # coefficient of x^2

    I0 = np.sin(mus) / lam
    I2 = ((mus**2 - 2.0) * np.sin(mus) + 2.0 * mus * np.cos(mus)) / lam**3
    norm = L / 2.0 + np.sin(2.0 * mus) / (4.0 * lam)
    A = (c0 * I0 + c2 * I2) / norm

    decay = np.exp(-ALPHA * lam**2 * t)
    theta = np.sum(A[:, None] * decay[:, None] * np.cos(np.outer(lam, x)),
                   axis=0)
    return steady(x, Bi, q) + theta


# ==============================================================================
# 3. MESH
# ==============================================================================
def make_mesh(N, ratio=1.0):
    """Graded cell-centred mesh; ratio > 1 clusters cells toward the surface."""
    w = np.ones(N) if abs(ratio - 1.0) < 1e-12 else ratio ** np.arange(N)[::-1]
    w = w / w.sum() * L
    xf = np.concatenate(([0.0], np.cumsum(w)))
    xf[-1] = L
    xc = 0.5 * (xf[:-1] + xf[1:])
    dxc = np.diff(xf)
    dxn = np.empty(N + 1)
    dxn[0] = xc[0] - xf[0]
    dxn[1:N] = np.diff(xc)
    dxn[N] = xf[-1] - xc[-1]
    return xc, xf, dxc, dxn


# ==============================================================================
# 4. THETA-SCHEME STEP
# ==============================================================================
def step(T_old, dt, dxc, dxn, beta=0.0, Bi=BI, q=QGEN, theta=1.0,
         tol=1e-11, max_inner=40):
    """One time step of the theta family (see Chapter 1, Example 1.3)."""
    N = len(T_old)
    T = T_old.copy()
    h_eff = Bi * K / L

    for inner in range(1, max_inner + 1):
        kc = K * (1.0 + beta * (T - T_INF))
        kf = np.empty(N + 1)
        kf[1:N] = 2.0 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])
        kf[0], kf[N] = kc[0], kc[-1]

        aW = np.zeros(N)
        aE = np.zeros(N)
        Sp = np.zeros(N)
        Su = np.full(N, q) * dxc            # source integral over each cell

        aW[1:] = kf[1:N] / dxn[1:N]
        aE[:-1] = kf[1:N] / dxn[1:N]
        # x = 0 is adiabatic: no link, nothing to add.
        U = 1.0 / (dxn[N] / kf[N] + 1.0 / h_eff)
        Su[-1] += U * T_INF
        Sp[-1] -= U

        aP0 = RHO * CP * dxc / dt
        aPs = aW + aE - Sp

        TW = np.concatenate(([0.0], T_old[:-1]))
        TE = np.concatenate((T_old[1:], [0.0]))
        F_old = aW * TW + aE * TE - aPs * T_old + Su

        aP = aP0 + theta * aPs
        b = aP0 * T_old + theta * Su + (1.0 - theta) * F_old

        ab = np.zeros((3, N))
        ab[0, 1:] = -theta * aE[:-1]
        ab[1, :] = aP
        ab[2, :-1] = -theta * aW[1:]
        T_new = solve_banded((1, 1), ab, b)

        change = np.max(np.abs(T_new - T))
        T = T_new
        if change < tol:
            break
    return T, inner


def integrate(N, ratio, dt, t_end, beta=0.0, Bi=BI, q=QGEN, theta=1.0,
              adaptive=False, rtol=1e-4, dt_min=1e-3, dt_max=500.0):
    """Integrate to t_end.  Returns (xc, T, diagnostics)."""
    xc, xf, dxc, dxn = make_mesh(N, ratio)
    T = np.full(N, T_INF)
    t, steps = 0.0, 0
    ht, hdt = [], []

    while t < t_end - 1e-12:
        trial = min(dt, t_end - t)
        if not adaptive:
            T, _ = step(T, trial, dxc, dxn, beta, Bi, q, theta)
            t += trial
            steps += 1
            continue
        while True:
            trial = min(trial, t_end - t)
            Tb, _ = step(T, trial, dxc, dxn, beta, Bi, q, 1.0)
            Th, _ = step(T, 0.5 * trial, dxc, dxn, beta, Bi, q, 1.0)
            Ts, _ = step(Th, 0.5 * trial, dxc, dxn, beta, Bi, q, 1.0)
            err = np.max(np.abs(Ts - Tb))
            scale = rtol * max(1.0, np.max(np.abs(Ts)))
            if err <= scale or trial <= dt_min * 1.000001:
                T = 2.0 * Ts - Tb            # Richardson-corrected update
                t += trial
                steps += 1
                ht.append(t)
                hdt.append(trial)
                fac = 0.9 * np.sqrt(scale / max(err, 1e-300))
                dt = float(np.clip(trial * np.clip(fac, 0.3, 2.0),
                                   dt_min, dt_max))
                break
            trial = max(dt_min, 0.5 * trial)

    return xc, T, {"steps": steps, "dxc": dxc, "xf": xf,
                   "hist_t": np.array(ht), "hist_dt": np.array(hdt)}


def norms(T, xc, t, dxc, Bi=BI):
    e = T - exact(xc, t, Bi=Bi)
    return np.sqrt(np.sum(e**2 * dxc) / L), np.max(np.abs(e))


# ==============================================================================
# 5. HEADER AND EXACT-SOLUTION VERIFICATION
# ==============================================================================
print("=" * 78)
print("EXAMPLE 2.3 -- TRANSIENT STARTUP OF THE HEATED PLATE")
print("=" * 78)
print(f"  alpha = {ALPHA:.6e} m^2/s    Bi = {BI:.6f}    tau = L^2/alpha "
      f"= {TAU:.2f} s")
print(f"  Steady:  T_L = {T_INF + QGEN*L/H:.6f} K, "
      f"T_max = {float(steady(np.array([0.0]))[0]):.6f} K")
print("-" * 78)
print("  Eigenvalue check:  mu tan(mu) = Bi")
print(f"  {'n':>3} {'mu_n':>14} {'residual':>13} {'bracket':>22}")
for i, mu in enumerate(eigenvalues(6), start=1):
    r = mu * np.tan(mu) - BI
    print(f"  {i:>3d} {mu:>14.10f} {r:>13.2e} "
          f"  ({(i-1)*np.pi:7.4f}, {(i-1)*np.pi+np.pi/2:7.4f})")

# closed-form coefficients versus quadrature
print("\n  Expansion coefficients: closed form vs quadrature")
mus = eigenvalues(5)
lam = mus / L
f = lambda s: T_INF - float(steady(np.array([s]))[0])
print(f"  {'mu_n':>14} {'A_n closed [K]':>17} {'A_n quad [K]':>16} {'diff':>11}")
c0 = -(QGEN * L / H) - QGEN * L * L / (2.0 * K)
c2 = QGEN / (2.0 * K)
for mu, lm in zip(mus, lam):
    I0 = np.sin(mu) / lm
    I2 = ((mu**2 - 2.0) * np.sin(mu) + 2.0 * mu * np.cos(mu)) / lm**3
    nrm = L / 2.0 + np.sin(2.0 * mu) / (4.0 * lm)
    A_form = (c0 * I0 + c2 * I2) / nrm
    A_quad = (quad(lambda s: f(s) * np.cos(lm * s), 0.0, L)[0]
              / quad(lambda s: np.cos(lm * s) ** 2, 0.0, L)[0])
    print(f"  {mu:>14.10f} {A_form:>17.10f} {A_quad:>16.10f} "
          f"{abs(A_form-A_quad):>11.2e}")

xp = np.linspace(0.0, L, 11)
print(f"\n  Series at t -> 0   : max|T - T_inf| = "
      f"{np.max(np.abs(exact(xp, 1e-9) - T_INF)):.3e} K  "
      "(compatible data: no Gibbs ringing)")
print(f"  Series at t -> inf : max|T - T_ss|  = "
      f"{np.max(np.abs(exact(xp, 60.0*TAU) - steady(xp))):.3e} K")

# ==============================================================================
# 6. SPATIAL ORDER (Crank-Nicolson so the time error is negligible)
# ==============================================================================
T_EVAL = 100.0
print("\n" + "=" * 78)
print(f"SPATIAL CONVERGENCE at t = {T_EVAL:.0f} s "
      f"(Fo = {ALPHA*T_EVAL/L**2:.4f}), Crank-Nicolson, dt = 0.02 s")
print("-" * 78)
print(f"  {'N':>5} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} {'p_inf':>7} "
      f"{'CPU [s]':>9}")
sp = []
for N in [10, 20, 40, 80, 160]:
    t0 = time.perf_counter()
    xc, T, dg = integrate(N, 1.0, 0.02, T_EVAL, theta=0.5)
    cpu = time.perf_counter() - t0
    l2, li = norms(T, xc, T_EVAL, dg["dxc"])
    p2 = np.log(sp[-1][1] / l2) / np.log(2.0) if sp else float("nan")
    pi_ = np.log(sp[-1][2] / li) / np.log(2.0) if sp else float("nan")
    sp.append((N, l2, li, p2, pi_, cpu))
    print(f"  {N:>5d} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} {pi_:>7.3f} "
          f"{cpu:>9.3f}")

mids = []
for N in [40, 80, 160]:
    xc, T, _ = integrate(N, 1.0, 0.02, T_EVAL, theta=0.5)
    mids.append(float(np.interp(L / 2.0, xc, T)))
m3, m2, m1 = mids
p_sp = np.log(abs((m3 - m2) / (m2 - m1))) / np.log(2.0)
m_rich = m1 + (m1 - m2) / (2.0**p_sp - 1.0)
m_ex = float(exact(np.array([L / 2.0]), T_EVAL)[0])
GCI = 1.25 * abs((m1 - m2) / m1) / (2.0**p_sp - 1.0) * 100.0
print("\n  Richardson extrapolation on T(L/2, t):")
print(f"    N=40/80/160 : {m3:.10f} / {m2:.10f} / {m1:.10f} K")
print(f"    observed order p        = {p_sp:.4f}")
print(f"    extrapolated            = {m_rich:.10f} K")
print(f"    exact                   = {m_ex:.10f} K")
print(f"    |extrapolated - exact|  = {abs(m_rich-m_ex):.3e} K")
print(f"    |finest      - exact|   = {abs(m1-m_ex):.3e} K")
print(f"    GCI_fine                = {GCI:.6f} %")

# ==============================================================================
# 7. TEMPORAL ORDER
# ==============================================================================
print("\n" + "=" * 78)
print("TEMPORAL CONVERGENCE at t = 100 s, N = 300")
print("  Measured against the same mesh at dt = 0.02 s (theta = 1/2), so the")
print("  common spatial error cancels and only the temporal error remains.")
print("-" * 78)
_, T_ref, _ = integrate(300, 1.0, 0.02, T_EVAL, theta=0.5)
print(f"  {'dt [s]':>9} {'Linf [K]':>14} {'p_t':>8} {'steps':>8}")
tm = []
for dt in [4.0, 2.0, 1.0, 0.5, 0.25]:
    _, Tb, dgb = integrate(300, 1.0, dt, T_EVAL, theta=1.0)
    e = np.max(np.abs(Tb - T_ref))
    p = np.log(tm[-1][1] / e) / np.log(2.0) if tm else float("nan")
    tm.append((dt, e, p, dgb["steps"]))
    print(f"  {dt:>9.3f} {e:>14.4e} {p:>8.3f} {dgb['steps']:>8d}")

# ==============================================================================
# 8. THE EXAMPLE 2.2 BOUNDARY ERROR, PREDICTED IN CLOSED FORM
# ==============================================================================
print("\n" + "=" * 78)
print("CLOSED-FORM PREDICTION OF THE STEADY-STATE DISCRETISATION ERROR")
print("-" * 78)
print("  In Example 2.2, Case A, the error was found to be a CONSTANT offset:")
print("  the L2 norm, the Linf norm and the mid-plane error were all equal.")
print("  The reason is that the interior discretisation is EXACT for a")
print("  parabola (a central second difference reproduces a quadratic exactly,")
print("  and the source integral is exact), so the only error enters through")
print("  the half-cell treatment at the convective face, and it shifts the")
print("  whole profile by a constant.")
print()
print("  Integrating the exact flux q''(x) = q''' x across the last half cell:")
print("      T_N - T_L = (q'''/2k) [ L^2 - (L - dx/2)^2 ]")
print("  whereas the scheme assumes a linear profile carrying the face flux:")
print("      (T_N - T_L)_FVM = q''' L dx / (2k)")
print("  The difference is  q''' dx^2 / (8k),  independent of L and h.")
print()
print(f"  {'N':>5} {'dx [mm]':>9} {'measured [K]':>15} {'predicted [K]':>15} "
      f"{'ratio':>9}")
for N in [10, 20, 40, 80, 160]:
    xc, T, dg = integrate(N, 1.0, 200.0, 80.0 * TAU, theta=1.0)
    meas = T[0] - float(steady(np.array([xc[0]]))[0])
    dx = L / N
    pred = QGEN * dx * dx / (8.0 * K)
    print(f"  {N:>5d} {dx*1e3:>9.4f} {meas:>15.6e} {pred:>15.6e} "
          f"{meas/pred:>9.6f}")

# ==============================================================================
# 9. MESH GRADING AND ADAPTIVE STEPPING
# ==============================================================================
print("\n" + "=" * 78)
print("MESH GRADING AND ADAPTIVE TIME STEPPING")
print("-" * 78)
print(f"  {'mesh':>24} {'L2 [K]':>13} {'Linf [K]':>13}")
for label, r in [("uniform (r = 1.00)", 1.00), ("graded  (r = 1.05)", 1.05),
                 ("graded  (r = 1.10)", 1.10), ("graded  (r = 1.20)", 1.20)]:
    xc, T, dg = integrate(40, r, 0.02, T_EVAL, theta=0.5)
    l2, li = norms(T, xc, T_EVAL, dg["dxc"])
    print(f"  {label:>24} {l2:>13.4e} {li:>13.4e}")

print(f"\n  Adaptive stepping, N = 80:")
print(f"  {'rtol':>10} {'Linf [K]':>13} {'steps':>8} {'dt range [s]':>24} "
      f"{'CPU [s]':>9}")
ad = {}
for rtol in [1e-3, 1e-4, 1e-5]:
    t0 = time.perf_counter()
    xc, T, dg = integrate(80, 1.0, 0.5, T_EVAL, adaptive=True, rtol=rtol)
    cpu = time.perf_counter() - t0
    _, li = norms(T, xc, T_EVAL, dg["dxc"])
    ad[rtol] = dg
    print(f"  {rtol:>10.0e} {li:>13.4e} {dg['steps']:>8d} "
          f"{dg['hist_dt'].min():>10.4f} - {dg['hist_dt'].max():<11.4f} "
          f"{cpu:>9.3f}")
t0 = time.perf_counter()
xcf, Tf, dgf = integrate(80, 1.0, 0.02, T_EVAL, theta=1.0)
cpuf = time.perf_counter() - t0
_, lif = norms(Tf, xcf, T_EVAL, dgf["dxc"])
print(f"\n  Fixed dt = 0.02 s: Linf = {lif:.4e} K, {dgf['steps']} steps, "
      f"{cpuf:.3f} s")
print(f"  Adaptive rtol=1e-5 reached comparable accuracy in "
      f"{ad[1e-5]['steps']} steps "
      f"({dgf['steps']/ad[1e-5]['steps']:.1f}x fewer)")

# ==============================================================================
# 10. CROSS-CHECKS AND SENSITIVITY
# ==============================================================================
print("\n" + "=" * 78)
print("CROSS-VALIDATION AGAINST EXAMPLES 2.1 AND 2.2")
print("-" * 78)
xc, T_long, dg = integrate(320, 1.0, 200.0, 80.0 * TAU, theta=1.0)
ss = steady(xc)
dx = L / 320
print(f"  Marched to t = 80 tau = {80*TAU:.0f} s on N = 320:")
print(f"    max|T_transient - T_ss(Example 2.1)| = "
      f"{np.max(np.abs(T_long - ss)):.6e} K")
print(f"    predicted constant offset q'''dx^2/8k = "
      f"{QGEN*dx*dx/(8*K):.6e} K")
print(f"    max|T_transient - T_series|          = "
      f"{np.max(np.abs(T_long - exact(xc, 80.0*TAU))):.3e} K")
q_removed = H * (brentq(lambda Ts: K / (0.5 * dx) * (T_long[-1] - Ts)
                        - H * (Ts - T_INF), 300.0, 1200.0, xtol=1e-13)
                 - T_INF)
print(f"    generated {QGEN*L:.6f} vs removed {q_removed:.6f} W/m^2 "
      f"(imbalance {abs(QGEN*L-q_removed):.3e})")

print("\n" + "-" * 78)
print("SENSITIVITY ANALYSIS")
print("-" * 78)
print("  (a) Biot number: steady temperatures and time to reach 99 % of the")
print("      steady mid-plane rise")
print(f"  {'Bi':>8} {'mu_1':>10} {'T_L [K]':>11} {'T_max [K]':>11} "
      f"{'Fo_99':>9}")
bi_rows = []
for Bi in [0.1, 0.3, BI, 1.5, 5.0, 20.0]:
    mu1 = eigenvalues(1, Bi)[0]
    h_eff = Bi * K / L
    T_L = T_INF + QGEN * L / h_eff
    T_mx = T_L + QGEN * L * L / (2.0 * K)
    # The window must cover the SLOWEST case.  The dominant time constant is
    # L^2/(alpha mu_1^2), and mu_1 -> 0 as Bi -> 0, so a poorly cooled plate
    # takes far longer to settle.  Sampling only a few tau and then calling
    # argmax would silently return index 0 when the threshold is never met,
    # reporting a spuriously FAST response for the slowest configuration.
    ts = np.linspace(1.0, 120.0 * TAU, 900)
    Tm = np.array([exact(np.array([0.0]), tt, 80, Bi)[0] for tt in ts])
    frac = (Tm - T_INF) / (T_mx - T_INF)
    reached = np.flatnonzero(frac >= 0.99)
    if reached.size == 0:
        raise RuntimeError(f"99 % never reached for Bi = {Bi}; widen window")
    t99 = ts[reached[0]]
    bi_rows.append((Bi, mu1, T_L, T_mx, ALPHA * t99 / L**2))
    print(f"  {Bi:>8.4f} {mu1:>10.6f} {T_L:>11.3f} {T_mx:>11.3f} "
          f"{ALPHA*t99/L**2:>9.4f}")

print("\n  (b) Conductivity model, steady mid-plane temperature (N = 320)")
print(f"  {'configuration':>30} {'T_max [K]':>12} {'shift [K]':>11}")
base = None
for label, beta in [("k constant (base)", 0.0),
                    ("k linear, beta = 9e-4", 9.0e-4),
                    ("k linear, beta = -9e-4", -9.0e-4)]:
    xcb, Tb, dgb = integrate(320, 1.0, 200.0, 80.0 * TAU, beta=beta,
                             theta=1.0)
    if base is None:
        base = Tb[0]
    print(f"  {label:>30} {Tb[0]:>12.4f} {Tb[0]-base:>+11.4f}")

# ==============================================================================
# 11. FIGURES
# ==============================================================================
times = [5.0, 20.0, 50.0, 150.0, 600.0]
xfine = np.linspace(0.0, L, 400)

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
cmap = plt.cm.plasma(np.linspace(0.05, 0.8, len(times)))
for col, tt in zip(cmap, times):
    ax[0].plot(xfine * 1e3, exact(xfine, tt), "-", lw=1.9, color=col,
               label=rf"$t = {tt:.0f}$ s ($Fo = {ALPHA*tt/L**2:.2f}$)")
    xcn, Tn, _ = integrate(40, 1.0, 0.25, tt, theta=0.5)
    ax[0].plot(xcn * 1e3, Tn, "o", ms=3.6, mfc="none", mew=1.0, color=col)
ax[0].plot(xfine * 1e3, steady(xfine), "k--", lw=1.4, label="steady state")
ax[0].set_xlabel(r"Position $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Startup transient\n(lines exact, symbols FVM $N=40$)")
ax[0].legend(fontsize=7.6, loc="lower left")
ax[0].set_xlim(0, L * 1e3)

_, _, dgU = integrate(30, 1.00, 1.0, 1.0)
_, _, dgG = integrate(30, 1.20, 1.0, 1.0)
ax[1].plot(dgU["xf"] * 1e3, np.ones_like(dgU["xf"]), "|", ms=16,
           color="#2166ac", mew=1.4)
ax[1].plot(dgG["xf"] * 1e3, 0.6 * np.ones_like(dgG["xf"]), "|", ms=16,
           color="#b2182b", mew=1.4)
ax[1].text(L * 1e3 * 0.5, 1.10, "uniform, $r = 1.00$", ha="center",
           color="#2166ac", fontsize=10)
ax[1].text(L * 1e3 * 0.5, 0.70, "graded to surface, $r = 1.20$", ha="center",
           color="#b2182b", fontsize=10)
ax[1].set_ylim(0.3, 1.35)
ax[1].set_yticks([])
ax[1].set_xlabel(r"Position $x$ [mm]")
ax[1].set_title("(b) Control-volume faces, $N = 30$")

for rtol, sty in [(1e-3, "-"), (1e-4, "--"), (1e-5, ":")]:
    d = ad[rtol]
    ax[2].semilogy(d["hist_t"], d["hist_dt"], sty, lw=1.7,
                   label=rf"$rtol = 10^{{{int(np.log10(rtol))}}}$")
ax[2].set_xlabel(r"Time $t$ [s]")
ax[2].set_ylabel(r"Adaptive time step $\Delta t$ [s]")
ax[2].set_title("(c) Adaptive step-size history")
ax[2].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 2.3 -- Transient startup, mesh grading, adaptive stepping",
             fontsize=12.5, y=1.03)
fig.savefig("fig_2_3a_transient.png")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
Ns = np.array([r[0] for r in sp])
dxs = L / Ns
ax[0].loglog(dxs * 1e3, [r[1] for r in sp], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(dxs * 1e3, [r[2] for r in sp], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$")
ax[0].loglog(dxs * 1e3, sp[0][1] * (dxs / dxs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [mm]")
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

Nb = np.array([10, 20, 40, 80, 160])
meas, pred = [], []
for N in Nb:
    xcb, Tb, _ = integrate(N, 1.0, 200.0, 80.0 * TAU, theta=1.0)
    meas.append(Tb[0] - float(steady(np.array([L / (2 * N)]))[0]))
    pred.append(QGEN * (L / N) ** 2 / (8.0 * K))
ax[2].loglog(L / Nb * 1e3, meas, "o", ms=8, mfc="none", mew=1.6,
             color="#b2182b", label="measured offset")
ax[2].loglog(L / Nb * 1e3, pred, "-", lw=1.8, color="#1b7837",
             label=r"predicted $q''''''\Delta x^2/8k$")
ax[2].set_xlabel(r"$\Delta x$ [mm]")
ax[2].set_ylabel("Steady-state offset [K]")
ax[2].set_title("(c) Boundary error, predicted vs measured")
ax[2].legend(fontsize=8.5, loc="lower right")

fig.suptitle("Example 2.3 -- Order of accuracy and the boundary error",
             fontsize=12.5, y=1.02)
fig.savefig("fig_2_3b_orders.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
bis = np.array([r[0] for r in bi_rows])
ax[0].loglog(bis, [r[3] - T_INF for r in bi_rows], "o-", lw=1.9, ms=6,
             color="#b2182b", label=r"$T_{max} - T_\infty$")
ax[0].loglog(bis, [r[2] - T_INF for r in bi_rows], "s--", lw=1.7, ms=6,
             color="#2166ac", label=r"$T_L - T_\infty$")
ax[0].axvline(BI, color="0.45", ls=":", lw=1.3)
ax[0].annotate(rf"design $Bi = {BI:.2f}$", xy=(BI * 1.1, 3), fontsize=9,
               color="0.35", rotation=90)
ax[0].set_xlabel(r"Biot number $Bi = hL/k$")
ax[0].set_ylabel("Temperature rise [K]")
ax[0].set_title("(a) Cooling effectiveness")
ax[0].legend(fontsize=9)

ax[1].semilogx(bis, [r[4] for r in bi_rows], "^-", lw=1.9, ms=6,
               color="#1b7837")
ax[1].axvline(BI, color="0.45", ls=":", lw=1.3)
ax[1].set_xlabel(r"Biot number $Bi = hL/k$")
ax[1].set_ylabel(r"$Fo_{99}$ to reach 99 % of steady rise")
ax[1].set_title("(b) Startup time versus Biot number")

fig.suptitle("Example 2.3 -- Sensitivity analysis", fontsize=12.5, y=1.02)
fig.savefig("fig_2_3c_sensitivity.png")
plt.close(fig)

print("\nFigures written: fig_2_3a_transient.png, fig_2_3b_orders.png, "
      "fig_2_3c_sensitivity.png")
print("=" * 78)
