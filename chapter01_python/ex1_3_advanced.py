"""
================================================================================
 EXAMPLE 1.3 -- ADVANCED TRANSIENT VERIFICATION STUDY
 Research-grade analysis of the Chapter 1 furnace wall
================================================================================

 OBJECTIVE
 ---------
 Examples 1.1 and 1.2 treated the wall at steady state.  Here the storage term
 is restored and the full verification apparatus of computational heat transfer
 is applied to the resulting initial-boundary-value problem:

   * exact eigenfunction (separation-of-variables) solution
   * fully implicit (backward Euler) time integration
   * non-uniform, geometrically graded mesh
   * adaptive time stepping by step doubling with a PI controller
   * separate SPATIAL and TEMPORAL order-of-accuracy measurements
   * Richardson extrapolation and the Grid Convergence Index
   * L2 and Linf error norms, automated error tables
   * CPU timings
   * consistency check against the Example 1.2 steady state
   * sensitivity analysis (Biot number, conductivity law, radiation)

 GOVERNING EQUATION
 ------------------
        dT      d  /     dT \
   rho c -- =  --- | k * -- |          0 < x < L ,  t > 0
        dt      dx \     dx /

 equivalently   dT/dt = alpha d2T/dx2 ,   alpha = k/(rho c)  [m^2/s]

 INITIAL CONDITION
 -----------------
   T(x, 0) = T_init                                        (uniform, cold wall)

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :  T(0, t) = T0                        (furnace lit at t = 0)
   x = L : -k dT/dx|_L = h (T_L - T_inf)        (verification case, eps = 0)
                       [ + eps sigma (T_L^4 - T_sur^4)  in the production case ]

 EXACT SOLUTION (verification case: constant k, linear Robin condition)
 ----------------------------------------------------------------------
 Split the field into the steady state plus a decaying transient,

        T(x,t) = T_ss(x) + theta(x,t) ,   T_ss(x) = T0 - G x ,  G = q''/k

 so that theta satisfies the HOMOGENEOUS boundary conditions

        theta(0,t) = 0 ,   -k dtheta/dx|_L = h theta(L,t)

 Separation of variables gives eigenfunctions X_n(x) = sin(lambda_n x) whose
 eigenvalues follow from the Robin condition.  With mu = lambda L and the Biot
 number Bi = hL/k this is the transcendental equation

        mu cot(mu) = -Bi        <=>       mu cos(mu) + Bi sin(mu) = 0

 which has exactly one root in each interval ((n-1/2)pi, n pi), n = 1, 2, ...
 The second form is used numerically because it is free of the poles of cot.

 The eigenfunctions are orthogonal on [0, L] with unit weight (a regular
 Sturm-Liouville system), so

        theta(x,t) = SUM_n  A_n sin(lambda_n x) exp(-alpha lambda_n^2 t)

        A_n = [ int_0^L f(x) sin(lambda_n x) dx ] / [ int_0^L sin^2(lambda_n x) dx ]

 with f(x) = T_init - T_ss(x) = (T_init - T0) + G x.  Both integrals are
 elementary:

        int_0^L sin(lam x) dx    = (1 - cos(lam L)) / lam
        int_0^L x sin(lam x) dx  = (sin(lam L) - lam L cos(lam L)) / lam^2
        int_0^L sin^2(lam x) dx  = L/2 - sin(2 lam L) / (4 lam)

 They are ALSO evaluated by quadrature in this script as an independent check.

 SYMBOLS (additional to Examples 1.1 and 1.2)
 --------------------------------------------
   rho      [kg/m^3]     density
   c        [J/(kg K)]   specific heat capacity
   alpha    [m^2/s]      thermal diffusivity, alpha = k/(rho c)
   Fo       [-]          Fourier number,  Fo = alpha t / L^2
   lambda_n [1/m]        n-th spatial eigenvalue
   mu_n     [-]          dimensionless eigenvalue, mu_n = lambda_n L
   A_n      [K]          n-th expansion coefficient
   dt       [s]          time step
   a_P0     [W/(m^2 K)]  unsteady (capacitance) coefficient, rho c dx / dt

 OUTPUTS
 -------
   fig_1_3a_transient.png    evolution, mesh, adaptive step history
   fig_1_3b_orders.png       spatial and temporal convergence
   fig_1_3c_sensitivity.png  parametric study

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
L = 0.05                 # [m]           wall thickness
K = 1.4                  # [W/(m K)]     thermal conductivity
RHO = 2000.0             # [kg/m^3]      density
CP = 880.0               # [J/(kg K)]    specific heat
ALPHA = K / (RHO * CP)   # [m^2/s]       thermal diffusivity

H = 25.0                 # [W/(m^2 K)]   convection coefficient
EPS = 0.85               # [-]           emissivity (production case only)
SIGMA = 5.670374419e-8   # [W/(m^2 K^4)]

T0 = 600.0               # [K]  inner-face temperature for t > 0
T_INF = 300.0            # [K]  ambient
T_SUR = 300.0            # [K]  surroundings
T_INIT = 300.0           # [K]  uniform initial temperature

BI = H * L / K                     # [-] Biot number
TAU_DIFF = L * L / ALPHA           # [s] diffusion time scale


# ==============================================================================
# 2. EXACT SOLUTION MACHINERY
# ==============================================================================
@lru_cache(maxsize=None)
def _eigenvalues_cached(n_terms, Bi):
    """First n_terms roots mu_n of  mu cos(mu) + Bi sin(mu) = 0.

    Exactly one root lies in each interval ((n-1/2)pi, n pi).  The endpoints
    are shifted by a small delta to keep a strict sign change at the bracket.
    """
    mus = np.empty(n_terms)
    f = lambda mu: mu * np.cos(mu) + Bi * np.sin(mu)
    d = 1e-10
    for n in range(1, n_terms + 1):
        lo = (n - 0.5) * np.pi + d
        hi = n * np.pi - d
        mus[n - 1] = brentq(f, lo, hi, xtol=1e-14)
    mus.flags.writeable = False
    return mus


def eigenvalues(n_terms, Bi=BI):
    """First n_terms roots mu_n of  mu cos(mu) + Bi sin(mu) = 0.

    Exactly one root lies in each interval ((n-1/2)pi, n pi).  Results are
    cached because the transient solution is evaluated at many times.
    """
    return _eigenvalues_cached(int(n_terms), float(Bi))


def steady_gradient(Bi=BI):
    """Steady-state gradient G = q''/k [K/m] for the LINEAR (eps = 0) case.

    Series resistance: q'' = (T0 - T_inf)/(L/k + 1/h).
    """
    h_eff = Bi * K / L
    q = (T0 - T_INF) / (L / K + 1.0 / h_eff)
    return q / K


def exact_solution(x, t, n_terms=300, Bi=BI):
    """Exact T(x,t) [K] by eigenfunction expansion.  x may be an array."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    G = steady_gradient(Bi)
    T_ss = T0 - G * x
    mus = eigenvalues(n_terms, Bi)
    lams = mus / L

    # closed-form integrals (see module docstring)
    I1 = (1.0 - np.cos(mus)) / lams
    I2 = (np.sin(mus) - mus * np.cos(mus)) / lams**2
    norm = L / 2.0 - np.sin(2.0 * mus) / (4.0 * lams)
    A = ((T_INIT - T0) * I1 + G * I2) / norm

    decay = np.exp(-ALPHA * lams**2 * t)                 # (n_terms,)
    theta = np.sum(A[:, None] * decay[:, None] * np.sin(np.outer(lams, x)),
                   axis=0)
    return T_ss + theta


def verify_coefficients(n_check=6, Bi=BI):
    """Cross-check the closed-form A_n against numerical quadrature."""
    G = steady_gradient(Bi)
    mus = eigenvalues(n_check, Bi)
    lams = mus / L
    f = lambda s: (T_INIT - T0) + G * s
    out = []
    for lam, mu in zip(lams, mus):
        num_q = quad(lambda s: f(s) * np.sin(lam * s), 0.0, L)[0]
        den_q = quad(lambda s: np.sin(lam * s)**2, 0.0, L)[0]
        A_quad = num_q / den_q
        I1 = (1.0 - np.cos(mu)) / lam
        I2 = (np.sin(mu) - mu * np.cos(mu)) / lam**2
        norm = L / 2.0 - np.sin(2.0 * mu) / (4.0 * lam)
        A_form = ((T_INIT - T0) * I1 + G * I2) / norm
        out.append((mu, A_form, A_quad, abs(A_form - A_quad)))
    return out


# ==============================================================================
# 3. NON-UNIFORM MESH GENERATION
# ==============================================================================
def make_mesh(N, ratio=1.0):
    """Geometrically graded cell-centred mesh on [0, L].

    ratio is the ratio of successive control-volume widths.  ratio = 1 recovers
    the uniform mesh; ratio > 1 clusters cells near x = 0, where the thermal
    front is steepest immediately after the step change in surface temperature.

    Returns xc (centres), xf (faces), dxc (cell widths), dxn (centre spacings,
    length N+1, including the two half-cell distances to the boundaries).
    """
    if abs(ratio - 1.0) < 1e-12:
        w = np.ones(N)
    else:
        w = ratio ** np.arange(N)
    w = w / np.sum(w) * L                     # cell widths, sum to L
    xf = np.concatenate(([0.0], np.cumsum(w)))
    xf[-1] = L                                # guard against round-off drift
    xc = 0.5 * (xf[:-1] + xf[1:])
    dxc = np.diff(xf)
    dxn = np.empty(N + 1)
    dxn[0] = xc[0] - xf[0]                    # half-cell, west boundary
    dxn[1:N] = np.diff(xc)                    # centre-to-centre spacings
    dxn[N] = xf[-1] - xc[-1]                  # half-cell, east boundary
    return xc, xf, dxc, dxn


# ==============================================================================
# 4. IMPLICIT FINITE VOLUME STEP
# ==============================================================================
def face_conductance(dxn, kk, i):
    """Conductance [W/(m^2 K)] of the link across interior face i.

    Patankar's harmonic (series-resistance) formula generalised to a
    non-uniform mesh: the two half-distances carry their own conductivities.
    For a uniform mesh with constant k this reduces to k/dx.
    """
    return kk / dxn[i]


def implicit_step(T_old, dt, xc, dxc, dxn, radiation=False, beta=0.0,
                  gamma=0.0, Bi=BI, tol=1e-11, max_inner=50, theta=1.0):
    """Advance one time step of the theta-family.  Returns (T_new, inners).

    Writing the semi-discrete control-volume balance as

        C_P dT_P/dt = F_P(T) ,  C_P = rho c dxc_P   [J/(m^2 K)]
        F_P(T)      = a_W T_W + a_E T_E - a_Ps T_P + S_u
        a_Ps        = a_W + a_E - S_P

    the theta-scheme evaluates the spatial operator as a weighted average of
    the new and the old time level,

        C_P (T_P^new - T_P^old)/dt = theta F_P(T^new) + (1-theta) F_P(T^old)

    which rearranges into the tridiagonal system

        (C_P/dt + theta a_Ps) T_P^new - theta a_W T_W^new - theta a_E T_E^new
              = (C_P/dt) T_P^old + theta S_u + (1-theta) F_P(T^old)

    theta = 1   -> fully implicit (backward Euler): first order in time,
                   unconditionally stable AND unconditionally bounded
    theta = 1/2 -> Crank-Nicolson: second order in time, unconditionally
                   stable but only conditionally bounded (it can ring on a
                   sharp front unless dt is modest)
    theta = 0   -> fully explicit: stable only for dt < dx^2/(2 alpha)

    Both are used below: Crank-Nicolson isolates the SPATIAL error for the
    grid-convergence study, backward Euler demonstrates first-order TEMPORAL
    convergence.  With radiation and/or k(T) the system is nonlinear, so each
    step carries an inner Picard loop; for the linear verification settings
    that loop converges in a single pass, which is checked below.
    """
    N = len(T_old)
    T = T_old.copy()
    h_eff = Bi * K / L                        # allows Biot sensitivity sweeps
    h_r = 0.0                                 # lagged radiative coefficient

    for inner in range(1, max_inner + 1):
        kc = K * (1.0 + beta * (T - T_INF) + gamma * (T - T_INF) ** 2)

        kf = np.empty(N + 1)
        kf[1:N] = 2.0 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])
        kf[0], kf[N] = kc[0], kc[-1]

        aW = np.zeros(N)
        aE = np.zeros(N)
        Sp = np.zeros(N)
        Su = np.zeros(N)

        aW[1:] = face_conductance(dxn, kf[1:N], np.arange(1, N))
        aE[:-1] = face_conductance(dxn, kf[1:N], np.arange(1, N))

        # west: Dirichlet through the half-cell distance dxn[0]
        a_bw = kf[0] / dxn[0]
        Su[0] += a_bw * T0
        Sp[0] -= a_bw

        # east: convection (+ optional radiation), half-cell in series
        if radiation:
            # Extrapolate from the last node to the SURFACE using the previous
            # sweep's h_r, then refresh h_r at that surface temperature.
            g = kf[N] / dxn[N]                    # half-cell conductance
            h_s_prev = h_eff + h_r
            T_sink_prev = (h_eff * T_INF + h_r * T_SUR) / h_s_prev
            T_face = (g * T[-1] + h_s_prev * T_sink_prev) / (g + h_s_prev)
            h_r = EPS * SIGMA * (T_face + T_SUR) * (T_face**2 + T_SUR**2)
        else:
            h_r = 0.0
        h_s = h_eff + h_r
        T_s = (h_eff * T_INF + h_r * T_SUR) / h_s
        U = 1.0 / (dxn[N] / kf[N] + 1.0 / h_s)
        Su[-1] += U * T_s
        Sp[-1] -= U

        aP0 = RHO * CP * dxc / dt          # C_P/dt   [W/(m^2 K)]
        aPs = aW + aE - Sp                 # spatial diagonal a_Ps

        # Explicit contribution F(T_old); neighbours are zero-padded because
        # the boundary volumes carry their links inside Su/Sp instead.
        TW_old = np.concatenate(([0.0], T_old[:-1]))
        TE_old = np.concatenate((T_old[1:], [0.0]))
        F_old = aW * TW_old + aE * TE_old - aPs * T_old + Su

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


def integrate(N, ratio, dt, t_end, radiation=False, beta=0.0, gamma=0.0,
              Bi=BI, adaptive=False, dt_min=1e-3, dt_max=200.0, rtol=1e-4,
              theta=1.0):
    """Integrate to t_end.  Returns (xc, T, diagnostics dict).

    ADAPTIVE STEPPING (step doubling).  Each candidate step is taken twice:
    once with dt and once as two steps of dt/2.  For a first-order method the
    difference between the two estimates is itself a first-order estimate of
    the local error, and Richardson correction gives a second-order answer at
    no extra assembly cost.  A PI controller then sets the next step:

        dt_new = dt * min(2, max(0.3, 0.9 * (tol_local/err)^(1/2)))

    The exponent 1/2 (rather than 1) is the standard damping for first-order
    step-doubling controllers and avoids oscillation of the step sequence.
    """
    xc, xf, dxc, dxn = make_mesh(N, ratio)
    T = np.full(N, T_INIT)
    t = 0.0
    steps = 0
    inner_total = 0
    hist_t, hist_dt, hist_err = [], [], []

    while t < t_end - 1e-12:
        dt_try = min(dt, t_end - t)

        if not adaptive:
            T, ni = implicit_step(T, dt_try, xc, dxc, dxn, radiation,
                                  beta, gamma, Bi, theta=theta)
            inner_total += ni
            t += dt_try
            steps += 1
            continue

        while True:
            dt_try = min(dt_try, t_end - t)
            T_big, n1 = implicit_step(T, dt_try, xc, dxc, dxn, radiation,
                                      beta, gamma, Bi, theta=1.0)
            T_h, n2 = implicit_step(T, 0.5 * dt_try, xc, dxc, dxn, radiation,
                                    beta, gamma, Bi, theta=1.0)
            T_small, n3 = implicit_step(T_h, 0.5 * dt_try, xc, dxc, dxn,
                                        radiation, beta, gamma, Bi, theta=1.0)
            inner_total += n1 + n2 + n3

            err = np.max(np.abs(T_small - T_big))          # local error [K]
            scale = rtol * max(1.0, np.max(np.abs(T_small)))
            if err <= scale or dt_try <= dt_min * 1.000001:
                # Richardson-corrected update: 2*T_small - T_big is O(dt^2)
                T = 2.0 * T_small - T_big
                t += dt_try
                steps += 1
                hist_t.append(t)
                hist_dt.append(dt_try)
                hist_err.append(err)
                fac = 0.9 * np.sqrt(scale / max(err, 1e-300))
                dt = float(np.clip(dt_try * np.clip(fac, 0.3, 2.0),
                                   dt_min, dt_max))
                break
            dt_try = max(dt_min, 0.5 * dt_try)

    diag = {"steps": steps, "inner_total": inner_total,
            "inner_per_step": inner_total / max(steps, 1), "xf": xf, "dxc": dxc,
            "hist_t": np.array(hist_t), "hist_dt": np.array(hist_dt),
            "hist_err": np.array(hist_err)}
    return xc, T, diag


def surface_temperature(T, dxc, radiation, beta=0.0, gamma=0.0, Bi=BI):
    """Outer-face temperature T_L [K] consistent with the discrete solution.

    Solves the half-cell energy balance exactly (no linearisation):

        (k_N / (dxc_N/2)) (T_N - T_L) = h (T_L - T_inf)
                                        [+ eps sigma (T_L^4 - T_sur^4)]
    """
    h_eff = Bi * K / L
    u = T[-1] - T_INF
    k_N = K * (1.0 + beta * u + gamma * u * u)
    g = k_N / (0.5 * dxc[-1])

    def res(TL):
        out = g * (T[-1] - TL) - h_eff * (TL - T_INF)
        if radiation:
            out -= EPS * SIGMA * (TL**4 - T_SUR**4)
        return out

    lo = min(T_INF, T_SUR, T[-1]) - 1.0
    hi = max(T0, T_INF, T_SUR, T[-1]) + 1.0
    return brentq(res, lo, hi, xtol=1e-13)


def error_norms(T, xc, t, dxc, Bi=BI):
    """Volume-weighted L2 and Linf error norms against the exact solution [K]."""
    Te = exact_solution(xc, t, Bi=Bi)
    e = T - Te
    l2 = np.sqrt(np.sum(e**2 * dxc) / L)
    return l2, np.max(np.abs(e))


# ==============================================================================
# 5. HEADER AND EXACT-SOLUTION VERIFICATION
# ==============================================================================
print("=" * 78)
print("EXAMPLE 1.3 -- ADVANCED TRANSIENT VERIFICATION")
print("=" * 78)
print(f"  alpha = {ALPHA:.6e} m^2/s     Bi = {BI:.6f}     "
      f"tau_diff = L^2/alpha = {TAU_DIFF:.2f} s")
print(f"  Steady gradient G = {steady_gradient():.6f} K/m,  "
      f"q'' = {K*steady_gradient():.6f} W/m^2")
print("-" * 78)
print("  Eigenvalue check:  mu_n cos(mu_n) + Bi sin(mu_n) = 0")
mus6 = eigenvalues(6)
print(f"  {'n':>3} {'mu_n':>14} {'residual':>13} {'bracket':>22}")
for i, mu in enumerate(mus6, start=1):
    r = mu * np.cos(mu) + BI * np.sin(mu)
    print(f"  {i:>3d} {mu:>14.10f} {r:>13.2e} "
          f"  ({(i-0.5)*np.pi:7.4f}, {i*np.pi:7.4f})")

print("\n  Expansion coefficients: closed form vs numerical quadrature")
print(f"  {'mu_n':>14} {'A_n closed [K]':>17} {'A_n quad [K]':>16} {'diff':>11}")
for mu, Af, Aq, d in verify_coefficients():
    print(f"  {mu:>14.10f} {Af:>17.10f} {Aq:>16.10f} {d:>11.2e}")

# Does the series reproduce the initial condition and the steady limit?
x_probe = np.linspace(0.0, L, 11)[1:]        # exclude the singular point x=0
T_early = exact_solution(x_probe, 1e-6)
T_late = exact_solution(x_probe, 50.0 * TAU_DIFF)
T_ss_ref = T0 - steady_gradient() * x_probe
print("\n  Behaviour as t -> 0.  The data are INCOMPATIBLE at the corner")
print("  (x,t) = (0,0): the initial condition gives 300 K there while the")
print("  boundary condition demands 600 K.  The eigenfunction series therefore")
print("  converges pointwise for every t > 0 but exhibits Gibbs ringing near")
print("  x = 0 as t -> 0, decaying as more terms are retained:")
for nt in [100, 300, 1000, 3000]:
    e_g = np.max(np.abs(exact_solution(x_probe, 1e-6, n_terms=nt) - T_INIT))
    print(f"    n_terms = {nt:>5d} : max|T - T_init| = {e_g:.4e} K")
print(f"  Away from the corner (x >= L/4) with 300 terms: "
      f"{np.max(np.abs(exact_solution(x_probe[x_probe >= L/4], 1e-6) - T_INIT)):.3e} K")
print(f"  Series at t -> inf : max|T - T_ss|   = "
      f"{np.max(np.abs(T_late - T_ss_ref)):.3e} K")

# ==============================================================================
# 6. SPATIAL ORDER OF ACCURACY (uniform mesh, tiny fixed dt)
# ==============================================================================
T_EVAL = 300.0                    # [s] evaluation time (Fo = 0.0955)
print("\n" + "=" * 78)
print(f"SPATIAL CONVERGENCE at t = {T_EVAL:.0f} s "
      f"(Fo = {ALPHA*T_EVAL/L**2:.4f}), uniform mesh")
print("  Crank-Nicolson (theta = 1/2), dt = 0.05 s: the temporal error is")
print("  O(dt^2) ~ 1e-6 K here, so the measured order is purely SPATIAL.")
print("-" * 78)
print(f"  {'N':>5} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} {'p_inf':>7} "
      f"{'steps':>7} {'CPU [s]':>9}")
sp_rows = []
for N in [10, 20, 40, 80, 160]:
    t0 = time.perf_counter()
    xc, T, dg = integrate(N, 1.0, 0.05, T_EVAL, theta=0.5)
    cpu = time.perf_counter() - t0
    l2, linf = error_norms(T, xc, T_EVAL, dg["dxc"])
    if sp_rows:
        p2 = np.log(sp_rows[-1][1] / l2) / np.log(2.0)
        pi_ = np.log(sp_rows[-1][2] / linf) / np.log(2.0)
    else:
        p2 = pi_ = float("nan")
    sp_rows.append((N, l2, linf, p2, pi_, dg["steps"], cpu))
    print(f"  {N:>5d} {l2:>13.4e} {p2:>7.3f} {linf:>13.4e} {pi_:>7.3f} "
          f"{dg['steps']:>7d} {cpu:>9.3f}")

# Richardson on the three finest spatial grids (probe: mid-plane temperature)
def midplane(N, dt=0.05, ratio=1.0):
    xc, T, dg = integrate(N, ratio, dt, T_EVAL, theta=0.5)
    return float(np.interp(L / 2.0, xc, T))

m1, m2, m3 = midplane(160), midplane(80), midplane(40)   # fine -> coarse
p_sp = np.log(abs((m3 - m2) / (m2 - m1))) / np.log(2.0)
m_rich = m1 + (m1 - m2) / (2.0**p_sp - 1.0)
GCI_sp = 1.25 * abs((m1 - m2) / m1) / (2.0**p_sp - 1.0) * 100.0
m_exact = float(exact_solution(np.array([L / 2.0]), T_EVAL)[0])
print("\n  Richardson extrapolation, mid-plane temperature T(L/2, t):")
print(f"    N= 40 : {m3:.10f} K")
print(f"    N= 80 : {m2:.10f} K")
print(f"    N=160 : {m1:.10f} K")
print(f"    observed spatial order p = {p_sp:.4f}")
print(f"    extrapolated             = {m_rich:.10f} K")
print(f"    exact                    = {m_exact:.10f} K")
print(f"    |extrapolated - exact|   = {abs(m_rich - m_exact):.3e} K")
print(f"    |finest      - exact|    = {abs(m1 - m_exact):.3e} K")
print(f"    GCI_fine                 = {GCI_sp:.6f} %")

# ==============================================================================
# 7. TEMPORAL ORDER OF ACCURACY (fine fixed mesh)
# ==============================================================================
print("\n" + "=" * 78)
print("TEMPORAL CONVERGENCE at t = 300 s, N = 400")
print("  Errors are measured against the SAME mesh at dt = 0.05 s (theta=1/2),")
print("  so the common spatial error cancels and only the TEMPORAL error is")
print("  seen.  Backward Euler (theta = 1) is formally FIRST order in time.")
print("-" * 78)
# The spatial error on N = 400 is ~5e-4 K and is COMMON to every run below,
# so it is removed by measuring each result against the same mesh integrated
# with a very small Crank-Nicolson step.  What remains is the temporal error.
xc_ref, T_ref, _ = integrate(400, 1.0, 0.05, T_EVAL, theta=0.5)

print(f"  {'dt [s]':>9} {'Linf [K]':>14} {'p_t':>8} {'steps':>8} {'CPU [s]':>9}")
tm_rows = []
for dt in [8.0, 4.0, 2.0, 1.0, 0.5]:
    t0 = time.perf_counter()
    _, T_be, dg_be = integrate(400, 1.0, dt, T_EVAL, theta=1.0)
    cpu = time.perf_counter() - t0
    e_be = np.max(np.abs(T_be - T_ref))
    p_be = (np.log(tm_rows[-1][1] / e_be) / np.log(2.0)) if tm_rows else float("nan")
    tm_rows.append((dt, e_be, p_be, dg_be["steps"], cpu))
    print(f"  {dt:>9.3f} {e_be:>14.4e} {p_be:>8.3f} {dg_be['steps']:>8d} "
          f"{cpu:>9.3f}")

# --- Boundedness: what theta really buys you --------------------------------
# The exact solution is monotone in time and confined to [T_INIT, T0].  A
# scheme is BOUNDED if the discrete solution respects that envelope.  Backward
# Euler does so for every dt; Crank-Nicolson does not, and rings near the
# front when dt exceeds roughly dx^2/(2 alpha) -- the same threshold that
# limits the explicit scheme, even though CN is unconditionally STABLE.
# Stability and boundedness are different properties: this is the practical
# reason Patankar recommends the fully implicit scheme for general use.
print("\n  Boundedness check at t = 32 s, N = 100.  Because the wall is heated")
print("  monotonically from one side, the exact profile T(x) is strictly")
print("  DECREASING in x at every instant.  A bounded scheme reproduces that;")
print("  an unbounded one produces wiggles.  Reported is the largest positive")
print("  jump  max_i (T_{i+1} - T_i),  which must be <= 0 for a monotone field.")
dx_probe = L / 100
dt_stab = dx_probe**2 / (2.0 * ALPHA)
print(f"  Explicit stability limit dx^2/(2 alpha) = {dt_stab:.3f} s")
print(f"\n  {'dt [s]':>9} {'dt/dt_stab':>12} {'BE violation [K]':>18} "
      f"{'CN violation [K]':>18}")
bnd_rows = []
for dt in [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]:
    _, T_be_b, _ = integrate(100, 1.0, dt, 32.0, theta=1.0)
    _, T_cn_b, _ = integrate(100, 1.0, dt, 32.0, theta=0.5)
    v_be = max(0.0, float(np.max(np.diff(T_be_b))))
    v_cn = max(0.0, float(np.max(np.diff(T_cn_b))))
    bnd_rows.append((dt, v_be, v_cn))
    print(f"  {dt:>9.3f} {dt/dt_stab:>12.2f} {v_be:>18.3e} {v_cn:>18.3e}")
print("\n  Backward Euler satisfies the discrete maximum principle for every")
print("  dt (all of its coefficients remain positive), so its violation column")
print("  is identically zero.  Crank-Nicolson is unconditionally STABLE but")
print("  not unconditionally BOUNDED: the coefficient (C_P/dt - (1-theta)a_Ps)")
print("  multiplying the OLD time level changes sign once dt grows large, and")
print("  the scheme then rings.  For this problem the onset is observed")
print("  between dt/dt_stab ~ 13 and ~25, i.e. an order of magnitude ABOVE the")
print("  explicit stability limit -- the boundedness threshold is a genuine")
print("  restriction but a far weaker one than explicit stability.  Stability")
print("  and boundedness are distinct properties, which is the practical")
print("  reason Patankar recommends the fully implicit scheme as the default.")

# ==============================================================================
# 8. NON-UNIFORM MESH AND ADAPTIVE TIME STEPPING
# ==============================================================================
print("\n" + "=" * 78)
print("NON-UNIFORM MESH AND ADAPTIVE TIME STEPPING")
print("-" * 78)
print(f"  {'mesh':>26} {'N':>5} {'L2 [K]':>12} {'Linf [K]':>12} "
      f"{'steps':>7} {'CPU [s]':>9}")
for label, ratio in [("uniform (r = 1.00)", 1.00),
                     ("graded  (r = 1.05)", 1.05),
                     ("graded  (r = 1.10)", 1.10)]:
    t0 = time.perf_counter()
    xc, T, dg = integrate(40, ratio, 0.05, T_EVAL, theta=0.5)
    cpu = time.perf_counter() - t0
    l2, linf = error_norms(T, xc, T_EVAL, dg["dxc"], BI)
    print(f"  {label:>26} {40:>5d} {l2:>12.4e} {linf:>12.4e} "
          f"{dg['steps']:>7d} {cpu:>9.3f}")

print("\n  Adaptive stepping (step doubling + Richardson correction), N = 80:")
print(f"  {'rtol':>10} {'Linf [K]':>13} {'steps':>7} {'dt_min [s]':>12} "
      f"{'dt_max [s]':>12} {'CPU [s]':>9}")
ad_runs = {}
for rtol in [1e-2, 1e-3, 1e-4, 1e-5]:
    t0 = time.perf_counter()
    xc, T, dg = integrate(80, 1.0, 0.5, T_EVAL, adaptive=True, rtol=rtol)
    cpu = time.perf_counter() - t0
    _, linf = error_norms(T, xc, T_EVAL, dg["dxc"])
    ad_runs[rtol] = dg
    print(f"  {rtol:>10.0e} {linf:>13.4e} {dg['steps']:>7d} "
          f"{dg['hist_dt'].min():>12.4f} {dg['hist_dt'].max():>12.4f} "
          f"{cpu:>9.3f}")

t0 = time.perf_counter()
xc_f, T_f, dg_f = integrate(80, 1.0, 0.01, T_EVAL)
cpu_fixed = time.perf_counter() - t0
_, linf_fixed = error_norms(T_f, xc_f, T_EVAL, dg_f["dxc"])
print(f"\n  Fixed dt = 0.01 s reference: Linf = {linf_fixed:.4e} K, "
      f"{dg_f['steps']} steps, {cpu_fixed:.3f} s")
best = ad_runs[1e-5]
print(f"  Adaptive rtol = 1e-5      : {best['steps']} steps "
      f"({dg_f['steps']/best['steps']:.1f}x fewer)")

# ==============================================================================
# 9. CONSISTENCY WITH EXAMPLE 1.2 (steady limit) AND FLUX EXACTNESS
# ==============================================================================
print("\n" + "=" * 78)
print("CONSISTENCY WITH EXAMPLE 1.2")
print("-" * 78)
xc, T_long, dg = integrate(80, 1.0, 2000.0, 40.0 * TAU_DIFF)
T_ss_series = exact_solution(xc, 40.0 * TAU_DIFF)
G = steady_gradient()
print(f"  Marching to t = 40 tau_diff = {40*TAU_DIFF:.0f} s and comparing with")
print("  the steady analytical profile of Example 1.1 (eps = 0 case):")
print(f"    max|T_transient - T_steady|  = "
      f"{np.max(np.abs(T_long - (T0 - G*xc))):.3e} K")
print(f"    max|T_transient - T_series|  = "
      f"{np.max(np.abs(T_long - T_ss_series)):.3e} K")

print("\n  Wall-flux behaviour under mesh refinement (steady limit):")
print("  Tests the Example 1.2 observation that a conductivity law LINEAR in T")
print("  yields the exact flux on every mesh, while a QUADRATIC law does not.")


def kirchhoff_psi(T, beta, gamma):
    """psi(T) = int_{T_INF}^{T} k(T') dT'  [W/m] for the quadratic k model."""
    u = T - T_INF
    return K * (u + 0.5 * beta * u**2 + gamma * u**3 / 3.0)


def exact_steady_flux(beta, gamma):
    """Exact steady flux [W/m^2] with convection only, via the Kirchhoff
    potential (psi is linear in x) plus a scalar root-find on the surface
    energy balance."""
    psi0 = kirchhoff_psi(T0, beta, gamma)

    def T_of_psi(p):
        return brentq(lambda TT: kirchhoff_psi(TT, beta, gamma) - p,
                      T_INF - 100.0, T0 + 100.0, xtol=1e-13)

    def res(q):
        return q - H * (T_of_psi(psi0 - q * L) - T_INF)

    return brentq(res, 1.0, psi0 / L, xtol=1e-12)


print(f"  {'conductivity law':>26} {'N':>5} {'q_wall [W/m^2]':>17} "
      f"{'error':>12} {'err ratio':>10} {'order':>7}")
for label, beta, gamma in [("k = const", 0.0, 0.0),
                           ("k linear   (b=1.2e-3)", 1.2e-3, 0.0),
                           ("k quadratic (g=2e-6)", 1.2e-3, 2.0e-6)]:
    q_ex = exact_steady_flux(beta, gamma)
    prev = None
    for N in [20, 40, 80, 160]:
        xc_q, Tq, dgq = integrate(N, 1.0, 2000.0, 60.0 * TAU_DIFF,
                                  beta=beta, gamma=gamma)
        u0 = Tq[0] - T_INF
        kc0 = K * (1.0 + beta * u0 + gamma * u0**2)
        q_wall = kc0 * (T0 - Tq[0]) / (0.5 * dgq["dxc"][0])
        err = abs(q_wall - q_ex)
        # Anything within ~1e-12 relative of the exact value is round-off, not
        # discretisation error, and no order may be inferred from it.
        at_roundoff = err < 1e-10 * max(1.0, abs(q_ex))
        if at_roundoff:
            ratio_s, order_s = "round-off", "exact"
        elif prev is None:
            ratio_s, order_s = "-", "-"
        else:
            ratio_s = f"{prev/err:.2f}"
            order_s = f"{np.log(prev/err)/np.log(2.0):.2f}"
        print(f"  {label if N == 20 else '':>26} {N:>5d} {q_wall:>17.9f} "
              f"{err:>12.3e} {ratio_s:>10} {order_s:>7}")
        prev = err
    print(f"  {'exact:':>26} {'':>5} {q_ex:>17.9f}")

# ==============================================================================
# 10. SENSITIVITY ANALYSIS
# ==============================================================================
print("\n" + "=" * 78)
print("SENSITIVITY ANALYSIS")
print("-" * 78)
print("  (a) Biot number -- effect on the steady outer-face temperature and on")
print("      the time to reach 99 % of the steady mid-plane temperature")
print(f"  {'Bi':>8} {'mu_1':>10} {'T_L,ss [K]':>12} {'t_99 [s]':>11} "
      f"{'Fo_99':>9}")
bi_list = [0.05, 0.1, 0.5, BI, 2.0, 10.0, 100.0]
bi_rows = []
for Bi in bi_list:
    mu1 = eigenvalues(1, Bi)[0]
    G_b = steady_gradient(Bi)
    TL = T0 - G_b * L
    ts = np.linspace(1.0, 8.0 * TAU_DIFF, 240)
    Tm = np.array([exact_solution(np.array([L / 2.0]), tt, 80, Bi)[0]
                   for tt in ts])
    Tm_ss = T0 - G_b * (L / 2.0)
    frac = (Tm - T_INIT) / (Tm_ss - T_INIT)
    idx = np.argmax(frac >= 0.99)
    t99 = ts[idx]
    bi_rows.append((Bi, mu1, TL, t99))
    print(f"  {Bi:>8.4f} {mu1:>10.6f} {TL:>12.4f} {t99:>11.2f} "
          f"{ALPHA*t99/L**2:>9.4f}")

print("\n  (b) Effect of radiation and of the conductivity law on the")
print("      steady outer-face temperature (N = 160, marched to steady state)")
print(f"  {'configuration':>34} {'T_L [K]':>12} {'shift vs base':>15}")
base_TL = None
for label, rad, beta, gamma in [
        ("convection only, k const  (base)", False, 0.0, 0.0),
        ("convection only, k linear", False, 1.2e-3, 0.0),
        ("convection only, k quadratic", False, 1.2e-3, 2.0e-6),
        ("convection + radiation, k const", True, 0.0, 0.0),
        ("convection + radiation, k linear", True, 1.2e-3, 0.0)]:
    xc_s, T_s, dg_s = integrate(160, 1.0, 2000.0, 60.0 * TAU_DIFF,
                                radiation=rad, beta=beta, gamma=gamma)
    TL_s = surface_temperature(T_s, dg_s["dxc"], rad, beta, gamma)
    if base_TL is None:
        base_TL = TL_s
    print(f"  {label:>34} {TL_s:>12.4f} {TL_s - base_TL:>+15.4f}")
    if label == "convection only, k const  (base)":
        TL_ref_11 = T0 - steady_gradient() * L
        print(f"  {'-> Example 1.1 analytical value:':>34} {TL_ref_11:>12.4f} "
              f"{'':>15}")
        print(f"  {'-> agreement:':>34} {abs(TL_s - TL_ref_11):>12.2e} K")

# ==============================================================================
# 11. FIGURES
# ==============================================================================
times = [30.0, 100.0, 300.0, 1000.0, 3000.0]
xc40, _, dg40 = integrate(40, 1.0, 0.01, 1.0)
xfine = np.linspace(1e-9, L, 400)

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))

cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(times)))
for col, tt in zip(cmap, times):
    ax[0].plot(xfine * 1e3, exact_solution(xfine, tt), "-", lw=1.9, color=col,
               label=f"$t = {tt:.0f}$ s ($Fo = {ALPHA*tt/L**2:.3f}$)")
    xcn, Tn, _ = integrate(40, 1.0, 0.5, tt)
    ax[0].plot(xcn * 1e3, Tn, "o", ms=3.6, mfc="none", mew=1.0, color=col)
ax[0].plot(xfine * 1e3, T0 - steady_gradient() * xfine, "k--", lw=1.4,
           label="steady state")
ax[0].set_xlabel(r"Position $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Transient evolution\n(lines exact, symbols FVM $N=40$)")
ax[0].legend(fontsize=7.6, loc="upper right")
ax[0].set_xlim(0, L * 1e3)

_, _, dgU = integrate(30, 1.00, 0.01, 1.0)
_, _, dgG = integrate(30, 1.10, 0.01, 1.0)
ax[1].plot(dgU["xf"] * 1e3, np.zeros_like(dgU["xf"]) + 1.0, "|", ms=16,
           color="#2166ac", mew=1.4)
ax[1].plot(dgG["xf"] * 1e3, np.zeros_like(dgG["xf"]) + 0.6, "|", ms=16,
           color="#b2182b", mew=1.4)
ax[1].text(L * 1e3 * 0.5, 1.10, "uniform, $r = 1.00$", ha="center",
           color="#2166ac", fontsize=10)
ax[1].text(L * 1e3 * 0.5, 0.70, "graded, $r = 1.10$", ha="center",
           color="#b2182b", fontsize=10)
ax[1].set_ylim(0.3, 1.35)
ax[1].set_yticks([])
ax[1].set_xlabel(r"Position $x$ [mm]")
ax[1].set_title("(b) Control-volume faces, $N = 30$")

for rtol, style in [(1e-3, "-"), (1e-4, "--"), (1e-5, ":")]:
    d = ad_runs[rtol]
    ax[2].semilogy(d["hist_t"], d["hist_dt"], style, lw=1.7,
                   label=rf"$rtol = 10^{{{int(np.log10(rtol))}}}$")
ax[2].set_xlabel(r"Time $t$ [s]")
ax[2].set_ylabel(r"Adaptive time step $\Delta t$ [s]")
ax[2].set_title("(c) Adaptive step-size history")
ax[2].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 1.3 -- Transient solution, mesh grading, adaptive stepping",
             fontsize=12.5, y=1.03)
fig.savefig("fig_1_3a_transient.png")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))

Ns = np.array([r[0] for r in sp_rows])
dxs = L / Ns
ax[0].loglog(dxs * 1e3, [r[1] for r in sp_rows], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(dxs * 1e3, [r[2] for r in sp_rows], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$")
ref = sp_rows[0][1] * (dxs / dxs[0])**2
ax[0].loglog(dxs * 1e3, ref, "k--", lw=1.3, label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [mm]")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title(f"(a) Spatial convergence ($p = {p_sp:.2f}$)")
ax[0].legend(fontsize=9, loc="lower right")

dts = np.array([r[0] for r in tm_rows])
ax[1].loglog(dts, [r[1] for r in tm_rows], "o-", lw=1.8, ms=6,
             color="#762a83", label=r"backward Euler ($\theta = 1$)")
ref_t = tm_rows[0][1] * (dts / dts[0])**1
ax[1].loglog(dts, ref_t, "k--", lw=1.3, label="slope 1")
ax[1].set_xlabel(r"$\Delta t$ [s]")
ax[1].set_ylabel("Error norm [K]")
ax[1].set_title("(b) Temporal convergence")
ax[1].legend(fontsize=9, loc="lower right")

bd = np.array(bnd_rows)
# Linear ordinate: the violations are exactly zero over most of the range,
# which a logarithmic axis cannot display.
ax[2].semilogx(bd[:, 0], bd[:, 2], "s-", lw=1.9, ms=7, color="#1b7837",
               label=r"Crank-Nicolson ($\theta = 1/2$)", zorder=3)
ax[2].semilogx(bd[:, 0], bd[:, 1], "o-", lw=1.9, ms=6, color="#762a83",
               label=r"backward Euler ($\theta = 1$)", zorder=4)
ax[2].axhline(0.0, color="k", lw=1.0, zorder=1)
ax[2].axvline(dt_stab, color="0.45", ls="--", lw=1.2, zorder=1)
ax[2].annotate(r"explicit limit $\Delta x^2/2\alpha$",
               xy=(dt_stab * 1.3, 0.06 * bd[:, 2].max()), fontsize=8,
               color="0.35", rotation=90)
ax[2].fill_between([bd[:, 0].min(), bd[:, 0].max()], 0, -8,
                   color="0.85", alpha=0.5, zorder=0)
ax[2].text(bd[:, 0].min() * 1.15, -5.0, "monotone (physical)", fontsize=8.5,
           color="0.35")
ax[2].set_ylim(-8, 1.18 * bd[:, 2].max())
ax[2].set_xlabel(r"Time step $\Delta t$ [s]")
ax[2].set_ylabel(r"$\max_i\,(T_{i+1} - T_i)$ [K]")
ax[2].set_title("(c) Boundedness violation, $N = 100$, $t = 32$ s")
ax[2].legend(fontsize=8.5, loc="upper left")

fig.suptitle("Example 1.3 -- Order-of-accuracy and boundedness verification",
             fontsize=12.5, y=1.02)
fig.savefig("fig_1_3b_orders.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

bis = np.array([r[0] for r in bi_rows])
ax[0].semilogx(bis, [r[2] for r in bi_rows], "o-", lw=1.9, ms=6,
               color="#b2182b")
ax[0].axvline(BI, color="0.4", ls="--", lw=1.1)
ax[0].annotate(rf"design $Bi = {BI:.2f}$", xy=(BI, 480), fontsize=9,
               color="0.3", rotation=90, ha="right")
ax[0].set_xlabel(r"Biot number $Bi = hL/k$ [-]")
ax[0].set_ylabel(r"Steady outer-face temperature $T_{L,ss}$ [K]")
ax[0].set_title("(a) Sensitivity to the Biot number")

ax[1].semilogx(bis, [ALPHA * r[3] / L**2 for r in bi_rows], "s-", lw=1.9,
               ms=6, color="#2166ac")
ax[1].axvline(BI, color="0.4", ls="--", lw=1.1)
ax[1].set_xlabel(r"Biot number $Bi = hL/k$ [-]")
ax[1].set_ylabel(r"$Fo_{99}$ to reach 99 % of steady mid-plane $T$ [-]")
ax[1].set_title("(b) Response time versus Biot number")

fig.suptitle("Example 1.3 -- Sensitivity analysis", fontsize=12.5, y=1.02)
fig.savefig("fig_1_3c_sensitivity.png")
plt.close(fig)

print("\nFigures written: fig_1_3a_transient.png, fig_1_3b_orders.png, "
      "fig_1_3c_sensitivity.png")
print("=" * 78)
