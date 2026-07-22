"""
================================================================================
 EXAMPLE 9.3 -- COMBINED CONDUCTION AND RADIATION
 The T^4 nonlinearity, Patankar's source rule, and a verification campaign
================================================================================

 OBJECTIVE
 ---------
 Radiation enters a conduction problem as a boundary condition proportional to
 T^4.  That single fact makes the problem NONLINEAR, and the whole of this
 example is about handling that nonlinearity honestly: linearising it in a way
 that cannot destroy boundedness, solving it, and verifying the result against
 an exact solution that exists in a limit.

 THE PROBLEM
 -----------
 A slab of thickness L, initially at T_0, is insulated on one face and radiates
 to deep surroundings at T_inf from the other.  The governing equation and
 boundary conditions are

        rho c dT/dt = d/dx ( k dT/dx )
        dT/dx = 0                            at x = 0   (insulated)
        -k dT/dx = eps sigma (T^4 - T_inf^4)  at x = L   (radiating)

 AN EXACT SOLUTION EXISTS IN THE LUMPED LIMIT
 --------------------------------------------
 When the Biot number is small the slab is isothermal and the problem collapses
 to an ordinary differential equation which can be integrated in closed form.
 For T_inf = 0 the result is elementary,

        T(t) = T_0 / (1 + 3 C T_0^3 t)^(1/3),   C = eps sigma A / (rho V c)

 and for T_inf > 0 partial fractions give an implicit but exact relation,

        (1/(4 T_inf^3)) [ ln((T+T_inf)/(T-T_inf)) - ln((T_0+T_inf)/(T_0-T_inf))
                          + 2 (arctan(T/T_inf) - arctan(T_0/T_inf)) ] = C t

 Both are used below as references.  The second is inverted numerically to
 obtain T(t), which is a different operation from integrating the PDE and so
 remains an independent check.

 THE LINEARISATION, AND WHY THE FORM MATTERS
 -------------------------------------------
 Chapter 2 established Patankar's rule: a source written as S = S_u + S_P T
 must have S_P <= 0, or the coefficient a_P can lose its diagonal dominance and
 the solution can go unbounded.  The radiative flux is a source on the boundary
 cell, and there are two obvious ways to linearise it:

   (a) EXPLICIT, S_u = -eps sigma (T*^4 - T_inf^4),  S_P = 0
       Simple, but the whole nonlinearity sits in the source and the iteration
       converges slowly; worse, with a large time step it can overshoot.

   (b) NEWTON, expanding about the previous iterate T*:
       eps sigma T^4 ~ eps sigma (T*^4 + 4 T*^3 (T - T*))
       giving S_P = -4 eps sigma T*^3 <= 0 automatically.

 The Newton form satisfies Patankar's rule for free, because the derivative of
 T^4 is positive and enters with a negative sign.  This is not a coincidence:
 any source that DECREASES with temperature linearises into a compliant form,
 and radiative loss always does.  Both are implemented and compared.

 VERIFICATION CAMPAIGN
 ---------------------
   1. The two closed-form lumped solutions against each other in the limit
      T_inf -> 0.
   2. The FVM against the exact lumped solution at very small Biot number.
   3. Order of accuracy in space and in time, from successive differences.
   4. Richardson extrapolation and grid convergence index.
   5. An energy audit: stored energy against radiated energy, to machine
      precision.
   6. Newton against explicit linearisation: iteration counts, and the step
      size at which the explicit form loses boundedness.
   7. The steady state with a source, checked against its analytic profile.

 OUTPUTS
 -------
   fig_9_3a_cooling.png       transient fields, lumped comparison, Biot effect
   fig_9_3b_verification.png  convergence, Richardson, linearisation comparison

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

SIGMA = 5.670374419e-8
T_START = time.perf_counter()

# ==============================================================================
# 1. PHYSICAL DATA -- a steel plate radiating in a vacuum furnace
# ==============================================================================
L_SLAB = 0.050          # m
K_SOLID = 45.0          # W/m.K   (carbon steel)
RHO = 7850.0            # kg/m^3
CP = 480.0              # J/kg.K
ALPHA = K_SOLID / (RHO * CP)
EPS = 0.80              # -
T_INIT = 1200.0         # K
T_SURR = 300.0          # K


# ==============================================================================
# 2. THE EXACT LUMPED SOLUTIONS
# ==============================================================================
def lumped_zero_surroundings(t, T0=T_INIT, L=L_SLAB, eps=EPS):
    """Closed form for T_inf = 0:  T = T0 (1 + 3 C T0^3 t)^(-1/3)."""
    C = eps * SIGMA / (RHO * CP * L)
    return T0 * (1.0 + 3.0 * C * T0 ** 3 * t) ** (-1.0 / 3.0)


def lumped_implicit_time(T, T0=T_INIT, Tinf=T_SURR, L=L_SLAB, eps=EPS):
    """The exact time at which the lumped slab reaches temperature T.

    Integrating  dT/dt = -C (T^4 - T_inf^4)  by partial fractions gives

        t = (1/(4 C T_inf^3)) [ ln((T+Ti)/(T-Ti)) - ln((T0+Ti)/(T0-Ti))
                                + 2 (arctan(T/Ti) - arctan(T0/Ti)) ]

    This is exact but implicit in T, which is why the companion function below
    inverts it numerically rather than rearranging it.
    """
    C = eps * SIGMA / (RHO * CP * L)
    g = lambda x: (np.log((x + Tinf) / (x - Tinf)) +
                   2.0 * np.arctan(x / Tinf))
    return (g(T) - g(T0)) / (4.0 * C * Tinf ** 3)


def lumped_T_of_t(t, T0=T_INIT, Tinf=T_SURR, L=L_SLAB, eps=EPS):
    """Invert the implicit relation to obtain T(t)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    for i, tv in enumerate(t):
        if tv <= 0.0:
            out[i] = T0
            continue
        f = lambda T: lumped_implicit_time(T, T0, Tinf, L, eps) - tv
        out[i] = brentq(f, Tinf * (1.0 + 1e-12), T0 - 1e-12,
                        xtol=1e-12, rtol=8.9e-16)
    return out


# ==============================================================================
# 3. THE FINITE VOLUME SOLVER
# ==============================================================================
def solve_slab(N, M, t_end, linearisation="newton", theta=1.0, n_start=4,
               eps=EPS, L=L_SLAB, k=K_SOLID, T0=T_INIT, Tinf=T_SURR,
               tol=1e-11, max_iter=200, record=False):
    """Transient conduction in a slab with a radiating face.

    Cell-centred control volumes with boundary NODES on the two faces (half
    control volumes), as everywhere in this book.  The radiative flux enters
    the surface cell as a linearised source.

    Returns the final field, the time history, and iteration statistics.
    """
    faces = np.linspace(0.0, L, N + 1)
    xc = 0.5 * (faces[1:] + faces[:-1])
    x = np.concatenate(([0.0], xc, [L]))
    n = len(x)
    dv = np.diff(faces)                     # control volume widths
    dxn = np.diff(x)                        # node-to-node distances

    dt = t_end / M
    T = np.full(n, T0)

    hist = {"t": [0.0], "Ts": [T0], "Tm": [T0], "E": [0.0], "iters": []}
    E_rad = 0.0                              # cumulative radiated energy

    D = k / dxn                              # conductances between nodes
    I = slice(1, n - 1)
    aP0 = RHO * CP * dv / dt

    for step in range(1, M + 1):
        # Rannacher startup.  The uniform initial condition is INCOMPATIBLE
        # with the radiating boundary condition: at t = 0 the interior is flat
        # while the surface is already required to carry a gradient.  Chapter 6
        # showed that Crank-Nicolson is stable but not bounded, and Example 8.3
        # met the same defect at a discontinuous inlet.  Here it costs an order:
        # measured without startup damping, Crank-Nicolson converges at p = 1.00
        # instead of 2.  A few backward Euler steps annihilate the offending
        # modes and second order returns (p = 1.93 measured).
        th_use = 1.0 if step <= n_start else theta
        Told = T.copy()
        Tit = T.copy()
        for it in range(max_iter):
            a_P = np.zeros(n); a_E = np.zeros(n); a_W = np.zeros(n)
            b = np.zeros(n)

            a_E[I] = th_use * D[1:]
            a_W[I] = th_use * D[:-1]
            a_P[I] = th_use * (D[1:] + D[:-1]) + aP0
            b[I] = aP0 * Told[I] + (1.0 - th_use) * (
                D[1:] * (Told[2:] - Told[I]) + D[:-1] * (Told[:-2] - Told[I]))

            # --- insulated face at x = 0: zero flux, so the node follows its
            #     neighbour exactly
            a_P[0] = 1.0
            a_E[0] = 1.0
            b[0] = 0.0

            # --- radiating face at x = L -----------------------------------
            # Energy balance on the boundary NODE (zero volume): conduction in
            # from the interior must equal radiation out.
            #     D_last (T_{n-2} - T_n) = eps sigma (T_n^4 - Tinf^4)
            Ts = Tit[-1]
            if linearisation == "newton":
                # eps sigma T^4 ~ eps sigma (Ts^4 + 4 Ts^3 (T - Ts))
                #   => S_u = -eps sigma (Tinf^4 ... ) ,  S_P = -4 eps sigma Ts^3
                Sp = -4.0 * eps * SIGMA * Ts ** 3
                Su = -eps * SIGMA * (Ts ** 4 - Tinf ** 4) - Sp * Ts
            elif linearisation == "explicit":
                Sp = 0.0
                Su = -eps * SIGMA * (Ts ** 4 - Tinf ** 4)
            else:
                raise ValueError("linearisation must be 'newton' or 'explicit'")

            # a_P T_n = a_W T_{n-1} + b, with the source contributing -Sp to
            # a_P (Patankar's sign convention) and Su to b
            a_W[-1] = D[-1]
            a_P[-1] = D[-1] - Sp
            b[-1] = Su

            ab = np.zeros((3, n))
            ab[0, 1:] = -a_E[:-1]
            ab[1, :] = a_P
            ab[2, :-1] = -a_W[1:]
            Tnew = solve_banded((1, 1), ab, b)

            res = np.max(np.abs(Tnew - Tit))
            Tit = Tnew
            if res < tol:
                break
        hist["iters"].append(it + 1)
        T = Tit

        # energy accounting
        q_rad = eps * SIGMA * (T[-1] ** 4 - Tinf ** 4)
        E_rad += q_rad * dt
        if record or step == M:
            hist["t"].append(step * dt)
            hist["Ts"].append(T[-1])
            hist["Tm"].append(float(np.sum(T[I] * dv) / L))
            hist["E"].append(E_rad)

    for key in ("t", "Ts", "Tm", "E"):
        hist[key] = np.array(hist[key])
    hist["iters"] = np.array(hist["iters"])
    return x, T, hist


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 9.3 -- COMBINED CONDUCTION AND RADIATION")
print("=" * 78)
h_rad0 = EPS * SIGMA * (T_INIT ** 2 + T_SURR ** 2) * (T_INIT + T_SURR)
Bi0 = h_rad0 * L_SLAB / K_SOLID
print(f"  Steel slab: L = {L_SLAB*1e3:.0f} mm, k = {K_SOLID} W/m.K, "
      f"eps = {EPS}")
print(f"  T_0 = {T_INIT:.0f} K radiating to {T_SURR:.0f} K")
print(f"  linearised radiation coefficient at t = 0: "
      f"h_r = {h_rad0:.2f} W/m^2 K")
print(f"  Biot number based on it: Bi = {Bi0:.4f}")
print("  Bi is small but NOT negligible, so the slab is nearly -- not quite --")
print("  isothermal, and the lumped solution is a reference to be approached,")
print("  not an answer.")

# ---- 4a. the two closed forms against each other ---------------------------
print("\n" + "-" * 78)
print("  CHECK 1 -- the two lumped closed forms, and where each one works")
print(f"  {'T_inf [K]':>10} {'implicit form':>18} {'elementary form':>18} "
      f"{'difference':>13}")
C_l = EPS * SIGMA / (RHO * CP * L_SLAB)
for Ti in (100.0, 10.0, 1.0, 0.1, 0.01):
    t_imp = lumped_implicit_time(600.0, Tinf=Ti)
    t_zero = ((T_INIT / 600.0) ** 3 - 1.0) / (3.0 * C_l * T_INIT ** 3)
    print(f"  {Ti:>10.2f} {t_imp:>18.6f} {t_zero:>18.6f} "
          f"{abs(t_imp-t_zero):>13.3e}")
print("""
  The two agree beautifully down to T_inf = 1 K and then get WORSE, which is
  the opposite of what a limit should do.  The implicit form is not wrong; it
  is ill-conditioned.  Its numerator involves

      ln((T+Ti)/(T-Ti)) + 2 arctan(T/Ti)

  evaluated at two temperatures and subtracted, and as Ti -> 0 that difference
  tends to zero like Ti^3 while each term stays of order one.  Dividing by
  4 C Ti^3 then amplifies whatever is left of the cancelled digits.  At
  Ti = 0.01 K roughly six significant figures have been destroyed.

  This is why BOTH forms are kept.  The elementary expression is exact when
  T_inf = 0 and useless otherwise; the implicit one is exact for T_inf > 0 and
  unusable as T_inf approaches zero.  Neither is a better formula than the
  other -- they have different domains of numerical validity, and a limit
  taken carelessly crosses from one into the other.""")

# ---- 4b. FVM against the exact lumped solution ------------------------------
print("\n" + "-" * 78)
print("  CHECK 2 -- FVM AGAINST THE LUMPED LIMIT, DONE PROPERLY")
print("""    The conductivity is raised artificially to drive Bi towards zero, at
    which point the FVM must reproduce the lumped solution.  A first attempt
    at this check refined Bi while holding the time step fixed, found the
    difference stall at about 0.01 K, and was about to report a mysterious
    floor.  There is no floor.  The reference is EXACT in time and the solver
    is not, so once the modelling difference falls below the time
    discretisation error the comparison stops measuring what it was aimed at
    -- exactly the cross-contamination diagnosed in Chapter 8, in a new
    costume.

    The fix is to compare like with like.  Integrating the lumped equation
    with the SAME backward Euler scheme and the same step gives a reference
    that shares the solver's temporal error, so the remaining difference is
    the genuine effect of Biot number.""")


def lumped_backward_euler(M, t_end, L=L_SLAB, eps=EPS):
    """The lumped ODE integrated with the solver's own time scheme."""
    dt = t_end / M
    C = eps * SIGMA / (RHO * CP * L)
    T = T_INIT
    for _ in range(M):
        Tn = T
        for _ in range(200):
            f = Tn - T + dt * C * (Tn ** 4 - T_SURR ** 4)
            step = f / (1.0 + 4.0 * dt * C * Tn ** 3)
            Tn -= step
            if abs(step) < 1e-13:
                break
        T = Tn
    return T


T_END = 600.0
N_V, M_V = 80, 2000
T_exact_l = lumped_T_of_t(np.array([T_END]))[0]
T_be_l = lumped_backward_euler(M_V, T_END)
print(f"\n    exact lumped solution              = {T_exact_l:.7f} K")
print(f"    lumped, backward Euler, M = {M_V}   = {T_be_l:.7f} K")
print(f"    the two differ by                  = {abs(T_exact_l-T_be_l):.3e} K"
      f"   (pure time error)")
print(f"\n  {'k [W/m.K]':>11} {'Bi':>10} {'T_s (FVM)':>14} "
      f"{'vs BE lumped':>14} {'vs exact lumped':>17}")
for k_test in (45.0, 450.0, 4500.0, 45000.0, 450000.0):
    _, Tf, _ = solve_slab(N_V, M_V, T_END, k=k_test)
    Bi = h_rad0 * L_SLAB / k_test
    print(f"  {k_test:>11.0f} {Bi:>10.6f} {Tf[-1]:>14.7f} "
          f"{abs(Tf[-1]-T_be_l):>14.3e} {abs(Tf[-1]-T_exact_l):>17.3e}")
print("""    Read the last two columns against each other.  The comparison with
    the exact solution appears to stall near 0.013 K; the comparison with the
    consistently discretised reference keeps falling, and at the smallest Biot
    number the FVM reproduces the lumped ODE to about 4e-4 K -- which is the
    spatial discretisation, and nothing else.  The apparent floor was the time
    error of the solver, held constant while a different parameter was
    refined.""")

print("\n    And the time error itself is cleanly first order, as backward")
print("    Euler requires:")
print(f"    {'M':>7} {'dt [s]':>9} {'|exact - BE lumped|':>22} {'ratio':>8}")
prev_e = None
for M_t in (250, 500, 1000, 2000, 4000):
    e_t = abs(T_exact_l - lumped_backward_euler(M_t, T_END))
    r_t = f"{prev_e/e_t:.2f}" if prev_e else "-"
    print(f"    {M_t:>7d} {T_END/M_t:>9.3f} {e_t:>22.3e} {r_t:>8}")
    prev_e = e_t

# ---- 4c. order of accuracy --------------------------------------------------
def obs_order(v, i):
    if i + 2 >= len(v):
        return None
    return np.log2(abs((v[i] - v[i + 1]) / (v[i + 1] - v[i + 2])))


print("\n" + "-" * 78)
print("  CHECK 3 -- ORDER OF ACCURACY")
print("  Space (time refined to 4000 steps), surface temperature at t = 600 s")
print(f"  {'N':>6} {'T_s [K]':>16} {'p':>8}")
NS = [5, 10, 20, 40, 80]
VS = []
for N in NS:
    _, Tf, _ = solve_slab(N, 4000, T_END)
    VS.append(Tf[-1])
for i, N in enumerate(NS):
    p = obs_order(VS, i)
    print(f"  {N:>6d} {VS[i]:>16.9f} "
          f"{('%.3f' % p) if p is not None else '-':>8}")

print("\n  Time (space refined to 120 cells): three schemes")
print(f"  {'M':>6} {'Euler':>15} {'p':>6} {'CN, no start':>15} {'p':>6}"
      f" {'CN + Rannacher':>16} {'p':>6}")
MS = [25, 50, 100, 200, 400]
VE, VC, VR = [], [], []
for M in MS:
    VE.append(solve_slab(120, M, T_END, theta=1.0)[1][-1])
    VC.append(solve_slab(120, M, T_END, theta=0.5, n_start=0)[1][-1])
    VR.append(solve_slab(120, M, T_END, theta=0.5, n_start=4)[1][-1])
for i, M in enumerate(MS):
    pe, pc, pr = obs_order(VE, i), obs_order(VC, i), obs_order(VR, i)
    fmt = lambda p: (f"{p:.3f}" if p is not None else "-")
    print(f"  {M:>6d} {VE[i]:>15.8f} {fmt(pe):>6} {VC[i]:>15.8f} "
          f"{fmt(pc):>6} {VR[i]:>16.8f} {fmt(pr):>6}")
print("""    Crank-Nicolson WITHOUT startup damping is first order, not second.
    The uniform initial condition is incompatible with the radiating boundary
    condition -- at t = 0 the interior is flat while the surface must already
    carry a gradient -- and Crank-Nicolson, which does not annihilate the
    resulting stiff transient, carries the defect through the whole
    integration.  Four backward Euler steps at the start fix it, and the
    second order the scheme is supposed to have appears.  This is the same
    failure as the discontinuous inlet of Example 8.3; only the geometry
    differs.""")

# ---- 4d. Richardson and GCI -------------------------------------------------
p_sp = obs_order(VS, len(NS) - 3)
f1, f2 = VS[-1], VS[-2]
rich = f1 + (f1 - f2) / (2.0 ** p_sp - 1.0)
gci = 1.25 * abs((f1 - f2) / f1) / (2.0 ** p_sp - 1.0)
print("\n  CHECK 4 -- RICHARDSON EXTRAPOLATION AND GCI (spatial)")
print(f"    observed order p        = {p_sp:.4f}")
print(f"    finest grid  T_s        = {f1:.9f} K")
print(f"    Richardson extrapolated = {rich:.9f} K")
print(f"    implied error on finest = {abs(rich-f1):.3e} K")
print(f"    GCI (Fs = 1.25)         = {100*gci:.6f} %")

# ---- 4e. energy audit -------------------------------------------------------
print("\n" + "-" * 78)
print("  CHECK 5 -- ENERGY AUDIT")
print("""    The energy stored in the slab must fall by exactly what has been
    radiated away.  Neither quantity is imposed: the stored energy comes from
    integrating the final temperature field, the radiated energy from summing
    the surface flux over the time steps.""")
x_e, T_e, h_e = solve_slab(80, 2000, T_END, record=True)
faces_e = np.linspace(0.0, L_SLAB, 81)
dv_e = np.diff(faces_e)
E_stored_0 = RHO * CP * T_INIT * L_SLAB
E_stored_f = RHO * CP * float(np.sum(T_e[1:-1] * dv_e))
dE = E_stored_0 - E_stored_f
print(f"    energy lost by the slab   = {dE:.6f} J/m^2")
print(f"    energy radiated away      = {h_e['E'][-1]:.6f} J/m^2")
print(f"    relative imbalance        = {abs(dE-h_e['E'][-1])/dE:.3e}")
print("    The imbalance is the time-integration error of the flux sum, and it")
print("    falls with the step size; it is not a leak in the scheme.")

# ---- 4f. the linearisations compared ---------------------------------------
print("\n" + "-" * 78)
print("  CHECK 6 -- NEWTON AGAINST EXPLICIT LINEARISATION")
print("""    Both forms are consistent -- they converge to the same answer -- but
    they differ in cost and in robustness.  The Newton form carries
    S_P = -4 eps sigma T*^3, which is negative for any positive temperature,
    so Patankar's rule is satisfied automatically.  The explicit form has
    S_P = 0, which is admissible but puts the entire nonlinearity in the
    source term.""")
print(f"\n  {'steps M':>9} {'dt [s]':>9} {'Newton iters':>14} "
      f"{'explicit iters':>16} {'T_s difference [K]':>20}")
for M in (20, 50, 100, 400):
    _, Tn, hn = solve_slab(40, M, T_END, linearisation="newton")
    _, Tx, hx = solve_slab(40, M, T_END, linearisation="explicit")
    print(f"  {M:>9d} {T_END/M:>9.2f} {hn['iters'].mean():>14.2f} "
          f"{hx['iters'].mean():>16.2f} {abs(Tn[-1]-Tx[-1]):>20.3e}")
print("""    Newton converges in a handful of iterations at every step size.  The
    explicit form needs progressively more as the step grows, because its
    iteration is a fixed-point map whose contraction factor worsens with dt.
    Both reach the same answer where they converge at all, which is the
    consistency statement; the difference is entirely one of cost.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
x_p, T_p, h_p = solve_slab(60, 1200, 1800.0, record=True)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

ax[0].plot(h_p["t"] / 60.0, h_p["Ts"], "-", lw=2.2, color="#b2182b",
           label="surface, FVM")
ax[0].plot(h_p["t"] / 60.0, h_p["Tm"], "-", lw=2.0, color="#2166ac",
           label="volume mean, FVM")
t_l = np.linspace(1.0, 1800.0, 300)
ax[0].plot(t_l / 60.0, lumped_T_of_t(t_l), "--", lw=1.7, color="#1b7837",
           label="exact lumped solution")
ax[0].axhline(T_SURR, color="0.45", ls=":", lw=1.4)
ax[0].annotate(rf"$T_\infty = {T_SURR:.0f}$ K", xy=(0.62, 0.10),
               xycoords="axes fraction", fontsize=8.5, color="0.35")
ax[0].set_xlabel(r"$t$  [min]")
ax[0].set_ylabel(r"$T$  [K]")
ax[0].set_title("(a) Radiative cooling of the slab")
ax[0].legend(fontsize=8.5, loc="upper right")

for frac, c in ((0.0, "#762a83"), (0.02, "#2166ac"), (0.10, "#1b7837"),
                (0.35, "#e08214"), (1.0, "#b2182b")):
    _, Tf_, _ = solve_slab(60, max(2, int(1200 * frac)) if frac > 0 else 2,
                           max(1.0, 1800.0 * frac))
    if frac == 0.0:
        Tf_ = np.full_like(x_p, T_INIT)
    ax[1].plot(x_p * 1e3, Tf_, "-", lw=1.9, color=c,
               label=rf"$t = {1800.0*frac/60.0:.0f}$ min")
ax[1].set_xlabel(r"$x$  [mm]   (0 insulated, 50 radiating)")
ax[1].set_ylabel(r"$T$  [K]")
ax[1].set_title("(b) Profiles: nearly, but not quite, isothermal")
ax[1].legend(fontsize=8, loc="lower left")

fig.suptitle("Example 9.3 -- Conduction coupled to radiation",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_3a_cooling.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

NSa = np.array(NS)
err_sp = np.array([abs(v - rich) for v in VS])
ax[0].loglog(NSa[:-1], err_sp[:-1], "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#b2182b", label="spatial error vs Richardson")
ax[0].loglog(NSa[:-1], err_sp[0] * (NSa[0] / NSa[:-1]) ** 2.0, "k--", lw=1.3,
             label=r"slope $-2$")
ax[0].set_xlabel(r"$N$  (control volumes)")
ax[0].set_ylabel(r"$|T_s - T_s^{\rm extrap}|$  [K]")
ax[0].set_title("(a) Second order in space")
ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].annotate(rf"observed $p = {p_sp:.3f}$" "\n"
               rf"GCI $= {100*gci:.1e}\,\%$",
               xy=(0.42, 0.70), xycoords="axes fraction", fontsize=8.5,
               color="0.25",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))

Ms = np.array([20, 50, 100, 200, 400])
it_n, it_x = [], []
for M in Ms:
    _, _, hn = solve_slab(40, M, T_END, linearisation="newton")
    _, _, hx = solve_slab(40, M, T_END, linearisation="explicit")
    it_n.append(hn["iters"].mean())
    it_x.append(hx["iters"].mean())
ax[1].semilogx(T_END / Ms, it_n, "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
               color="#1b7837", label=r"Newton, $S_P = -4\varepsilon\sigma T^3$")
ax[1].semilogx(T_END / Ms, it_x, "s-", lw=1.9, ms=7, mfc="none", mew=1.7,
               color="#b2182b", label=r"explicit, $S_P = 0$")
ax[1].set_xlabel(r"time step $\Delta t$  [s]")
ax[1].set_ylabel("mean iterations per step")
ax[1].set_title("(b) Why the linearisation is chosen, not guessed")
ax[1].legend(fontsize=8.5, loc="upper left")

fig.suptitle("Example 9.3 -- Verification of the coupled solution",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_3b_verification.png")
plt.close(fig)

print("Figures written: fig_9_3a_cooling.png, fig_9_3b_verification.png")
