"""
================================================================================
 EXAMPLE 8.1 -- THE LAMINAR BOUNDARY LAYER ON A FLAT PLATE
 The Blasius and Pohlhausen similarity solutions, computed and verified
================================================================================

 OBJECTIVE
 ---------
 Chapters 1 to 7 treated h as a given quantity.  This chapter computes it.  For
 the laminar boundary layer on a flat plate the governing partial differential
 equations collapse to ORDINARY ones under a similarity transformation, and the
 result is the classical correlation

        Nu_x = 0.332 Re_x^(1/2) Pr^(1/3)

 which is normally quoted.  Here it is DERIVED: the Blasius momentum equation
 and the Pohlhausen energy equation are integrated numerically, the boundary
 layer thickness and skin friction are recovered, and the Pr^(1/3) exponent is
 tested against the computed solution over four decades of Prandtl number.

 GOVERNING EQUATIONS
 -------------------
 With the similarity variable and stream function

        eta = y sqrt(U / (nu x)) ,      psi = sqrt(nu U x) f(eta)

 the momentum equation becomes the BLASIUS equation

        f''' + (1/2) f f'' = 0 ,   f(0) = f'(0) = 0 ,  f'(inf) = 1

 and, with theta = (T - T_s)/(T_inf - T_s), the energy equation becomes the
 POHLHAUSEN equation

        theta'' + (Pr/2) f theta' = 0 ,   theta(0) = 0 ,  theta(inf) = 1

 The energy equation is LINEAR in theta once f is known, which is why the
 thermal problem is far easier than the momentum problem and why Pr enters as a
 single parameter.

 METHOD
 ------
 Both are two-point boundary value problems solved by SHOOTING: guess the
 missing initial condition, integrate, and correct the guess by the secant
 method until the far-field condition is met.  The integrator is a classical
 fourth-order Runge-Kutta written out explicitly rather than called from a
 library, since its structure matters for the error discussion.

 The Blasius problem has an exact scaling property that provides a free check:
 if F(eta) solves the equation with F''(0) = 1 and F'(inf) = c, then
 f(eta) = a F(a eta) with a = c^(-1/2) solves it with f'(inf) = 1.  So the
 correct f''(0) can be obtained in ONE integration, and the shooting result
 must agree with it.

 SYMBOLS (all SI)
 ----------------
   U       [m/s]      free-stream velocity
   nu      [m^2/s]    kinematic viscosity
   alpha   [m^2/s]    thermal diffusivity
   Pr      [-]        Prandtl number nu/alpha
   Re_x    [-]        Reynolds number U x / nu
   Nu_x    [-]        local Nusselt number h x / k
   Cf_x    [-]        local skin friction coefficient
   delta   [m]        velocity boundary layer thickness (u = 0.99 U)
   delta_t [m]        thermal boundary layer thickness
   eta     [-]        similarity variable
   f, theta[-]        Blasius and Pohlhausen functions

 OUTPUTS
 -------
   fig_8_1a_profiles.png   velocity and temperature similarity profiles
   fig_8_1b_correlation.png  the Pr exponent tested, and boundary layer growth

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

# --- NumPy compatibility -----------------------------------------------------
# The trapezoidal rule was renamed in NumPy 2.0: `np.trapz` became
# `np.trapezoid`, and the old name now emits a DeprecationWarning.  Neither
# spelling works everywhere -- `np.trapezoid` does not exist before 2.0 and
# raises AttributeError there.  Binding the name once, here, lets the rest of
# the script run unchanged on both.  (An earlier version of this file used
# `np.trapezoid` directly, which ran clean on NumPy 2.x and failed outright on
# 1.x.  Testing on the version you happen to have installed is not the same as
# testing on the version your reader has installed.)
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ETA_MAX = 10.0          # far field; f' is within 1e-12 of 1 well before this
N_STEP = 4000


# ==============================================================================
# 1. FOURTH-ORDER RUNGE-KUTTA
# ==============================================================================
def rk4(rhs, y0, t0, t1, n):
    """Classical RK4 on a uniform grid.  Returns the full trajectory.

    Written out rather than imported because its order (four) and its step
    count enter the error budget of everything below, and because the shooting
    residual must be differentiated with respect to the initial guess -- which
    is easier to reason about when the integrator is explicit.
    """
    t = np.linspace(t0, t1, n + 1)
    h = (t1 - t0) / n
    y = np.empty((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = rhs(t[i], y[i])
        k2 = rhs(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = rhs(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = rhs(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


# ==============================================================================
# 2. THE BLASIUS PROBLEM
# ==============================================================================
def blasius_rhs(eta, y):
    """y = [f, f', f''];  f''' = -(1/2) f f''."""
    return np.array([y[1], y[2], -0.5 * y[0] * y[2]])


def blasius_shoot(s, n=N_STEP):
    """Integrate with f''(0) = s and return the trajectory."""
    return rk4(blasius_rhs, np.array([0.0, 0.0, s]), 0.0, ETA_MAX, n)


def solve_blasius(n=N_STEP, tol=1e-13, max_iter=60):
    """Secant shooting on f''(0) so that f'(eta_max) = 1."""
    s0, s1 = 0.3, 0.35
    F = lambda s: blasius_shoot(s, n)[1][-1, 1] - 1.0
    g0, g1 = F(s0), F(s1)
    for _ in range(max_iter):
        s2 = s1 - g1 * (s1 - s0) / (g1 - g0)
        g2 = F(s2)
        s0, g0, s1, g1 = s1, g1, s2, g2
        if abs(g2) < tol:
            break
    eta, y = blasius_shoot(s1, n)
    return s1, eta, y


def blasius_by_scaling(n=N_STEP):
    """f''(0) in ONE integration, using the exact scaling property.

    Integrating with F''(0) = 1 gives some F'(inf) = c.  Because the Blasius
    equation is invariant under f(eta) -> a f(a eta), the solution with
    f'(inf) = 1 has f''(0) = c^(-3/2).  This is an independent route to the
    same number and needs no iteration at all.
    """
    _, Y = rk4(blasius_rhs, np.array([0.0, 0.0, 1.0]), 0.0, ETA_MAX, n)
    c = Y[-1, 1]
    return c ** (-1.5)


# ==============================================================================
# 3. THE POHLHAUSEN PROBLEM
# ==============================================================================
DELTA_STAR = 1.7207876575       # exact displacement constant: f -> eta - beta


def f_extended(e, eta, f_traj):
    """Blasius f(eta), continued analytically beyond the computed range.

    THIS FUNCTION EXISTS BECAUSE OF A BUG.  The first version of this example
    integrated the thermal problem on the same domain as the momentum problem,
    0 <= eta <= 10.  That is ample for the VELOCITY layer at any Pr, and it is
    ample for the thermal layer when Pr is of order one or larger.  It is
    badly insufficient for liquid metals: the thermal layer thickness scales
    as Pr^(-1/2), so at Pr = 0.001 it reaches eta of order 30, and truncating
    at 10 amputates two thirds of it.

    The symptom was diagnostic.  theta'(0) did not fall as Pr fell; it flat-
    tened at about 0.10 for both Pr = 0.01 and Pr = 0.001, and the computed
    layer ratio delta/delta_t froze at 0.496 for both.  A quantity that stops
    responding to the parameter that should control it is not physics -- it is
    a boundary sitting where the physics has not finished.  Had that output
    been accepted, the example would have "demonstrated" a failure of the
    Pr^(1/3) correlation that was really a failure of the domain.

    Outside the computed range the Blasius solution approaches its exact
    asymptote f -> eta - beta, with beta = 1.7207876 the displacement
    constant, so the continuation costs nothing and is accurate to the
    precision of the asymptote itself.
    """
    e = np.asarray(e, dtype=float)
    e_end = eta[-1]
    inner = np.interp(np.minimum(e, e_end), eta, f_traj[:, 0])
    return np.where(e <= e_end, inner, e - DELTA_STAR)


def pohlhausen_quadrature(eta, f_traj, Pr, n=40000):
    """theta'(0) in closed form -- no shooting, no iteration.

    Integrating theta'' + (Pr/2) f theta' = 0 once gives

        theta'(eta) = theta'(0) exp[ -(Pr/2) int_0^eta f d(eta') ]

    and imposing theta(inf) = 1 gives the EXACT quadrature

        theta'(0) = 1 / int_0^inf exp[ -(Pr/2) int_0^eta f ] d(eta)

    This is an independent route to the same number: it shares no code path
    with the shooting solver, so agreement between the two is a genuine check
    rather than a restatement.  It also fixes the far-field question honestly,
    because the integrand's decay can be measured instead of assumed.
    """
    # Far field: for large eta the exponent behaves as -(Pr/4) eta^2, so the
    # integrand is Gaussian.  Solve -(Pr/4) e^2 = -45 for a safe cut-off.
    e_max = max(12.0, np.sqrt(4.0 * 45.0 / Pr) + DELTA_STAR)
    e = np.linspace(0.0, e_max, n + 1)
    F = f_extended(e, eta, f_traj)
    # cumulative integral of f by the trapezoidal rule
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(e))))
    integrand = np.exp(-0.5 * Pr * cum)
    tail = integrand[-1]
    denom = trapezoid(integrand, e)
    return 1.0 / denom, tail, e, integrand


def solve_pohlhausen(eta, f_traj, Pr, tol=1e-13, max_iter=60):
    """theta'' + (Pr/2) f theta' = 0 with theta(0)=0, theta(inf)=1.

    Solved by shooting on theta'(0), on a domain sized for the THERMAL layer
    rather than the velocity layer -- see f_extended for why that distinction
    is not cosmetic.  The problem is linear in theta, so the secant iteration
    converges in one step; it is kept for uniformity with the Blasius solver.
    """
    # Domain: the THERMAL layer scales as Pr^(-1/2) at small Pr, so the far
    # field must grow as Pr falls.  At large Pr the layer is thin and a short
    # domain suffices -- and is in fact required, see the step size below.
    e_max = max(6.0, 8.0 / np.sqrt(Pr))

    # Step size: the decay rate of the energy equation is (Pr/2) f, which at
    # the far edge is about (Pr/2)(e_max - beta).  Explicit RK4 is stable only
    # for h*lambda < 2.78, so at Pr = 1000 a step chosen for ACCURACY alone
    # overshoots the STABILITY limit and the integration returns NaN.  This is
    # the stiffness that Chapter 6 met in time integration, reappearing here in
    # a spatial variable: the same arithmetic, a different independent
    # variable.  Sizing n from the eigenvalue removes it.
    lam = 0.5 * Pr * max(1.0, e_max - DELTA_STAR)
    n = max(4000, int(e_max * lam / 2.0))

    def rhs(e, y):
        return np.array([y[1], -0.5 * Pr * float(f_extended(e, eta, f_traj)) * y[1]])

    def shoot(g):
        return rk4(rhs, np.array([0.0, g]), 0.0, e_max, n)

    g0, g1 = 0.2, 0.4
    F = lambda g: shoot(g)[1][-1, 0] - 1.0
    h0, h1 = F(g0), F(g1)
    for _ in range(max_iter):
        g2 = g1 - h1 * (g1 - g0) / (h1 - h0)
        h2 = F(g2)
        g0, h0, g1, h1 = g1, h1, g2, h2
        if abs(h2) < tol:
            break
    et, Y = shoot(g1)
    return g1, et, Y


# ==============================================================================
# 4. SOLVE AND VERIFY
# ==============================================================================
t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 8.1 -- THE LAMINAR FLAT-PLATE BOUNDARY LAYER")
print("=" * 78)

fpp0, eta, Yb = solve_blasius()
fpp0_scaled = blasius_by_scaling()
print(f"  BLASIUS.  f''(0) by shooting          = {fpp0:.12f}")
print(f"            f''(0) by the scaling law   = {fpp0_scaled:.12f}")
print(f"            difference                  = {abs(fpp0-fpp0_scaled):.3e}")
print("  The two routes share no iteration and agree to twelve figures.")

# ODE residual of the computed solution
fp = np.interp(eta, eta, Yb[:, 1])
d3 = np.gradient(np.gradient(np.gradient(Yb[:, 0], eta), eta), eta)
res = np.max(np.abs(d3[5:-5] + 0.5 * Yb[5:-5, 0] * Yb[5:-5, 2]))
print(f"\n  Verification  max|f''' + f f''/2|     = {res:.3e}")
print(f"  Verification  f'(eta_max) - 1          = {Yb[-1,1]-1.0:.3e}")
print(f"  Verification  f''(eta_max)             = {Yb[-1,2]:.3e} (-> 0)")

# convergence of the RK4 integrator
print(f"\n  RK4 convergence of f''(0):")
hdr = "f''(0)"
print(f"  {'steps':>8} {hdr:>18} {'error':>13} {'ratio':>8}")
prev = None
ref = fpp0
for n in [125, 250, 500, 1000, 2000]:
    s, _, _ = solve_blasius(n)
    e = abs(s - ref)
    r = f"{prev/e:.2f}" if prev and e > 0 else "-"
    print(f"  {n:>8d} {s:>18.12f} {e:>13.3e} {r:>8}")
    prev = e
print("  The ratio approaches 16 per halving, confirming fourth order.")

# --- boundary layer thickness and skin friction -------------------------------
eta99 = np.interp(0.99, Yb[:, 1], eta)
print("\n" + "-" * 78)
print("  BOUNDARY LAYER THICKNESS AND SKIN FRICTION")
print(f"    eta at u/U = 0.99             = {eta99:.6f}")
print(f"    delta/x    = {eta99:.4f} / sqrt(Re_x)")
print(f"    textbooks usually quote 5.0/sqrt(Re_x); the computed value is")
print(f"    {eta99:.3f}, and the difference is rounding, not physics.")
print(f"    C_f,x sqrt(Re_x) = 2 f''(0)   = {2*fpp0:.8f}  (quoted 0.664)")
# displacement and momentum thickness
d_star = trapezoid(1.0 - Yb[:, 1], eta)
theta_m = trapezoid(Yb[:, 1] * (1.0 - Yb[:, 1]), eta)
print(f"    displacement thickness  delta*/x sqrt(Re_x) = {d_star:.6f} "
      f"(exact 1.72079)")
print(f"    momentum thickness      theta/x  sqrt(Re_x) = {theta_m:.6f} "
      f"(exact 0.664)")
print(f"    shape factor H = delta*/theta               = "
      f"{d_star/theta_m:.6f} (exact 2.59)")

# ==============================================================================
# 5. THE THERMAL PROBLEM AND THE Pr EXPONENT
# ==============================================================================
print("\n" + "-" * 78)
print("  POHLHAUSEN.  Nu_x / sqrt(Re_x) = theta'(0), computed for each Pr")
print("  and compared with the classical correlation 0.332 Pr^(1/3).")
print(f"\n  {'Pr':>10} {'shooting':>13} {'quadrature':>13} {'diff':>10} "
      f"{'0.332Pr^1/3':>13} {'error':>9} {'d/d_t':>10}")
Prs = [0.001, 0.01, 0.1, 0.7, 1.0, 6.0, 10.0, 100.0, 1000.0]
rows = []
for Pr in Prs:
    tp0, et, Yt = solve_pohlhausen(eta, Yb, Pr)
    tq0, tail, _, _ = pohlhausen_quadrature(eta, Yb, Pr)
    corr = 0.332 * Pr ** (1.0 / 3.0)
    eta_t = np.interp(0.99, Yt[:, 0], et)
    rows.append((Pr, tp0, corr, eta99 / eta_t, et, Yt, eta_t))
    print(f"  {Pr:>10.3f} {tp0:>13.6f} {tq0:>13.6f} {abs(tp0-tq0):>10.1e} "
          f"{corr:>13.6f} {100*(corr-tp0)/tp0:>8.2f}% {eta99/eta_t:>10.4f}")

print("\n  The shooting and quadrature columns agree to at least six significant")
print("  figures at every Pr.  They share no code path -- one iterates on an")
print("  initial slope, the other evaluates a closed-form integral -- so the")
print("  agreement is an independent check and not a restatement.")
pr1 = [q for q in rows if q[0] == 1.0][0][1]
print(f"\n  EXACT IDENTITY AT Pr = 1.  When Pr = 1 the energy equation and the")
print(f"  differentiated momentum equation are the same equation with the same")
print(f"  boundary conditions, so theta must equal f' identically and")
print(f"  theta'(0) must equal f''(0).  Computed: {pr1:.9f} vs {fpp0:.9f},")
print(f"  differing by {abs(pr1-fpp0):.1e}.  This is the sharpest check available")
print("  here, because it is an identity rather than a comparison.")

print("\n  READING THE TABLE.  The Pr^(1/3) form is excellent for Pr of order")
print("  one and above -- within about 2 % from Pr = 0.7 to 1000, which covers")
print("  air, water, and most oils.  It fails for LIQUID METALS, where it")
print("  OVERPREDICTS badly -- by more than 90 % at Pr = 0.001:")
for Pr in (0.001, 0.01):
    r = [q for q in rows if q[0] == Pr][0]
    print(f"    Pr = {Pr:<6g}: computed {r[1]:.6f},  "
          f"0.332 Pr^(1/3) = {r[2]:.6f} ({100*(r[2]/r[1]-1):+6.1f} %),  "
          f"0.564 Pr^(1/2) = {0.564*Pr**0.5:.6f} "
          f"({100*(0.564*Pr**0.5/r[1]-1):+6.1f} %)")
print("  The reason is in the last column: for small Pr the thermal layer is")
print("  far THICKER than the velocity layer, so almost all of the heated")
print("  fluid is moving at essentially the free-stream speed.  The velocity")
print("  profile that produced the 1/3 exponent is then irrelevant, and the")
print("  correct limit is the SLUG-FLOW result Nu_x = 0.564 (Re_x Pr)^(1/2),")
print("  which the table confirms to within a few per cent at Pr = 0.001.")

# a better-fitting exponent, obtained by regression rather than assertion
sel = [(r[0], r[1]) for r in rows if r[0] >= 0.7]
lp = np.log([s[0] for s in sel])
lt = np.log([s[1] for s in sel])
slope, inter = np.polyfit(lp, lt, 1)
print(f"\n  Fitting theta'(0) = C Pr^n over Pr >= 0.7 gives")
print(f"    n = {slope:.5f}  (the classical 1/3 = {1/3:.5f})")
print(f"    C = {np.exp(inter):.5f}  (the classical 0.332)")

# --- the domain-truncation experiment, run deliberately ----------------------
print("\n" + "-" * 78)
print("  WHY THE DOMAIN IS SIZED BY Pr.  Repeating Pr = 0.001 on progressively")
print("  longer domains shows what a truncated far field does to the answer.")
print(f"  {'eta_max':>9} {'theta_p(0)':>13} {'error vs converged':>20}")
tq_ref = pohlhausen_quadrature(eta, Yb, 0.001)[0]
for em in [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]:
    n_e = max(4000, int(2000 * em / 12.0))
    def rhs_t(e, y, Pr=0.001):
        return np.array([y[1],
                         -0.5 * Pr * float(f_extended(e, eta, Yb)) * y[1]])
    _, Yq = rk4(rhs_t, np.array([0.0, 1.0]), 0.0, em, n_e)
    g = 1.0 / Yq[-1, 0]          # linearity: rescale a unit shot
    print(f"  {em:>9.1f} {g:>13.6f} {100*(g/tq_ref-1):>19.2f}%")
print("  At eta_max = 10 the answer is high by a factor of about five and,")
print("  worse, it is INSENSITIVE to Pr -- which is exactly what made the")
print("  error look like a physical plateau instead of a numerical artefact.")

print(f"\n  CPU time = {time.perf_counter()-t0:.2f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax[0].plot(Yb[:, 1], eta, "-", lw=2.2, color="#b2182b", label=r"$u/U = f'$")
ax[0].plot(Yb[:, 2] / fpp0, eta, "--", lw=1.6, color="#4d4d4d",
           label=r"$f''/f''(0)$ (shear)")
ax[0].axhline(eta99, color="#1b7837", ls=":", lw=1.6)
ax[0].annotate(rf"$\eta_{{99}} = {eta99:.3f}$", xy=(0.99, eta99),
               xytext=(0.30, eta99 + 1.4), fontsize=9.5, color="#1b7837",
               arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.0))
ax[0].set_xlabel(r"$u/U$,  $f''/f''(0)$")
ax[0].set_ylabel(r"Similarity variable $\eta = y\sqrt{U/\nu x}$")
ax[0].set_title("(a) The Blasius velocity profile")
ax[0].legend(fontsize=9, loc="upper right")
ax[0].set_ylim(0, 8)

for (Pr, tp0, _c, _r, et, Yt, _e), col in zip(
        [r for r in rows if r[0] in (0.01, 0.7, 10.0, 1000.0)],
        ["#1b7837", "#2166ac", "#b2182b", "#762a83"]):
    ax[1].plot(Yt[:, 0], et, "-", lw=1.9, color=col, label=rf"$Pr = {Pr:g}$")
ax[1].plot(Yb[:, 1], eta, "--", lw=1.4, color="0.45", label=r"$u/U$")
ax[1].set_xlabel(r"$\theta = (T - T_s)/(T_\infty - T_s)$")
ax[1].set_ylabel(r"$\eta$   (logarithmic)")
ax[1].set_title("(b) Thermal profiles: $Pr$ sets the layer ratio")
ax[1].legend(fontsize=8.5, loc="upper left")
# A LOGARITHMIC ordinate is essential here.  The thermal layer spans eta of
# order 0.5 at Pr = 1000 and eta of order 60 at Pr = 0.01 -- more than two
# decades.  On a linear axis sized for one, the other either collapses onto
# the wall or runs off the top of the frame, which is what the first version
# of this figure did.
ax[1].set_yscale("log")
ax[1].set_ylim(0.02, 200)

fig.suptitle("Example 8.1 -- Similarity solutions for the laminar boundary layer",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_1a_profiles.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
Pr_arr = np.array([r[0] for r in rows])
tp_arr = np.array([r[1] for r in rows])
ax[0].loglog(Pr_arr, tp_arr, "o", ms=8, mfc="none", mew=1.8, color="#b2182b",
             label="computed Pohlhausen")
pp = np.logspace(-3, 3, 200)
ax[0].loglog(pp, 0.332 * pp ** (1 / 3), "-", lw=1.8, color="#2166ac",
             label=r"$0.332\,Pr^{1/3}$")
ax[0].loglog(pp, 0.564 * pp ** 0.5, "--", lw=1.6, color="#1b7837",
             label=r"$0.564\,Pr^{1/2}$ (slug flow)")
ax[0].set_xlabel(r"Prandtl number $Pr$")
ax[0].set_ylabel(r"$Nu_x / \sqrt{Re_x}$")
ax[0].set_title("(a) The $Pr^{1/3}$ form and where it fails")
ax[0].legend(fontsize=8.5, loc="upper left")

Rex = np.logspace(3, 6, 200)
for Pr, col in [(0.7, "#2166ac"), (7.0, "#b2182b")]:
    tp0 = solve_pohlhausen(eta, Yb, Pr)[0]
    ax[1].loglog(Rex, tp0 * np.sqrt(Rex), "-", lw=2.0, color=col,
                 label=rf"$Nu_x$, $Pr = {Pr:g}$")
ax[1].loglog(Rex, eta99 / np.sqrt(Rex) * 1e3, "--", lw=1.8, color="#4d4d4d",
             label=r"$10^3\,\delta/x$")
ax[1].axvline(5e5, color="0.4", ls=":", lw=1.6)
ax[1].annotate("laminar theory\nends here\n$Re_x \\approx 5\\times10^5$",
               xy=(5e5, 0.30), xycoords=("data", "axes fraction"),
               xytext=(0.52, 0.16), textcoords="axes fraction",
               fontsize=8.5, color="0.30", ha="center",
               arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0),
               bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))
ax[1].set_xlabel(r"$Re_x$")
ax[1].set_ylabel(r"$Nu_x$   and   $10^3\,\delta/x$")
ax[1].set_title("(b) Growth of the layer and of $Nu_x$")
ax[1].legend(fontsize=8.5, loc="upper left")

fig.suptitle("Example 8.1 -- Testing the classical correlation",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_1b_correlation.png")
plt.close(fig)

print("Figures written: fig_8_1a_profiles.png, fig_8_1b_correlation.png")
