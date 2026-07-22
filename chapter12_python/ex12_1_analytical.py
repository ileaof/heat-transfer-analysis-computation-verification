"""
================================================================================
 EXAMPLE 12.1 -- CONVECTION AND DIFFUSION, AND THE SCHEMES THAT DISCRETISE IT
 The one equation whose exact solution exposes every discretisation choice
================================================================================

 OBJECTIVE
 ---------
 Every finite volume computation in this book has discretised diffusion, and
 several have discretised convection.  This chapter steps back to the equation
 that contains both, and to the single dimensionless group -- the Peclet number
 -- that decides which of the two dominates and therefore how the equation must
 be discretised.

 The steady one-dimensional convection-diffusion equation

        d/dx ( rho u phi ) = d/dx ( Gamma d phi/dx )

 with phi fixed at the two ends has an EXACT solution, and that exact solution
 is what makes this the ideal instrument for studying discretisation.  Every
 scheme can be measured against it, at every Peclet number, with no ambiguity
 about what "correct" means.

 THE EXACT SOLUTION
 ------------------
 With phi(0) = phi_0, phi(L) = phi_L, and constant rho u and Gamma,

        phi(x) - phi_0     exp(Pe x/L) - 1
        --------------- =  ---------------
        phi_L - phi_0       exp(Pe) - 1

 where Pe = rho u L / Gamma is the Peclet number.  At small Pe the profile is
 the straight line of pure diffusion; at large Pe it is flat across the domain
 and turns up sharply in a thin layer near the outflow boundary -- convection
 sweeps phi downstream and diffusion resists only in the last sliver.

 THE SCHEMES (Patankar's family)
 -------------------------------
 Each scheme is a rule for the face value of phi carried by the flow, expressed
 through the coefficient a_E = D A(|Pe_cell|) + max(-F, 0), with F = rho u the
 convective strength, D = Gamma/dx the diffusive conductance, and A a function
 of the cell Peclet number Pe_cell = F/D:

   central     A = 1 - 0.5 |Pe|        second order, but UNBOUNDED for |Pe|>2
   upwind      A = 1                    first order, always bounded, false diffusion
   hybrid      A = max(0, 1 - 0.5|Pe|)  central below |Pe|=2, upwind above
   power-law   A = max(0, (1-0.1|Pe|)^5)  Patankar's smooth fit to exponential
   exponential A = |Pe| / (exp(|Pe|)-1)  EXACT for this 1-D problem

 The exponential scheme is exact here because it is derived from the exact
 solution above.  That is also its weakness, exposed in Example 12.2: exactness
 in one dimension does not survive into two.

 VERIFICATION
 ------------
   1. The exponential scheme reproduces the exact solution to machine precision
      at every Peclet number and every grid -- because it IS the exact solution.
   2. Central differencing OSCILLATES for cell Peclet numbers above 2, and the
      example locates that threshold exactly.
   3. Order of accuracy of each scheme, measured against the exact solution.
   4. The boundedness of each scheme, tested by whether any node leaves the
      range set by the boundary values.

 OUTPUTS
 -------
   fig_12_1a_profiles.png    exact solution and the schemes at several Peclet
   fig_12_1b_schemes.png     the A(Pe) functions, and convergence orders

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
T_START = time.perf_counter()

PHI0, PHIL = 0.0, 1.0
LENGTH = 1.0


# ==============================================================================
# 1. THE EXACT SOLUTION
# ==============================================================================
def phi_exact(x, Pe, L=LENGTH):
    """Exact steady convection-diffusion profile between phi0 and phiL.

    Written to be stable at Pe = 0 (pure diffusion) and at large |Pe|, where a
    naive (exp(Pe x/L) - 1)/(exp(Pe) - 1) overflows.  The large-|Pe| limits are
    obtained by dividing through so that no growing exponential survives, and
    BOTH must still honour phi(0) = phi_0 and phi(L) = phi_L:

        Pe -> +inf :  frac = exp(Pe(xi - 1))          (0 at xi=0, 1 at xi=1)
        Pe -> -inf :  frac = 1 - exp(Pe xi)           (0 at xi=0, 1 at xi=1)

    A first draft wrote the negative-Pe limit as exp(Pe xi), which is 1 at
    xi = 0 -- violating phi(0) = phi_0 -- and produced an error of exactly one
    at the inflow boundary.  The lesson is small and sharp: an asymptotic form
    must reproduce the boundary conditions of the thing it approximates, and
    the cheapest check of that is to evaluate it at the boundaries.
    """
    x = np.asarray(x, dtype=float)
    xi = x / L
    if abs(Pe) < 1e-8:
        frac = xi
    elif Pe > 40:
        frac = np.exp(Pe * (xi - 1.0))
    elif Pe < -40:
        frac = 1.0 - np.exp(Pe * xi)
    else:
        frac = np.expm1(Pe * xi) / np.expm1(Pe)
    return PHI0 + (PHIL - PHI0) * frac


# ==============================================================================
# 2. THE A(|Pe|) FUNCTIONS
# ==============================================================================
def A_central(p):
    return 1.0 - 0.5 * np.abs(p)


def A_upwind(p):
    return np.ones_like(np.asarray(p, dtype=float))


def A_hybrid(p):
    return np.maximum(0.0, 1.0 - 0.5 * np.abs(p))


def A_powerlaw(p):
    return np.maximum(0.0, (1.0 - 0.1 * np.abs(p)) ** 5)


def A_exponential(p):
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    small = np.abs(p) < 1e-8
    out[small] = 1.0 - 0.5 * np.abs(p[small])          # limit as p -> 0
    ps = p[~small]
    out[~small] = np.abs(ps) / np.expm1(np.abs(ps))
    return out


SCHEMES = {
    "central": A_central,
    "upwind": A_upwind,
    "hybrid": A_hybrid,
    "power-law": A_powerlaw,
    "exponential": A_exponential,
}


# ==============================================================================
# 3. THE SOLVER
# ==============================================================================
def solve_cd(N, Pe, scheme, L=LENGTH):
    """Finite volume solution on N cells with boundary nodes on the faces.

    F = rho u is set from Pe = F L / Gamma with Gamma = 1 and L = 1, so F = Pe.
    The cell Peclet number is F/D = F dx / Gamma = Pe/N.
    """
    Gamma = 1.0
    F = Pe * Gamma / L                       # rho u
    faces = np.linspace(0.0, L, N + 1)
    xc = 0.5 * (faces[1:] + faces[:-1])
    x = np.concatenate(([0.0], xc, [L]))
    n = len(x)
    dxn = np.diff(x)
    Afun = SCHEMES[scheme]

    D = Gamma / dxn                          # conductance between nodes
    p_cell = F / D                           # cell Peclet at each face
    # Patankar's deferred form: a_E = D A(|Pe|) + max(-F,0); a_W = D A(|Pe|)+max(F,0)
    aE_face = D * Afun(np.abs(p_cell)) + np.maximum(-F, 0.0)
    aW_face = D * Afun(np.abs(p_cell)) + np.maximum(F, 0.0)

    a_P = np.zeros(n); a_E = np.zeros(n); a_W = np.zeros(n); b = np.zeros(n)
    for i in range(1, n - 1):
        a_E[i] = aE_face[i]
        a_W[i] = aW_face[i - 1]
        a_P[i] = a_E[i] + a_W[i]             # no source; F cancels by continuity
    a_P[0] = 1.0; b[0] = PHI0
    a_P[-1] = 1.0; b[-1] = PHIL

    ab = np.zeros((3, n))
    ab[0, 1:] = -a_E[:-1]
    ab[1, :] = a_P
    ab[2, :-1] = -a_W[1:]
    phi = solve_banded((1, 1), ab, b)
    return x, phi


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 12.1 -- CONVECTION-DIFFUSION AND ITS DISCRETISATION")
print("=" * 78)

print("\n" + "-" * 78)
print("  CHECK 1 -- THE EXPONENTIAL SCHEME IS EXACT IN 1-D")
print("""    The exponential scheme is derived from the exact solution, so it must
    reproduce it to round-off on ANY grid, however coarse, at ANY Peclet number.
    This is the strongest test a scheme can pass -- and Example 12.2 shows why
    it is not the end of the story.""")
print(f"\n  {'Pe':>8} {'N':>5} {'max|phi - exact|':>18}")
for Pe in (1.0, 10.0, 100.0, -50.0):
    for N in (5, 20):
        x, phi = solve_cd(N, Pe, "exponential")
        err = np.max(np.abs(phi - phi_exact(x, Pe)))
        print(f"  {Pe:>8.1f} {N:>5d} {err:>18.3e}")

print("\n" + "-" * 78)
print("  CHECK 2 -- CENTRAL DIFFERENCING OSCILLATES ABOVE CELL Pe = 2")
print("""    Central differencing is second order and, on paper, the most accurate.
    But its coefficient a_E = D(1 - 0.5|Pe_cell|) goes NEGATIVE when the cell
    Peclet number exceeds 2, and a negative neighbour coefficient violates
    Patankar's boundedness rule: the solution can then overshoot and oscillate.
    The threshold is exact, and the table locates it.""")
print(f"\n  {'N':>5} {'cell Pe = Pe/N':>15} {'min phi':>12} {'max phi':>12} "
      f"{'bounded?':>10}")
Pe = 50.0
for N in (5, 10, 20, 25, 26, 30, 50, 100):
    x, phi = solve_cd(N, Pe, "central")
    cellPe = Pe / N
    bounded = (phi.min() >= PHI0 - 1e-9) and (phi.max() <= PHIL + 1e-9)
    print(f"  {N:>5d} {cellPe:>15.4f} {phi.min():>12.5f} {phi.max():>12.5f} "
          f"{str(bounded):>10}")
print("""    The solution is bounded exactly when cell Pe <= 2, i.e. N >= Pe/2 = 25,
    and oscillates below that.  The wiggles are not a coding error; they are the
    scheme telling the truth about a matrix that has lost its diagonal
    dominance.  Refining the grid past N = 25 cures them -- which is the
    practical rule: central differencing needs cell Pe below 2.""")

print("\n" + "-" * 78)
print("  CHECK 3 -- ORDER OF ACCURACY OF EACH SCHEME")
print("""    Measured against the exact solution at a moderate Peclet number where
    all schemes are bounded, so the comparison is of ACCURACY, not stability.""")
Pe = 10.0


def L2_error(N, Pe, scheme):
    x, phi = solve_cd(N, Pe, scheme)
    e = phi[1:-1] - phi_exact(x[1:-1], Pe)
    dv = np.diff(np.linspace(0, LENGTH, N + 1))
    return np.sqrt(np.sum(e ** 2 * dv))


print(f"\n  {'scheme':>13} {'N=20':>11} {'N=40':>11} {'N=80':>11} "
      f"{'N=160':>11} {'order':>7}")
for scheme in SCHEMES:
    errs = [L2_error(N, Pe, scheme) for N in (20, 40, 80, 160)]
    order = np.log2(errs[-2] / errs[-1]) if errs[-1] > 1e-15 else float("nan")
    cells = "  ".join(f"{e:.2e}" for e in errs)
    print(f"  {scheme:>13} {errs[0]:>11.2e} {errs[1]:>11.2e} {errs[2]:>11.2e} "
          f"{errs[3]:>11.2e} {order:>7.2f}")
print("""    Central and exponential are second order (exponential's error is
    round-off, so its "order" is meaningless and it is exact).  Upwind is first
    order.  Hybrid and power-law are second order where they act as central and
    first order where they switch to upwind, so their measured order sits
    between the two and depends on how much of the domain is in each regime.""")

print("\n" + "-" * 78)
print("  CHECK 4 -- BOUNDEDNESS ACROSS THE PECLET RANGE")
print("""    A bounded scheme keeps every node within [phi_0, phi_L].  The table
    marks which schemes stay bounded on a fixed 20-cell grid as Peclet climbs
    and the cell Peclet number passes 2.""")
print(f"\n  {'Pe':>8} {'cell Pe':>9} " +
      " ".join(f"{s[:5]:>7}" for s in SCHEMES))
for Pe in (5.0, 20.0, 40.0, 80.0, 200.0):
    N = 20
    marks = []
    for scheme in SCHEMES:
        x, phi = solve_cd(N, Pe, scheme)
        ok = (phi.min() >= PHI0 - 1e-9) and (phi.max() <= PHIL + 1e-9)
        marks.append("  ok " if ok else " OSC ")
    print(f"  {Pe:>8.1f} {Pe/N:>9.2f} " + " ".join(f"{m:>7}" for m in marks))
print("    Only central differencing loses boundedness, and only once cell Pe")
print("    exceeds 2.  Every scheme in Patankar's family below it stays bounded")
print("    by construction, which is exactly why the family exists.")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

xf = np.linspace(0, 1, 400)
for Pe, c in ((1.0, "#2166ac"), (5.0, "#1b7837"), (20.0, "#e08214"),
              (100.0, "#b2182b")):
    ax[0].plot(xf, phi_exact(xf, Pe), "-", lw=2.0, color=c,
               label=rf"$Pe = {Pe:g}$")
ax[0].set_xlabel(r"$x/L$")
ax[0].set_ylabel(r"$\phi$")
ax[0].set_title("(a) The exact solution: convection wins downstream")
ax[0].legend(fontsize=8.5, loc="upper left")

# a coarse grid where central oscillates and upwind smears
Pe = 50.0
N = 10
xe = np.linspace(0, 1, 400)
ax[1].plot(xe, phi_exact(xe, Pe), "-", lw=2.0, color="0.3", label="exact")
for scheme, c, mk in (("central", "#b2182b", "o"),
                      ("upwind", "#2166ac", "s"),
                      ("power-law", "#1b7837", "^")):
    x, phi = solve_cd(N, Pe, scheme)
    ax[1].plot(x, phi, mk + "-", ms=5, mfc="none", mew=1.3, lw=1.2, color=c,
               label=scheme)
ax[1].axhline(0, color="0.6", lw=0.8, ls=":")
ax[1].axhline(1, color="0.6", lw=0.8, ls=":")
ax[1].annotate("central overshoots\n(cell $Pe = 5 > 2$)", xy=(0.55, 0.55),
               xycoords="axes fraction", fontsize=8.2, color="#b2182b",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.28"))
ax[1].set_xlabel(r"$x/L$")
ax[1].set_ylabel(r"$\phi$")
ax[1].set_title(rf"(b) Schemes on a coarse grid, $Pe = {Pe:g}$, $N = {N}$")
ax[1].legend(fontsize=8.5, loc="upper left")

fig.suptitle("Example 12.1 -- Convection-diffusion and its exact solution",
             fontsize=12.5, y=1.08)
fig.savefig("fig_12_1a_profiles.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

pp = np.linspace(0, 12, 400)
for scheme, c in (("central", "#b2182b"), ("upwind", "#2166ac"),
                  ("hybrid", "#1b7837"), ("power-law", "#e08214"),
                  ("exponential", "#762a83")):
    ax[0].plot(pp, SCHEMES[scheme](pp), "-", lw=2.0, color=c, label=scheme)
ax[0].axhline(0, color="0.5", lw=1.0)
ax[0].axvline(2, color="0.5", lw=1.0, ls=":")
ax[0].annotate("central goes\nnegative here", xy=(2, 0), xytext=(4.5, -0.9),
               fontsize=8.2, color="#b2182b",
               arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.0))
ax[0].set_xlabel(r"cell Peclet number $|Pe|$")
ax[0].set_ylabel(r"$A(|Pe|)$")
ax[0].set_title("(a) The scheme functions of Patankar's family")
ax[0].set_ylim(-1.2, 1.05)
ax[0].legend(fontsize=8.5, loc="upper right")

Ns = np.array([20, 40, 80, 160, 320])
Pe = 10.0
for scheme, c, mk in (("central", "#b2182b", "o"),
                      ("upwind", "#2166ac", "s"),
                      ("power-law", "#1b7837", "^")):
    errs = np.array([L2_error(N, Pe, scheme) for N in Ns])
    ax[1].loglog(Ns, errs, mk + "-", lw=1.8, ms=6, mfc="none", mew=1.5,
                 color=c, label=scheme)
ax[1].loglog(Ns, 3e-2 * (Ns[0] / Ns) ** 1.0, "k--", lw=1.1, label="slope -1")
ax[1].loglog(Ns, 3e-2 * (Ns[0] / Ns) ** 2.0, "k:", lw=1.3, label="slope -2")
ax[1].set_xlabel(r"$N$  (cells)")
ax[1].set_ylabel(r"$L_2$ error vs exact")
ax[1].set_title(rf"(b) Order of accuracy, $Pe = {Pe:g}$")
ax[1].legend(fontsize=8, loc="lower left")

fig.suptitle("Example 12.1 -- The schemes, compared",
             fontsize=12.5, y=1.08)
fig.savefig("fig_12_1b_schemes.png")
plt.close(fig)

print("Figures written: fig_12_1a_profiles.png, fig_12_1b_schemes.png")
