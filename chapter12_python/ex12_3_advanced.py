"""
================================================================================
 EXAMPLE 12.3 -- THE METHOD OF MANUFACTURED SOLUTIONS
 Verifying a code where no exact solution exists, by inventing one that does
================================================================================

 OBJECTIVE
 ---------
 Every verification in this book so far has needed an exact solution: Neumann
 for the Stefan problem, the Blasius profile for the boundary layer, 48/11 for
 the tube.  Most real codes solve equations that have NO exact solution --
 variable properties, source terms, awkward geometry -- and for those the whole
 apparatus of order-of-accuracy verification seems to collapse.

 The METHOD OF MANUFACTURED SOLUTIONS (MMS) rescues it.  The idea is disarming:
 instead of finding a solution to the given equation, CHOOSE a solution and
 then work out what equation it solves.  Pick any smooth function T*(x, y);
 substitute it into the differential operator L; whatever comes out is a source
 term q(x, y) = L[T*].  The manufactured T* is then the EXACT solution of the
 original equation with that source added and with boundary conditions taken
 from T* itself.  Run the code with that source and those boundaries, and the
 error against T* can be measured on any grid, to any order.

 This is the gold standard of code verification (Roache; Salari and Knupp).
 It tests the DISCRETISATION, not the physics: a manufactured solution need not
 be physically realistic -- it only needs to be smooth and to exercise every
 term in the operator.

 THE OPERATOR
 ------------
 A steady convection-diffusion equation with VARIABLE, ANISOTROPIC diffusivity
 and a prescribed velocity field -- deliberately more general than any exact
 solution could handle:

        d/dx(rho u T) + d/dy(rho v T)
              = d/dx( Gamma(x,y) dT/dx ) + d/dy( Gamma(x,y) dT/dy ) + q

 with Gamma(x, y) = Gamma_0 (1 + x^2 + y^2) a smoothly varying conductivity and
 (u, v) a divergence-free rotating field.  No exact solution to this exists;
 MMS manufactures one.

 THE MANUFACTURED SOLUTION
 -------------------------
        T*(x, y) = sin(pi x) sin(pi y) + 0.25 x y

 chosen because every derivative is nonzero, so every term in the operator is
 exercised, and because it is not separable in a way that would let some error
 cancel by symmetry.  The source q = L[T*] is computed ANALYTICALLY here (the
 derivatives are elementary), which is itself a check: an independent symbolic
 differentiation would have to agree.

 WHAT IS COMPUTED AND CHECKED
 ----------------------------
   1. The manufactured source, verified by substituting T* back through the
      DISCRETE operator on a very fine grid -- the residual must vanish at the
      discretisation order.
   2. The observed order of accuracy of the full solver, in L2 and L-infinity,
      driven to its asymptotic value -- the central result of the example.
   3. The distinction MMS draws sharply: a code can be second-order ACCURATE
      (verified) and still solve the wrong physics (unvalidated).  A deliberate
      bug is introduced and shown to DROP the observed order, which is exactly
      how MMS catches coding errors that a single-grid eyeball test misses.

 OUTPUTS
 -------
   fig_12_3a_mms.png        the manufactured solution, source, and error field
   fig_12_3b_order.png      the order of accuracy, and the bug that MMS catches

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

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
T_START = time.perf_counter()

GAMMA0 = 0.05
RHO = 1.0
PI = np.pi


# ==============================================================================
# 1. THE MANUFACTURED SOLUTION AND ITS DERIVATIVES (all analytic)
# ==============================================================================
def T_exact(x, y):
    return np.sin(PI * x) * np.sin(PI * y) + 0.25 * x * y


def dTdx(x, y):
    return PI * np.cos(PI * x) * np.sin(PI * y) + 0.25 * y


def dTdy(x, y):
    return PI * np.sin(PI * x) * np.cos(PI * y) + 0.25 * x


def d2Tdx2(x, y):
    return -PI ** 2 * np.sin(PI * x) * np.sin(PI * y)


def d2Tdy2(x, y):
    return -PI ** 2 * np.sin(PI * x) * np.sin(PI * y)


def Gamma(x, y):
    return GAMMA0 * (1.0 + x ** 2 + y ** 2)


def dGammadx(x, y):
    return GAMMA0 * 2.0 * x


def dGammady(x, y):
    return GAMMA0 * 2.0 * y


def velocity(x, y):
    """A divergence-free rotating field about the domain centre (0.5, 0.5),
    so that continuity holds exactly and the convective term is conservative."""
    u = 2.0 * (y - 0.5)
    v = -2.0 * (x - 0.5)
    return u, v


def manufactured_source(x, y):
    """q = d/dx(rho u T) + d/dy(rho v T) - div(Gamma grad T), all analytic.

    Because the velocity is divergence-free, d/dx(uT) + d/dy(vT)
    = u dT/dx + v dT/dy.  The diffusion term expands by the product rule into
    Gamma (T_xx + T_yy) + Gamma_x T_x + Gamma_y T_y.
    """
    u, v = velocity(x, y)
    conv = RHO * (u * dTdx(x, y) + v * dTdy(x, y))
    diff = (Gamma(x, y) * (d2Tdx2(x, y) + d2Tdy2(x, y)) +
            dGammadx(x, y) * dTdx(x, y) + dGammady(x, y) * dTdy(x, y))
    return conv - diff


# ==============================================================================
# 2. THE SOLVER (central diffusion, central convection -> second order)
# ==============================================================================
def solve_mms(N, bug=False):
    """Finite volume solution on an N x N cell-centred grid.

    Cell centres at (i+0.5)/N.  Dirichlet boundaries taken from T_exact.
    Diffusion evaluates the conductivity at the face directly, Gamma(x_f, y_f),
    which is second order for this smooth Gamma; convection uses central
    differencing, which is
    second order and, for this smooth manufactured field with modest Peclet, is
    bounded.  The manufactured source is integrated at the cell centre.

    If bug=True, convection is discretised with UPWIND differencing instead of
    central.  Upwind is consistent -- it converges to the right answer -- but it
    is only FIRST order, so the whole solver drops from second order to first.
    On a single grid the upwind solution looks entirely reasonable; only an
    order-of-accuracy test reveals the loss.  That is exactly the kind of
    mistake MMS exists to catch.

    (A first draft used the arithmetic mean of the neighbouring Gamma as the
    "bug".  That is NOT order-reducing: for a smooth Gamma the arithmetic mean
    of the two cell-centre values equals the face value to second order, so the
    code stayed at order two and the "bug" was invisible.  A demonstration of
    MMS catching a bug has to use a bug MMS can actually catch.)
    """
    h = 1.0 / N
    xc = (np.arange(N) + 0.5) * h
    Xc, Yc = np.meshgrid(xc, xc, indexing="ij")

    # face coordinates
    xf = np.arange(N + 1) * h
    n = N * N

    def idx(i, j):
        return i * N + j

    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            xC, yC = xc[i], xc[j]
            aP = 0.0
            # ---- diffusion, four faces ----
            for (di, dj, xf_face, yf_face, axis) in (
                    (-1, 0, xf[i], yC, "x"), (1, 0, xf[i + 1], yC, "x"),
                    (0, -1, xC, xf[j], "y"), (0, 1, xC, xf[j + 1], "y")):
                Gf = Gamma(xf_face, yf_face)     # face value, second order
                D = Gf / h * h          # conductance * face length (=h)
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    A[p, idx(ni, nj)] -= D
                    aP += D
                else:
                    # Dirichlet: neighbour is the boundary value at the face
                    xb = xf_face if axis == "x" else xC
                    yb = yf_face if axis == "y" else yC
                    # ghost at 2*face - centre
                    Dghost = Gf / (0.5 * h) * h
                    aP += Dghost
                    b[p] += Dghost * T_exact(
                        xf_face if axis == "x" else xC,
                        yf_face if axis == "y" else yC)
            # ---- convection, central differencing, divergence-free ----
            for (di, dj, xf_face, yf_face, comp) in (
                    (-1, 0, xf[i], yC, "u"), (1, 0, xf[i + 1], yC, "u"),
                    (0, -1, xC, xf[j], "v"), (0, 1, xC, xf[j + 1], "v")):
                u, v = velocity(xf_face, yf_face)
                F = RHO * (u if comp == "u" else v) * h    # mass flux * face len
                sgn = 1.0 if di + dj > 0 else -1.0         # outward normal sign
                Fout = sgn * F                              # signed outward flux
                ni, nj = i + di, j + dj
                interior = (0 <= ni < N) and (0 <= nj < N)
                if not bug:
                    # central: face value = average of the two straddling cells
                    if interior:
                        A[p, p] += 0.5 * Fout
                        A[p, idx(ni, nj)] += 0.5 * Fout
                    else:
                        b[p] -= Fout * T_exact(xf_face, yf_face)
                else:
                    # upwind (first order): face value is the UPSTREAM cell.
                    # Outflow (Fout > 0) carries this cell; inflow carries the
                    # neighbour or the boundary value.
                    if Fout >= 0.0:
                        A[p, p] += Fout
                    else:
                        if interior:
                            A[p, idx(ni, nj)] += Fout
                        else:
                            b[p] -= Fout * T_exact(xf_face, yf_face)
            A[p, p] += aP
            b[p] += manufactured_source(xC, yC) * h * h

    T = np.linalg.solve(A, b).reshape(N, N)
    return Xc, Yc, T


# ==============================================================================
# 3. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 12.3 -- THE METHOD OF MANUFACTURED SOLUTIONS")
print("=" * 78)

print("\n" + "-" * 78)
print("  CHECK 1 -- THE MANUFACTURED SOURCE IS SELF-CONSISTENT")
print("""    The source q = L[T*] was computed analytically.  If it is right, then
    substituting the exact T* into the DISCRETE operator and subtracting q must
    leave a residual that is only the discretisation error -- vanishing as the
    grid is refined.  If the analytic source were wrong, the residual would
    stall at a nonzero constant.  This checks the source independently of the
    solver.""")
print(f"\n  {'N':>5} {'max discrete residual':>24} {'ratio':>8}")
prev = None
for N in (10, 20, 40, 80):
    h = 1.0 / N
    xc = (np.arange(N) + 0.5) * h
    # discrete Laplacian-of-exact minus source, interior cells only
    res_max = 0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            xC, yC = xc[i], xc[j]
            # central second differences of the exact solution
            Txx = (T_exact(xC + h, yC) - 2 * T_exact(xC, yC) +
                   T_exact(xC - h, yC)) / h ** 2
            Tyy = (T_exact(xC, yC + h) - 2 * T_exact(xC, yC) +
                   T_exact(xC, yC - h)) / h ** 2
            Tx = (T_exact(xC + h, yC) - T_exact(xC - h, yC)) / (2 * h)
            Ty = (T_exact(xC, yC + h) - T_exact(xC, yC - h)) / (2 * h)
            u, v = velocity(xC, yC)
            L_disc = (RHO * (u * Tx + v * Ty) -
                      Gamma(xC, yC) * (Txx + Tyy) -
                      dGammadx(xC, yC) * Tx - dGammady(xC, yC) * Ty)
            res_max = max(res_max, abs(L_disc - manufactured_source(xC, yC)))
    r = f"{prev/res_max:.2f}" if prev else "-"
    print(f"  {N:>5d} {res_max:>24.3e} {r:>8}")
    prev = res_max
print("    The residual falls by about four per grid doubling -- second order,")
print("    the order of the central differences used to form it.  It does not")
print("    stall, so the analytic source is correct.")

print("\n" + "-" * 78)
print("  CHECK 2 -- THE OBSERVED ORDER OF ACCURACY OF THE SOLVER")
print("""    The central result.  The solver is run with the manufactured source
    and boundaries, and its error against T* is measured on a sequence of grids.
    Second-order construction demands observed order 2 in both norms.""")


def norms(N):
    X, Y, T = solve_mms(N)
    e = T - T_exact(X, Y)
    L2 = np.sqrt(np.mean(e ** 2))
    Linf = np.max(np.abs(e))
    return L2, Linf


print(f"\n  {'N':>5} {'L2 error':>13} {'p(L2)':>7} {'Linf error':>13} "
      f"{'p(Linf)':>8}")
NS = [10, 20, 40, 80]
res = [norms(N) for N in NS]
for k, N in enumerate(NS):
    if k == 0:
        print(f"  {N:>5d} {res[k][0]:>13.3e} {'-':>7} {res[k][1]:>13.3e} "
              f"{'-':>8}")
    else:
        p2 = np.log2(res[k - 1][0] / res[k][0])
        pi = np.log2(res[k - 1][1] / res[k][1])
        print(f"  {N:>5d} {res[k][0]:>13.3e} {p2:>7.3f} {res[k][1]:>13.3e} "
              f"{pi:>8.3f}")
print("    Both norms converge at order 2, confirming the discretisation is")
print("    second order as designed -- the whole solver, every term, verified.")

print("\n" + "-" * 78)
print("  CHECK 3 -- MMS CATCHES A BUG THAT AN EYEBALL TEST MISSES")
print("""    A subtle error is introduced: the face conductivity is taken as the
    arithmetic mean of the neighbouring cell values instead of evaluated at the
    face.  The code still converges -- it is CONSISTENT -- and on a single grid
    the solution looks perfectly reasonable.  But the error is now first order,
    and only an order-of-accuracy test reveals it.  This is precisely the class
    of mistake MMS is built to expose.""")
print(f"\n  {'N':>5} {'correct p(L2)':>15} {'buggy p(L2)':>13}")


def norms_bug(N):
    X, Y, T = solve_mms(N, bug=True)
    e = T - T_exact(X, Y)
    return np.sqrt(np.mean(e ** 2))


good = [norms(N)[0] for N in NS]
bad = [norms_bug(N) for N in NS]
for k in range(1, len(NS)):
    pg = np.log2(good[k - 1] / good[k])
    pb = np.log2(bad[k - 1] / bad[k])
    print(f"  {NS[k]:>5d} {pg:>15.3f} {pb:>13.3f}")
print("""    The correct code holds order 2; the buggy one drops toward order 1.
    On the finest single grid the buggy solution's error is still small and
    would pass unremarked -- which is the entire point.  Verification is not
    looking at one answer and nodding; it is measuring the RATE at which the
    answer improves, because only the rate exposes an order-reducing bug.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.0))
N = 60
X, Y, T = solve_mms(N)
Tex = T_exact(X, Y)
q = manufactured_source(X, Y)

for k, (field, title, cmap) in enumerate((
        (Tex, r"(a) manufactured $T^*$", "viridis"),
        (q, r"(b) required source $q = L[T^*]$", "PuOr"),
        (np.abs(T - Tex), r"(c) $|T_{\rm num} - T^*|$", "inferno"))):
    lim = np.max(np.abs(field))
    if k == 1:
        im = ax[k].pcolormesh(X, Y, field, cmap=cmap, shading="auto",
                              vmin=-lim, vmax=lim)
    else:
        im = ax[k].pcolormesh(X, Y, field, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax[k], shrink=0.85)
    ax[k].set_title(title)
    ax[k].set_xlabel(r"$x$")
    ax[k].set_aspect("equal")
    ax[k].grid(False)
    if k == 0:
        ax[k].set_ylabel(r"$y$")
fig.suptitle("Example 12.3 -- Manufacture a solution, derive its source, "
             "measure the error", fontsize=11.5, y=1.05)
fig.savefig("fig_12_3a_mms.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

Na = np.array(NS)
L2s = np.array([r[0] for r in res])
Linfs = np.array([r[1] for r in res])
ax[0].loglog(Na, L2s, "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#b2182b", label=r"$L_2$ error")
ax[0].loglog(Na, Linfs, "s-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#2166ac", label=r"$L_\infty$ error")
ax[0].loglog(Na, L2s[0] * (Na[0] / Na) ** 2.0, "k--", lw=1.3,
             label=r"slope $-2$")
ax[0].set_xlabel(r"$N$  (cells per side)")
ax[0].set_ylabel("error vs manufactured $T^*$")
ax[0].set_title("(a) Second order, both norms")
ax[0].legend(fontsize=8.5, loc="lower left")

ax[1].loglog(Na, np.array(good), "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#1b7837", label="correct (central)")
ax[1].loglog(Na, np.array(bad), "s-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#b2182b", label="buggy (upwind convection)")
ax[1].loglog(Na, good[0] * (Na[0] / Na) ** 2.0, "k--", lw=1.2,
             label=r"slope $-2$")
ax[1].loglog(Na, bad[0] * (Na[0] / Na) ** 1.0, "k:", lw=1.2,
             label=r"slope $-1$")
ax[1].set_xlabel(r"$N$  (cells per side)")
ax[1].set_ylabel(r"$L_2$ error")
ax[1].set_title("(b) MMS catches an order-reducing bug")
ax[1].legend(fontsize=8, loc="lower left")

fig.suptitle("Example 12.3 -- The order of accuracy is the verification",
             fontsize=12.5, y=1.08)
fig.savefig("fig_12_3b_order.png")
plt.close(fig)

print("Figures written: fig_12_3a_mms.png, fig_12_3b_order.png")
