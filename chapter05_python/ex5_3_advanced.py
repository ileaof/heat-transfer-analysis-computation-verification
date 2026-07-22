"""
================================================================================
 EXAMPLE 5.3 -- ADVANCED VERIFICATION OF THE 2-D POISSON SOLVER
 Manufactured solutions, conjugate gradients, and mesh anisotropy
================================================================================

 OBJECTIVE
 ---------
 Example 5.2 verified the Laplace solver against a single-term separation-of-
 variables solution.  Here the source term is restored -- the POISSON equation
 -- and three questions are settled that Example 5.2 could not reach:

   1. Is the scheme still second order with a non-zero source and with
      MIXED boundary conditions (Dirichlet, Neumann and Robin on different
      edges)?  Verified by the Method of Manufactured Solutions of Chapter 3.
   2. How does the CONJUGATE GRADIENT method compare with SOR, and how does
      each scale with mesh size?
   3. What happens when the control volumes are strongly ANISOTROPIC
      (dx >> dy), as they must be in thin geometries?

 GOVERNING EQUATION
 ------------------
        d2T/dx2 + d2T/dy2 + S(x,y)/k = 0

 THE MANUFACTURED FIELD
 ----------------------
   T_mms(x,y) = Ta + A sin(pi x/W) cos(pi y/(2H))
                   + B (x/W)^2 (1 - y/H)
                   + C (x/W)(y/H)

 chosen so that every term of the operator is exercised and so that the
 boundary data are non-trivial on all four edges.  Applying the Laplacian:

   term 1 : -A [ (pi/W)^2 + (pi/(2H))^2 ] sin(pi x/W) cos(pi y/(2H))
   term 2 : (2B/W^2)(1 - y/H)
   term 3 : 0                          (bilinear terms are harmonic)

 so the required source is

   S(x,y) = -k [ -A((pi/W)^2 + (pi/(2H))^2) sin cos + (2B/W^2)(1 - y/H) ]

 BOUNDARY CONDITIONS (deliberately mixed)
 ----------------------------------------
   x = 0 : Dirichlet,  T = T_mms(0,y)
   x = W : Dirichlet,  T = T_mms(W,y)
   y = 0 : Neumann,    -k dT/dy = q_s(x) taken from T_mms
   y = H : Robin,      -k dT/dy = h (T - T_inf_eff(x)) with T_inf_eff chosen
                       so the condition holds identically

 A code that passes MMS with all three kinds of boundary condition present has
 had every boundary branch exercised, which no single physical test does.

 SYMBOLS -- see Examples 5.1 and 5.2; additionally
   A, B, C, Ta  manufactured-solution parameters [K]
   S            manufactured volumetric source [W/m^3]
   AR           cell aspect ratio dx/dy [-]

 OUTPUTS
 -------
   fig_5_3a_mms.png        manufactured field, source and error
   fig_5_3b_orders.png     convergence and solver scaling
   fig_5_3c_anisotropy.png effect of cell aspect ratio

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
})

# ==============================================================================
# 1. DATA
# ==============================================================================
W, H = 0.40, 0.30
K = 15.0
HCONV = 250.0
TA, AMP, BQ, CQ = 350.0, 80.0, 40.0, 25.0

KX, KY = np.pi / W, np.pi / (2.0 * H)


# ==============================================================================
# 2. MANUFACTURED SOLUTION, ITS DERIVATIVES AND ITS SOURCE
# ==============================================================================
def T_mms(x, y):
    return (TA + AMP * np.sin(KX * x) * np.cos(KY * y)
            + BQ * (x / W) ** 2 * (1.0 - y / H) + CQ * (x / W) * (y / H))


def dTdy_mms(x, y):
    return (-AMP * KY * np.sin(KX * x) * np.sin(KY * y)
            - BQ * (x / W) ** 2 / H + CQ * (x / W) / H)


def source(x, y):
    """S [W/m^3] making T_mms exact."""
    lap = (-AMP * (KX**2 + KY**2) * np.sin(KX * x) * np.cos(KY * y)
           + 2.0 * BQ / W**2 * (1.0 - y / H))
    return -K * lap


def T_inf_eff(x):
    """Sink temperature making the Robin condition at y = H exact."""
    return T_mms(x, H) + (K / HCONV) * dTdy_mms(x, H)


# ==============================================================================
# 3. MESH, COEFFICIENTS, SOLVERS
# ==============================================================================
def mesh(Nx, Ny):
    dx, dy = W / Nx, H / Ny
    return (np.arange(Nx) + 0.5) * dx, (np.arange(Ny) + 0.5) * dy, dx, dy


def coefficients(Nx, Ny):
    xc, yc, dx, dy = mesh(Nx, Ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    aE = np.full((Nx, Ny), K * dy / dx)
    aW = np.full((Nx, Ny), K * dy / dx)
    aN = np.full((Nx, Ny), K * dx / dy)
    aS = np.full((Nx, Ny), K * dx / dy)
    Sp = np.zeros((Nx, Ny))
    b = source(X, Y) * dx * dy                     # exact cell integral? see note

    # The source is integrated by the midpoint rule, which is second order --
    # consistent with the rest of the scheme.  Using the exact integral would
    # not raise the overall order, because the flux approximation is also
    # second order.
    gx, gy = 2.0 * K * dy / dx, 2.0 * K * dx / dy

    # x = 0 and x = W : Dirichlet from the manufactured field
    aW[0, :] = 0.0
    Sp[0, :] -= gx
    b[0, :] += gx * T_mms(0.0, yc)
    aE[-1, :] = 0.0
    Sp[-1, :] -= gx
    b[-1, :] += gx * T_mms(W, yc)

    # y = 0 : Neumann.  The prescribed flux enters b directly; no Sp term,
    # which is why a pure-Neumann problem would be singular.
    q_s = -K * dTdy_mms(xc, 0.0)                   # [W/m^2] entering the face
    aS[:, 0] = 0.0
    b[:, 0] += q_s * dx

    # y = H : Robin, half-cell in series with the film
    U = 1.0 / (dy / (2.0 * K) + 1.0 / HCONV)
    aN[:, -1] = 0.0
    Sp[:, -1] -= U * dx
    b[:, -1] += U * dx * T_inf_eff(xc)

    return aE, aW, aN, aS, aE + aW + aN + aS - Sp, b


def apply_A(T, aE, aW, aN, aS, aP):
    r = aP * T
    r[:-1, :] -= aE[:-1, :] * T[1:, :]
    r[1:, :] -= aW[1:, :] * T[:-1, :]
    r[:, :-1] -= aN[:, :-1] * T[:, 1:]
    r[:, 1:] -= aS[:, 1:] * T[:, :-1]
    return r


def solve_sor(Nx, Ny, omega=None, tol=1e-11, max_sweeps=100000):
    aE, aW, aN, aS, aP, b = coefficients(Nx, Ny)
    if omega is None:
        omega = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))
    T = np.full((Nx, Ny), TA, dtype=float)
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    colours = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]
    scale = np.max(np.abs(b))
    hist = []
    for sweep in range(1, max_sweeps + 1):
        for c in colours:
            s = np.zeros_like(T)
            s[:-1, :] += aE[:-1, :] * T[1:, :]
            s[1:, :] += aW[1:, :] * T[:-1, :]
            s[:, :-1] += aN[:, :-1] * T[:, 1:]
            s[:, 1:] += aS[:, 1:] * T[:, :-1]
            T = np.where(c, T + omega * ((s + b) / aP - T), T)
        if sweep % 5 == 0:
            r = np.max(np.abs(b - apply_A(T, aE, aW, aN, aS, aP))) / scale
            hist.append((sweep, r))
            if r < tol:
                break
    return T, {"iters": sweep, "hist": np.array(hist), "omega": omega}


def solve_cg(Nx, Ny, tol=1e-11, max_iter=20000):
    """Conjugate gradients with Jacobi (diagonal) preconditioning.

    The matrix is symmetric positive definite -- a consequence of the
    control-volume discretisation of a self-adjoint operator with these
    boundary conditions -- so CG applies.  It is written out in full rather
    than called from a library because the structure matters: each iteration
    costs one matrix-vector product, which for this five-point stencil is four
    shifted multiplications and no storage of the matrix at all.
    """
    aE, aW, aN, aS, aP, b = coefficients(Nx, Ny)
    T = np.full((Nx, Ny), TA, dtype=float)
    scale = np.max(np.abs(b))
    r = b - apply_A(T, aE, aW, aN, aS, aP)
    z = r / aP                                  # Jacobi preconditioner
    p = z.copy()
    rz = np.sum(r * z)
    hist = []
    for it in range(1, max_iter + 1):
        Ap = apply_A(p, aE, aW, aN, aS, aP)
        alpha = rz / np.sum(p * Ap)
        T += alpha * p
        r -= alpha * Ap
        res = np.max(np.abs(r)) / scale
        hist.append((it, res))
        if res < tol:
            break
        z = r / aP
        rz_new = np.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return T, {"iters": it, "hist": np.array(hist)}


def errors(T, Nx, Ny):
    xc, yc, dx, dy = mesh(Nx, Ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    e = T - T_mms(X, Y)
    return np.sqrt(np.mean(e**2)), np.max(np.abs(e))


# ==============================================================================
# 4. VERIFY THE MANUFACTURED SOLUTION ITSELF
# ==============================================================================
print("=" * 78)
print("EXAMPLE 5.3 -- MMS VERIFICATION OF THE 2-D POISSON SOLVER")
print("=" * 78)
xs = np.linspace(0.01, W - 0.01, 300)
ys = np.linspace(0.01, H - 0.01, 240)
X, Y = np.meshgrid(xs, ys, indexing="ij")
hx, hy = xs[1] - xs[0], ys[1] - ys[0]
F = T_mms(X, Y)
lap = ((F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / hx**2
       + (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / hy**2)
res = K * lap + source(X[1:-1, 1:-1], Y[1:-1, 1:-1])
print(f"  PDE residual  max|k lap T + S| / max|S| = "
      f"{np.max(np.abs(res))/np.max(np.abs(source(X, Y))):.3e}")
xb = np.linspace(0, W, 101)
rob = np.max(np.abs(-K * dTdy_mms(xb, H) - HCONV * (T_mms(xb, H) - T_inf_eff(xb))))
print(f"  Robin condition residual at y = H       = {rob:.3e}")
print(f"  Boundary kinds exercised: Dirichlet (x=0, x=W), Neumann (y=0), "
      f"Robin (y=H)")

# ==============================================================================
# 5. ORDER OF ACCURACY
# ==============================================================================
print("\n" + "-" * 78)
print("  GRID CONVERGENCE (SOR, tol = 1e-11)")
print(f"  {'Nx x Ny':>10} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} "
      f"{'p_inf':>7} {'iters':>7} {'CPU [s]':>9}")
rows = []
for Nx, Ny in [(16, 12), (32, 24), (64, 48), (128, 96)]:
    t0 = time.perf_counter()
    T, info = solve_sor(Nx, Ny)
    cpu = time.perf_counter() - t0
    l2, li = errors(T, Nx, Ny)
    if rows:
        p2 = np.log(rows[-1]["l2"] / l2) / np.log(2.0)
        pi_ = np.log(rows[-1]["li"] / li) / np.log(2.0)
    else:
        p2 = pi_ = float("nan")
    rows.append({"Nx": Nx, "Ny": Ny, "h": W / Nx, "l2": l2, "li": li, "T": T,
                 "iters": info["iters"], "cpu": cpu})
    print(f"  {f'{Nx} x {Ny}':>10} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} "
          f"{pi_:>7.3f} {info['iters']:>7d} {cpu:>9.3f}")

def probe(Nx, Ny):
    T, _ = solve_sor(Nx, Ny, tol=1e-12)
    xc, yc, _, _ = mesh(Nx, Ny)
    col = np.array([np.interp(0.5 * H, yc, T[i, :]) for i in range(Nx)])
    return float(np.interp(0.5 * W, xc, col))

v = [probe(n, m) for n, m in [(32, 24), (64, 48), (128, 96)]]
p_obs = np.log(abs((v[0] - v[1]) / (v[1] - v[2]))) / np.log(2.0)
v_rich = v[2] + (v[2] - v[1]) / (2.0**p_obs - 1.0)
v_ex = float(T_mms(0.5 * W, 0.5 * H))
GCI = 1.25 * abs((v[2] - v[1]) / v[2]) / (2.0**p_obs - 1.0) * 100.0
print(f"\n  Richardson at the fixed point (W/2, H/2):")
print(f"    {v[0]:.9f} / {v[1]:.9f} / {v[2]:.9f} K")
print(f"    observed order p       = {p_obs:.4f}")
print(f"    extrapolated           = {v_rich:.9f} K")
print(f"    manufactured (exact)   = {v_ex:.9f} K")
print(f"    |extrapolated - exact| = {abs(v_rich - v_ex):.3e} K")
print(f"    |finest      - exact|  = {abs(v[2] - v_ex):.3e} K")
print(f"    GCI_fine               = {GCI:.6f} %")

# ==============================================================================
# 6. SOLVER SCALING
# ==============================================================================
print("\n" + "-" * 78)
print("  SOR versus PRECONDITIONED CONJUGATE GRADIENTS (tol = 1e-10)")
print(f"  {'Nx x Ny':>10} {'SOR its':>9} {'SOR [s]':>9} {'CG its':>8} "
      f"{'CG [s]':>9} {'speed-up':>10}")
scal = []
for Nx, Ny in [(16, 12), (32, 24), (64, 48), (128, 96)]:
    t0 = time.perf_counter()
    _, si = solve_sor(Nx, Ny, tol=1e-10)
    ts = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, ci = solve_cg(Nx, Ny, tol=1e-10)
    tc = time.perf_counter() - t0
    scal.append((Nx, si["iters"], ts, ci["iters"], tc))
    print(f"  {f'{Nx} x {Ny}':>10} {si['iters']:>9d} {ts:>9.3f} "
          f"{ci['iters']:>8d} {tc:>9.3f} {ts/tc:>9.2f}x")
print("  Both scale far better than the O(N^2) of Jacobi, but CG needs no")
print("  problem-dependent parameter: SOR must be given a good omega, and a")
print("  poor choice costs an order of magnitude (Example 5.2).")

# ==============================================================================
# 7. MESH ANISOTROPY
# ==============================================================================
print("\n" + "-" * 78)
print("  CELL ASPECT RATIO.  Total cell count held near 3072.")
print(f"  {'Nx x Ny':>10} {'dx/dy':>8} {'L2 [K]':>13} {'Linf [K]':>13} "
      f"{'SOR its':>9}")
aniso = []
for Nx, Ny in [(16, 192), (32, 96), (64, 48), (128, 24), (256, 12)]:
    T, info = solve_sor(Nx, Ny, tol=1e-10, max_sweeps=200000)
    l2, li = errors(T, Nx, Ny)
    ar = (W / Nx) / (H / Ny)
    aniso.append((ar, l2, li, info["iters"], Nx, Ny))
    print(f"  {f'{Nx} x {Ny}':>10} {ar:>8.3f} {l2:>13.4e} {li:>13.4e} "
          f"{info['iters']:>9d}")
best = min(aniso, key=lambda t: t[1])
print(f"  Most accurate at dx/dy = {best[0]:.3f} ({best[4]}x{best[5]}).")
print("  Note that the optimum is NOT square cells.  The truncation error")
print("  carries dx^2 and dy^2 weighted by the solution's own curvature in")
print("  each direction, and this manufactured field varies over a full")
print("  half-period in x but only a quarter-period in y, so it tolerates")
print("  coarser cells in y.  Accuracy degrades in both directions away from")
print("  that optimum: making one spacing small while the other stays large")
print("  leaves the larger term untouched, so the effort is wasted.")
print("  The practical rule is to match cell spacing to where the SOLUTION")
print("  varies, not to make cells square for their own sake.")
print("  The iteration count also rises, since the effective coupling weakens")
print("  along the stretched direction.")
print("=" * 78)

# ==============================================================================
# 8. FIGURES
# ==============================================================================
Nx, Ny = 64, 48
T, _ = solve_sor(Nx, Ny, tol=1e-12)
xc, yc, dx, dy = mesh(Nx, Ny)
XG, YG = np.meshgrid(xc, yc, indexing="ij")

fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.9),
    constrained_layout=True)
for a, F_, ttl, cm in [(ax[0], T_mms(XG, YG), "(a) Manufactured field [K]",
                        "inferno"),
                       (ax[1], source(XG, YG) * 1e-3,
                        "(b) Required source [kW m$^{-3}$]", "viridis"),
                       (ax[2], (T - T_mms(XG, YG)) * 1e3,
                        "(c) Error, $64\\times48$ [mK]", "RdBu_r")]:
    cf = a.contourf(XG * 1e3, YG * 1e3, F_, levels=22, cmap=cm)
    a.set_xlabel(r"$x$ [mm]"); a.set_ylabel(r"$y$ [mm]")
    a.set_title(ttl); a.set_aspect("equal"); a.grid(False)
    fig.colorbar(cf, ax=a, fraction=0.046, pad=0.03)
fig.suptitle("Example 5.3 -- Manufactured solution for the 2-D Poisson equation",
             fontsize=12.5, y=1.04)
fig.savefig("fig_5_3a_mms.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
hs = np.array([r["h"] for r in rows])
ax[0].loglog(hs * 1e3, [r["l2"] for r in rows], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(hs * 1e3, [r["li"] for r in rows], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$")
ax[0].loglog(hs * 1e3, rows[0]["l2"] * (hs / hs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [mm]")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title(f"(a) MMS convergence ($p = {p_obs:.2f}$)")
ax[0].legend(fontsize=9, loc="lower right")

ns = np.array([s[0] for s in scal])
ax[1].loglog(ns, [s[1] for s in scal], "o-", lw=1.8, ms=6, color="#b2182b",
             label="SOR sweeps")
ax[1].loglog(ns, [s[3] for s in scal], "s-", lw=1.8, ms=6, color="#1b7837",
             label="CG iterations")
ax[1].loglog(ns, scal[0][1] * (ns / ns[0]), "k--", lw=1.2, label=r"$O(N)$")
ax[1].set_xlabel(r"$N_x$")
ax[1].set_ylabel("Iterations to $10^{-10}$")
ax[1].set_title("(b) Solver scaling")
ax[1].legend(fontsize=9, loc="upper left")

fig.suptitle("Example 5.3 -- Order of accuracy and solver scaling",
             fontsize=12.5, y=1.02)
fig.savefig("fig_5_3b_orders.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
ars = np.array([a[0] for a in aniso])
ax[0].loglog(ars, [a[1] for a in aniso], "o-", lw=1.9, ms=7, color="#2166ac",
             label=r"$\|e\|_2$")
ax[0].loglog(ars, [a[2] for a in aniso], "s-", lw=1.9, ms=7, color="#b2182b",
             label=r"$\|e\|_\infty$")
ax[0].axvline(1.0, color="0.45", ls="--", lw=1.3)
ax[0].annotate("square cells", xy=(1.06, min(a[1] for a in aniso) * 1.4),
               fontsize=9, color="0.35", rotation=90)
ax[0].set_xlabel(r"Cell aspect ratio $\Delta x/\Delta y$")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title("(a) Accuracy at fixed cell count")
ax[0].legend(fontsize=9)

ax[1].loglog(ars, [a[3] for a in aniso], "^-", lw=1.9, ms=7, color="#1b7837")
ax[1].axvline(1.0, color="0.45", ls="--", lw=1.3)
ax[1].set_xlabel(r"Cell aspect ratio $\Delta x/\Delta y$")
ax[1].set_ylabel("SOR sweeps to $10^{-10}$")
ax[1].set_title("(b) Iteration cost")

fig.suptitle("Example 5.3 -- The price of stretched cells", fontsize=12.5,
             y=1.08)
fig.savefig("fig_5_3c_anisotropy.png")
plt.close(fig)

print("\nFigures written: fig_5_3a_mms.png, fig_5_3b_orders.png, "
      "fig_5_3c_anisotropy.png")
