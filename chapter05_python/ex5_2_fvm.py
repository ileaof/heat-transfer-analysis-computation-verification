"""
================================================================================
 EXAMPLE 5.2 -- TWO-DIMENSIONAL FINITE VOLUME SOLUTION
 and why the tridiagonal solver of Chapters 1-4 no longer works
================================================================================

 OBJECTIVE
 ---------
 Solve the Example 5.1 plate by the control-volume method, and confront the
 structural change that two dimensions bring: each control volume now has FOUR
 neighbours instead of two, so the matrix is pentadiagonal rather than
 tridiagonal and the Thomas algorithm no longer applies.

 This is the point in the book where ITERATIVE solvers become necessary, and
 three are implemented and compared: Jacobi, Gauss-Seidel and successive
 over-relaxation (SOR).

 GOVERNING EQUATION
 ------------------
        d2T/dx2 + d2T/dy2 = 0        on 0 < x < W, 0 < y < H

 BOUNDARY CONDITIONS
 -------------------
   T = T1 on the left, right and bottom edges
   T = T1 + dT sin(pi x/W) on the top edge      (Case S of Example 5.1)

 The sinusoidal top is used because its exact solution is a single term with no
 truncation error and no corner singularity, so any discrepancy is the scheme's
 own.  Case U is revisited at the end to show what a singular corner does to
 the measured order of accuracy.

 DISCRETISATION
 --------------
 Integrating over a control volume of size dx by dy (unit depth):

        [k dT/dx]_e dy - [k dT/dx]_w dy
      + [k dT/dy]_n dx - [k dT/dy]_s dx = 0

 With linear profiles between nodes,

        a_P T_P = a_E T_E + a_W T_W + a_N T_N + a_S T_S + b

        a_E = a_W = k dy/dx ,   a_N = a_S = k dx/dy ,
        a_P = a_E + a_W + a_N + a_S - S_P

 Dirichlet edges are imposed through the half-cell device of Chapter 2, e.g.
 at the west edge  S_P -= 2 k dy/dx,  S_u += (2 k dy/dx) T_wall.

 WHY THE THOMAS ALGORITHM FAILS
 ------------------------------
 In one dimension the unknowns form a chain and the matrix is tridiagonal, so
 Gaussian elimination costs O(N).  In two dimensions, numbering the unknowns
 row by row, the north and south neighbours sit Nx entries away from the
 diagonal.  The matrix is still sparse -- five non-zeros per row -- but its
 BANDWIDTH is Nx, and banded elimination costs O(N Nx^2) with O(N Nx) fill-in.
 For a 200 x 200 mesh that is 1.6e9 stored values.  Iterative methods, which
 never form the factorisation at all, become the practical choice.

 THE THREE ITERATIVE SCHEMES
 ---------------------------
   Jacobi           update every node from OLD neighbours
   Gauss-Seidel     update using new values as soon as they are available
   SOR              extrapolate past the Gauss-Seidel update by a factor omega

 SOR reduces to Gauss-Seidel at omega = 1 and diverges for omega >= 2.  The
 optimal omega is found here by direct measurement rather than by formula.

 SYMBOLS -- see Example 5.1; additionally
   Nx, Ny  [-]   control volumes in each direction
   dx, dy  [m]   control-volume dimensions
   omega   [-]   over-relaxation factor
   rho     [-]   measured asymptotic convergence factor per sweep

 OUTPUTS
 -------
   fig_5_2a_verification.png   field, error map and residual histories
   fig_5_2b_solvers.png        solver comparison and optimal omega

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

W, H = 0.40, 0.30
K = 15.0
T1 = 300.0
T2 = 400.0
DT = 100.0


# ==============================================================================
# 1. EXACT SOLUTIONS (Example 5.1)
# ==============================================================================
def T_sin(x, y):
    lam = np.pi / W
    ratio = np.exp(lam * (y - H)) * (1.0 - np.exp(-2.0 * lam * y)) / (
        1.0 - np.exp(-2.0 * lam * H))
    return T1 + DT * np.sin(lam * x) * ratio


def T_uni(x, y, n_terms=400):
    out = np.zeros(np.shape(x))
    for n in range(1, n_terms + 1, 2):
        lam = n * np.pi / W
        ratio = np.exp(lam * (y - H)) * (1.0 - np.exp(-2.0 * lam * y)) / (
            1.0 - np.exp(-2.0 * lam * H))
        out = out + (4.0 / (n * np.pi)) * np.sin(lam * x) * ratio
    return T1 + (T2 - T1) * out


# ==============================================================================
# 2. MESH AND COEFFICIENTS
# ==============================================================================
def mesh(Nx, Ny):
    dx, dy = W / Nx, H / Ny
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    return xc, yc, dx, dy


def coefficients(Nx, Ny, case="S"):
    """Return aE, aW, aN, aS, aP, b for the plate (unit depth)."""
    xc, yc, dx, dy = mesh(Nx, Ny)
    aE = np.full((Nx, Ny), K * dy / dx)
    aW = np.full((Nx, Ny), K * dy / dx)
    aN = np.full((Nx, Ny), K * dx / dy)
    aS = np.full((Nx, Ny), K * dx / dy)
    Sp = np.zeros((Nx, Ny))
    b = np.zeros((Nx, Ny))

    # west and east edges: Dirichlet at T1, half-cell conductance
    gx = 2.0 * K * dy / dx
    aW[0, :] = 0.0
    Sp[0, :] -= gx
    b[0, :] += gx * T1
    aE[-1, :] = 0.0
    Sp[-1, :] -= gx
    b[-1, :] += gx * T1

    # bottom edge: Dirichlet at T1
    gy = 2.0 * K * dx / dy
    aS[:, 0] = 0.0
    Sp[:, 0] -= gy
    b[:, 0] += gy * T1

    # top edge: prescribed profile
    T_top = (T1 + DT * np.sin(np.pi * xc / W)) if case == "S" else np.full(Nx, T2)
    aN[:, -1] = 0.0
    Sp[:, -1] -= gy
    b[:, -1] += gy * T_top

    aP = aE + aW + aN + aS - Sp
    return aE, aW, aN, aS, aP, b


def residual(T, aE, aW, aN, aS, aP, b):
    """Unnormalised residual field [W/m]."""
    R = b - aP * T
    R[:-1, :] += aE[:-1, :] * T[1:, :]
    R[1:, :] += aW[1:, :] * T[:-1, :]
    R[:, :-1] += aN[:, :-1] * T[:, 1:]
    R[:, 1:] += aS[:, 1:] * T[:, :-1]
    return R


# ==============================================================================
# 3. THE THREE ITERATIVE SOLVERS
# ==============================================================================
def solve(Nx, Ny, method="sor", omega=1.7, tol=1e-10, max_sweeps=200000,
          case="S"):
    """Iterative solution.  Returns (T, info).

    Jacobi is written with whole-array numpy operations.  Gauss-Seidel and SOR
    need the newest values as they are produced, which a single vectorised
    statement cannot express; they are implemented in RED-BLACK ordering, in
    which the mesh is chequered and each colour updated in one vectorised pass
    using only the other colour.  Red-black Gauss-Seidel is mathematically a
    Gauss-Seidel variant and converges at the same asymptotic rate, while
    remaining fully vectorised.
    """
    aE, aW, aN, aS, aP, b = coefficients(Nx, Ny, case)
    T = np.full((Nx, Ny), T1, dtype=float)
    scale = np.max(np.abs(b))
    hist = []

    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    red = (ii + jj) % 2 == 0
    black = ~red

    def neighbour_sum(Tc):
        s = np.zeros_like(Tc)
        s[:-1, :] += aE[:-1, :] * Tc[1:, :]
        s[1:, :] += aW[1:, :] * Tc[:-1, :]
        s[:, :-1] += aN[:, :-1] * Tc[:, 1:]
        s[:, 1:] += aS[:, 1:] * Tc[:, :-1]
        return s

    for sweep in range(1, max_sweeps + 1):
        if method == "jacobi":
            T_new = (neighbour_sum(T) + b) / aP
            T = T_new
        else:
            w = 1.0 if method == "gs" else omega
            for colour in (red, black):
                Tgs = (neighbour_sum(T) + b) / aP
                T = np.where(colour, T + w * (Tgs - T), T)

        if sweep % 5 == 0 or sweep == 1:
            r = np.max(np.abs(residual(T, aE, aW, aN, aS, aP, b))) / scale
            hist.append((sweep, r))
            if r < tol:
                break

    R = residual(T, aE, aW, aN, aS, aP, b)
    return T, {"sweeps": sweep, "hist": np.array(hist),
               "res": np.max(np.abs(R)) / scale}


# ==============================================================================
# 4. VERIFICATION AND GRID CONVERGENCE
# ==============================================================================
print("=" * 78)
print("EXAMPLE 5.2 -- TWO-DIMENSIONAL FINITE VOLUME METHOD")
print("=" * 78)
print(f"  Plate {W} x {H} m, k = {K} W/(m K); sinusoidal top (Case S)")
print("\n  Matrix structure: 5 non-zeros per row, bandwidth Nx.")
for N in [20, 50, 100, 200]:
    n = N * N
    print(f"    {N:>4} x {N:<4} : {n:>7} unknowns, {5*n:>8} non-zeros, "
          f"banded storage {n*(2*N+1):>11,} values")
print("  The tridiagonal solver of Chapters 1-4 cannot be used; the last")
print("  column shows why direct banded elimination is unattractive too.")

print("\n  GRID CONVERGENCE, Case S (SOR, tol = 1e-10)")
print(f"  {'Nx x Ny':>10} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} "
      f"{'p_inf':>7} {'sweeps':>8} {'CPU [s]':>9}")
rows = []
for Nx, Ny in [(16, 12), (32, 24), (64, 48), (128, 96)]:
    t0 = time.perf_counter()
    om = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))     # classical estimate
    T, info = solve(Nx, Ny, "sor", om, tol=1e-11)
    cpu = time.perf_counter() - t0
    xc, yc, dx, dy = mesh(Nx, Ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    e = T - T_sin(X, Y)
    l2 = np.sqrt(np.mean(e**2))
    li = np.max(np.abs(e))
    if rows:
        p2 = np.log(rows[-1]["l2"] / l2) / np.log(2.0)
        pi_ = np.log(rows[-1]["li"] / li) / np.log(2.0)
    else:
        p2 = pi_ = float("nan")
    rows.append({"Nx": Nx, "Ny": Ny, "h": dx, "l2": l2, "li": li, "T": T,
                 "X": X, "Y": Y, "sweeps": info["sweeps"], "cpu": cpu,
                 "omega": om, "hist": info["hist"]})
    print(f"  {f'{Nx} x {Ny}':>10} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} "
          f"{pi_:>7.3f} {info['sweeps']:>8d} {cpu:>9.3f}")

# Richardson extrapolation on the centre-point temperature
def centre(Nx, Ny):
    """Temperature at the FIXED physical point (W/2, H/2).

    Richardson extrapolation compares the same physical location across
    meshes.  Reading the value at "the middle cell" instead moves the sample
    point with the mesh, which contaminates the comparison with a first-order
    positional error and makes a second-order scheme report p = 1.  Bilinear
    interpolation to a fixed point removes it.
    """
    om = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))
    T, _ = solve(Nx, Ny, "sor", om, tol=1e-12)
    xc, yc, _, _ = mesh(Nx, Ny)
    col = np.array([np.interp(H / 2.0, yc, T[i, :]) for i in range(Nx)])
    return float(np.interp(W / 2.0, xc, col))

vals = [centre(Nx, Ny) for Nx, Ny in [(32, 24), (64, 48), (128, 96)]]
f3, f2, f1 = vals
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
print("\n  Richardson extrapolation on T at the fixed point (W/2, H/2):")
print(f"    32x24 / 64x48 / 128x96 : {f3:.8f} / {f2:.8f} / {f1:.8f} K")
print(f"    observed order p = {p_obs:.4f}")
print(f"    extrapolated     = {f_rich:.8f} K")
print(f"    exact            = {float(T_sin(np.array([W/2]), np.array([H/2]))[0]):.8f} K")
print(f"    |extrap - exact| = "
      f"{abs(f_rich - float(T_sin(np.array([W/2]), np.array([H/2]))[0])):.3e} K")

# global energy balance on the finest mesh
best = rows[-1]
Nx, Ny = best["Nx"], best["Ny"]
aE, aW, aN, aS, aP, b = coefficients(Nx, Ny, "S")
T = best["T"]
xc, yc, dx, dy = mesh(Nx, Ny)
q_top = np.sum(2 * K * dx / dy * ((T1 + DT * np.sin(np.pi * xc / W)) - T[:, -1]))
q_bot = np.sum(2 * K * dx / dy * (T[:, 0] - T1))
q_left = np.sum(2 * K * dy / dx * (T[0, :] - T1))
q_right = np.sum(2 * K * dy / dx * (T[-1, :] - T1))
print(f"\n  Global energy balance on the {Nx}x{Ny} mesh (per unit depth):")
print(f"    in through the top      = {q_top:.8f} W/m")
print(f"    out bottom+left+right   = {q_bot+q_left+q_right:.8f} W/m")
print(f"    imbalance               = {abs(q_top-(q_bot+q_left+q_right)):.3e} W/m")

# ==============================================================================
# 5. SOLVER COMPARISON
# ==============================================================================
print("\n" + "-" * 78)
print("  SOLVER COMPARISON on a 48 x 36 mesh, tolerance 1e-8")
Nx, Ny = 48, 36
om_theory = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))
print(f"  {'method':>16} {'sweeps':>9} {'CPU [s]':>9} {'vs Jacobi':>11}")
base = None
solver_hists = {}
for name, meth, om in [("Jacobi", "jacobi", 1.0),
                       ("Gauss-Seidel", "gs", 1.0),
                       (f"SOR, w={om_theory:.4f}", "sor", om_theory)]:
    t0 = time.perf_counter()
    T_s, inf = solve(Nx, Ny, meth, om, tol=1e-8, max_sweeps=60000)
    cpu = time.perf_counter() - t0
    if base is None:
        base = inf["sweeps"]
    solver_hists[name] = inf["hist"]
    print(f"  {name:>16} {inf['sweeps']:>9d} {cpu:>9.3f} "
          f"{base/inf['sweeps']:>10.1f}x")

print(f"\n  Optimal omega found by direct measurement:")
print(f"  {'omega':>8} {'sweeps':>9}")
om_scan, sweep_scan = [], []
for om in np.arange(1.0, 1.99, 0.05):
    _, inf = solve(Nx, Ny, "sor", float(om), tol=1e-8, max_sweeps=60000)
    om_scan.append(float(om))
    sweep_scan.append(inf["sweeps"])
best_i = int(np.argmin(sweep_scan))
for om, sw in zip(om_scan[::3], sweep_scan[::3]):
    print(f"  {om:>8.2f} {sw:>9d}")
print(f"  measured optimum : omega = {om_scan[best_i]:.3f} "
      f"({sweep_scan[best_i]} sweeps)")
print(f"  classical formula: omega = {om_theory:.4f}")
print("  2/(1 + sin(pi/N)) predicts the optimum well; note how sharply the")
print("  cost rises past it, which is why over-relaxing too far is worse than")
print("  not relaxing enough.")

# ==============================================================================
# 6. WHAT A SINGULAR CORNER DOES TO THE MEASURED ORDER
# ==============================================================================
print("\n" + "-" * 78)
print("  CASE U: the same scheme on the problem with discontinuous corners")
print(f"  {'Nx x Ny':>10} {'L2 [K]':>13} {'p_L2':>7} {'Linf [K]':>13} {'p_inf':>7}")
prev2 = previ = None
for Nx, Ny in [(16, 12), (32, 24), (64, 48), (128, 96)]:
    om = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))
    T, _ = solve(Nx, Ny, "sor", om, tol=1e-11, case="U")
    xc, yc, _, _ = mesh(Nx, Ny)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    e = T - T_uni(X, Y)
    l2, li = np.sqrt(np.mean(e**2)), np.max(np.abs(e))
    p2 = np.log(prev2 / l2) / np.log(2.0) if prev2 else float("nan")
    pi_ = np.log(previ / li) / np.log(2.0) if previ else float("nan")
    print(f"  {f'{Nx} x {Ny}':>10} {l2:>13.4e} {p2:>7.3f} {li:>13.4e} {pi_:>7.3f}")
    prev2, previ = l2, li
print("  The order is degraded by the corner singularity: the exact solution")
print("  is not smooth there, so the Taylor expansion underlying second-order")
print("  accuracy does not hold.  This is a property of the PROBLEM.  Reporting")
print("  an order from such a case, without saying so, would be misleading --")
print("  which is exactly why Case S was constructed for the verification.")
print("=" * 78)

# ==============================================================================
# 7. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 3, figsize=(18.0, 5.2),
    constrained_layout=True)
r = rows[-1]
cf = ax[0].contourf(r["X"] * 1e3, r["Y"] * 1e3, r["T"], levels=24,
                    cmap="inferno")
ax[0].set_xlabel(r"$x$ [mm]"); ax[0].set_ylabel(r"$y$ [mm]")
ax[0].set_title(f"(a) FVM solution, {r['Nx']}x{r['Ny']}")
ax[0].set_aspect("equal"); ax[0].grid(False)
fig.colorbar(cf, ax=ax[0], label=r"$T$ [K]", fraction=0.046, pad=0.03)

err = r["T"] - T_sin(r["X"], r["Y"])
cf2 = ax[1].contourf(r["X"] * 1e3, r["Y"] * 1e3, err * 1e3, levels=24,
                     cmap="RdBu_r")
ax[1].set_xlabel(r"$x$ [mm]"); ax[1].set_ylabel(r"$y$ [mm]")
ax[1].set_title("(b) Error field [mK]")
ax[1].set_aspect("equal"); ax[1].grid(False)
fig.colorbar(cf2, ax=ax[1], fraction=0.046, pad=0.03)

for rr, col in zip(rows, plt.cm.viridis(np.linspace(0.05, 0.8, len(rows)))):
    hh = rr["hist"]
    ax[2].semilogy(hh[:, 0], hh[:, 1], "-", lw=1.6, color=col,
                   label=f"{rr['Nx']}x{rr['Ny']}")
ax[2].set_xlabel("SOR sweep")
ax[2].set_ylabel(r"$\|R\|_\infty$ (normalised)")
ax[2].set_title("(c) Residual histories")
ax[2].legend(fontsize=8)

fig.suptitle("Example 5.2 -- Two-dimensional finite volume verification",
             fontsize=12.5, y=1.04)
fig.savefig("fig_5_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
for (name, hh), col in zip(solver_hists.items(),
                           ["#4d4d4d", "#2166ac", "#b2182b"]):
    ax[0].semilogy(hh[:, 0], hh[:, 1], "-", lw=1.8, color=col, label=name)
ax[0].set_xlabel("Sweep")
ax[0].set_ylabel(r"$\|R\|_\infty$ (normalised)")
ax[0].set_title("(a) Three iterative solvers, $48\\times36$")
ax[0].legend(fontsize=9)
ax[0].set_xlim(0, 4000)

ax[1].plot(om_scan, sweep_scan, "o-", lw=1.8, ms=5, color="#b2182b")
ax[1].axvline(om_theory, color="#2166ac", ls="--", lw=1.5,
              label=r"$2/(1+\sin(\pi/N))$")
ax[1].plot([om_scan[best_i]], [sweep_scan[best_i]], "*", ms=15,
           color="#1b7837", label="measured optimum", zorder=5)
ax[1].set_yscale("log")
ax[1].set_xlabel(r"Over-relaxation factor $\omega$")
ax[1].set_ylabel("Sweeps to $10^{-8}$")
ax[1].set_title("(b) The optimum is sharp")
ax[1].legend(fontsize=9)

fig.suptitle("Example 5.2 -- Iterative solvers in two dimensions",
             fontsize=12.5, y=1.02)
fig.savefig("fig_5_2b_solvers.png")
plt.close(fig)

print("Figures written: fig_5_2a_verification.png, fig_5_2b_solvers.png")
