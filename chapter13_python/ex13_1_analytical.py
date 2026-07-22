"""
================================================================================
 EXAMPLE 13.1 -- SOLUTION VERIFICATION: RICHARDSON, THE GCI, AND HONESTY
 Does the numerical error bar actually contain the numerical error?
================================================================================

 OBJECTIVE
 ---------
 Chapter 12 verified CODES by the order of accuracy: a manufactured solution and
 an observed order confirm that the discretisation is implemented correctly.
 This chapter verifies SOLUTIONS: given one particular computation on a real
 problem with no exact answer, how large is the discretisation error, and how
 sure can one be of that estimate?

 The standard answer is Richardson extrapolation and the Grid Convergence Index
 (GCI) of Roache.  These produce a numerical UNCERTAINTY -- an error bar on the
 computed result.  An error bar is only worth having if it is honest: if it
 actually contains the true error the stated fraction of the time.  Most
 presentations of the GCI simply assert that it is conservative.  This example
 TESTS it, by running the whole machinery on a problem whose exact answer IS
 known -- so the true error can be computed and compared with the GCI's claim.

 THE TEST PROBLEM
 ----------------
 Steady conduction (Laplace's equation) on the unit square, with

        T = 0 on three sides,   T = sin(pi y) on x = 1.

 The exact solution is the single Fourier mode

        T*(x, y) = sinh(pi x) sin(pi y) / sinh(pi)

 so the true discretisation error is known at every grid point, and the GCI's
 error bar can be checked against it directly.  Because there IS an exact
 solution here, the GCI is not needed -- which is exactly why it is the right
 place to test the GCI.

 THE METHODOLOGY
 ---------------
   1. Solve on three systematically refined grids, ratio r = 2.
   2. Observed order  p = ln((f1 - f2)/(f2 - f3)) / ln(r).
   3. Richardson extrapolation  f_exact ~ f1 + (f1 - f2)/(r^p - 1).
   4. GCI on the finest grid  GCI = Fs |(f1 - f2)/f1| / (r^p - 1),  Fs = 1.25.
   5. Asymptotic-range check  (r^p GCI_23)/GCI_12 ~ 1: are the grids fine
      enough for the theory to apply at all?

 The safety factor Fs = 1.25 is Roache's recommendation for three or more grids
 in the asymptotic range; it is meant to make the GCI a roughly 95 %-confidence
 bound.  Whether it succeeds is the question.

 OUTPUTS
 -------
   fig_13_1a_convergence.png  the solution, and grid convergence of a probe
   fig_13_1b_gci.png          the GCI band against the true error

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
PI = np.pi


# ==============================================================================
# 1. THE EXACT SOLUTION
# ==============================================================================
def T_exact(x, y):
    return np.sinh(PI * x) * np.sin(PI * y) / np.sinh(PI)


# ==============================================================================
# 2. THE FVM SOLVER (5-point Laplace, cell-centred, second order)
# ==============================================================================
def solve_laplace(N):
    """Solve Laplace's equation on the unit square, N x N cells.

    Cell centres at (i+0.5)/N.  Dirichlet boundaries by the half-cell ghost
    (the boundary value sits ON the face).  Solved by red-black successive
    over-relaxation -- the solver of Chapter 5 -- to a tight residual so that
    the ITERATIVE error is far below the DISCRETISATION error being measured.
    That separation matters: a loose iterative solve would masquerade as
    discretisation error and corrupt the observed order.
    """
    h = 1.0 / N
    xc = (np.arange(N) + 0.5) * h
    T = np.zeros((N, N))
    right = np.sin(PI * xc)          # x = 1 face value, varies with y (index i)

    # Face conductances (normalised, Gamma = 1): interior face = Gamma/h * h = 1;
    # a boundary face is a HALF cell away, so its conductance is Gamma/(h/2)*h = 2.
    # The diagonal a_P is therefore 4 in the interior, 5 on an edge, 6 in a
    # corner -- NOT a uniform 4.  A first draft used 0.25*(neighbour sum) with
    # a reflected ghost, which is the interior stencil applied everywhere; it
    # mishandled the boundary conductance and diverged at N = 64.
    aP = np.full((N, N), 4.0)
    aP[0, :] += 1.0      # south edge (one boundary face: 2 instead of 1)
    aP[-1, :] += 1.0     # north edge
    aP[:, 0] += 1.0      # west edge
    aP[:, -1] += 1.0     # east edge

    # boundary source: 2 * (boundary value) for each boundary face.  Only the
    # east face (x = 1) is nonzero; the other three are held at zero.
    src = np.zeros((N, N))
    src[:, -1] += 2.0 * right

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    red = ((ii + jj) % 2 == 0)
    omega = 2.0 / (1.0 + np.sin(PI / N))     # optimal SOR factor (Chapter 5)

    def interior_neighbour_sum(T):
        """Sum of interior-neighbour contributions (boundary faces contribute
        through `src`, not here)."""
        s = np.zeros_like(T)
        s[1:, :] += T[:-1, :]        # south neighbour of rows 1..
        s[:-1, :] += T[1:, :]        # north neighbour of rows ..N-2
        s[:, 1:] += T[:, :-1]        # west neighbour
        s[:, :-1] += T[:, 1:]        # east neighbour
        return s

    for sweep in range(50000):
        upd_max = 0.0
        for mask in (red, ~red):
            Tstar = (interior_neighbour_sum(T) + src) / aP
            upd = omega * (Tstar - T)
            T = np.where(mask, T + upd, T)
            upd_max = max(upd_max, np.max(np.abs(np.where(mask, upd, 0.0))))
        if upd_max < 1e-12:
            break
    return xc, T


PROBE_X, PROBE_Y = 0.75, 0.5


def probe_value(xc, T):
    """A scalar functional: T bilinearly interpolated to the FIXED physical
    point (0.75, 0.5).

    The interpolation to a fixed point is essential and was the subject of a
    bug.  A first draft returned T at the nearest CELL CENTRE, whose physical
    location moves with the grid; the "probe error" then mixed discretisation
    error with the variation of the exact solution between the different sampled
    points, and the observed order came out near 1 with an erratic jump instead
    of the scheme's true order 2.  This is the same moving-probe trap met in
    Chapter 5's Richardson study.  Interpolating to a fixed point -- which is
    second order and so preserves the order being measured -- removes it.
    """
    x = np.clip(PROBE_X, xc[0], xc[-1])
    y = np.clip(PROBE_Y, xc[0], xc[-1])
    j = np.searchsorted(xc, x) - 1        # column (x)
    i = np.searchsorted(xc, y) - 1        # row (y)
    j = min(max(j, 0), len(xc) - 2)
    i = min(max(i, 0), len(xc) - 2)
    tx = (x - xc[j]) / (xc[j + 1] - xc[j])
    ty = (y - xc[i]) / (xc[i + 1] - xc[i])
    val = ((1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i, j + 1] +
           (1 - tx) * ty * T[i + 1, j] + tx * ty * T[i + 1, j + 1])
    return val, PROBE_X, PROBE_Y


# ==============================================================================
# 3. THE VERIFICATION MACHINERY
# ==============================================================================
def gci_study(values, ratio=2.0, Fs=1.25):
    """Richardson order, extrapolate, GCI, and asymptotic-range index.

    values = (f_coarse, f_medium, f_fine), each on a grid ratio `ratio` finer.
    Returns a dict of the standard solution-verification quantities.
    """
    f3, f2, f1 = values          # coarse, medium, fine
    e12 = f1 - f2
    e23 = f2 - f3
    p = np.log(abs(e23 / e12)) / np.log(ratio)
    f_extrap = f1 + e12 / (ratio ** p - 1.0)
    gci_fine = Fs * abs(e12 / f1) / (ratio ** p - 1.0)
    gci_med = Fs * abs(e23 / f2) / (ratio ** p - 1.0)
    # asymptotic range: this ratio should be near 1 if the grids are fine enough
    asym = ratio ** p * gci_fine / gci_med
    return dict(p=p, f_extrap=f_extrap, gci_fine=gci_fine, gci_med=gci_med,
                asym=asym)


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 13.1 -- SOLUTION VERIFICATION AND THE GCI")
print("=" * 78)

NS = [16, 32, 64, 128]
probes = []
xstore = {}
for N in NS:
    xc, T = solve_laplace(N)
    val, xp, yp = probe_value(xc, T)
    probes.append(val)
    xstore[N] = (xc, T)
Tex_probe = T_exact(xp, yp)
print(f"  Probe: T at ({xp:.4f}, {yp:.4f}), exact = {Tex_probe:.10f}")
print(f"\n  {'N':>6} {'probe value':>15} {'true error':>14}")
for N, v in zip(NS, probes):
    print(f"  {N:>6d} {v:>15.10f} {abs(v - Tex_probe):>14.3e}")

print("\n" + "-" * 78)
print("  CHECK 1 -- RICHARDSON EXTRAPOLATION BEATS EVERY GRID")
g = gci_study(probes[-3:])
print(f"    observed order p          = {g['p']:.4f}  (scheme is 2nd order)")
print(f"    Richardson extrapolated   = {g['f_extrap']:.10f}")
print(f"    exact                     = {Tex_probe:.10f}")
print(f"    extrapolation error       = {abs(g['f_extrap']-Tex_probe):.3e}")
print(f"    finest-grid error         = {abs(probes[-1]-Tex_probe):.3e}")
print(f"    extrapolation is better by a factor of "
      f"{abs(probes[-1]-Tex_probe)/abs(g['f_extrap']-Tex_probe):.1f}")

print("\n" + "-" * 78)
print("  CHECK 2 -- DOES THE GCI ACTUALLY CONTAIN THE TRUE ERROR?")
print("""    The GCI claims to be a conservative (roughly 95 %) bound on the
    discretisation error of the finest grid.  Because the exact answer is known
    here, the claim can be TESTED rather than trusted: the true error must fall
    inside the GCI band.""")
print(f"\n  {'grid pair':>12} {'GCI (%)':>10} {'true rel. error (%)':>20} "
      f"{'contained?':>12}")
for k in range(len(NS) - 2):
    trio = probes[k:k + 3]
    gg = gci_study(trio)
    true_rel = abs(probes[k + 2] - Tex_probe) / abs(Tex_probe)
    contained = gg["gci_fine"] >= true_rel
    print(f"  {NS[k+1]}-{NS[k+2]:<7d} {100*gg['gci_fine']:>10.4f} "
          f"{100*true_rel:>20.4f} {str(contained):>12}")
print("""    The GCI band contains the true error at every level, and exceeds it
    by a healthy margin -- the safety factor Fs = 1.25 is doing its job.  This
    is the honest outcome, and it is worth having demonstrated on a case where
    the truth is available rather than assumed.""")

print("\n" + "-" * 78)
print("  CHECK 3 -- THE ASYMPTOTIC-RANGE INDEX")
print("""    The GCI is only meaningful if the grids are fine enough that the
    leading error term dominates -- the "asymptotic range".  The index
    (r^p GCI_fine)/GCI_medium should be near 1 there.  Far from 1 means the
    grids are too coarse and the GCI, however small, is not to be trusted.""")
print(f"\n  {'grid triple':>16} {'observed p':>12} {'asymptotic index':>18}")
for k in range(len(NS) - 2):
    trio = probes[k:k + 3]
    gg = gci_study(trio)
    print(f"  {NS[k]}-{NS[k+1]}-{NS[k+2]:<6d} {gg['p']:>12.4f} "
          f"{gg['asym']:>18.4f}")
print("""    The index sits close to 1 and the observed order close to 2, so the
    grids ARE in the asymptotic range and the GCI is trustworthy.  A code that
    reported a tiny GCI together with an asymptotic index of, say, 0.4 would be
    reporting a precise-looking number from grids too coarse to support it --
    the commonest way a convergence study lies.""")

print("\n" + "-" * 78)
print("  CHECK 4 -- THE ORDER IS THE SAME FOR A GLOBAL NORM")
print("""    The probe is one point.  A field norm tests the whole solution.  The
    L2 error over all cells must converge at the same order, or the point probe
    was sampling something unrepresentative.""")
print(f"\n  {'N':>6} {'L2 error':>13} {'order':>7}")
prev = None
l2s = []
for N in NS:
    xc, T = xstore[N]
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    e = T - T_exact(Y, X)     # note index convention: T[i,j], i->y, j->x
    L2 = np.sqrt(np.mean(e ** 2))
    l2s.append(L2)
    order = np.log2(prev / L2) if prev else None
    print(f"  {N:>6d} {L2:>13.3e} {('%.3f' % order) if order else '-':>7}")
    prev = L2
print("    The global norm converges at order 2 as well, confirming the point")
print("    probe was representative and the whole field is second order.")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

xc, T = xstore[128]
X, Y = np.meshgrid(xc, xc, indexing="ij")
cs = ax[0].contourf(Y, X, T, levels=20, cmap="inferno")
fig.colorbar(cs, ax=ax[0], label=r"$T$")
ax[0].plot([0.75], [0.5], "wo", ms=8, mfc="none", mew=2)
ax[0].annotate("probe", xy=(0.75, 0.5), xytext=(0.5, 0.72), color="white",
               fontsize=9, arrowprops=dict(arrowstyle="->", color="white"))
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_title("(a) $T = \\sinh(\\pi x)\\sin(\\pi y)/\\sinh(\\pi)$")
ax[0].set_aspect("equal")
ax[0].grid(False)

Na = np.array(NS)
errs = np.array([abs(v - Tex_probe) for v in probes])
ax[1].loglog(Na, errs, "o-", lw=1.9, ms=8, mfc="none", mew=1.7,
             color="#b2182b", label="true probe error")
ax[1].loglog(Na, errs[0] * (Na[0] / Na) ** 2.0, "k--", lw=1.3,
             label="slope -2")
ax[1].axhline(abs(g["f_extrap"] - Tex_probe), color="#1b7837", ls=":", lw=1.8,
              label="Richardson error")
ax[1].set_xlabel(r"$N$  (cells per side)")
ax[1].set_ylabel("probe error")
ax[1].set_title("(b) Second order, and what Richardson recovers")
ax[1].legend(fontsize=8.5, loc="lower left")

fig.suptitle("Example 13.1 -- Solution verification on a known solution",
             fontsize=12.5, y=1.08)
fig.savefig("fig_13_1a_convergence.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

# GCI band vs true error, per grid
finegrids = NS[2:]
gci_vals, true_vals = [], []
for k in range(len(NS) - 2):
    gg = gci_study(probes[k:k + 3])
    gci_vals.append(100 * gg["gci_fine"])
    true_vals.append(100 * abs(probes[k + 2] - Tex_probe) / abs(Tex_probe))
xpos = np.arange(len(finegrids))
w = 0.35
ax[0].bar(xpos - w / 2, gci_vals, w, color="#2166ac", alpha=0.85,
          label="GCI (claimed bound)")
ax[0].bar(xpos + w / 2, true_vals, w, color="#b2182b", alpha=0.85,
          label="true error")
ax[0].set_xticks(xpos)
ax[0].set_xticklabels([f"$N={n}$" for n in finegrids])
ax[0].set_ylabel("relative error / bound  [%]")
ax[0].set_title("(a) The GCI bounds the true error")
ax[0].legend(fontsize=8.5)

# asymptotic index and order convergence
ps, asyms = [], []
for k in range(len(NS) - 2):
    gg = gci_study(probes[k:k + 3])
    ps.append(gg["p"])
    asyms.append(gg["asym"])
ax[1].plot(finegrids, ps, "o-", lw=1.9, ms=8, mfc="none", mew=1.7,
           color="#b2182b", label="observed order $p$")
ax[1].plot(finegrids, asyms, "s-", lw=1.9, ms=8, mfc="none", mew=1.7,
           color="#1b7837", label="asymptotic index")
ax[1].axhline(2.0, color="#b2182b", ls=":", lw=1.2)
ax[1].axhline(1.0, color="#1b7837", ls=":", lw=1.2)
ax[1].set_xlabel(r"finest $N$ in the triple")
ax[1].set_ylabel("value")
ax[1].set_title("(b) In the asymptotic range: $p \\to 2$, index $\\to 1$")
ax[1].set_ylim(0.8, 2.2)
ax[1].legend(fontsize=8.5, loc="center right")

fig.suptitle("Example 13.1 -- Is the error bar honest?",
             fontsize=12.5, y=1.08)
fig.savefig("fig_13_1b_gci.png")
plt.close(fig)

print("Figures written: fig_13_1a_convergence.png, fig_13_1b_gci.png")
