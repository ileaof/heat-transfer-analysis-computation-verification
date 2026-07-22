"""
================================================================================
 EXAMPLE 10.3 -- CROSSFLOW WITH BOTH FLUIDS UNMIXED
 A two-dimensional exchanger, and the correlation put on trial
================================================================================

 OBJECTIVE
 ---------
 Example 10.1 tested the three classical effectiveness relations against a
 physical bound -- parallel flow is the worst arrangement, counterflow the best,
 crossflow between them -- and found that the widely used crossflow CORRELATION
 violates that bound over roughly half its parameter space, dipping below
 parallel flow at small NTU.  The excursions were small, at most 6e-3 in
 effectiveness, but a fitted expression that breaks a bound it cannot physically
 break deserves to be checked against the thing it was fitted to.

 This example computes the crossflow answer directly, from the governing
 equations, and then puts the correlation on trial against it.

 GOVERNING EQUATIONS
 -------------------
 Both fluids unmixed means each stream retains a transverse temperature profile:
 the hot stream, flowing in x, varies with y as well, because different parts of
 it meet cold fluid at different temperatures.  With xi = x/L and eta = y/W,

        dT_h/dxi  = -NTU_h (T_h - T_c)       NTU_h = UA / C_h
        dT_c/deta = +NTU_c (T_h - T_c)       NTU_c = UA / C_c

        T_h(0, eta) = T_h,in     for all eta
        T_c(xi, 0)  = T_c,in     for all xi

 This is a hyperbolic system in two dimensions.  Information travels in +xi for
 the hot stream and in +eta for the cold one, so both boundary conditions are
 supplied at the "upstream" edges and the solution can be swept out from the
 corner (0, 0) in one pass -- no linear system, no iteration.  That is a rarity
 and worth noticing: the two-dimensional problem is CHEAPER than the
 one-dimensional counterflow problem of Example 10.2, which was a boundary value
 problem requiring a simultaneous solve.

 The outlet temperatures are the transverse MEANS at the far edges, because the
 streams mix only after leaving:

        T_h,out = mean over eta of T_h(1, eta)
        T_c,out = mean over xi  of T_c(xi, 1)

 VERIFICATION
 ------------
   1. The exact limit C_r -> 0, where every arrangement gives 1 - exp(-NTU).
   2. Energy balance between the two streams.
   3. Order of accuracy, Richardson extrapolation, GCI.
   4. The physical ordering, which the EXACT solution must respect at every
      point -- the test the correlation failed in Example 10.1.
   5. The correlation's error, mapped over the whole (NTU, C_r) plane.

 OUTPUTS
 -------
   fig_10_3a_field.png       the two temperature fields and the outlet profiles
   fig_10_3b_correlation.png convergence, and the correlation's error map

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


# ==============================================================================
# 1. REFERENCE RELATIONS
# ==============================================================================
def eps_counterflow(NTU, Cr):
    if abs(Cr - 1.0) < 1e-10:
        return NTU / (1.0 + NTU)
    e = np.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


def eps_parallel(NTU, Cr):
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


def eps_crossflow_corr(NTU, Cr):
    """The fitted correlation examined in Example 10.1."""
    if Cr < 1e-10:
        return 1.0 - np.exp(-NTU)
    return 1.0 - np.exp((1.0 / Cr) * NTU ** 0.22 *
                        (np.exp(-Cr * NTU ** 0.78) - 1.0))


# ==============================================================================
# 2. THE CROSSFLOW SOLVER
# ==============================================================================
def solve_crossflow(N, NTU, Cr, theta=0.5):
    """Sweep the hyperbolic system out from the corner.

    Works in the normalised temperature  t = (T - T_c,in)/(T_h,in - T_c,in),
    so t_h = 1 on the inlet edge and t_c = 0 on the other.  C_min is taken as
    C_h, so NTU_h = NTU and NTU_c = NTU * Cr.

    The cell update is a theta-weighted trapezoidal rule along each stream:
    theta = 1/2 gives second order, theta = 1 gives first-order upwind.  Both
    are available so that the order can be MEASURED rather than assumed.
    """
    h = 1.0 / N
    NTU_h, NTU_c = NTU, NTU * Cr

    th = np.zeros((N + 1, N + 1))     # hot,  indexed [i (xi), j (eta)]
    tc = np.zeros((N + 1, N + 1))     # cold

    # ---- boundary conditions ------------------------------------------------
    # ONLY two edges are given: the hot stream enters along xi = 0, and the
    # cold stream enters along eta = 0.  The OTHER two edges, eta = 0 for the
    # hot stream and xi = 0 for the cold one, are not boundaries at all -- they
    # are interior lines that must be solved.  A first draft of this function
    # left them at zero, which silently removed a whole edge of hot fluid from
    # the problem; the energy balance then failed by 40 %, which is how it was
    # found.
    th[0, :] = 1.0
    tc[:, 0] = 0.0

    a = theta * h

    # ---- the two lines that are single equations, not systems ---------------
    # Along eta = 0 the cold stream sits at its inlet, t_c = 0, so the hot
    # stream obeys  d(t_h)/d(xi) = -NTU_h t_h  by itself.
    for i in range(1, N + 1):
        rh = th[i - 1, 0] - (1 - theta) * h * NTU_h * th[i - 1, 0]
        th[i, 0] = rh / (1.0 + a * NTU_h)
    # Along xi = 0 the hot stream sits at its inlet, t_h = 1, so
    # d(t_c)/d(eta) = NTU_c (1 - t_c).
    for j in range(1, N + 1):
        rc = (tc[0, j - 1] +
              (1 - theta) * h * NTU_c * (1.0 - tc[0, j - 1]) + a * NTU_c)
        tc[0, j] = rc / (1.0 + a * NTU_c)

    # ---- the interior: a 2x2 system in each cell ---------------------------
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            # th[i,j] and tc[i,j] each depend on the other, so the two-by-two
            # cell system is solved exactly rather than iterated.
            rh = th[i - 1, j] - (1 - theta) * h * NTU_h * (th[i - 1, j]
                                                           - tc[i - 1, j])
            rc = tc[i, j - 1] + (1 - theta) * h * NTU_c * (th[i, j - 1]
                                                           - tc[i, j - 1])
            A11, A12 = 1.0 + a * NTU_h, -a * NTU_h
            A21, A22 = -a * NTU_c, 1.0 + a * NTU_c
            det = A11 * A22 - A12 * A21
            th[i, j] = (A22 * rh - A12 * rc) / det
            tc[i, j] = (A11 * rc - A21 * rh) / det

    # transverse means at the outlet edges (the streams mix only on leaving)
    eta = np.linspace(0.0, 1.0, N + 1)
    th_out = trapezoid(th[N, :], eta)      # mean over the whole outlet edge
    tc_out = trapezoid(tc[:, N], eta)
    eps = 1.0 - th_out                     # since C_h = C_min and t_h,in = 1
    return th, tc, eps, th_out, tc_out


# ==============================================================================
# 3. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 10.3 -- CROSSFLOW, BOTH FLUIDS UNMIXED")
print("=" * 78)

print("\n" + "-" * 78)
print("  CHECK 1 -- THE EXACT LIMIT C_r -> 0")
print("""    When one stream has infinite capacity it stays at its inlet
    temperature, the arrangement becomes irrelevant, and every exchanger obeys
    eps = 1 - exp(-NTU).  This is the one point where the crossflow answer is
    known in closed form, and it tests the solver end to end.""")
print(f"\n  {'NTU':>7} {'N':>6} {'eps (solver)':>15} {'1 - exp(-NTU)':>16} "
      f"{'error':>12}")
for NTU_t in (0.5, 2.0, 5.0):
    ex = 1.0 - np.exp(-NTU_t)
    for N in (100, 400):
        _, _, e, _, _ = solve_crossflow(N, NTU_t, 0.0)
        print(f"  {NTU_t if N == 100 else '':>7} {N:>6d} {e:>15.10f} "
              f"{ex:>16.10f} {abs(e-ex):>12.3e}")

print("\n" + "-" * 78)
print("  CHECK 2 -- ENERGY BALANCE")
print("""    What the hot stream gives up, the cold stream must take up, and
    neither side of that statement is imposed by the solver.

    With C_h taken as C_min, C_c = C_h / C_r, so the balance reads
    (1 - t_h,out) = t_c,out / C_r.  A first draft wrote C_r * t_c,out -- the
    ratio the wrong way up -- and reported a 40 % imbalance that was entirely
    the fault of the check, not of the solution.""")
print(f"  {'NTU':>7} {'C_r':>6} {'1 - t_h,out':>14} {'t_c,out / C_r':>15} "
      f"{'imbalance':>12}")
for NTU_t, Cr_t in ((1.0, 0.5), (3.0, 0.8), (5.0, 1.0), (0.3, 0.25)):
    _, _, e, tho, tco = solve_crossflow(400, NTU_t, Cr_t)
    lhs, rhs = 1.0 - tho, tco / Cr_t
    print(f"  {NTU_t:>7.2f} {Cr_t:>6.2f} {lhs:>14.9f} {rhs:>15.9f} "
          f"{abs(lhs-rhs):>12.2e}")

print("\n" + "-" * 78)
print("  CHECK 3 -- ORDER OF ACCURACY")


def obs_order(v, i):
    if i + 2 >= len(v):
        return None
    return np.log2(abs((v[i] - v[i + 1]) / (v[i + 1] - v[i + 2])))


for th_v, name in ((1.0, "upwind (theta = 1)"),
                   (0.5, "trapezoidal (theta = 1/2)")):
    NS = [10, 20, 40, 80, 160]
    V = [solve_crossflow(N, 2.0, 0.8, theta=th_v)[2] for N in NS]
    print(f"\n    {name}")
    print(f"    {'N':>6} {'eps':>15} {'p':>8}")
    for i, N in enumerate(NS):
        p = obs_order(V, i)
        print(f"    {N:>6d} {V[i]:>15.10f} "
              f"{('%.3f' % p) if p is not None else '-':>8}")
    p_obs = obs_order(V, len(NS) - 3)
    rich = V[-1] + (V[-1] - V[-2]) / (2.0 ** p_obs - 1.0)
    gci = 1.25 * abs((V[-1] - V[-2]) / V[-1]) / (2.0 ** p_obs - 1.0)
    print(f"    observed order p       = {p_obs:.4f}")
    print(f"    Richardson extrapolate = {rich:.10f}")
    print(f"    GCI on the finest grid = {100*gci:.6f} %")
    if th_v == 0.5:
        EPS_REF = rich

print("\n" + "-" * 78)
print("  CHECK 4 -- THE ORDERING, NOW WITH THE EXACT CROSSFLOW SOLUTION")
print("""    Example 10.1 found the crossflow CORRELATION violating the bound
    eps_parallel <= eps_crossflow <= eps_counterflow on about half the domain.
    The exact solution has no such licence.  If the solver reproduces the
    violation, the bound was wrong; if it does not, the correlation was.""")
viol = 0
n_pts = 0
worst = 0.0
worst_at = None
for NTU_t in (0.05, 0.1, 0.3, 1.0, 2.0, 5.0):
    for Cr_t in (0.1, 0.25, 0.5, 0.75, 1.0):
        n_pts += 1
        _, _, e_x, _, _ = solve_crossflow(300, NTU_t, Cr_t)
        e_p = eps_parallel(NTU_t, Cr_t)
        e_c = eps_counterflow(NTU_t, Cr_t)
        gap = min(e_x - e_p, e_c - e_x)
        if gap < -1e-9:
            viol += 1
            if -gap > worst:
                worst, worst_at = -gap, (NTU_t, Cr_t)
print(f"\n    points tested            = {n_pts}")
print(f"    ordering violations      = {viol}")
if viol:
    print(f"    worst excursion          = {worst:.3e} at NTU = "
          f"{worst_at[0]}, C_r = {worst_at[1]}")

print("\n    A closer look at the points where the correlation failed:")
print(f"    {'NTU':>7} {'C_r':>6} {'parallel':>11} {'exact cross':>13} "
      f"{'correlation':>13} {'counter':>11}")
for NTU_t, Cr_t in ((0.05, 1.0), (0.1, 0.5), (0.3, 1.0), (2.0, 0.5)):
    _, _, e_x, _, _ = solve_crossflow(400, NTU_t, Cr_t)
    print(f"    {NTU_t:>7.2f} {Cr_t:>6.2f} {eps_parallel(NTU_t, Cr_t):>11.7f} "
          f"{e_x:>13.7f} {eps_crossflow_corr(NTU_t, Cr_t):>13.7f} "
          f"{eps_counterflow(NTU_t, Cr_t):>11.7f}")

print("\n" + "-" * 78)
print("  CHECK 5 -- THE CORRELATION PUT ON TRIAL")
print(f"  {'NTU':>7} {'C_r':>6} {'exact':>12} {'correlation':>13} "
      f"{'error':>11} {'rel. error':>12}")
rows = []
for NTU_t in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
    for Cr_t in (0.25, 0.5, 1.0):
        _, _, e_x, _, _ = solve_crossflow(300, NTU_t, Cr_t)
        e_corr = eps_crossflow_corr(NTU_t, Cr_t)
        rows.append((NTU_t, Cr_t, e_x, e_corr))
        print(f"  {NTU_t:>7.2f} {Cr_t:>6.2f} {e_x:>12.7f} {e_corr:>13.7f} "
              f"{e_corr-e_x:>11.2e} {abs(e_corr/e_x-1):>12.3e}")
errs = np.array([abs(r[3] - r[2]) for r in rows])
rels = np.array([abs(r[3] / r[2] - 1.0) for r in rows])
print(f"\n    worst absolute error in eps = {errs.max():.3e}")
print(f"    worst relative error        = {100*rels.max():.3f} %")
# Summarise the table by NTU band rather than in one lump, because the
# behaviour is not uniform across it.
lo = [r for r in rows if r[0] < 1.0]
hi = [r for r in rows if r[0] >= 1.0]
rel_lo = max(abs(r[3] / r[2] - 1.0) for r in lo)
rel_hi = max(abs(r[3] / r[2] - 1.0) for r in hi)
print(f"\n    worst relative error, NTU < 1   = {100*rel_lo:.2f} %")
print(f"    worst relative error, NTU >= 1  = {100*rel_hi:.2f} %")
print("""
    THE VERDICT, STATED AGAINST THE TABLE RATHER THAN AROUND IT.  A first
    draft of this paragraph claimed the correlation was "good to a fraction of
    a per cent over NTU between 0.5 and 5".  The table says otherwise: the
    worst relative error in that very range, 3.3 %, occurs at NTU = 0.5 with
    C_r = 1.  Writing a summary that the numbers above it contradict is the
    easiest mistake in this book to make and the easiest to catch.

    What the numbers actually support is this.  The ABSOLUTE error in
    effectiveness never exceeds 0.011 anywhere in the table, which is small
    against the uncertainty of any real U.  The RELATIVE error stays under
    1.7 % for NTU >= 1 and grows to 3.3 % below it, because effectiveness
    itself is small there and a fixed absolute error becomes a large fraction
    of it.  The sign also turns over: the correlation underestimates at low
    NTU and overestimates around NTU = 2 to 3.

    So the correlation is fit for the purpose it is used for, and it is not
    fit for the purpose of respecting a bound.  That is not a contradiction --
    it is what being a correlation means.  The only way to know which purpose
    it serves is to compute the thing it approximates, which is what this
    example did.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
NF, NTU_F, CR_F = 200, 3.0, 0.8
th_f, tc_f, eps_f, tho_f, tco_f = solve_crossflow(NF, NTU_F, CR_F)
xi = np.linspace(0.0, 1.0, NF + 1)

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0),
                       gridspec_kw={"width_ratios": [1, 1, 0.9]})
for k, (fld, name, cmap) in enumerate(((th_f, "hot stream $t_h$", "inferno"),
                                       (tc_f, "cold stream $t_c$", "viridis"))):
    cs = ax[k].pcolormesh(xi, xi, fld.T, cmap=cmap, shading="auto",
                          vmin=0.0, vmax=1.0)
    fig.colorbar(cs, ax=ax[k])
    ax[k].set_xlabel(r"$\xi = x/L$   (hot flows $\rightarrow$)")
    ax[k].set_ylabel(r"$\eta = y/W$   (cold flows $\uparrow$)")
    ax[k].set_title(f"({'ab'[k]}) {name}")
    ax[k].set_aspect("equal")
    ax[k].grid(False)

ax[2].plot(th_f[NF, :], xi, "-", lw=2.2, color="#b2182b",
           label=r"$t_h$ leaving at $\xi = 1$")
ax[2].plot(tc_f[:, NF], xi, "-", lw=2.2, color="#2166ac",
           label=r"$t_c$ leaving at $\eta = 1$")
ax[2].axvline(tho_f, color="#b2182b", ls="--", lw=1.4)
ax[2].axvline(tco_f, color="#2166ac", ls="--", lw=1.4)
ax[2].set_xlabel("normalised temperature")
ax[2].set_ylabel("position across the outlet edge")
ax[2].set_title("(c) Outlet profiles and their means")
ax[2].legend(fontsize=7.8, loc="upper right")
ax[2].annotate("dashed: mixed\nmean temperature",
               xy=(0.06, 0.06), xycoords="axes fraction", fontsize=7.8,
               color="0.3",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.25"))

fig.suptitle(rf"Example 10.3 -- Crossflow, both unmixed: "
             rf"NTU = {NTU_F:g}, $C_r$ = {CR_F:g}, $\varepsilon$ = {eps_f:.4f}",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_3a_field.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

NS = np.array([10, 20, 40, 80, 160])
for th_v, name, c in ((1.0, r"upwind, $\theta = 1$", "#2166ac"),
                      (0.5, r"trapezoidal, $\theta = 1/2$", "#b2182b")):
    errs_c = [abs(solve_crossflow(int(N), 2.0, 0.8, theta=th_v)[2] - EPS_REF)
              for N in NS]
    ax[0].loglog(NS, errs_c, "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
                 color=c, label=name)
ax[0].loglog(NS, 3e-2 * (NS[0] / NS) ** 1.0, "k--", lw=1.2,
             label=r"slope $-1$")
ax[0].loglog(NS, 3e-3 * (NS[0] / NS) ** 2.0, "k:", lw=1.4,
             label=r"slope $-2$")
ax[0].set_xlabel(r"$N$  (cells per direction)")
ax[0].set_ylabel(r"$|\varepsilon - \varepsilon_{\rm extrap}|$")
ax[0].set_title("(a) The scheme delivers the order it claims")
ax[0].legend(fontsize=8, loc="lower left")

ntu_g = np.logspace(np.log10(0.05), np.log10(6.0), 26)
cr_g = np.linspace(0.05, 1.0, 20)
err_map = np.zeros((len(cr_g), len(ntu_g)))
for i, Cr_t in enumerate(cr_g):
    for j, NTU_t in enumerate(ntu_g):
        _, _, e_x, _, _ = solve_crossflow(120, NTU_t, Cr_t)
        err_map[i, j] = (eps_crossflow_corr(NTU_t, Cr_t) - e_x)
lim = np.max(np.abs(err_map))
cs = ax[1].pcolormesh(ntu_g, cr_g, err_map, cmap="RdBu_r", shading="auto",
                      vmin=-lim, vmax=lim)
cb = fig.colorbar(cs, ax=ax[1])
cb.set_label(r"correlation $-$ exact  (in $\varepsilon$)")
ax[1].set_xscale("log")
ax[1].set_xlabel("NTU")
ax[1].set_ylabel(r"$C_r$")
ax[1].set_title("(b) Where the correlation is wrong, and by how much")
ax[1].grid(False)

fig.suptitle("Example 10.3 -- Verification, and the correlation on trial",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_3b_correlation.png")
plt.close(fig)

print("Figures written: fig_10_3a_field.png, fig_10_3b_correlation.png")
