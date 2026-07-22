"""
================================================================================
 EXAMPLE 9.2 -- VIEW FACTORS AND THE RADIOSITY ENCLOSURE
 Geometry computed, not quoted; and an enclosure solved and checked
================================================================================

 OBJECTIVE
 ---------
 Radiation exchange between surfaces separates cleanly into two problems: a
 GEOMETRIC one (what fraction of what leaves surface i arrives at surface j)
 and a RADIATIVE one (given those fractions and the surface properties, what
 are the net heat rates).  This example does both from scratch.

 Textbooks supply view factors from charts and formula tables.  Here they are
 computed by direct integration and then checked against Hottel's crossed-string
 construction, which is exact for any two-dimensional geometry.  Two independent
 routes to the same matrix.

 THE GEOMETRIC PROBLEM
 ---------------------
 For surfaces infinitely long in z, the view factor between differential strips
 reduces to a one-dimensional integral,

        dF_{d1-d2} = cos(theta_1) cos(theta_2) / (2 s) ds_2

 and the finite-surface factor follows by integrating over both surfaces and
 dividing by the area of the first.  The factor of 2 and the single power of s
 (rather than s^2, as in three dimensions) are consequences of integrating the
 third dimension out analytically; they are NOT assumed correct here, they are
 tested against crossed strings below.

 HOTTEL'S CROSSED-STRING METHOD
 ------------------------------
 For any two-dimensional geometry in which the surfaces see each other without
 obstruction,

        F_12 = (sum of crossed strings - sum of uncrossed strings) / (2 L_1)

 This is exact and needs no integration at all.  It is one of the most elegant
 results in the subject: a double integral replaced by measuring four lengths.

 THE RADIATIVE PROBLEM
 ---------------------
 For a diffuse-gray enclosure of N surfaces, the radiosity of surface i is

        J_i = eps_i sigma T_i^4 + (1 - eps_i) sum_j F_ij J_j

 which is a linear system in J.  The net heat rate follows from

        q_i = A_i (J_i - sum_j F_ij J_j)

 a form that holds for black and gray surfaces alike, unlike the more familiar
 A eps/(1-eps) (sigma T^4 - J), which divides by zero when eps = 1.

 VERIFICATION STRATEGY
 ---------------------
   1. Reciprocity  A_i F_ij = A_j F_ji           (integration is symmetric)
   2. Summation    sum_j F_ij = 1                (the enclosure is closed)
   3. Crossed strings vs integration             (independent geometry routes)
   4. Isothermal enclosure  =>  every q_i = 0    (a strong global test:
      no arrangement of emissivities may move heat with no driving potential)
   5. Energy closure  sum_i q_i = 0              (nothing is created)
   6. Two-surface enclosure against its closed form
   7. Black-surface limit  eps -> 1  =>  J_i -> sigma T_i^4

 OUTPUTS
 -------
   fig_9_2a_geometry.png     the enclosure, view factor matrix, convergence
   fig_9_2b_enclosure.png    radiosities and net fluxes, and the checks

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

SIGMA = 5.670374419e-8          # W/m^2 K^4  (Example 9.1 computes this)
T_START = time.perf_counter()


# ==============================================================================
# 1. GEOMETRY -- a two-dimensional enclosure defined by its corner points
# ==============================================================================
# A long duct of triangular cross-section.  Three flat surfaces, each seeing
# the other two.  Chosen because every view factor is nonzero, the crossed
# string construction is trivial to write down, and for a triangle the exact
# answer is available in closed form as well:  F_12 = (L1 + L2 - L3)/(2 L1).
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.30, 0.80]])


def surfaces_from_polygon(pts):
    """Return the list of (start, end) segments forming a closed polygon."""
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n]) for i in range(n)]


def seg_length(seg):
    return float(np.hypot(*(seg[1] - seg[0])))


def _panels(n, graded):
    """Panel midpoints and widths on [0,1], optionally clustered at both ends.

    The graded distribution maps a uniform parameter through
    u -> (1 - cos(pi u))/2, which packs panels quadratically towards u = 0 and
    u = 1.  That is exactly where the trouble is: see view_factor_integral.
    """
    e = np.linspace(0.0, 1.0, n + 1)
    if graded:
        e = 0.5 * (1.0 - np.cos(np.pi * e))
    return 0.5 * (e[1:] + e[:-1]), np.diff(e)


def view_factor_integral(seg_i, seg_j, n=400, graded=False):
    """F_ij between two 2-D surfaces by direct double integration.

    The kernel cos(theta_i) cos(theta_j) / (2 s) is evaluated at panel
    midpoints and summed.  Normals point INTO the enclosure, which for a
    polygon traversed anticlockwise means rotating the tangent by -90 degrees.

    ON THE ORDER OF ACCURACY.  A first draft of this example asserted that the
    midpoint rule would be second order here "on a smooth kernel".  Measured,
    it is FIRST order, and the assertion was simply wrong: the kernel is not
    smooth.  Adjacent sides of a polygon meet at a shared vertex where the
    separation s goes to zero, so the integrand is UNBOUNDED there.  The
    double integral still converges -- 1/s is integrable in two dimensions --
    but an unbounded integrand destroys the formal order of any equal-weight
    rule, and the observed ratio settles on 2 per doubling rather than 4.

    Two remedies are offered.  Grading the panels towards the endpoints puts
    resolution where the singularity is; and since the uniform-panel error is
    cleanly first order, Richardson extrapolation removes almost all of it for
    the cost of one extra solve.  Both are exercised below.
    """
    ai, bi = seg_i
    aj, bj = seg_j
    Li, Lj = seg_length(seg_i), seg_length(seg_j)

    ti = (bi - ai) / Li
    tj = (bj - aj) / Lj
    # Inward normal for an anticlockwise polygon: rotate the tangent by +90
    # degrees, (tx, ty) -> (-ty, tx).  A first draft rotated by -90 and so used
    # OUTWARD normals throughout.  The view factors were unaffected, because
    # the kernel contains the PRODUCT cos_i cos_j and is invariant when both
    # normals are flipped -- which is precisely why the error survived every
    # numerical check and was caught only by drawing the arrows on a figure.
    ni = np.array([-ti[1], ti[0]])
    nj = np.array([-tj[1], tj[0]])

    si, wi = _panels(n, graded)
    sj, wj = _panels(n, graded)
    Pi = ai[None, :] + np.outer(si, bi - ai)
    Pj = aj[None, :] + np.outer(sj, bj - aj)

    d = Pj[None, :, :] - Pi[:, None, :]      # vector from i to j
    s = np.linalg.norm(d, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_i = (d @ ni) / s
        cos_j = (-d @ nj) / s
        kern = cos_i * cos_j / (2.0 * s)
    kern = np.where(np.isfinite(kern), kern, 0.0)
    kern = np.maximum(kern, 0.0)             # surfaces cannot see behind them

    W = np.outer(wi * Li, wj * Lj)
    return float(np.sum(kern * W) / Li)


def view_factor_strings(pts, i, j):
    """Hottel's crossed-string construction for a convex polygon.

    For two sides of a triangle sharing the vertex arrangement below, the
    crossed strings are the two remaining sides and the uncrossed strings
    degenerate to the shared vertices, giving the classical

        F_ij = (L_i + L_j - L_k) / (2 L_i)
    """
    n = len(pts)
    segs = surfaces_from_polygon(pts)
    L = [seg_length(s) for s in segs]
    if n == 3:
        k = 3 - i - j
        return (L[i] + L[j] - L[k]) / (2.0 * L[i])
    raise NotImplementedError("closed form given here only for the triangle")


# ==============================================================================
# 2. BUILD AND CHECK THE VIEW FACTOR MATRIX
# ==============================================================================
def view_factor_matrix(pts, n=400, graded=True, richardson=True):
    """Assemble F, optionally graded and Richardson-extrapolated.

    With uniform panels the error is first order, so the combination
    2 F(n) - F(n/2) cancels the leading term.  With graded panels the error is
    already much smaller and the extrapolation is a refinement rather than a
    rescue.
    """
    segs = surfaces_from_polygon(pts)
    N = len(segs)

    def build(m):
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    M[i, j] = view_factor_integral(segs[i], segs[j], m, graded)
        return M

    F = build(n)
    if richardson:
        F = 2.0 * F - build(n // 2)
    return F


print("=" * 78)
print("EXAMPLE 9.2 -- VIEW FACTORS AND THE RADIOSITY ENCLOSURE")
print("=" * 78)

segs = surfaces_from_polygon(TRIANGLE)
A = np.array([seg_length(s) for s in segs])      # area per unit depth
N = len(segs)
print(f"  Triangular duct, surface lengths (area per unit depth):")
for i, a in enumerate(A):
    print(f"    surface {i+1}: L = {a:.6f} m")

F_raw = view_factor_matrix(TRIANGLE, n=600, graded=False, richardson=False)
F = view_factor_matrix(TRIANGLE, n=600, graded=True, richardson=True)
print("\n  VIEW FACTOR MATRIX (600 graded panels per side, Richardson)")
print("     " + "".join(f"{'F_'+str(i+1)+'j':>12}" for i in range(N)))
for i in range(N):
    print(f"  i={i+1}" + "".join(f"{F[i, j]:>12.8f}" for j in range(N)))

print("\n  CHECK 1 -- summation rule, sum_j F_ij = 1")
print(f"    {'row':>5} {'uniform panels':>18} {'graded + Richardson':>22}")
for i in range(N):
    print(f"    {i+1:>5d} {abs(F_raw[i].sum()-1.0):>18.2e} "
          f"{abs(F[i].sum()-1.0):>22.2e}")
print("    The summation rule is NOT imposed; it is a consequence of the")
print("    geometry, so its residual measures the quadrature error directly.")

print("\n  CHECK 2 -- reciprocity, A_i F_ij = A_j F_ji")
rec = 0.0
for i in range(N):
    for j in range(i + 1, N):
        d = abs(A[i] * F[i, j] - A[j] * F[j, i])
        rec = max(rec, d)
        print(f"    ({i+1},{j+1}): {A[i]*F[i,j]:.10f} vs {A[j]*F[j,i]:.10f}"
              f"   diff {d:.2e}")

print("\n  CHECK 3 -- crossed strings, an independent route to the geometry")
print(f"    {'pair':>8} {'integration':>14} {'crossed strings':>18} {'diff':>11}")
worst = 0.0
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        fs = view_factor_strings(TRIANGLE, i, j)
        worst = max(worst, abs(F[i, j] - fs))
        print(f"    F_{i+1}{j+1:<5d} {F[i, j]:>14.8f} {fs:>18.8f} "
              f"{abs(F[i, j]-fs):>11.2e}")
print(f"    worst disagreement = {worst:.2e}")
print("    The integration knows nothing about Hottel's construction and the")
print("    construction performs no integration.  Agreement here validates the")
print("    kernel -- including the factor of 2 and the single power of s that")
print("    distinguish the two-dimensional case from the three-dimensional one.")

# convergence of the integrated view factors
print("\n  CONVERGENCE OF THE INTEGRATION, three panel strategies")
F12_exact = view_factor_strings(TRIANGLE, 0, 1)
print(f"    {'panels':>7} {'uniform':>12} {'p':>6} {'graded':>12} {'p':>6}"
      f" {'unif.+Rich.':>13} {'p':>6}")
pu = pg = pr = None
for n in (50, 100, 200, 400, 800):
    fu = view_factor_integral(segs[0], segs[1], n, graded=False)
    fg = view_factor_integral(segs[0], segs[1], n, graded=True)
    fr = 2.0 * fu - view_factor_integral(segs[0], segs[1], n // 2, graded=False)
    eu, eg, er = (abs(fu - F12_exact), abs(fg - F12_exact),
                  abs(fr - F12_exact))
    su = f"{np.log2(pu/eu):.2f}" if pu else "-"
    sg = f"{np.log2(pg/eg):.2f}" if pg else "-"
    sr = f"{np.log2(pr/er):.2f}" if pr else "-"
    print(f"    {n:>7d} {eu:>12.2e} {su:>6} {eg:>12.2e} {sg:>6} "
          f"{er:>13.2e} {sr:>6}")
    pu, pg, pr = eu, eg, er
print("""    Uniform panels give first order, not second, because of the vertex
    singularity.  Grading recovers second order by resolving it.  Richardson
    extrapolation, which costs one extra solve and assumes only that the
    uniform error is cleanly first order, does about as well as grading --
    and the fact that BOTH remedies land on the crossed-string value is
    itself evidence that the singularity was the whole of the problem.""")


# ==============================================================================
# 3. THE RADIOSITY SYSTEM
# ==============================================================================
def solve_enclosure(F, A, T, eps):
    """Solve the diffuse-gray enclosure for radiosity and net heat rate.

        J_i - (1 - eps_i) sum_j F_ij J_j = eps_i sigma T_i^4

    The net rate is formed as q_i = A_i (J_i - sum_j F_ij J_j), which is the
    difference between what leaves surface i and what arrives at it.  The
    alternative expression A_i eps_i/(1 - eps_i) (sigma T_i^4 - J_i) is
    algebraically identical for eps < 1 and DIVIDES BY ZERO for a black
    surface, so it is not used here.
    """
    N = len(A)
    M = np.eye(N) - (1.0 - eps)[:, None] * F
    rhs = eps * SIGMA * T ** 4
    J = np.linalg.solve(M, rhs)
    G = F @ J                      # irradiation
    q = A * (J - G)
    return J, G, q


print("\n" + "=" * 78)
print("  THE RADIOSITY ENCLOSURE")
print("=" * 78)

T_surf = np.array([1000.0, 500.0, 750.0])
eps_surf = np.array([0.85, 0.40, 0.60])
print(f"  T   = {T_surf}  K")
print(f"  eps = {eps_surf}")

J, G, q = solve_enclosure(F, A, T_surf, eps_surf)
print(f"\n  {'i':>3} {'T [K]':>8} {'eps':>6} {'E_b [W/m2]':>13} "
      f"{'J [W/m2]':>13} {'q [W/m]':>13}")
for i in range(N):
    print(f"  {i+1:>3d} {T_surf[i]:>8.0f} {eps_surf[i]:>6.2f} "
          f"{SIGMA*T_surf[i]**4:>13.2f} {J[i]:>13.2f} {q[i]:>13.3f}")

print("\n  CHECK 4 -- energy closure, sum_i q_i = 0")
print(f"    sum q = {q.sum():.6e} W/m")
print(f"    relative to the largest |q| = {abs(q.sum())/np.max(np.abs(q)):.3e}")

print("\n  CHECK 5 -- an isothermal enclosure must transfer nothing")
print("    Every surface at 800 K, emissivities left deliberately unequal.")
J_iso, G_iso, q_iso = solve_enclosure(F, A, np.full(N, 800.0), eps_surf)
print(f"    max |q_i|          = {np.max(np.abs(q_iso)):.3e} W/m")
print(f"    max |J_i - sigma T^4| = "
      f"{np.max(np.abs(J_iso - SIGMA*800.0**4)):.3e} W/m^2")
print("""    This is the strongest single test in the example.  With no
    temperature difference anywhere, no arrangement of emissivities may drive
    a net flux; and every radiosity must collapse onto the blackbody value
    regardless of eps.  It exercises the view factors, the matrix assembly and
    the solve simultaneously, and a sign error in any of them shows up here.""")

print("\n  TRACING THE RESIDUALS OF CHECKS 4 AND 5")
print("""    Neither residual is zero, and neither should be taken on trust.  If
    they come from the view factor quadrature, then refining it must drive
    them down at the same rate as the summation-rule defect.  If they come
    from anything else -- a sign, an index, the solve -- they will not move.""")
print(f"    {'panels':>7} {'max |1 - row sum|':>19} {'isothermal max|q|':>19}"
      f" {'sum q (real case)':>19}")
for n_v in (50, 100, 200, 400, 800):
    Fv = view_factor_matrix(TRIANGLE, n=n_v, graded=False, richardson=False)
    rs = np.max(np.abs(Fv.sum(axis=1) - 1.0))
    _, _, qi = solve_enclosure(Fv, A, np.full(N, 800.0), eps_surf)
    _, _, qr = solve_enclosure(Fv, A, T_surf, eps_surf)
    print(f"    {n_v:>7d} {rs:>19.3e} {np.max(np.abs(qi)):>19.3e} "
          f"{abs(qr.sum()):>19.3e}")
print("""    All three columns fall together, by the same factor of two per
    doubling.  The spurious flux in an isothermal enclosure is therefore not a
    defect of the radiosity solver at all: it is the summation-rule error
    entering through the geometry, and it inherits the geometry's order of
    accuracy exactly.  With the graded and extrapolated matrix used above,
    both residuals are some two orders of magnitude smaller.""")

print("\n  CHECK 6 -- the black limit, eps -> 1, must give J = sigma T^4")
J_bl, _, q_bl = solve_enclosure(F, A, T_surf, np.ones(N))
print(f"    max |J - sigma T^4| = {np.max(np.abs(J_bl - SIGMA*T_surf**4)):.3e}")
print("    Note the solver was not special-cased for eps = 1; the (1 - eps)")
print("    factor simply annihilates the coupling term.")

print("\n  CHECK 7 -- two-surface enclosure against the closed form")
print("""    For two gray surfaces forming a complete enclosure,

        q_1 = sigma (T_1^4 - T_2^4) /
              [ (1-e1)/(A1 e1) + 1/(A1 F12) + (1-e2)/(A2 e2) ]

    Built here as two concentric infinite plates of unit area, for which
    F_12 = 1 exactly.""")
A2 = np.array([1.0, 1.0])
F2 = np.array([[0.0, 1.0], [1.0, 0.0]])
T2 = np.array([900.0, 400.0])
e2 = np.array([0.75, 0.35])
_, _, q2 = solve_enclosure(F2, A2, T2, e2)
denom = ((1 - e2[0]) / (A2[0] * e2[0]) + 1.0 / (A2[0] * F2[0, 1]) +
         (1 - e2[1]) / (A2[1] * e2[1]))
q_closed = SIGMA * (T2[0] ** 4 - T2[1] ** 4) / denom
print(f"    solver     q_1 = {q2[0]:.9f} W/m^2")
print(f"    closed form    = {q_closed:.9f} W/m^2")
print(f"    difference     = {abs(q2[0]-q_closed):.3e}")
print(f"    and q_2 = {q2[1]:.9f}, so q_1 + q_2 = {q2.sum():.3e}")

# ---- radiation shields ------------------------------------------------------
print("\n" + "-" * 78)
print("  APPLICATION: RADIATION SHIELDS")
print("""    A shield is a thin sheet placed between two surfaces.  It has TWO
    faces, and the radiosity formulation above gives every surface exactly
    ONE.  A first draft of this example modelled each shield as a single
    surface of the enclosure and reported that ten shields of emissivity 0.05
    cut the flux by a factor of four.  The correct factor is nearer forty.
    The error was structural, not numerical: no refinement of the view factors
    would have found it, because the geometry was never the problem.

    For plane parallel surfaces the exchange is a chain of resistances in
    series, and the shields simply add links:

        q = sigma (T_1^4 - T_2^4) /
            [ (1/e_1 + 1/e_2 - 1) + sum_shields (2/e_s - 1) ]

    the 2/e_s - 1 per shield being the two faces.  The zero-shield case of
    this formula is checked against the enclosure solver, which is where the
    two treatments do agree.""")


def shield_flux(T1, T2, e1, e2, n_shield, e_s):
    """Series-resistance result for n_shield shields between parallel plates."""
    R = 1.0 / e1 + 1.0 / e2 - 1.0 + n_shield * (2.0 / e_s - 1.0)
    return SIGMA * (T1 ** 4 - T2 ** 4) / R


q0_formula = shield_flux(T2[0], T2[1], e2[0], e2[1], 0, 1.0)
print(f"\n    zero shields, series formula  = {q0_formula:.9f} W/m^2")
print(f"    zero shields, enclosure solver = {q2[0]:.9f} W/m^2")
print(f"    difference                     = {abs(q0_formula-q2[0]):.3e}")
print("    The two agree exactly when the model is one the solver can express.")

print(f"\n    {'shields':>8} {'eps_s = 0.05':>16} {'reduction':>11}"
      f" {'eps_s = 0.5':>14} {'reduction':>11}")
for ns in (0, 1, 2, 5, 10):
    qa = shield_flux(T2[0], T2[1], e2[0], e2[1], ns, 0.05)
    qb = shield_flux(T2[0], T2[1], e2[0], e2[1], ns, 0.50)
    print(f"    {ns:>8d} {qa:>16.4f} {qa/q0_formula:>11.4f} "
          f"{qb:>14.4f} {qb/q0_formula:>11.4f}")
print("""    The emissivity of the shield matters far more than the number of
    them: one shield at eps = 0.05 outperforms ten at eps = 0.5.  This is why
    multilayer insulation uses aluminised film rather than many sheets of
    ordinary foil, and why a single low-emissivity coating on a window does
    more than a second pane of plain glass.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

poly = np.vstack([TRIANGLE, TRIANGLE[0]])
ax[0].plot(poly[:, 0], poly[:, 1], "-", lw=2.5, color="#2166ac")
cols = ["#b2182b", "#1b7837", "#e08214"]
for i, (a, b) in enumerate(segs):
    mid = 0.5 * (a + b)
    ax[0].plot([a[0], b[0]], [a[1], b[1]], "-", lw=4.0, color=cols[i],
               solid_capstyle="butt", alpha=0.85)
    t = (b - a) / seg_length((a, b))
    nrm = np.array([-t[1], t[0]])
    ax[0].annotate("", xy=mid + 0.16 * nrm, xytext=mid,
                   arrowprops=dict(arrowstyle="->", color=cols[i], lw=1.5))
    ax[0].text(*(mid + 0.075 * nrm), f"{i+1}", color=cols[i], fontsize=11,
               ha="center", va="center", fontweight="bold")
    # labels OUTSIDE the enclosure: placing them along +nrm sends all three
    # towards the centroid, where they pile on top of one another
    ax[0].text(*(mid - 0.16 * nrm), f"$T={T_surf[i]:.0f}$ K\n"
               rf"$\varepsilon={eps_surf[i]:.2f}$", color=cols[i],
               fontsize=8.0, ha="center", va="center")
ax[0].set_aspect("equal")
ax[0].set_xlabel(r"$x$  [m]")
ax[0].set_ylabel(r"$y$  [m]")
ax[0].set_title("(a) The enclosure, normals pointing inward")
ax[0].set_xlim(-0.34, 1.34)
ax[0].set_ylim(-0.30, 1.02)

ns_list = np.array([25, 50, 100, 200, 400, 800])
e_uni = np.array([abs(view_factor_integral(segs[0], segs[1], n, graded=False)
                      - F12_exact) for n in ns_list])
e_gra = np.array([abs(view_factor_integral(segs[0], segs[1], n, graded=True)
                      - F12_exact) for n in ns_list])
e_ric = np.array([abs(2.0 * view_factor_integral(segs[0], segs[1], n, False)
                      - view_factor_integral(segs[0], segs[1], n // 2, False)
                      - F12_exact) for n in ns_list])
ax[1].loglog(ns_list, e_uni, "o-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#b2182b", label="uniform panels")
ax[1].loglog(ns_list, e_gra, "s-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#1b7837", label="graded panels")
ax[1].loglog(ns_list, e_ric, "^-", lw=1.9, ms=7, mfc="none", mew=1.7,
             color="#2166ac", label="uniform + Richardson")
ax[1].loglog(ns_list, e_uni[0] * (ns_list[0] / ns_list) ** 1.0, "k--", lw=1.2,
             label=r"slope $-1$")
ax[1].loglog(ns_list, e_gra[0] * (ns_list[0] / ns_list) ** 2.0, "k:", lw=1.4,
             label=r"slope $-2$")
ax[1].set_xlabel("panels per surface")
ax[1].set_ylabel(r"$|F_{12}^{\rm num} - F_{12}^{\rm strings}|$")
ax[1].set_title("(b) A vertex singularity costs an order")
ax[1].legend(fontsize=7.5, loc="lower left")

fig.suptitle("Example 9.2 -- View factors computed two independent ways",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_2a_geometry.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

idx = np.arange(N)
w = 0.27
ax[0].bar(idx - w, SIGMA * T_surf ** 4, w, color="#b2182b", alpha=0.85,
          label=r"$E_b = \sigma T^4$")
ax[0].bar(idx, J, w, color="#2166ac", alpha=0.85, label=r"radiosity $J$")
ax[0].bar(idx + w, G, w, color="#1b7837", alpha=0.85,
          label=r"irradiation $G$")
ax[0].set_xticks(idx)
ax[0].set_xticklabels([f"surface {i+1}" for i in idx])
ax[0].set_ylabel(r"[W m$^{-2}$]")
ax[0].set_title("(a) Emissive power, radiosity, irradiation")
ax[0].legend(fontsize=8.5)

qcol = ["#b2182b" if v > 0 else "#2166ac" for v in q]
ax[1].bar(idx, q, 0.55, color=qcol, alpha=0.9)
ax[1].axhline(0.0, color="0.3", lw=1.1)
for i, v in enumerate(q):
    ax[1].annotate(f"{v:,.0f}", xy=(i, v), xytext=(0, 6 if v > 0 else -14),
                   textcoords="offset points", ha="center", fontsize=8.5)
ax[1].set_xticks(idx)
ax[1].set_xticklabels([f"surface {i+1}" for i in idx])
ax[1].set_ylabel(r"net heat rate  [W m$^{-1}$ of duct]")
ax[1].set_ylim(min(q) * 1.35, max(q) * 1.18)
ax[1].set_title("(b) Net exchange: the three sum to zero")
ax[1].annotate(rf"$\sum q_i = {q.sum():.1e}$ W m$^{{-1}}$",
               xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9,
               color="0.25",
               bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))

fig.suptitle("Example 9.2 -- The diffuse-gray enclosure, solved and checked",
             fontsize=12.5, y=1.08)
fig.savefig("fig_9_2b_enclosure.png")
plt.close(fig)

print("Figures written: fig_9_2a_geometry.png, fig_9_2b_enclosure.png")
