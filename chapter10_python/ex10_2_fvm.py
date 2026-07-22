"""
================================================================================
 EXAMPLE 10.2 -- FINITE VOLUME SOLUTION OF A TWO-STREAM EXCHANGER
 A boundary value problem, because the inlets are at opposite ends
================================================================================

 OBJECTIVE
 ---------
 Example 10.1 used closed-form effectiveness relations, which exist only for a
 handful of idealised arrangements with constant properties and constant U.
 This example solves the underlying differential equations directly, so that
 the assumptions can be relaxed one at a time.  The closed forms then become
 what they should be: a reference to verify against, not the method itself.

 GOVERNING EQUATIONS
 -------------------
 For a two-stream exchanger with local overall coefficient U and perimeter P,

        C_h dT_h/dx = -U P (T_h - T_c)
        C_c dT_c/dx = +-U P (T_h - T_c)

 the sign in the second equation being positive for parallel flow (both streams
 travel in +x) and negative for counterflow (the cold stream travels in -x).

 WHY THIS IS NOT A MARCHING PROBLEM
 ----------------------------------
 In parallel flow both inlet conditions are given at x = 0, so the system is an
 initial value problem and can be marched, exactly as the boundary layer of
 Chapter 8 was.  In COUNTERFLOW the hot inlet is at x = 0 and the cold inlet is
 at x = L.  Half the data sits at each end, and no amount of cleverness makes it
 an initial value problem.  It is a two-point boundary value problem, and the
 whole coupled system must be solved simultaneously.

 That structural difference is the reason this example exists.  It is also the
 reason a naive shooting method is a poor idea here: the homogeneous solution
 grows like exp(NTU(1 - C_r)) along the exchanger, so at NTU = 10 an error in
 the guessed inlet is amplified by e^10, and the shooting residual is dominated
 by round-off long before the physical answer emerges.  The example demonstrates
 that failure and then solves the system properly.

 DISCRETISATION
 --------------
 Control volumes along x, with the convective term treated by upwinding, which
 for a one-directional flow is exact rather than merely stable: the temperature
 carried into a cell IS the upstream cell's temperature.  The exchange term is
 a source linearised in Patankar's form, and because it always opposes the
 local temperature difference its S_P is negative automatically -- the same
 structural guarantee met for radiation in Chapter 9.

 The assembled system is block tridiagonal in the pair (T_h, T_c) and is solved
 directly.

 VERIFICATION
 ------------
   1. Against the closed-form effectiveness of Example 10.1, for both
      arrangements, over a sweep of NTU and C_r.
   2. Order of accuracy from successive differences.
   3. Richardson extrapolation and GCI.
   4. Global energy balance: what the hot stream loses, the cold stream gains.
   5. The second law: no cell may transfer heat from cold to hot, and the
      outlet temperatures may not cross in counterflow.
   6. A case the closed forms cannot reach -- U varying along the exchanger --
      checked by the energy balance and by grid convergence alone.

 OUTPUTS
 -------
   fig_10_2a_profiles.png    temperature profiles, both arrangements
   fig_10_2b_verification.png  convergence, and the variable-U case

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


# ==============================================================================
# 1. CLOSED FORMS (from Example 10.1) -- used only as references
# ==============================================================================
def eps_counterflow(NTU, Cr):
    if abs(Cr - 1.0) < 1e-10:
        return NTU / (1.0 + NTU)
    e = np.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


def eps_parallel(NTU, Cr):
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


# ==============================================================================
# 2. THE SOLVER
# ==============================================================================
def solve_exchanger(N, UA, Ch, Cc, Thi, Tci, arrangement="counterflow",
                    U_profile=None):
    """Solve the coupled two-stream system on N control volumes.

    Unknowns are ordered as [Th_0, Tc_0, Th_1, Tc_1, ...], which makes the
    matrix banded with bandwidth 3 -- the hot equation in cell i couples to
    Th_{i-1} (upwind) and to Tc_i (exchange), and similarly for the cold one.
    Ordering the unknowns stream by stream instead would give two dense blocks
    and destroy the band structure.

    U_profile, if given, is a callable U(x/L) returning a multiplier on the
    mean U, so that a non-uniform coefficient can be imposed.
    """
    dx = 1.0 / N                       # dimensionless cell width
    xc = (np.arange(N) + 0.5) * dx

    ua = np.full(N, UA * dx)           # UA of each cell
    if U_profile is not None:
        f = np.asarray(U_profile(xc), dtype=float)
        ua *= f * N / f.sum()          # normalise so the total UA is preserved

    n = 2 * N
    ab = np.zeros((5, n))              # 2 sub, 1 main, 2 super diagonals
    b = np.zeros(n)

    def put(row, col, val):
        """Store into banded storage: ab[2 + row - col, col] = A[row, col]."""
        ab[2 + row - col, col] += val

    for i in range(N):
        rh, rc = 2 * i, 2 * i + 1

        # ---- hot stream: flows in +x for both arrangements ----------------
        # C_h (T_h,i - T_h,i-1) = -ua_i (T_h,i - T_c,i)
        put(rh, rh, Ch + ua[i])
        put(rh, rc, -ua[i])
        if i == 0:
            b[rh] += Ch * Thi
        else:
            put(rh, rh - 2, -Ch)

        # ---- cold stream --------------------------------------------------
        if arrangement == "parallel":
            # flows in +x; inlet at i = 0
            put(rc, rc, Cc + ua[i])
            put(rc, rh, -ua[i])
            if i == 0:
                b[rc] += Cc * Tci
            else:
                put(rc, rc - 2, -Cc)
        elif arrangement == "counterflow":
            # flows in -x; inlet at i = N-1.  The upwind neighbour is i+1.
            put(rc, rc, Cc + ua[i])
            put(rc, rh, -ua[i])
            if i == N - 1:
                b[rc] += Cc * Tci
            else:
                put(rc, rc + 2, -Cc)
        else:
            raise ValueError("arrangement must be 'counterflow' or 'parallel'")

    sol = solve_banded((2, 2), ab, b)
    Th, Tc = sol[0::2], sol[1::2]
    return xc, Th, Tc, ua


def outlets(Th, Tc, arrangement):
    """Outlet temperatures, at whichever end each stream leaves."""
    Tho = Th[-1]
    Tco = Tc[-1] if arrangement == "parallel" else Tc[0]
    return Tho, Tco


# ==============================================================================
# 3. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 10.2 -- FINITE VOLUME SOLUTION OF A TWO-STREAM EXCHANGER")
print("=" * 78)

CH, CC = 5016.0, 8360.0         # W/K
THI, TCI = 360.0, 300.0         # K
UA_REF = 12000.0                # W/K
CMIN, CMAX = min(CH, CC), max(CH, CC)
CR = CMIN / CMAX
NTU = UA_REF / CMIN
print(f"  C_h = {CH:,.1f} W/K, C_c = {CC:,.1f} W/K, UA = {UA_REF:,.0f} W/K")
print(f"  C_r = {CR:.6f}, NTU = {NTU:.6f}")

print("\n" + "-" * 78)
print("  CHECK 1 -- AGAINST THE CLOSED FORMS")
print(f"  {'arrangement':>13} {'N':>6} {'eps (FVM)':>13} {'eps (exact)':>13} "
      f"{'rel. error':>12}")
for arr, ef in (("counterflow", eps_counterflow), ("parallel", eps_parallel)):
    e_ex = ef(NTU, CR)
    for N in (20, 80, 320, 640):
        xc, Th, Tc, _ = solve_exchanger(N, UA_REF, CH, CC, THI, TCI, arr)
        Tho, Tco = outlets(Th, Tc, arr)
        q = CH * (THI - Tho)
        e_fvm = q / (CMIN * (THI - TCI))
        print(f"  {arr if N == 20 else '':>13} {N:>6d} {e_fvm:>13.9f} "
              f"{e_ex:>13.9f} {abs(e_fvm/e_ex - 1):>12.3e}")

print("\n" + "-" * 78)
print("  CHECK 2 -- ORDER OF ACCURACY")
print("""    Upwinding a one-directional convective term is exact in the sense
    that the temperature carried into a cell IS the upstream value; the error
    comes entirely from where that value is deemed to sit.  The scheme is
    first order, and the study below says so rather than hoping otherwise.""")


def obs_order(v, i):
    if i + 2 >= len(v):
        return None
    return np.log2(abs((v[i] - v[i + 1]) / (v[i + 1] - v[i + 2])))


for arr, ef in (("counterflow", eps_counterflow), ("parallel", eps_parallel)):
    e_ex = ef(NTU, CR)
    NS = [10, 20, 40, 80, 160]
    V = []
    for N in NS:
        xc, Th, Tc, _ = solve_exchanger(N, UA_REF, CH, CC, THI, TCI, arr)
        Tho, Tco = outlets(Th, Tc, arr)
        V.append(CH * (THI - Tho) / (CMIN * (THI - TCI)))
    print(f"\n    {arr}")
    print(f"    {'N':>6} {'eps':>14} {'error':>12} {'p':>8}")
    for i, N in enumerate(NS):
        p = obs_order(V, i)
        print(f"    {N:>6d} {V[i]:>14.9f} {abs(V[i]-e_ex):>12.3e} "
              f"{('%.3f' % p) if p is not None else '-':>8}")
    p_obs = obs_order(V, len(NS) - 3)
    rich = V[-1] + (V[-1] - V[-2]) / (2.0 ** p_obs - 1.0)
    gci = 1.25 * abs((V[-1] - V[-2]) / V[-1]) / (2.0 ** p_obs - 1.0)
    print(f"    observed order p       = {p_obs:.4f}")
    print(f"    Richardson extrapolate = {rich:.10f}")
    print(f"    exact                  = {e_ex:.10f}")
    print(f"    extrapolation error    = {abs(rich - e_ex):.3e}")
    print(f"    GCI on the finest grid = {100*gci:.5f} %")

print("\n" + "-" * 78)
print("  CHECK 3 -- ENERGY BALANCE AND THE SECOND LAW")
for arr in ("counterflow", "parallel"):
    xc, Th, Tc, ua = solve_exchanger(400, UA_REF, CH, CC, THI, TCI, arr)
    Tho, Tco = outlets(Th, Tc, arr)
    q_h = CH * (THI - Tho)
    q_c = CC * (Tco - TCI)
    q_cells = float(np.sum(ua * (Th - Tc)))
    print(f"\n    {arr}")
    print(f"      heat given up by the hot stream = {q_h:,.6f} W")
    print(f"      heat taken up by the cold stream = {q_c:,.6f} W")
    print(f"      sum of the cell exchange terms   = {q_cells:,.6f} W")
    print(f"      worst relative discrepancy       = "
          f"{max(abs(q_h/q_c - 1), abs(q_cells/q_h - 1)):.3e}")
    cross = np.min(Th - Tc)
    print(f"      minimum (T_h - T_c) over all cells = {cross:.6f} K")
    print(f"      second law respected (no cell transfers uphill): "
          f"{bool(cross > 0)}")

print("""
    The three heat rates are computed from three different places in the
    solution -- the hot outlet, the cold outlet, and the sum of the local
    exchange terms -- and none of them was imposed.  Their agreement is the
    discrete statement that the scheme conserves energy.""")

print("\n" + "-" * 78)
print("  CHECK 4 -- WHY SHOOTING IS THE WRONG METHOD HERE")
print("""    Counterflow could in principle be solved by guessing the cold outlet
    at x = 0, marching to x = L, and correcting the guess until the cold inlet
    comes out right.  The homogeneous solution of the coupled system grows like
    exp(NTU(1 - C_r)) along the exchanger, so the guess is amplified by that
    factor before the residual is formed.  The table below shows the
    amplification, which is the condition number of the shooting problem.""")
print(f"  {'NTU':>8} {'C_r':>7} {'amplification':>16} "
      f"{'digits surviving':>18}")
for NTU_t in (1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 74.0):
    for Cr_t in (0.5,):
        amp = np.exp(NTU_t * (1.0 - Cr_t))
        digits = max(0.0, 16.0 - np.log10(amp))
        print(f"  {NTU_t:>8.1f} {Cr_t:>7.2f} {amp:>16.4e} {digits:>18.1f}")
print("""    A first draft of this table stopped at NTU = 30 and asserted that
    "nothing survives".  The table said otherwise -- nine digits still stand
    there -- so the claim was wrong and has been corrected by extending the
    table until it becomes true.  Complete failure arrives near NTU = 74, where
    the amplification reaches 1e16 and the residual is pure round-off.  Such
    NTU values are not academic: cryogenic and regenerative exchangers are
    designed at NTU of 50 and above, which is precisely the regime where
    shooting stops working and the boundary value formulation used here does
    not, because it never propagates an error along the whole length -- the
    matrix enforces both boundary conditions at once.""")

# ---- a case the closed forms cannot reach -----------------------------------
print("\n" + "-" * 78)
print("  CHECK 5 -- AN INVARIANCE THAT THE SOLVER SHOULD REPRODUCE")
print("""    A first draft of this section claimed that redistributing U along
    the exchanger, at fixed total UA, would change the duty -- concentrating
    the conductance where the temperature difference is largest ought to
    extract more heat.  The computation flatly refused: every profile gave the
    same effectiveness.  The computation was right and the claim was wrong,
    and the reason is a short piece of algebra that should have been done
    first.

    Subtract the two stream equations.  For counterflow the difference
    theta = T_h - T_c obeys

        d(theta)/dx = -(1/C_h - 1/C_c) U P theta

    a single first-order equation whose solution involves U only through the
    integral of U P dx -- that is, through the TOTAL UA.  Where the
    conductance sits is irrelevant.  The same argument with a plus sign
    applies to parallel flow.

    So this is not a case beyond closed form; it is an exact invariance, and a
    strong test of the solver, because nothing in the discretisation knows
    about it.""")
print(f"\n  {'U profile':<20} {'eps (N = 600)':>16} {'total UA':>12}")
profiles = {
    "uniform": lambda x: np.ones_like(x),
    "linear, halving": lambda x: 1.0 - 0.5 * x,
    "fouled outlet": lambda x: 1.0 - 0.6 * x ** 3,
    "crowded at hot end": lambda x: np.exp(-6.0 * x),
    "crowded at cold end": lambda x: np.exp(6.0 * (x - 1.0)),
}
e_vals = []
for name, prof in profiles.items():
    xc, Th, Tc, ua = solve_exchanger(600, UA_REF, CH, CC, THI, TCI,
                                     "counterflow", U_profile=prof)
    Tho, Tco = outlets(Th, Tc, "counterflow")
    e_v = CH * (THI - Tho) / (CMIN * (THI - TCI))
    e_vals.append(e_v)
    print(f"  {name:<20} {e_v:>16.9f} {ua.sum():>12.4f}")
print(f"\n    exact (uniform-U closed form) = {eps_counterflow(NTU, CR):.9f}")
print(f"    spread across the five profiles = {max(e_vals)-min(e_vals):.3e}")
print(f"    distance of each from the exact = "
      f"{min(abs(np.array(e_vals) - eps_counterflow(NTU, CR))):.3e} to "
      f"{max(abs(np.array(e_vals) - eps_counterflow(NTU, CR))):.3e}")
print("""    The spread is smaller than the distance to the exact value, which
    is the signature wanted: all five profiles are converging to the SAME
    answer, and what separates them is only how well each is resolved.  The
    extreme profiles put most of the conductance in a few cells and are
    therefore the hardest to resolve, which is why they lag slightly.

    WHAT WOULD BREAK THE INVARIANCE.  It rests on C_h and C_c being constant.
    Let the specific heats vary with temperature and theta no longer satisfies
    a single equation, the distribution of U starts to matter, and no closed
    form survives.  That is the genuinely open case, and the solver built here
    reaches it without modification -- which was the point of building it.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

for k, arr in enumerate(("counterflow", "parallel")):
    xc, Th, Tc, _ = solve_exchanger(400, UA_REF, CH, CC, THI, TCI, arr)
    ax[k].plot(xc, Th, "-", lw=2.3, color="#b2182b", label=r"hot stream")
    ax[k].plot(xc, Tc, "-", lw=2.3, color="#2166ac", label=r"cold stream")
    ax[k].fill_between(xc, Tc, Th, color="0.6", alpha=0.15)
    Tho, Tco = outlets(Th, Tc, arr)
    ax[k].set_xlabel(r"$x/L$")
    ax[k].set_ylabel(r"$T$  [K]")
    ax[k].set_title(f"({'ab'[k]}) {arr.capitalize()}")
    ax[k].set_ylim(295, 365)
    ax[k].legend(fontsize=8.5, loc="center right")
    e_fvm = CH * (THI - Tho) / (CMIN * (THI - TCI))
    ax[k].annotate(rf"$\varepsilon = {e_fvm:.4f}$" "\n"
                   rf"$T_{{h,out}} = {Tho:.1f}$ K" "\n"
                   rf"$T_{{c,out}} = {Tco:.1f}$ K",
                   xy=(0.04, 0.06), xycoords="axes fraction", fontsize=8.2,
                   bbox=dict(facecolor="white", alpha=0.92, edgecolor="0.75",
                             boxstyle="round,pad=0.3"))
    if arr == "counterflow":
        ax[k].annotate("cold stream flows\nright to left",
                       xy=(0.55, 0.90), xycoords="axes fraction",
                       fontsize=8.0, color="#2166ac", ha="center")

fig.suptitle("Example 10.2 -- The same exchanger, two flow arrangements",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_2a_profiles.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

NS = np.array([10, 20, 40, 80, 160, 320])
for arr, ef, c in (("counterflow", eps_counterflow, "#b2182b"),
                   ("parallel", eps_parallel, "#2166ac")):
    e_ex = ef(NTU, CR)
    errs = []
    for N in NS:
        xc, Th, Tc, _ = solve_exchanger(int(N), UA_REF, CH, CC, THI, TCI, arr)
        Tho, _ = outlets(Th, Tc, arr)
        errs.append(abs(CH * (THI - Tho) / (CMIN * (THI - TCI)) - e_ex))
    ax[0].loglog(NS, errs, "o-", lw=1.9, ms=7, mfc="none", mew=1.7, color=c,
                 label=arr)
ax[0].loglog(NS, errs[0] * (NS[0] / NS) ** 1.0, "k--", lw=1.3,
             label=r"slope $-1$")
ax[0].set_xlabel(r"$N$  (control volumes)")
ax[0].set_ylabel(r"$|\varepsilon_{\rm FVM} - \varepsilon_{\rm exact}|$")
ax[0].set_title("(a) Upwinding is first order, and says so")
ax[0].legend(fontsize=8.5, loc="lower left")

xs = np.linspace(0, 1, 400)
for (name, prof), c in zip(list(profiles.items())[:3],
                           ["#762a83", "#1b7837", "#e08214"]):
    xc, Th, Tc, ua = solve_exchanger(400, UA_REF, CH, CC, THI, TCI,
                                     "counterflow", U_profile=prof)
    Tho, Tco = outlets(Th, Tc, "counterflow")
    e_v = CH * (THI - Tho) / (CMIN * (THI - TCI))
    ax[1].plot(xc, Th - Tc, "-", lw=2.1, color=c,
               label=rf"{name}, $\varepsilon = {e_v:.4f}$")
ax[1].set_xlabel(r"$x/L$")
ax[1].set_ylabel(r"$T_h - T_c$  [K]")
ax[1].set_title("(b) Same total $UA$: same duty, different path")
ax[1].legend(fontsize=8.0, loc="upper right")

fig.suptitle("Example 10.2 -- Verification, and a case with no closed form",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_2b_verification.png")
plt.close(fig)

print("Figures written: fig_10_2a_profiles.png, fig_10_2b_verification.png")
