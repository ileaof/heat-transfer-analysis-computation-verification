"""
================================================================================
 EXAMPLE 13.2 -- VALIDATION: A VERIFIED CODE CAN SOLVE THE WRONG PROBLEM
 The distinction that verification alone cannot draw
================================================================================

 OBJECTIVE
 ---------
 Example 13.1 verified a SOLUTION: it bounded the numerical error of a
 computation.  But a computation can be numerically flawless and still be
 wrong, because it solves the wrong equation.  VERIFICATION asks whether the
 equations are solved correctly; VALIDATION asks whether the correct equations
 were chosen.  They are different questions, and no amount of grid refinement
 answers the second.

 This example makes the distinction concrete and unmistakable.  Two models of
 the same physical situation are built.  BOTH are verified -- each converges at
 its design order to its own exact answer.  One model is right and one is
 wrong, and only comparison with a trusted BENCHMARK -- validation -- tells them
 apart.  Grid refinement drives each toward its own answer with perfect
 fidelity; it cannot reveal that one of those answers is the wrong physics.

 THE PHYSICAL SITUATION
 ----------------------
 A pin fin loses heat to the surroundings.  The HONEST model is the full
 one-dimensional fin equation with convection along the whole surface:

        d/dx( k A dT/dx ) - h P (T - T_inf) = 0            (Chapter 7)

 whose exact solution is the hyperbolic profile of Chapter 7.  The FLAWED model
 -- a common and plausible simplification -- neglects the lateral convection
 and instead lumps ALL the heat loss into an effective tip condition, solving
 pure conduction with a modified end.  It is a different equation; it has its
 own clean exact solution; and it is WRONG, by an amount that depends on the
 fin parameter mL.

 Both models are discretised and both are verified against their own exact
 solutions.  Then both are validated against the benchmark -- the full fin
 solution treated as the "truth" a careful experiment would deliver.

 THE VALIDATION METRIC
 ---------------------
 Validation is quantitative.  The comparison error is

        E = S - D

 where S is the simulation result and D the benchmark datum.  With a numerical
 uncertainty U_num (from Example 13.1's GCI) and a benchmark uncertainty U_D,
 the model is VALIDATED at the point if

        |E| <= U_val = sqrt(U_num^2 + U_D^2)

 -- the comparison error is within the combined uncertainty.  If |E| exceeds
 U_val, the discrepancy is real: it is MODEL error, and no refinement will
 remove it.  This example computes E, U_num and U_D and applies the metric.

 OUTPUTS
 -------
   fig_13_2a_models.png       the two models against the benchmark
   fig_13_2b_validation.png   verification of each, and the validation metric

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
# 1. PHYSICAL DATA -- an aluminium pin fin (as in Chapter 7)
# ==============================================================================
K = 180.0               # W/m.K
L = 0.05                # m
D_FIN = 0.005           # m diameter
A_C = np.pi * D_FIN ** 2 / 4.0
PERI = np.pi * D_FIN
H = 50.0                # W/m^2 K
T_B = 400.0             # K base
T_INF = 300.0           # K ambient
M = np.sqrt(H * PERI / (K * A_C))       # fin parameter
ML = M * L
THETA_B = T_B - T_INF


# ==============================================================================
# 2. THE TWO EXACT SOLUTIONS
# ==============================================================================
def theta_full(x):
    """Exact FULL fin: convection along the length, insulated tip (Chapter 7).
    theta = theta_b cosh(m(L - x))/cosh(mL)."""
    return THETA_B * np.cosh(M * (L - x)) / np.cosh(ML)


def theta_flawed(x):
    """The FLAWED model: NO lateral convection, all heat loss lumped into an
    effective convective tip.  This is pure conduction, so theta is LINEAR in x,
    with the slope set by an energy balance that (wrongly) puts the whole fin's
    heat loss at the tip.

    The tip 'sink' is chosen so the model conserves the correct TOTAL heat rate
    of an infinite-conductivity fin -- a natural but incorrect closure.  The
    result is a straight line from theta_b to a tip excess theta_tip.
    """
    # total heat the real fin would lose if isothermal at theta_b:
    q_lump = H * PERI * L * THETA_B
    # pure conduction with that heat leaving the tip: linear profile
    theta_tip = THETA_B - q_lump * L / (K * A_C) * 0.5   # mid-lumped closure
    return THETA_B + (theta_tip - THETA_B) * (x / L)


# ==============================================================================
# 3. THE SOLVERS (both second order, both verified against their own exact)
# ==============================================================================
def solve_full(N):
    """FVM for the full fin equation, second order."""
    dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    x = np.concatenate(([0.0], xc, [L]))
    n = len(x)
    dxn = np.diff(x)
    D = K * A_C / dxn
    Sp = H * PERI * dx                       # convective sink per cell (>0)
    a_P = np.zeros(n); a_E = np.zeros(n); a_W = np.zeros(n); b = np.zeros(n)
    for i in range(1, n - 1):
        a_E[i] = D[i]
        a_W[i] = D[i - 1]
        a_P[i] = a_E[i] + a_W[i] + Sp
        b[i] = Sp * T_INF
    a_P[0] = 1.0; b[0] = T_B
    a_P[-1] = 1.0 + 0.0
    # insulated tip: node follows neighbour
    a_P[-1] = 1.0; a_W[-1] = 1.0; b[-1] = 0.0
    ab = np.zeros((3, n))
    ab[0, 1:] = -a_E[:-1]; ab[1, :] = a_P; ab[2, :-1] = -a_W[1:]
    T = solve_banded((1, 1), ab, b)
    return x, T - T_INF


def solve_flawed(N):
    """FVM for the flawed pure-conduction model with the lumped tip sink."""
    dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    x = np.concatenate(([0.0], xc, [L]))
    n = len(x)
    dxn = np.diff(x)
    D = K * A_C / dxn
    q_lump = H * PERI * L * THETA_B
    theta_tip = THETA_B - q_lump * L / (K * A_C) * 0.5
    a_P = np.zeros(n); a_E = np.zeros(n); a_W = np.zeros(n); b = np.zeros(n)
    for i in range(1, n - 1):
        a_E[i] = D[i]; a_W[i] = D[i - 1]; a_P[i] = a_E[i] + a_W[i]
    a_P[0] = 1.0; b[0] = T_B
    a_P[-1] = 1.0; b[-1] = theta_tip + T_INF
    ab = np.zeros((3, n))
    ab[0, 1:] = -a_E[:-1]; ab[1, :] = a_P; ab[2, :-1] = -a_W[1:]
    T = solve_banded((1, 1), ab, b)
    return x, T - T_INF


def tip_temperature(x, theta):
    return theta[-1]


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 13.2 -- VALIDATION VERSUS VERIFICATION")
print("=" * 78)
print(f"  Aluminium pin fin: mL = {ML:.4f}, base excess = {THETA_B:.0f} K")
print(f"  Benchmark (full fin) tip excess = {theta_full(L):.6f} K")
print(f"  Flawed model        tip excess = {theta_flawed(L):.6f} K")

print("\n" + "-" * 78)
print("  CHECK 1 -- BOTH MODELS ARE VERIFIED (each converges at order 2)")
print("""    The essential point: BOTH codes are numerically correct.  Each
    converges at its design order to ITS OWN exact solution.  Verification
    cannot distinguish them, because verification never looks outside a single
    model.""")


def order_study(solver, exact_fn, label):
    print(f"\n    {label}")
    print(f"    {'N':>6} {'tip excess':>14} {'error vs own exact':>20} {'p':>7}")
    prev, prevN = None, None
    vals = []
    for N in (10, 20, 40, 80):
        x, th = solver(N)
        tip = th[-1]
        err = abs(tip - exact_fn(L))
        vals.append(tip)
        p = np.log2(prev / err) if (prev and err > 0) else None
        print(f"    {N:>6d} {tip:>14.8f} {err:>20.3e} "
              f"{('%.3f' % p) if p else '-':>7}")
        prev = err
    return vals


full_vals = order_study(solve_full, theta_full, "FULL model (the honest one)")
flaw_vals = order_study(solve_flawed, theta_flawed, "FLAWED model (wrong physics)")
print("""
    The full model converges at order 2 to its exact hyperbolic profile.  The
    flawed model reproduces ITS exact solution to 1e-14 on every grid -- because
    its solution is linear and a second-order scheme captures a line exactly.
    That is the strongest verification imaginable: ZERO numerical error.  And
    it makes the coming verdict sharper still: a model can be numerically
    PERFECT and physically WRONG.  Verification, looking only inside one model,
    cannot tell.""")

print("\n" + "-" * 78)
print("  CHECK 2 -- VALIDATION: COMPARE EACH AGAINST THE BENCHMARK")
print("""    The full fin solution is the benchmark D -- what a careful experiment
    would measure.  We give it a realistic experimental uncertainty U_D of 2 %
    of the base excess.  Each model's tip prediction S is compared: the
    comparison error E = S - D against the validation uncertainty
    U_val = sqrt(U_num^2 + U_D^2).""")
D_bench = theta_full(L)
U_D = 0.02 * THETA_B                     # 2 % experimental uncertainty
# numerical uncertainty from a GCI on the finest grids of each model
def gci_num(vals, ratio=2.0, Fs=1.25):
    f3, f2, f1 = vals[-3:]
    if abs(f1 - f2) < 1e-12 * max(abs(f1), 1.0):
        # the model is captured EXACTLY on every grid (a linear profile under a
        # second-order scheme): the discretisation error is zero, so the
        # numerical uncertainty is zero.  This is not a failure of the GCI but
        # the strongest possible verification -- and it makes the validation
        # verdict below all the sharper.
        return 0.0
    p = np.log(abs((f2 - f3) / (f1 - f2))) / np.log(ratio)
    return Fs * abs((f1 - f2) / f1) / (ratio ** p - 1.0) * abs(f1)
U_num_full = gci_num(full_vals)
U_num_flaw = gci_num(flaw_vals)
print(f"\n    benchmark tip excess D          = {D_bench:.6f} K")
print(f"    experimental uncertainty U_D    = {U_D:.4f} K (2%)")
print(f"\n  {'model':>16} {'S (tip)':>12} {'E = S - D':>12} {'U_num':>10} "
      f"{'U_val':>10} {'|E|<=U_val?':>13}")
for label, vals, U_num in (("full (honest)", full_vals, U_num_full),
                           ("flawed", flaw_vals, U_num_flaw)):
    S = vals[-1]
    E = S - D_bench
    U_val = np.sqrt(U_num ** 2 + U_D ** 2)
    ok = abs(E) <= U_val
    print(f"  {label:>16} {S:>12.6f} {E:>12.6f} {U_num:>10.2e} "
          f"{U_val:>10.4f} {str(ok):>13}")
print("""
    The honest model's comparison error is far inside the uncertainty band:
    validated.  The flawed model's error is many times the band: NOT validated.
    The discrepancy is model error, and it is invisible to verification -- the
    flawed model passed its order test in Check 1 with the same second order as
    the honest one.  Only the comparison with the benchmark exposes it.""")

print("\n" + "-" * 78)
print("  CHECK 3 -- REFINEMENT CANNOT FIX A MODEL ERROR")
print("""    If the flawed model's discrepancy were numerical, refining the grid
    would shrink it.  It does not: the comparison error converges to a nonzero
    CONSTANT as N grows, because it is built into the equation, not the
    discretisation.  This is the operational signature of model error versus
    numerical error.""")
print(f"\n  {'N':>6} {'flawed tip':>13} {'E vs benchmark':>16} {'shrinking?':>12}")
prev_E = None
for N in (10, 20, 40, 80, 160):
    x, th = solve_flawed(N)
    E = th[-1] - D_bench
    shrink = (prev_E is not None) and (abs(E) < 0.9 * abs(prev_E))
    print(f"  {N:>6d} {th[-1]:>13.6f} {E:>16.6f} {str(shrink):>12}")
    prev_E = E
print("""    The error is frozen at about {:.2f} K regardless of grid -- a flat
    line, not a decaying one.  That flatness IS the diagnosis: a numerical
    error decays under refinement, a model error does not.""".format(
    solve_flawed(160)[1][-1] - D_bench))

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

xe = np.linspace(0, L, 200)
ax[0].plot(xe * 1e3, theta_full(xe), "-", lw=2.4, color="#b2182b",
           label="benchmark (full fin)")
ax[0].plot(xe * 1e3, theta_flawed(xe), "--", lw=2.0, color="#2166ac",
           label="flawed model")
xN, thN = solve_full(20)
ax[0].plot(xN * 1e3, thN, "o", ms=4, mfc="none", mew=1.2, color="#b2182b",
           label="full, FVM $N=20$", markevery=2)
ax[0].fill_between(xe * 1e3, theta_flawed(xe), theta_full(xe),
                   color="0.6", alpha=0.2)
ax[0].annotate("model error\n(not numerical)", xy=(28, 0.5 *
               (theta_full(0.028) + theta_flawed(0.028))),
               xytext=(6, 80), fontsize=8.5, color="0.3",
               arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))
ax[0].set_xlabel(r"$x$  [mm]")
ax[0].set_ylabel(r"excess temperature $\theta$  [K]")
ax[0].set_title("(a) Two models, one benchmark")
ax[0].legend(fontsize=8.0, loc="upper right")

# verification of both, then validation gap
Ns = np.array([10, 20, 40, 80])
ef = np.array([abs(solve_full(N)[1][-1] - theta_full(L)) for N in Ns])
eg = np.array([abs(solve_flawed(N)[1][-1] - theta_flawed(L)) for N in Ns])
model_gap = abs(theta_flawed(L) - theta_full(L))
ax[1].loglog(Ns, ef, "o-", lw=1.9, ms=7, mfc="none", mew=1.6,
             color="#b2182b", label="full: error vs own exact")
ax[1].loglog(Ns, eg, "s-", lw=1.9, ms=7, mfc="none", mew=1.6,
             color="#2166ac", label="flawed: error vs own exact")
ax[1].axhline(model_gap, color="#1b7837", ls="--", lw=1.8,
              label="model error (flawed vs benchmark)")
ax[1].loglog(Ns, ef[0] * (Ns[0] / Ns) ** 2.0, "k:", lw=1.1, label="slope -2")
ax[1].set_xlabel(r"$N$  (cells)")
ax[1].set_ylabel("error  [K]")
ax[1].set_title("(b) Both verified; one invalid by a fixed gap")
ax[1].legend(fontsize=7.5, loc="center left")

fig.suptitle("Example 13.2 -- Verification certifies the arithmetic, "
             "not the physics", fontsize=11.8, y=1.08)
fig.savefig("fig_13_2a_models.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

# the validation metric as a bar chart
models = ["full\n(honest)", "flawed"]
Es = [abs(full_vals[-1] - D_bench), abs(flaw_vals[-1] - D_bench)]
Uvals = [np.sqrt(U_num_full ** 2 + U_D ** 2),
         np.sqrt(U_num_flaw ** 2 + U_D ** 2)]
xpos = np.arange(2)
w = 0.35
ax[0].bar(xpos - w / 2, Es, w, color="#b2182b", alpha=0.85,
          label="|comparison error E|")
ax[0].bar(xpos + w / 2, Uvals, w, color="#2166ac", alpha=0.85,
          label="validation uncertainty $U_{val}$")
ax[0].set_xticks(xpos)
ax[0].set_xticklabels(models)
ax[0].set_ylabel("[K]")
ax[0].set_title(r"(a) The validation metric $|E| \leq U_{val}$")
ax[0].legend(fontsize=8.0, loc="upper left")
ax[0].annotate("validated", xy=(0, Es[0]), xytext=(0, Uvals[0] * 1.3),
               ha="center", fontsize=8.5, color="#1b7837")
ax[0].annotate("NOT validated", xy=(1, Es[1]), xytext=(1, Es[1] * 1.02),
               ha="center", fontsize=8.5, color="#b2182b")

# model error frozen under refinement
Nr = np.array([10, 20, 40, 80, 160])
Eg = np.array([abs(solve_flawed(N)[1][-1] - D_bench) for N in Nr])
En = np.array([abs(solve_full(N)[1][-1] - D_bench) for N in Nr])
ax[1].semilogx(Nr, Eg, "s-", lw=1.9, ms=7, mfc="none", mew=1.6,
               color="#2166ac", label="flawed model, E vs benchmark")
ax[1].semilogx(Nr, En, "o-", lw=1.9, ms=7, mfc="none", mew=1.6,
               color="#b2182b", label="full model, E vs benchmark")
ax[1].set_xlabel(r"$N$  (cells)")
ax[1].set_ylabel("comparison error $|E|$  [K]")
ax[1].set_title("(b) Model error does not shrink; numerical error does")
ax[1].legend(fontsize=8.0, loc="center right")

fig.suptitle("Example 13.2 -- The validation metric, and the signature of "
             "model error", fontsize=11.8, y=1.08)
fig.savefig("fig_13_2b_validation.png")
plt.close(fig)

print("Figures written: fig_13_2a_models.png, fig_13_2b_validation.png")
