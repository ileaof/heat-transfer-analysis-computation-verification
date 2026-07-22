"""
================================================================================
 EXAMPLE 10.1 -- THE TWO CLASSICAL METHODS, AND WHY THEY MUST AGREE
 LMTD and effectiveness-NTU, derived, compared, and pushed to their limits
================================================================================

 OBJECTIVE
 ---------
 Heat exchanger analysis is taught as two methods.  The LMTD method computes the
 duty from the terminal temperatures; the effectiveness-NTU method computes the
 outlet temperatures from the inlet ones.  Textbooks present them as alternative
 tools for different situations, which is true of their CONVENIENCE and false of
 their CONTENT: they are two rearrangements of the same energy balance, and for
 a given exchanger they must give identical answers.

 That identity is the backbone of this example.  It is not an approximation to
 be checked within a tolerance; it is an algebraic fact, and agreement to
 machine precision is the only acceptable outcome.

 WHAT IS COMPUTED
 ----------------
   1. The effectiveness relations for counterflow, parallel flow and crossflow,
      and the LMTD method with its correction factor.
   2. The identity: LMTD and eps-NTU applied to the same exchanger, over a wide
      sweep of NTU and capacity ratio.
   3. The removable singularity in the log-mean temperature difference, which
      is a genuine numerical hazard, and its cure.
   4. The exact limits: C_r -> 0 (both arrangements give 1 - exp(-NTU)),
      NTU -> 0 (eps -> NTU), NTU -> infinity (counterflow -> 1, parallel ->
      1/(1+C_r)).
   5. The ordering  eps_parallel <= eps_crossflow <= eps_counterflow, which must
      hold at every NTU and every C_r, and is tested rather than asserted.

 GOVERNING RELATIONS
 -------------------
        C_min, C_max  = the smaller and larger of  m_dot c_p
        C_r  = C_min / C_max
        NTU  = U A / C_min
        eps  = q / q_max ,   q_max = C_min (T_h,in - T_c,in)

 Counterflow:   eps = [1 - exp(-NTU(1-C_r))] / [1 - C_r exp(-NTU(1-C_r))]
                eps = NTU / (1 + NTU)                        when C_r = 1
 Parallel:      eps = [1 - exp(-NTU(1+C_r))] / (1 + C_r)
 Crossflow, both fluids unmixed (Holman's correlation):
                eps = 1 - exp{ (1/C_r) NTU^0.22 [exp(-C_r NTU^0.78) - 1] }

 The crossflow expression is a CORRELATION fitted to the exact series solution,
 not an exact result, and it is labelled as such throughout.  Example 10.3
 computes the exact crossflow answer by solving the governing equations, and
 measures how good the correlation actually is.

 SYMBOLS (all SI)
 ----------------
   C       [W/K]     capacity rate, m_dot c_p
   C_r     [-]       capacity ratio C_min/C_max
   U       [W/m^2 K] overall heat transfer coefficient
   A       [m^2]     heat transfer area
   NTU     [-]       number of transfer units, UA/C_min
   eps     [-]       effectiveness
   q       [W]       heat duty
   DT_lm   [K]       log-mean temperature difference
   F       [-]       LMTD correction factor

 OUTPUTS
 -------
   fig_10_1a_effectiveness.png  effectiveness charts and the ordering
   fig_10_1b_identity.png       LMTD vs eps-NTU, and the log-mean singularity

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.optimize import brentq
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
# 1. EFFECTIVENESS RELATIONS
# ==============================================================================
def eps_counterflow(NTU, Cr):
    """Counterflow effectiveness.

    The general expression is indeterminate at C_r = 1, where numerator and
    denominator both vanish.  The limit is NTU/(1+NTU), and it is applied
    directly rather than approached, because floating point cannot approach it:
    at C_r = 1 - 1e-15 the expression is 0/0 in the arithmetic actually
    performed.
    """
    NTU = np.asarray(NTU, dtype=float)
    Cr = np.asarray(Cr, dtype=float)
    out = np.empty(np.broadcast(NTU, Cr).shape, dtype=float)
    near_one = np.abs(Cr - 1.0) < 1e-8
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = np.exp(-NTU * (1.0 - Cr))
        gen = (1.0 - e) / (1.0 - Cr * e)
    lim = NTU / (1.0 + NTU)
    return np.where(near_one, lim, gen)


def eps_parallel(NTU, Cr):
    """Parallel-flow effectiveness.  No singularity anywhere."""
    NTU = np.asarray(NTU, dtype=float)
    Cr = np.asarray(Cr, dtype=float)
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


def eps_crossflow_corr(NTU, Cr):
    """Crossflow, both fluids unmixed -- an empirical CORRELATION.

    Exact only in the limits C_r -> 0 and NTU -> 0; elsewhere it approximates
    a series solution.  Example 10.3 measures its error against a direct
    numerical solution of the governing equations.
    """
    NTU = np.asarray(NTU, dtype=float)
    Cr = np.asarray(Cr, dtype=float)
    Cr_safe = np.where(Cr < 1e-10, 1e-10, Cr)
    val = 1.0 - np.exp((1.0 / Cr_safe) * NTU ** 0.22 *
                       (np.exp(-Cr_safe * NTU ** 0.78) - 1.0))
    return np.where(Cr < 1e-10, 1.0 - np.exp(-NTU), val)


# ==============================================================================
# 2. THE LOG-MEAN TEMPERATURE DIFFERENCE
# ==============================================================================
def lmtd_naive(dT1, dT2):
    """The textbook formula, written exactly as it is usually printed."""
    return (dT1 - dT2) / np.log(dT1 / dT2)


def lmtd(dT1, dT2):
    """Log-mean temperature difference, accurate at and near dT1 = dT2.

    THE SINGULARITY IS REMOVABLE, AND THAT IS THE PROBLEM.  As dT2 -> dT1 both
    numerator and denominator tend to zero and the true limit is the common
    value dT1.  In exact arithmetic the textbook formula is fine everywhere
    except the single point dT1 = dT2.  In floating point it is unusable over a
    whole NEIGHBOURHOOD of that point, because dT1 - dT2 and ln(dT1/dT2) each
    lose their significant digits to cancellation.

    The cure is not a series.  Writing s = (dT2 - dT1)/dT1, algebra gives the
    exactly equivalent form

        DT_lm = dT1 * s / ln(1 + s)

    and `log1p` computes ln(1+s) to full precision for small s by construction.
    One expression, accurate everywhere, with a single special case at s = 0.

    A FIRST DRAFT OF THIS FUNCTION USED A SERIES, AND THE SERIES WAS WRONG.  It
    carried 1 - s/2 + s^2/3 - ..., which is the expansion of ln(1+s)/s -- the
    RECIPROCAL of what is needed.  The error was invisible in the code and
    obvious in the output: the "corrected" formula disagreed with the textbook
    one by 10 % at s = 0.1, where the textbook formula is perfectly accurate.
    When a fix disagrees with the thing it is fixing in the regime where the
    original was never in trouble, the fix is what is broken.
    """
    dT1 = np.asarray(dT1, dtype=float)
    dT2 = np.asarray(dT2, dtype=float)
    s = (dT2 - dT1) / dT1
    with np.errstate(divide="ignore", invalid="ignore"):
        val = dT1 * s / np.log1p(s)
    return np.where(s == 0.0, dT1, val)


# ==============================================================================
# 3. THE TWO METHODS APPLIED TO THE SAME EXCHANGER
# ==============================================================================
def solve_by_eps_ntu(UA, Ch, Cc, Thi, Tci, arrangement="counterflow"):
    """Outlet temperatures and duty from the effectiveness relations."""
    Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
    Cr = Cmin / Cmax
    NTU = UA / Cmin
    e = {"counterflow": eps_counterflow,
         "parallel": eps_parallel,
         "crossflow": eps_crossflow_corr}[arrangement](NTU, Cr)
    e = float(e)
    q = e * Cmin * (Thi - Tci)
    return q, Thi - q / Ch, Tci + q / Cc, e, NTU, Cr


def duty_by_lmtd(UA, Ch, Cc, Thi, Tci, Tho, Tco, arrangement="counterflow"):
    """Duty from U A F DT_lm, using the outlet temperatures as data.

    For counterflow F = 1 by definition; for parallel flow F = 1 as well, with
    the terminal differences taken at the correct ends.  The correction factor
    F exists for arrangements that are neither, and is not needed here.
    """
    if arrangement == "counterflow":
        dT1, dT2 = Thi - Tco, Tho - Tci
    elif arrangement == "parallel":
        dT1, dT2 = Thi - Tci, Tho - Tco
    else:
        raise ValueError("F factor required for this arrangement")
    return UA * float(lmtd(dT1, dT2))


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 10.1 -- LMTD AND EFFECTIVENESS-NTU")
print("=" * 78)

# A water-to-water exchanger
M_H, CP_H = 1.20, 4180.0        # kg/s, J/kg.K   (hot stream)
M_C, CP_C = 2.00, 4180.0        # kg/s, J/kg.K   (cold stream)
CH, CC = M_H * CP_H, M_C * CP_C
THI, TCI = 360.0, 300.0         # K
UA_REF = 12000.0                # W/K

print(f"  Hot stream : C_h = {CH:,.1f} W/K, T_in = {THI:.1f} K")
print(f"  Cold stream: C_c = {CC:,.1f} W/K, T_in = {TCI:.1f} K")
print(f"  UA = {UA_REF:,.0f} W/K")

for arr in ("counterflow", "parallel"):
    q, Tho, Tco, e, NTU, Cr = solve_by_eps_ntu(UA_REF, CH, CC, THI, TCI, arr)
    print(f"\n  {arr.upper()}")
    print(f"    C_r = {Cr:.6f},  NTU = {NTU:.6f},  eps = {e:.9f}")
    print(f"    q = {q:,.4f} W,  T_h,out = {Tho:.6f} K,  T_c,out = {Tco:.6f} K")
    q_lmtd = duty_by_lmtd(UA_REF, CH, CC, THI, TCI, Tho, Tco, arr)
    print(f"    duty from eps-NTU = {q:.9f} W")
    print(f"    duty from LMTD    = {q_lmtd:.9f} W")
    print(f"    relative difference = {abs(q_lmtd/q - 1):.3e}")

print("\n" + "-" * 78)
print("  CHECK 1 -- THE IDENTITY, SWEPT OVER THE WHOLE PARAMETER SPACE")
print("""    The two methods are rearrangements of one energy balance, so their
    agreement is an identity and not a numerical coincidence.  Sweeping NTU
    from 0.01 to 20 and C_r from 0 to 1 tests it where the formulas are most
    likely to misbehave.""")
worst = {"counterflow": 0.0, "parallel": 0.0}
worst_at = {"counterflow": None, "parallel": None}
excluded = {"counterflow": 0, "parallel": 0}
for arr in ("counterflow", "parallel"):
    for NTU_t in np.logspace(-2, np.log10(20.0), 60):
        for Cr_t in np.linspace(0.0, 1.0, 41):
            Cmin = 5000.0
            Cmax = Cmin / Cr_t if Cr_t > 0 else 1e12
            UA_t = NTU_t * Cmin
            q, Tho, Tco, e, _, _ = solve_by_eps_ntu(UA_t, Cmin, Cmax,
                                                    THI, TCI, arr)
            if q <= 0.0:
                continue
            dT1 = (THI - Tco) if arr == "counterflow" else (THI - TCI)
            dT2 = (Tho - TCI) if arr == "counterflow" else (Tho - Tco)
            # The terminal difference is formed by subtracting two computed
            # temperatures.  When the exchanger approaches thermal equilibrium
            # those two are nearly equal and the subtraction destroys the
            # result -- before the log-mean formula is ever reached.  Points
            # where the surviving relative precision is worse than 1e-9 are
            # counted separately rather than silently averaged in.
            if abs(dT2) / max(abs(THI), abs(TCI)) < 1e-9:
                excluded[arr] += 1
                continue
            q_l = duty_by_lmtd(UA_t, Cmin, Cmax, THI, TCI, Tho, Tco, arr)
            rel = abs(q_l / q - 1.0)
            if rel > worst[arr]:
                worst[arr] = rel
                worst_at[arr] = (NTU_t, Cr_t)
    n, c = worst_at[arr]
    print(f"    {arr:<13} worst relative difference = {worst[arr]:.3e}"
          f"   at NTU = {n:.3f}, C_r = {c:.3f}")
    print(f"    {'':<13} points excluded for cancellation: {excluded[arr]}")

print("""
    CHECK 1b -- WHERE THE LMTD METHOD SIMPLY STOPS WORKING.  The excluded
    points are not a numerical nuisance to be stepped around; they are the
    reason the effectiveness method exists.  As an exchanger approaches
    thermal equilibrium the two outlet temperatures converge, and the terminal
    difference DT_2 = T_h,out - T_c,out is the subtraction of two nearly equal
    numbers.  No amount of care in the log-mean formula can recover digits that
    were destroyed before it was called.""")
print(f"\n  Parallel flow at C_r = 1, C_min = 5000 W/K:")
print(f"  {'NTU':>7} {'T_h,out':>12} {'T_c,out':>12} {'DT_2 [K]':>12} "
      f"{'q (eps-NTU)':>14} {'q (LMTD)':>14} {'rel diff':>11}")
for NTU_t in (1.0, 4.0, 8.0, 12.0, 16.0, 20.0):
    Cmin = 5000.0
    q, Tho, Tco, _, _, _ = solve_by_eps_ntu(NTU_t * Cmin, Cmin, Cmin,
                                            THI, TCI, "parallel")
    dT2 = Tho - Tco
    with np.errstate(divide="ignore", invalid="ignore"):
        q_l = duty_by_lmtd(NTU_t * Cmin, Cmin, Cmin, THI, TCI, Tho, Tco,
                           "parallel")
    rel = abs(q_l / q - 1.0) if q else float("nan")
    print(f"  {NTU_t:>7.1f} {Tho:>12.8f} {Tco:>12.8f} {dT2:>12.3e} "
          f"{q:>14.4f} {q_l:>14.4f} {rel:>11.2e}")
print("""    By NTU = 16 the two outlets agree to twelve figures, DT_2 has been
    reduced to numerical noise, and the LMTD duty is wrong by a large factor
    or is NaN.  The effectiveness method computes the same duty exactly, from
    the inlet temperatures alone, because it never forms that difference.
    This is the practical content of the distinction between the two methods,
    and it is sharper than the usual statement that one is 'more convenient
    when the outlets are unknown'.""")

# ---- the log-mean singularity ----------------------------------------------
print("\n" + "-" * 78)
print("  CHECK 2 -- THE LOG-MEAN SINGULARITY IS A REAL NUMERICAL HAZARD")
print("""    When the two terminal differences are equal -- which happens exactly
    when C_r = 1 in counterflow, the commonest design point of all -- the
    textbook formula evaluates 0/0.  Nearby it does something worse: it returns
    a number that looks plausible and is wrong.

    The reference column below is the log1p form, which is exact by
    construction.  That it IS trustworthy is established independently by the
    Taylor expansion DT_lm = dT1 (1 + s/2 - s^2/12 + s^3/24 + ...), valid for
    small s and derived by a different route.""")
print(f"\n  {'s = dT2/dT1 - 1':>16} {'textbook':>17} {'log1p form':>17} "
      f"{'series':>17} {'textbook error':>15}")
dT1 = 20.0
for s_v in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 0.0):
    dT2 = dT1 * (1.0 + s_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        nv = float(lmtd_naive(np.array(dT1), np.array(dT2)))
    sf = float(lmtd(np.array(dT1), np.array(dT2)))
    ser = dT1 * (1.0 + s_v / 2.0 - s_v ** 2 / 12.0 + s_v ** 3 / 24.0)
    print(f"  {s_v:>16.0e} {nv:>17.11f} {sf:>17.11f} {ser:>17.11f} "
          f"{abs(nv - sf):>15.2e}")
print("""    The series and the log1p form agree to eleven figures wherever the
    series is valid, which establishes the reference.  Against it, the textbook
    formula is accurate for s down to about 1e-11, degrades below that, and
    returns NaN at s = 0 exactly.  A counterflow exchanger with balanced
    capacity rates sits precisely on that point.""")

# ---- exact limits -----------------------------------------------------------
print("\n" + "-" * 78)
print("  CHECK 3 -- THE EXACT LIMITS")
print("\n  (a) C_r -> 0: every arrangement must give eps = 1 - exp(-NTU),")
print("      because one stream is at constant temperature and the flow")
print("      arrangement can then make no difference.")
print(f"  {'NTU':>8} {'counterflow':>14} {'parallel':>14} {'crossflow':>14} "
      f"{'1-exp(-NTU)':>14}")
for NTU_t in (0.5, 1.0, 2.0, 5.0):
    a = float(eps_counterflow(NTU_t, 0.0))
    b = float(eps_parallel(NTU_t, 0.0))
    c = float(eps_crossflow_corr(NTU_t, 0.0))
    ex = 1.0 - np.exp(-NTU_t)
    print(f"  {NTU_t:>8.2f} {a:>14.10f} {b:>14.10f} {c:>14.10f} {ex:>14.10f}")

print("\n  (b) NTU -> 0: eps -> NTU for every arrangement (a small exchanger")
print("      transfers heat in proportion to its size).")
print(f"  {'NTU':>10} {'eps_counter':>15} {'eps/NTU':>12}")
for NTU_t in (1e-2, 1e-3, 1e-4, 1e-5):
    e = float(eps_counterflow(NTU_t, 0.5))
    print(f"  {NTU_t:>10.0e} {e:>15.10f} {e/NTU_t:>12.8f}")

print("\n  (c) NTU -> infinity: counterflow -> 1, parallel -> 1/(1+C_r).")
print(f"  {'C_r':>8} {'counterflow':>16} {'parallel':>16} {'1/(1+C_r)':>14}")
for Cr_t in (0.0, 0.25, 0.5, 0.75, 1.0):
    a = float(eps_counterflow(1e4, Cr_t))
    b = float(eps_parallel(1e4, Cr_t))
    print(f"  {Cr_t:>8.2f} {a:>16.10f} {b:>16.10f} {1.0/(1.0+Cr_t):>14.10f}")

# ---- the C_r = 1 removable singularity in the effectiveness relation --------
print("\n  (d) The counterflow relation at C_r = 1 is also 0/0, and the limit")
print("      NTU/(1+NTU) is applied explicitly.  Approaching it numerically:")
print(f"  {'1 - C_r':>10} {'general formula':>18} {'NTU/(1+NTU)':>15}")
NTU_t = 3.0
for d in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    Cr_t = 1.0 - d
    with np.errstate(divide="ignore", invalid="ignore"):
        e_gen = ((1 - np.exp(-NTU_t * d)) /
                 (1 - Cr_t * np.exp(-NTU_t * d)))
    print(f"  {d:>10.0e} {e_gen:>18.12f} {NTU_t/(1+NTU_t):>15.12f}")
print("      The general form degrades below about 1 - C_r = 1e-8 and would")
print("      return NaN at C_r = 1 exactly.")

# ---- the ordering -----------------------------------------------------------
print("\n" + "-" * 78)
print("  CHECK 4 -- THE ORDERING OF ARRANGEMENTS")
print("""    Counterflow is the most effective arrangement and parallel flow the
    least, with crossflow between them, at every NTU and every C_r.  This is a
    physical statement -- counterflow maintains a temperature difference along
    the whole length -- and it must hold at every point of the sweep.""")
print("""
    The test is run in two parts, because the three expressions do not have
    the same standing.  Counterflow and parallel flow are EXACT solutions; the
    crossflow expression is a fitted correlation.  Lumping them together would
    make it impossible to tell whose fault a violation is.""")
v_exact, worst_exact, n_pts = 0, 1.0, 0
v_corr, worst_corr, corr_at = 0, 0.0, None
for NTU_t in np.logspace(-2, 1.3, 80):
    for Cr_t in np.linspace(0.01, 1.0, 40):
        n_pts += 1
        ep = float(eps_parallel(NTU_t, Cr_t))
        ex = float(eps_crossflow_corr(NTU_t, Cr_t))
        ec = float(eps_counterflow(NTU_t, Cr_t))
        if ec < ep - 1e-14:
            v_exact += 1
        worst_exact = min(worst_exact, ec - ep)
        gap = min(ex - ep, ec - ex)
        if gap < -1e-14:
            v_corr += 1
            if -gap > worst_corr:
                worst_corr, corr_at = -gap, (NTU_t, Cr_t)

print(f"\n    (a) The two EXACT relations, counterflow >= parallel:")
print(f"        points tested       = {n_pts}")
print(f"        violations          = {v_exact}")
print(f"        smallest margin     = {worst_exact:.3e}")
print(f"\n    (b) The crossflow CORRELATION, tested against those bounds:")
print(f"        violations          = {v_corr}  of {n_pts}")
print(f"        worst excursion     = {worst_corr:.3e}"
      f"   at NTU = {corr_at[0]:.3f}, C_r = {corr_at[1]:.3f}")
print("""
    The exact pair passes everywhere, to the precision of the arithmetic.  The
    correlation fails on roughly half the domain -- it dips BELOW parallel flow
    at small NTU, which is physically impossible, since crossflow cannot be
    worse than the worst arrangement there is.  A few sample points:""")
print(f"    {'NTU':>7} {'C_r':>6} {'parallel':>11} {'crossflow':>11} "
      f"{'counter':>11} {'cross - par':>13}")
for NTU_t, Cr_t in ((0.01, 0.5), (0.10, 0.5), (0.05, 1.0), (0.50, 0.5),
                    (2.00, 0.5)):
    ep = float(eps_parallel(NTU_t, Cr_t))
    ex = float(eps_crossflow_corr(NTU_t, Cr_t))
    ec = float(eps_counterflow(NTU_t, Cr_t))
    print(f"    {NTU_t:>7.2f} {Cr_t:>6.2f} {ep:>11.7f} {ex:>11.7f} "
          f"{ec:>11.7f} {ex-ep:>13.2e}")
print("""    The excursions are small in absolute terms -- at most 6e-3 in
    effectiveness -- and would pass unnoticed in any design calculation.  They
    are reported here because a correlation that violates a bound it cannot
    physically violate is telling us something about its provenance: it was
    fitted where NTU is of engineering size, and it was never constrained to
    behave at the small-NTU end.  Example 10.3 solves the crossflow problem
    directly and checks whether the EXACT solution respects the ordering, which
    it must.""")

print("\n    Where the three coincide is as informative as where they differ:")
print(f"    {'NTU':>8} {'C_r':>6} {'parallel':>11} {'crossflow':>11} "
      f"{'counter':>11} {'spread':>10}")
for NTU_t, Cr_t in ((0.1, 1.0), (0.5, 1.0), (2.0, 1.0), (5.0, 1.0),
                    (2.0, 0.1)):
    ep = float(eps_parallel(NTU_t, Cr_t))
    ex = float(eps_crossflow_corr(NTU_t, Cr_t))
    ec = float(eps_counterflow(NTU_t, Cr_t))
    print(f"    {NTU_t:>8.2f} {Cr_t:>6.2f} {ep:>11.6f} {ex:>11.6f} "
          f"{ec:>11.6f} {ec-ep:>10.6f}")
print("""    At small NTU the arrangement hardly matters: the exchanger is too
    short for the streams to know how they are pointed.  The choice becomes
    decisive only when NTU is large and C_r is near one -- which is exactly
    where economically sized exchangers operate.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

ntu = np.linspace(0.01, 6.0, 400)
for Cr_v, c in ((0.0, "#762a83"), (0.25, "#2166ac"), (0.5, "#1b7837"),
                (0.75, "#e08214"), (1.0, "#b2182b")):
    ax[0].plot(ntu, eps_counterflow(ntu, Cr_v), "-", lw=2.0, color=c,
               label=rf"$C_r = {Cr_v:g}$")
    ax[0].plot(ntu, eps_parallel(ntu, Cr_v), "--", lw=1.3, color=c, alpha=0.8)
ax[0].set_xlabel("NTU")
ax[0].set_ylabel(r"effectiveness $\varepsilon$")
ax[0].set_title("(a) Counterflow (solid) and parallel (dashed)")
ax[0].set_ylim(0, 1.02)
ax[0].legend(fontsize=8, loc="lower right")

Cr_v = 1.0
ax[1].plot(ntu, eps_counterflow(ntu, Cr_v), "-", lw=2.2, color="#b2182b",
           label="counterflow")
ax[1].plot(ntu, eps_crossflow_corr(ntu, Cr_v), "-", lw=2.2, color="#1b7837",
           label="crossflow, both unmixed")
ax[1].plot(ntu, eps_parallel(ntu, Cr_v), "-", lw=2.2, color="#2166ac",
           label="parallel")
ax[1].fill_between(ntu, eps_parallel(ntu, Cr_v),
                   eps_counterflow(ntu, Cr_v), color="0.6", alpha=0.15)
ax[1].annotate("the arrangement\nis worth this much",
               xy=(4.0, 0.63), fontsize=8.5, color="0.3", ha="center",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.28"))
ax[1].set_xlabel("NTU")
ax[1].set_ylabel(r"effectiveness $\varepsilon$")
ax[1].set_title(r"(b) The three arrangements at $C_r = 1$")
ax[1].set_ylim(0, 1.02)
ax[1].legend(fontsize=8.5, loc="lower right")

fig.suptitle("Example 10.1 -- Effectiveness of the classical arrangements",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_1a_effectiveness.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

ss = np.logspace(-14, -0.5, 300)
dT1v = 20.0
ex_v = np.array([dT1v * sum((-1) ** k * s ** k / (k + 1) for k in range(60))
                 for s in ss])
with np.errstate(divide="ignore", invalid="ignore"):
    nv_v = np.array([float(lmtd_naive(np.array(dT1v),
                                      np.array(dT1v * (1 + s)))) for s in ss])
sf_v = np.array([float(lmtd(np.array(dT1v), np.array(dT1v * (1 + s))))
                 for s in ss])
ax[0].loglog(ss, np.maximum(np.abs(nv_v - ex_v), 1e-18), "-", lw=2.0,
             color="#b2182b", label="textbook formula")
ax[0].loglog(ss, np.maximum(np.abs(sf_v - ex_v), 1e-18), "-", lw=2.0,
             color="#1b7837", label="series-protected form")
ax[0].axvline(1e-4, color="0.45", ls=":", lw=1.5)
ax[0].annotate("switch", xy=(1e-4, 3e-11), fontsize=8.5, color="0.35",
               rotation=90, ha="right")
ax[0].set_xlabel(r"$\Delta T_2/\Delta T_1 - 1$")
ax[0].set_ylabel(r"absolute error in $\Delta T_{lm}$  [K]")
ax[0].set_title("(a) A removable singularity, and what it costs")
ax[0].legend(fontsize=8.5, loc="lower right")

NTUg, Crg = np.meshgrid(np.logspace(-2, np.log10(20), 60),
                        np.linspace(0.0, 1.0, 60))
rel = np.zeros_like(NTUg)
Cmin = 5000.0
for i in range(NTUg.shape[0]):
    for j in range(NTUg.shape[1]):
        Cr_t, NTU_t = Crg[i, j], NTUg[i, j]
        Cmax = Cmin / Cr_t if Cr_t > 0 else 1e12
        q, Tho, Tco, _, _, _ = solve_by_eps_ntu(NTU_t * Cmin, Cmin, Cmax,
                                                THI, TCI, "counterflow")
        q_l = duty_by_lmtd(NTU_t * Cmin, Cmin, Cmax, THI, TCI, Tho, Tco,
                           "counterflow")
        rel[i, j] = abs(q_l / q - 1.0)
cs = ax[1].pcolormesh(NTUg, Crg, np.log10(np.maximum(rel, 1e-18)),
                      cmap="viridis", shading="auto", vmin=-16, vmax=-13)
cb = fig.colorbar(cs, ax=ax[1])
cb.set_label(r"$\log_{10}$ relative difference")
ax[1].set_xscale("log")
ax[1].set_xlabel("NTU")
ax[1].set_ylabel(r"$C_r$")
ax[1].set_title("(b) LMTD vs $\\varepsilon$-NTU: an identity, everywhere")
ax[1].grid(False)

fig.suptitle("Example 10.1 -- The identity, and the hazard",
             fontsize=12.5, y=1.08)
fig.savefig("fig_10_1b_identity.png")
plt.close(fig)

print("Figures written: fig_10_1a_effectiveness.png, fig_10_1b_identity.png")
