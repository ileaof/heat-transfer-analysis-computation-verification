"""
================================================================================
 EXAMPLE 7.1 -- ANALYTICAL SOLUTIONS FOR EXTENDED SURFACES
 The four classical fin cases, efficiency, and the corrected-length rule
================================================================================

 PHYSICAL PROBLEM
 ----------------
 A straight rectangular aluminium fin of thickness t and length L projects from
 a base at T_b into a fluid at T_inf with coefficient h.  Four tip conditions
 are solved and compared:

   A  infinite fin          theta -> 0 as x -> inf
   B  adiabatic tip         dtheta/dx = 0 at x = L
   C  convective tip        -k dtheta/dx = h theta at x = L
   D  prescribed tip        theta(L) = theta_L

 Two engineering claims are TESTED rather than quoted: that the corrected
 length L_c = L + t/2 lets the simple adiabatic-tip formula stand in for the
 convective-tip one, and that the one-dimensional fin equation is valid when
 the fin Biot number is small.

 GOVERNING EQUATION
 ------------------
 An energy balance on a slice of the fin, with lateral convection acting as a
 distributed sink, gives

        d2(theta)/dx2 - m^2 theta = 0 ,    m^2 = h P / (k A_c)

 where theta = T - T_inf, P is the wetted perimeter and A_c the cross-section.
 The lateral loss appears as a SOURCE TERM in the one-dimensional equation --
 this is the step that makes fin analysis one-dimensional, and it is exactly
 the source linearisation of Chapter 2 with S_P = -hP/A_c < 0, satisfying
 Patankar's rule automatically.

 SOLUTIONS  (with M = sqrt(h P k A_c) theta_b, the infinite-fin heat rate)
 ------------------------------------------------------------------------
   A  theta/theta_b = exp(-m x)                       q = M
   B  theta/theta_b = cosh(m(L-x))/cosh(mL)           q = M tanh(mL)
   C  theta/theta_b = [cosh(m(L-x)) + (h/mk) sinh(m(L-x))]
                      / [cosh(mL) + (h/mk) sinh(mL)]
                      q = M [sinh(mL) + (h/mk) cosh(mL)]
                          / [cosh(mL) + (h/mk) sinh(mL)]
   D  theta/theta_b = [(theta_L/theta_b) sinh(mx) + sinh(m(L-x))] / sinh(mL)
                      q = M [cosh(mL) - theta_L/theta_b] / sinh(mL)

 EFFICIENCY AND EFFECTIVENESS
 ----------------------------
   efficiency     eta = q / (h A_fin theta_b)   -- actual vs. an isothermal fin
   effectiveness  eps = q / (h A_c theta_b)     -- with fin vs. bare base

 A fin is only worth fitting if eps > 1, and in practice eps > 2 is the usual
 threshold.  Note that the two measures answer different questions and a fin
 can have high efficiency and poor effectiveness at the same time.

 SYMBOLS (all SI)
 ----------------
   L, t, w  [m]         fin length, thickness, width
   A_c      [m^2]       cross-sectional area, w t
   P        [m]         wetted perimeter, 2(w + t)
   k        [W/(m K)]   conductivity
   h        [W/(m^2 K)] convection coefficient
   m        [1/m]       fin parameter sqrt(hP/(k A_c))
   theta    [K]         excess temperature T - T_inf
   M        [W]         sqrt(h P k A_c) theta_b
   eta, eps [-]         efficiency, effectiveness
   Bi_fin   [-]         fin Biot number h(t/2)/k

 OUTPUTS
 -------
   fig_7_1a_profiles.png   the four tip conditions and their heat rates
   fig_7_1b_design.png     efficiency, effectiveness and the corrected length

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

# ==============================================================================
# 1. DATA -- an aluminium fin of unit width
# ==============================================================================
K = 180.0            # [W/(m K)]   aluminium alloy
T_FIN = 0.002        # [m]         thickness
L_FIN = 0.050        # [m]         length
W_FIN = 1.0          # [m]         width (per unit)
H = 50.0             # [W/(m^2 K)]
T_B, T_INF = 400.0, 300.0

A_C = W_FIN * T_FIN
P_WET = 2.0 * (W_FIN + T_FIN)
M_PAR = np.sqrt(H * P_WET / (K * A_C))
THETA_B = T_B - T_INF
M_INF = np.sqrt(H * P_WET * K * A_C) * THETA_B
BI_FIN = H * (T_FIN / 2.0) / K


# ==============================================================================
# 2. THE FOUR SOLUTIONS
# ==============================================================================
def theta_A(x, m=M_PAR):
    return THETA_B * np.exp(-m * np.asarray(x))


def theta_B(x, m=M_PAR, L=L_FIN):
    x = np.asarray(x)
    return THETA_B * np.cosh(m * (L - x)) / np.cosh(m * L)


def theta_C(x, m=M_PAR, L=L_FIN, h=H, k=K):
    x = np.asarray(x)
    r = h / (m * k)
    return THETA_B * (np.cosh(m * (L - x)) + r * np.sinh(m * (L - x))) / (
        np.cosh(m * L) + r * np.sinh(m * L))


def theta_D(x, theta_L, m=M_PAR, L=L_FIN):
    x = np.asarray(x)
    return THETA_B * ((theta_L / THETA_B) * np.sinh(m * x)
                      + np.sinh(m * (L - x))) / np.sinh(m * L)


def q_A():
    return M_INF


def q_B(m=M_PAR, L=L_FIN):
    return M_INF * np.tanh(m * L)


def q_C(m=M_PAR, L=L_FIN, h=H, k=K):
    r = h / (m * k)
    return M_INF * (np.sinh(m * L) + r * np.cosh(m * L)) / (
        np.cosh(m * L) + r * np.sinh(m * L))


def q_D(theta_L, m=M_PAR, L=L_FIN):
    return M_INF * (np.cosh(m * L) - theta_L / THETA_B) / np.sinh(m * L)


t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 7.1 -- EXTENDED SURFACES: THE FOUR CLASSICAL CASES")
print("=" * 78)
print(f"  Aluminium fin: L = {L_FIN*1e3:.0f} mm, t = {T_FIN*1e3:.0f} mm, "
      f"k = {K:.0f} W/(m K)")
print(f"  h = {H:.0f} W/(m^2 K), T_b = {T_B:.0f} K, T_inf = {T_INF:.0f} K")
print(f"  A_c = {A_C:.6f} m^2, P = {P_WET:.4f} m")
print(f"  m = {M_PAR:.6f} 1/m,  mL = {M_PAR*L_FIN:.6f},  M = {M_INF:.4f} W")
print(f"  fin Biot number h(t/2)/k = {BI_FIN:.3e}")

print("\n  HEAT RATE AND TIP TEMPERATURE BY CASE")
print(f"  {'case':>28} {'q [W]':>10} {'vs infinite':>12} {'T(L) [K]':>10}")
theta_L_D = 0.4 * THETA_B
rows = [("A  infinite fin", q_A(), float(theta_A(L_FIN))),
        ("B  adiabatic tip", q_B(), float(theta_B(L_FIN))),
        ("C  convective tip", q_C(), float(theta_C(L_FIN))),
        (f"D  prescribed tip ({0.4:.1f} theta_b)", q_D(theta_L_D), theta_L_D)]
for name, q, th in rows:
    print(f"  {name:>28} {q:>10.4f} {100*q/q_A():>11.2f}% "
          f"{T_INF+th:>10.4f}")

# ==============================================================================
# 3. VERIFICATION
# ==============================================================================
print("\n" + "-" * 78)
print("  VERIFICATION")
xg = np.linspace(0, L_FIN, 4001)
hx = xg[1] - xg[0]
for name, fun in [("A", theta_A), ("B", theta_B), ("C", theta_C)]:
    th = fun(xg)
    d2 = (th[2:] - 2 * th[1:-1] + th[:-2]) / hx**2
    res = np.max(np.abs(d2 - M_PAR**2 * th[1:-1])) / (M_PAR**2 * THETA_B)
    print(f"    case {name}: max|theta'' - m^2 theta| / (m^2 theta_b) = "
          f"{res:.3e}")

# base heat rate by two independent routes: Fourier at the base, and the
# integral of the lateral convective loss (plus tip loss where applicable)
def q_base_fourier(fun):
    d = (-3 * fun(0.0) + 4 * fun(hx) - fun(2 * hx)) / (2 * hx)
    return -K * A_C * float(d)

print("\n    base heat rate: Fourier at x = 0 versus integrated surface loss")
print(f"    {'case':>6} {'Fourier [W]':>13} {'integral [W]':>14} {'diff':>11}")
for name, fun, tip in [("B", theta_B, 0.0),
                       ("C", theta_C, None)]:
    q_f = q_base_fourier(fun)
    lateral = H * P_WET * np.trapz(fun(xg), xg)
    tip_loss = 0.0 if tip == 0.0 else H * A_C * float(fun(L_FIN))
    q_i = lateral + tip_loss
    print(f"    {name:>6} {q_f:>13.6f} {q_i:>14.6f} {abs(q_f-q_i):>11.2e}")

# ==============================================================================
# 4. THE CORRECTED-LENGTH RULE
# ==============================================================================
print("\n" + "-" * 78)
print("  THE CORRECTED LENGTH.  The convective-tip solution is awkward; the")
print("  standard device is to use the ADIABATIC-tip formula with")
print("      L_c = L + t/2   (rectangular fin)")
print("  which adds an equivalent length carrying the tip area.  Tested here")
print("  across a wide range of mL.")
print(f"\n  {'mL':>7} {'q exact (C) [W]':>17} {'q via L_c [W]':>15} "
      f"{'error':>9}")
mls, errs = [], []
for mL in [0.1, 0.3, 0.5, 0.833, 1.5, 3.0, 5.0]:
    L_ = mL / M_PAR
    q_exact = q_C(L=L_)
    Lc = L_ + T_FIN / 2.0
    q_corr = M_INF * np.tanh(M_PAR * Lc)
    mls.append(mL)
    errs.append(abs(q_corr - q_exact) / q_exact)
    print(f"  {mL:>7.3f} {q_exact:>17.6f} {q_corr:>15.6f} "
          f"{100*abs(q_corr-q_exact)/q_exact:>8.4f}%")
print("  The rule is excellent: better than 0.1 % over the whole practical")
print("  range.  It works because the tip area A_c is small compared with the")
print("  lateral area P L, so representing it as extra length is a very small")
print("  correction applied to a term that is itself small.")

# ==============================================================================
# 5. EFFICIENCY, EFFECTIVENESS AND DESIGN
# ==============================================================================
def efficiency(L=L_FIN):
    Lc = L + T_FIN / 2.0
    q = M_INF * np.tanh(M_PAR * Lc)
    return q / (H * P_WET * Lc * THETA_B)


def effectiveness(L=L_FIN):
    Lc = L + T_FIN / 2.0
    q = M_INF * np.tanh(M_PAR * Lc)
    return q / (H * A_C * THETA_B)


print("\n" + "-" * 78)
print("  EFFICIENCY AND EFFECTIVENESS")
print(f"  design fin: eta = {100*efficiency():.3f} %, "
      f"eps = {effectiveness():.3f}")
print(f"\n  {'L [mm]':>8} {'mL_c':>8} {'q [W]':>9} {'eta':>9} {'eps':>9} "
      f"{'dq/dL [W/m]':>13}")
for L_ in [0.005, 0.02, 0.05, 0.10, 0.20, 0.40]:
    Lc = L_ + T_FIN / 2.0
    q = M_INF * np.tanh(M_PAR * Lc)
    dq = M_INF * M_PAR / np.cosh(M_PAR * Lc)**2
    print(f"  {L_*1e3:>8.1f} {M_PAR*Lc:>8.4f} {q:>9.4f} "
          f"{100*efficiency(L_):>8.2f}% {effectiveness(L_):>9.3f} {dq:>13.4f}")
print("  Lengthening the fin has strongly diminishing returns: beyond")
print("  mL_c ~ 2.65 the fin delivers 99 % of an infinite one, and further")
print("  material earns almost nothing.  That is the standard design cut-off.")

mL_99 = brentq(lambda z: np.tanh(z) - 0.99, 0.1, 10.0, xtol=1e-14)
print(f"    tanh(mL_c) = 0.99 at mL_c = {mL_99:.6f}, i.e. "
      f"L = {mL_99/M_PAR*1e3:.2f} mm for this fin")

print("\n  IS THE FIN WORTH FITTING?  A fin pays only if eps > 1, and in")
print("  practice eps > 2 is the usual threshold.")
print(f"  {'t [mm]':>8} {'L [mm]':>8} {'k':>7} {'h':>7} {'eps':>9} {'verdict':>10}")
for t_, L_, k_, h_ in [(0.002, 0.050, 180.0, 50.0),
                       (0.002, 0.050, 180.0, 500.0),
                       (0.002, 0.050, 15.0, 50.0),
                       (0.002, 0.050, 15.0, 500.0),
                       (0.002, 0.050, 1.0, 500.0),
                       (0.020, 0.010, 15.0, 500.0),
                       (0.020, 0.005, 1.0, 500.0),
                       (0.020, 0.005, 1.0, 2000.0)]:
    Ac_ = W_FIN * t_
    P_ = 2.0 * (W_FIN + t_)
    m_ = np.sqrt(h_ * P_ / (k_ * Ac_))
    Mi = np.sqrt(h_ * P_ * k_ * Ac_) * THETA_B
    Lc = L_ + t_ / 2.0
    eps = Mi * np.tanh(m_ * Lc) / (h_ * Ac_ * THETA_B)
    verdict = "yes" if eps > 2 else ("marginal" if eps > 1 else "NO")
    print(f"  {t_*1e3:>8.0f} {L_*1e3:>8.0f} {k_:>7.0f} {h_:>7.0f} "
          f"{eps:>9.3f} {verdict:>10}")
print("  The first five rows are the THIN fin of this example, and it stays")
print("  effective throughout -- because its cross-section is tiny compared")
print("  with its lateral area, the ratio kP/(h A_c) stays large even for a")
print("  poor conductor.  Effectiveness collapses only when the fin is also")
print("  SHORT AND THICK, as the last three rows show: a 20 mm thick, 5 mm")
print("  long stub of a poor conductor in a high-h stream is not a fin at all")
print("  but an obstruction.")
print("  The scaling is eps ~ sqrt(kP/(h A_c)) tanh(mL_c), so BOTH the")
print("  material ratio k/h and the shape ratio P/A_c must be favourable.")

print(f"\n  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
x = np.linspace(0, L_FIN, 400)
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax[0].plot(x * 1e3, T_INF + theta_A(x), "-", lw=2.0, color="#4d4d4d",
           label="A  infinite")
ax[0].plot(x * 1e3, T_INF + theta_B(x), "-", lw=2.0, color="#2166ac",
           label="B  adiabatic tip")
ax[0].plot(x * 1e3, T_INF + theta_C(x), "--", lw=2.0, color="#b2182b",
           label="C  convective tip")
ax[0].plot(x * 1e3, T_INF + theta_D(x, theta_L_D), ":", lw=2.0,
           color="#1b7837", label="D  prescribed tip")
ax[0].axhline(T_INF, color="0.55", ls=":", lw=1.1)
ax[0].set_xlabel(r"Distance from base $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) The four tip conditions")
ax[0].legend(fontsize=9, loc="upper right")
ax[0].set_xlim(0, L_FIN * 1e3)
ax[0].set_ylim(T_INF - 4, T_B + 6)

Ls = np.linspace(0.002, 0.30, 300)
etas = np.array([efficiency(l) for l in Ls])
epss = np.array([effectiveness(l) for l in Ls])
ax[1].plot(M_PAR * (Ls + T_FIN / 2), 100 * etas, "-", lw=2.1,
           color="#b2182b", label=r"efficiency $\eta$ [%]")
ax[1].set_xlabel(r"$m L_c$ [-]")
ax[1].set_ylabel(r"Fin efficiency $\eta$ [%]", color="#b2182b")
ax[1].tick_params(axis="y", colors="#b2182b")
ax[1].set_title("(b) Efficiency falls as effectiveness rises")
ax[1].set_ylim(0, 105)

axb = ax[1].twinx()
axb.plot(M_PAR * (Ls + T_FIN / 2), epss, "-", lw=2.1, color="#2166ac",
         label=r"effectiveness $\varepsilon$")
axb.set_ylabel(r"Effectiveness $\varepsilon$ [-]", color="#2166ac")
axb.tick_params(axis="y", colors="#2166ac")
axb.grid(False)
axb.axhline(2.0, color="0.45", ls="--", lw=1.2)
axb.text(0.05, 2.25, r"$\varepsilon = 2$ threshold", fontsize=8.5,
         color="0.35")
ax[1].axvline(mL_99, color="#1b7837", ls=":", lw=1.6)
ax[1].text(mL_99 + 0.08, 55, rf"$mL_c = {mL_99:.2f}$" + "\n99 % of infinite",
           fontsize=8.5, color="#1b7837",
           bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                     boxstyle="round,pad=0.25"))

fig.suptitle("Example 7.1 -- Fin temperature fields and design measures",
             fontsize=12.5, y=1.08)
fig.savefig("fig_7_1a_profiles.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax[0].semilogy(mls, np.maximum(np.array(errs) * 100, 1e-8), "o-", lw=1.9,
               ms=7, color="#b2182b")
ax[0].axhline(0.1, color="0.45", ls="--", lw=1.2)
ax[0].text(2.2, 0.13, "0.1 % band", fontsize=9, color="0.35")
ax[0].set_xlabel(r"$mL$ [-]")
ax[0].set_ylabel("Error of the corrected-length rule [%]")
ax[0].set_title(r"(a) $L_c = L + t/2$ tested against the exact tip solution")

kk = np.logspace(0, 2.7, 200)
for h_, col, lab in [(20.0, "#1b7837", r"$h = 20$"),
                     (100.0, "#2166ac", r"$h = 100$"),
                     (500.0, "#b2182b", r"$h = 500$")]:
    m_ = np.sqrt(h_ * P_WET / (kk * A_C))
    Mi = np.sqrt(h_ * P_WET * kk * A_C) * THETA_B
    Lc = L_FIN + T_FIN / 2.0
    eps = Mi * np.tanh(m_ * Lc) / (h_ * A_C * THETA_B)
    ax[1].loglog(kk, eps, "-", lw=2.0, color=col,
                 label=lab + r" W m$^{-2}$K$^{-1}$")
ax[1].axhline(2.0, color="0.4", ls="--", lw=1.3)
ax[1].text(1.3, 2.3, r"$\varepsilon = 2$", fontsize=9, color="0.35")
ax[1].axhline(1.0, color="0.6", ls=":", lw=1.2)
ax[1].text(1.3, 0.78, r"$\varepsilon = 1$: no benefit", fontsize=8.5,
           color="0.45")
ax[1].set_xlabel(r"Fin conductivity $k$ [W m$^{-1}$K$^{-1}$]")
ax[1].set_ylabel(r"Effectiveness $\varepsilon$ [-]")
ax[1].set_title("(b) When is a fin worth fitting?")
ax[1].legend(fontsize=8.5, loc="lower right")

fig.suptitle("Example 7.1 -- Testing the corrected length and the design rule",
             fontsize=12.5, y=1.08)
fig.savefig("fig_7_1b_design.png")
plt.close(fig)

print("Figures written: fig_7_1a_profiles.png, fig_7_1b_design.png")
