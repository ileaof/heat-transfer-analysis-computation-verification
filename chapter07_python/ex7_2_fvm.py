"""
================================================================================
 EXAMPLE 7.2 -- FINITE VOLUME SOLUTION OF THE FIN EQUATION
 Lateral convection as a linearised sink, and a radiating fin
================================================================================

 OBJECTIVE
 ---------
 The fin equation is the first problem in this book whose source term depends
 on the SOLUTION.  Lateral convection removes h P (T - T_inf) per unit length,
 so in Patankar's linearised form S = S_u + S_P T,

        S_u = + h P T_inf / A_c ,      S_P = - h P / A_c  < 0

 The sink is automatically compliant with Patankar's rule S_P <= 0, which is
 why fin problems converge so readily.  This example verifies that treatment
 against all three tip conditions of Example 7.1, then makes the source
 genuinely NONLINEAR by adding radiation, where the rule must be enforced
 deliberately rather than inherited.

 GOVERNING EQUATION
 ------------------
        d/dx ( k A_c dT/dx ) - h P (T - T_inf) = 0

 with, for the radiating case, an extra sink

        - eps sigma P (T^4 - T_sur^4)

 DISCRETISATION
 --------------
 Integrating over a control volume of width dx:

        [k A_c dT/dx]_e - [k A_c dT/dx]_w + (S_u + S_P T_P) A_c dx = 0

 giving a_P T_P = a_W T_W + a_E T_E + b with

        a_W = a_E = k A_c / dx ,   a_P = a_W + a_E - S_P A_c dx ,
        b = S_u A_c dx

 BOUNDARY CONDITIONS
 -------------------
   x = 0 : Dirichlet at T_b, half-cell conductance 2 k A_c / dx
   x = L : adiabatic (case B)  -- omit the link
           convective (case C) -- U = 1/(dx/(2 k A_c) + 1/(h A_c)) on the
                                  TIP area A_c, not the lateral area

 SYMBOLS -- see Example 7.1; additionally
   N       [-]   number of control volumes
   S_u,S_P       linearised source intercept and slope
   eps     [-]   emissivity for the radiating case

 OUTPUTS
 -------
   fig_7_2a_verification.png   profiles and residual histories
   fig_7_2b_convergence.png    grid convergence for both tip conditions

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

K = 180.0
T_FIN, L_FIN, W_FIN = 0.002, 0.050, 1.0
H = 50.0
T_B, T_INF = 400.0, 300.0
A_C = W_FIN * T_FIN
P_WET = 2.0 * (W_FIN + T_FIN)
M_PAR = np.sqrt(H * P_WET / (K * A_C))
THETA_B = T_B - T_INF
M_INF = np.sqrt(H * P_WET * K * A_C) * THETA_B

EPS_RAD = 0.85
SIGMA = 5.670374419e-8
T_SUR = 300.0


# ==============================================================================
# 1. EXACT SOLUTIONS (Example 7.1)
# ==============================================================================
def exact_B(x):
    return T_INF + THETA_B * np.cosh(M_PAR * (L_FIN - np.asarray(x))) / np.cosh(
        M_PAR * L_FIN)


def exact_C(x):
    r = H / (M_PAR * K)
    x = np.asarray(x)
    return T_INF + THETA_B * (
        np.cosh(M_PAR * (L_FIN - x)) + r * np.sinh(M_PAR * (L_FIN - x))) / (
        np.cosh(M_PAR * L_FIN) + r * np.sinh(M_PAR * L_FIN))


def q_exact_B():
    return M_INF * np.tanh(M_PAR * L_FIN)


def q_exact_C():
    r = H / (M_PAR * K)
    return M_INF * (np.sinh(M_PAR * L_FIN) + r * np.cosh(M_PAR * L_FIN)) / (
        np.cosh(M_PAR * L_FIN) + r * np.sinh(M_PAR * L_FIN))


# ==============================================================================
# 2. FVM SOLVER
# ==============================================================================
def solve_fin(N, tip="C", radiation=False, tol=1e-11, tol_T=1e-10,
              max_iter=200):
    dx = L_FIN / N
    xc = (np.arange(N) + 0.5) * dx
    T = np.full(N, 0.5 * (T_B + T_INF))
    hist = []

    for it in range(1, max_iter + 1):
        aW = np.zeros(N)
        aE = np.zeros(N)
        Sp = np.zeros(N)
        Su = np.zeros(N)

        g = K * A_C / dx
        aW[1:] = g
        aE[:-1] = g

        # lateral convection: a linear sink, S_P < 0 automatically
        Sp -= H * P_WET * dx
        Su += H * P_WET * dx * T_INF

        if radiation:
            # Radiation is linearised about the current iterate in Newton form,
            # h_r = eps sigma (T + T_sur)(T^2 + T_sur^2), which keeps S_P < 0.
            # Writing the T^4 term directly with a positive slope would violate
            # Patankar's rule and can diverge; Exercise 7.C4 explores that.
            h_r = EPS_RAD * SIGMA * (T + T_SUR) * (T**2 + T_SUR**2)
            Sp -= h_r * P_WET * dx
            Su += h_r * P_WET * dx * T_SUR

        # base: Dirichlet through the half cell
        gb = 2.0 * K * A_C / dx
        Sp[0] -= gb
        Su[0] += gb * T_B

        # tip
        if tip == "C":
            U = 1.0 / (dx / (2.0 * K * A_C) + 1.0 / (H * A_C))
            Sp[-1] -= U
            Su[-1] += U * T_INF
        # tip == "B": adiabatic, nothing to add

        aP = aW + aE - Sp
        R = np.zeros(N)
        R[1:-1] = (aW[1:-1] * T[:-2] + aE[1:-1] * T[2:] - aP[1:-1] * T[1:-1]
                   + Su[1:-1])
        R[0] = aE[0] * T[1] - aP[0] * T[0] + Su[0]
        R[-1] = aW[-1] * T[-2] - aP[-1] * T[-1] + Su[-1]
        res = np.max(np.abs(R)) / (H * P_WET * L_FIN * THETA_B)
        hist.append((it, res))

        ab = np.zeros((3, N))
        ab[0, 1:] = -aE[:-1]
        ab[1, :] = aP
        ab[2, :-1] = -aW[1:]
        T_new = solve_banded((1, 1), ab, Su)
        dT = np.max(np.abs(T_new - T))
        T = T_new
        if res < tol and dT < tol_T:
            break

    q_base = 2.0 * K * A_C / dx * (T_B - T[0])
    return xc, T, {"it": it, "hist": np.array(hist), "dx": dx, "q": q_base}


# ==============================================================================
# 3. VERIFICATION
# ==============================================================================
print("=" * 78)
print("EXAMPLE 7.2 -- FINITE VOLUME SOLUTION OF THE FIN EQUATION")
print("=" * 78)
print(f"  mL = {M_PAR*L_FIN:.6f},  M = {M_INF:.4f} W")
print(f"  Source linearisation: S_P = -hP/A_c = "
      f"{-H*P_WET/A_C:.3f} W/(m^3 K) < 0, so Patankar's rule holds by")
print("  construction and no under-relaxation is needed for the linear case.")

print("\n  GRID CONVERGENCE")
print(f"  {'tip':>4} {'N':>5} {'Linf [K]':>13} {'p':>7} {'q [W]':>11} "
      f"{'q err [W]':>12} {'its':>4}")
store = {}
for tip, exact_fun, q_ex in [("B", exact_B, q_exact_B()),
                             ("C", exact_C, q_exact_C())]:
    rows = []
    for N in [10, 20, 40, 80, 160, 320]:
        xc, T, info = solve_fin(N, tip)
        e = np.max(np.abs(T - exact_fun(xc)))
        p = np.log(rows[-1][1] / e) / np.log(2.0) if rows else float("nan")
        rows.append((N, e, p, info["q"], info["dx"], xc, T, info["hist"]))
        print(f"  {tip if N == 10 else '':>4} {N:>5d} {e:>13.4e} {p:>7.3f} "
              f"{info['q']:>11.5f} {abs(info['q']-q_ex):>12.3e} "
              f"{info['it']:>4d}")
    store[tip] = rows

# energy balance on the finest mesh
xc, T, info = solve_fin(320, "C")
dx = info["dx"]
lateral = np.sum(H * P_WET * dx * (T - T_INF))
tip_loss = H * A_C * (T[-1] - T_INF)   # approximately; exact uses the face
print(f"\n  Energy balance, N = 320, convective tip:")
print(f"    q at the base           = {info['q']:.8f} W")
print(f"    lateral + tip loss      = {lateral + tip_loss:.8f} W")
print(f"    imbalance               = {abs(info['q']-lateral-tip_loss):.3e} W")
print("    (the small residue is the half-cell tip treatment, which is")
print("     second order and vanishes under refinement)")

# ==============================================================================
# 4. THE RADIATING FIN -- a genuinely nonlinear sink
# ==============================================================================
print("\n" + "-" * 78)
print("  A RADIATING FIN.  Adding radiation to the surroundings makes the")
print("  sink nonlinear.  Linearising it in Newton form keeps S_P < 0, so")
print("  Patankar's rule is preserved and the Picard iteration converges.")
print(f"  eps = {EPS_RAD}, T_sur = {T_SUR:.0f} K")
print(f"\n  {'N':>5} {'q [W]':>11} {'T_tip [K]':>11} {'its':>5} "
      f"{'change vs no rad.':>18}")
prev = None
rad_rows = []
for N in [20, 40, 80, 160, 320]:
    xc, T, info = solve_fin(N, "C", radiation=True)
    rad_rows.append((N, info["q"], T[-1], info["dx"], xc, T))
    print(f"  {N:>5d} {info['q']:>11.5f} {T[-1]:>11.5f} {info['it']:>5d} "
          f"{100*(info['q']/q_exact_C()-1):>17.2f}%")
print("  Radiation adds about 12 % to the heat rate here even though the fin")
print("  is only 100 K above ambient -- the same lesson as Chapter 1, that")
print("  radiation is not negligible merely because temperatures are modest.")
print("  Note the tip temperature falls by 3 K as well: the extra sink steepens")
print("  the profile, so the fin is also slightly less efficient.")

# Richardson on the radiating case, which has no closed form
f1, f2, f3 = rad_rows[-1][1], rad_rows[-2][1], rad_rows[-3][1]
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
print(f"\n  Richardson extrapolation on the radiating fin (no exact solution):")
print(f"    N = 80/160/320 : {f3:.8f} / {f2:.8f} / {f1:.8f} W")
print(f"    observed order p = {p_obs:.4f}")
print(f"    extrapolated q   = {f_rich:.8f} W")
print(f"    |finest - extrapolated| = {abs(f1-f_rich):.3e} W")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
xp = np.linspace(0, L_FIN, 400)
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax[0].plot(xp * 1e3, exact_B(xp), "-", lw=2.0, color="#2166ac",
           label="exact, adiabatic tip")
rB = [r for r in store["B"] if r[0] == 10][0]
ax[0].plot(rB[5] * 1e3, rB[6], "o", ms=6, mfc="none", mew=1.4,
           color="#2166ac", label="FVM, $N=10$")
ax[0].plot(xp * 1e3, exact_C(xp), "--", lw=2.0, color="#b2182b",
           label="exact, convective tip")
rC = [r for r in store["C"] if r[0] == 10][0]
ax[0].plot(rC[5] * 1e3, rC[6], "s", ms=6, mfc="none", mew=1.4,
           color="#b2182b", label="FVM, $N=10$")
rr = rad_rows[-1]
ax[0].plot(rr[4] * 1e3, rr[5], "-", lw=1.8, color="#1b7837",
           label="with radiation")
ax[0].set_xlabel(r"Distance from base $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) FVM against the exact solutions")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].set_xlim(0, L_FIN * 1e3)

hB = store["B"][1][7]
hC = store["C"][1][7]
_, _, infoR = solve_fin(20, "C", radiation=True)
ax[1].semilogy(hB[:, 0], np.maximum(hB[:, 1], 1e-18), "o-", lw=1.6, ms=5,
               color="#2166ac", label="linear, adiabatic tip")
ax[1].semilogy(hC[:, 0], np.maximum(hC[:, 1], 1e-18), "s-", lw=1.6, ms=5,
               color="#b2182b", label="linear, convective tip")
ax[1].semilogy(infoR["hist"][:, 0], np.maximum(infoR["hist"][:, 1], 1e-18),
               "^-", lw=1.6, ms=5, color="#1b7837", label="with radiation")
ax[1].axhline(1e-11, color="0.45", ls="--", lw=1.2, label="tolerance")
ax[1].set_xlabel("Picard iteration")
ax[1].set_ylabel(r"$\|R\|_\infty$ (normalised)")
ax[1].set_title("(b) Residual histories, $N = 20$")
ax[1].legend(fontsize=8.5, loc="upper right")

fig.suptitle("Example 7.2 -- Finite volume verification of the fin equation",
             fontsize=12.5, y=1.08)
fig.savefig("fig_7_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
dxs = np.array([r[4] for r in store["B"]])
ax[0].loglog(dxs * 1e3, [r[1] for r in store["B"]], "o-", lw=1.8, ms=6,
             color="#2166ac", label="adiabatic tip")
ax[0].loglog(dxs * 1e3, [r[1] for r in store["C"]], "s-", lw=1.8, ms=6,
             color="#b2182b", label="convective tip")
ax[0].loglog(dxs * 1e3, store["B"][0][1] * (dxs / dxs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [mm]")
ax[0].set_ylabel(r"$\|e\|_\infty$ [K]")
ax[0].set_title("(a) Grid convergence, both tip conditions")
ax[0].legend(fontsize=9, loc="lower right")

qs = np.array([r[1] for r in rad_rows])
Ns = np.array([r[0] for r in rad_rows])
ax[1].semilogx(Ns, qs, "o-", lw=1.9, ms=7, color="#1b7837",
               label="radiating fin, FVM")
ax[1].axhline(f_rich, color="#b2182b", ls="--", lw=1.5,
              label="Richardson extrapolated")
ax[1].set_xlabel(r"Number of control volumes $N$")
ax[1].set_ylabel(r"Base heat rate $q$ [W]")
ax[1].set_title(rf"(b) Radiating fin, $p_{{obs}} = {p_obs:.2f}$")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 7.2 -- Convergence with linear and nonlinear sinks",
             fontsize=12.5, y=1.08)
fig.savefig("fig_7_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_7_2a_verification.png, fig_7_2b_convergence.png")
