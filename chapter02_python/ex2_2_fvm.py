"""
================================================================================
 EXAMPLE 2.2 -- FINITE VOLUME SOLUTION WITH A SOURCE TERM
 The Example 2.1 plate solved by Patankar's control-volume method
================================================================================

 OBJECTIVE
 ---------
 Introduce the treatment of a VOLUMETRIC SOURCE in the finite volume method,
 then verify it in two configurations:

   Case A :  k = const                       -> exact solution is parabolic
   Case B :  k(T) = k0 [1 + beta (T - Tref)] -> exact solution follows from the
                                                Kirchhoff transformation, which
                                                survives the presence of a
                                                uniform source

 GOVERNING EQUATION
 ------------------
        d  /        dT \
       --- | k(T) * -- |  + q''' = 0 ,      0 < x < L
        dx \        dx /

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :   dT/dx = 0                       (symmetry, adiabatic)
   x = L :  -k dT/dx|_L = h (T_L - T_inf)    (convective)

 THE SOURCE TERM IN THE FINITE VOLUME METHOD
 -------------------------------------------
 Integrating the governing equation over a control volume of width dx gives

        (k dT/dx)|_e - (k dT/dx)|_w + INTEGRAL_w^e q''' dx = 0

 The source enters as its INTEGRAL over the cell, not as a point value.  For a
 uniform source that integral is exactly q''' * dx, so no approximation is made
 here at all -- an advantage of the control-volume formulation over a finite
 difference scheme, which would sample q''' at the node.

 Patankar writes a general source in linearised form

        S = S_u + S_P T_P ,        with the requirement  S_P <= 0

 so that it contributes S_u*dx to b and -S_P*dx to a_P.  The sign requirement
 keeps a_P positive and hence preserves diagonal dominance; a positive S_P can
 destroy convergence, and Exercise 2.P2 constructs such a case.  Here the source
 is independent of temperature, so S_u = q''', S_P = 0.

 KIRCHHOFF TRANSFORMATION WITH A SOURCE
 --------------------------------------
 With psi = INTEGRAL k dT the operator linearises even though a source is
 present, because the source term does not involve k:

        d2(psi)/dx2 + q''' = 0

 so psi is exactly PARABOLIC in x.  Applying the adiabatic condition at x = 0,

        psi(x) = psi(0) - q''' x^2 / 2

 The surface condition remains nonlinear in T and fixes psi(0): the surface
 temperature T_L = psi^-1(psi(0) - q''' L^2/2) must satisfy
 q''' L = h (T_L - T_inf), a single scalar equation solved by Brent's method.

 SYMBOLS -- see Example 2.1; additionally
   N        [-]         number of control volumes
   dx       [m]         control-volume width
   S_u,S_P              linearised source intercept and slope
   psi      [W/m]       Kirchhoff potential

 OUTPUTS
 -------
   fig_2_2a_verification.png   profiles and residual histories
   fig_2_2b_convergence.png    grid convergence and Richardson extrapolation

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.integrate import quad

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. PROBLEM DATA (identical to Example 2.1)
# ==============================================================================
L = 0.02
K0 = 15.0
QGEN = 5.0e6
H = 500.0
T_INF = 350.0
TREF = 350.0
BETA_A = 0.0            # Case A: constant conductivity
BETA_B = 9.0e-4         # Case B: +0.09 % per kelvin (austenitic stainless)


def k_of_T(T, beta):
    """k(T) = K0 [1 + beta (T - TREF)]  [W/(m K)]."""
    return K0 * (1.0 + beta * (T - TREF))


# ==============================================================================
# 2. EXACT SOLUTIONS
# ==============================================================================
def kirchhoff(T, beta):
    """psi(T) = int_TREF^T k dT'  [W/m]."""
    u = T - TREF
    return K0 * (u + 0.5 * beta * u * u)


def kirchhoff_inverse(psi, beta):
    """Invert psi -> T, taking the physically admissible root."""
    if beta == 0.0:
        return TREF + psi / K0
    return TREF + (-K0 + np.sqrt(K0 * K0 + 2.0 * K0 * beta * psi)) / (K0 * beta)


def exact_surface_temperature(beta):
    """T_L is fixed by the surface balance alone and is INDEPENDENT of k.

    Whatever happens inside the plate, every watt generated must leave through
    the surface: q''' L = h (T_L - T_inf).  The conductivity model therefore
    cannot change T_L -- only the internal profile and hence T_max.
    """
    return T_INF + QGEN * L / H


def exact_temperature(x, beta):
    """Exact field T(x) [K] via the Kirchhoff potential."""
    T_L = exact_surface_temperature(beta)
    psi_L = kirchhoff(T_L, beta)
    # psi(x) = psi(0) - q''' x^2/2  and  psi(L) = psi_L
    psi_0 = psi_L + QGEN * L * L / 2.0
    psi = psi_0 - QGEN * np.asarray(x) ** 2 / 2.0
    return kirchhoff_inverse(psi, beta)


# ==============================================================================
# 3. MESH
# ==============================================================================
def make_mesh(N):
    """Uniform cell-centred mesh of N control volumes on [0, L]."""
    xf = np.linspace(0.0, L, N + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    return xc, xf, L / N


# ==============================================================================
# 4. ASSEMBLY
# ==============================================================================
def assemble(T, dx, N, beta):
    """Coefficients of  a_P T_P = a_W T_W + a_E T_E + b."""
    k_cell = k_of_T(T, beta)

    k_face = np.zeros(N + 1)
    k_face[1:N] = 2.0 * k_cell[:-1] * k_cell[1:] / (k_cell[:-1] + k_cell[1:])
    k_face[0] = k_cell[0]
    k_face[N] = k_cell[-1]

    aW = np.zeros(N)
    aE = np.zeros(N)
    Sp = np.zeros(N)
    b = np.zeros(N)

    aW[1:] = k_face[1:N] / dx
    aE[:-1] = k_face[1:N] / dx

    # SOURCE: integral over the cell is exactly q''' * dx (uniform source)
    b += QGEN * dx

    # WEST boundary x = 0: adiabatic.  Nothing to add -- a zero-flux face is
    # represented by simply omitting the link, which is why the finite volume
    # method handles Neumann conditions so cleanly.
    # (aW[0] is already zero.)

    # EAST boundary x = L: convection, half-cell resistance in series with film
    U = 1.0 / (dx / (2.0 * k_face[N]) + 1.0 / H)
    b[-1] += U * T_INF
    Sp[-1] -= U

    aP = aW + aE - Sp
    return aW, aE, aP, b


def solve_tridiagonal(aW, aE, aP, b):
    """Thomas algorithm via banded storage."""
    N = len(aP)
    ab = np.zeros((3, N))
    ab[0, 1:] = -aE[:-1]
    ab[1, :] = aP
    ab[2, :-1] = -aW[1:]
    return solve_banded((1, 1), ab, b)


def solve_fvm(N, beta, tol=1e-11, tol_T=1e-9, max_iter=200, T_guess=None,
              verbose=False):
    """Full solve with Picard iteration and residual monitoring."""
    xc, xf, dx = make_mesh(N)
    T = np.full(N, T_INF + 100.0) if T_guess is None else T_guess.copy()
    q_ref = QGEN * L
    history = []

    for it in range(1, max_iter + 1):
        aW, aE, aP, b = assemble(T, dx, N, beta)

        R = np.zeros(N)
        R[1:-1] = (aW[1:-1] * T[:-2] + aE[1:-1] * T[2:]
                   - aP[1:-1] * T[1:-1] + b[1:-1])
        R[0] = aE[0] * T[1] - aP[0] * T[0] + b[0]
        R[-1] = aW[-1] * T[-2] - aP[-1] * T[-1] + b[-1]
        res = np.max(np.abs(R)) / q_ref

        T_new = solve_tridiagonal(aW, aE, aP, b)
        dT = np.max(np.abs(T_new - T))
        T = T_new
        history.append((it, res, dT))
        if verbose:
            print(f"    it {it:3d}   |R|inf/q_ref = {res:.3e}   "
                  f"max|dT| = {dT:.3e} K")
        if res < tol and dT < tol_T:
            break

    # surface temperature from the exact half-cell balance
    k_N = k_of_T(T[-1], beta)
    T_L = brentq(lambda Ts: k_N / (0.5 * dx) * (T[-1] - Ts)
                 - H * (Ts - T_INF),
                 T_INF - 10.0, T[-1] + 10.0, xtol=1e-13)
    q_out = H * (T_L - T_INF)
    generated = QGEN * L

    info = {"iterations": it, "history": np.array(history), "dx": dx,
            "T_L": T_L, "T_max": T[0], "q_out": q_out,
            "imbalance": abs(generated - q_out),
            "T_bar": dx * np.sum(T) / L}
    return xc, T, info


# ==============================================================================
# 5. CASE A -- CONSTANT CONDUCTIVITY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 2.2 -- FINITE VOLUME METHOD WITH A VOLUMETRIC SOURCE")
print("=" * 78)
print("\nCASE A: constant conductivity (exact solution is parabolic)")
print("-" * 78)
print("  Outer-iteration history, N = 20 (linear problem: one solve suffices)")
xcA, TA, infoA = solve_fvm(20, BETA_A, verbose=True)
TA_ex = exact_temperature(xcA, BETA_A)
print("-" * 78)
print(f"  Converged in {infoA['iterations']} iterations")
print(f"  max|T_FVM - T_exact| = {np.max(np.abs(TA - TA_ex)):.6e} K")
print(f"  T_L  (FVM) = {infoA['T_L']:.10f} K   "
      f"(exact {exact_surface_temperature(BETA_A):.10f})")
print(f"  Global balance: generated {QGEN*L:.6f} vs removed "
      f"{infoA['q_out']:.6f} W/m^2, imbalance {infoA['imbalance']:.3e}")

print("\n  Grid convergence, Case A:")
print(f"  {'N':>5} {'dx [mm]':>9} {'L2 [K]':>13} {'p_L2':>7} "
      f"{'Linf [K]':>13} {'p_inf':>7} {'T_max err [K]':>14}")
gridsA, rowsA = [10, 20, 40, 80, 160, 320], []
for N in gridsA:
    xc, T, info = solve_fvm(N, BETA_A)
    Te = exact_temperature(xc, BETA_A)
    e = T - Te
    l2 = np.sqrt(np.sum(e**2) / N)
    linf = np.max(np.abs(e))
    if rowsA:
        p2 = np.log(rowsA[-1]["L2"] / l2) / np.log(2.0)
        pi_ = np.log(rowsA[-1]["Linf"] / linf) / np.log(2.0)
    else:
        p2 = pi_ = float("nan")
    rowsA.append({"N": N, "dx": info["dx"], "L2": l2, "Linf": linf,
                  "Tmax_err": abs(T[0] - exact_temperature(xc[0], BETA_A)),
                  "xc": xc, "T": T})
    print(f"  {N:>5d} {info['dx']*1e3:>9.4f} {l2:>13.4e} {p2:>7.3f} "
          f"{linf:>13.4e} {pi_:>7.3f} {rowsA[-1]['Tmax_err']:>14.4e}")

# ==============================================================================
# 6. CASE B -- TEMPERATURE-DEPENDENT CONDUCTIVITY
# ==============================================================================
print("\n" + "=" * 78)
print(f"CASE B: k(T) = K0[1 + beta(T - TREF)], beta = {BETA_B:.1e} 1/K")
print("-" * 78)
TL_B = exact_surface_temperature(BETA_B)
Tmax_B = float(exact_temperature(np.array([0.0]), BETA_B)[0])
Tmax_A = float(exact_temperature(np.array([0.0]), BETA_A)[0])
print(f"  Exact surface temperature T_L   = {TL_B:.10f} K")
print(f"    (identical to Case A: the surface balance does not involve k)")
print(f"  Exact mid-plane  Case A         = {Tmax_A:.6f} K")
print(f"  Exact mid-plane  Case B         = {Tmax_B:.6f} K  "
      f"({Tmax_B-Tmax_A:+.4f} K)")

print(f"\n  {'N':>5} {'dx [mm]':>9} {'L2 [K]':>13} {'p_L2':>7} "
      f"{'Linf [K]':>13} {'p_inf':>7} {'its':>4} {'CPU [ms]':>9}")
grids, rows = [10, 20, 40, 80, 160, 320, 640], []
t_start = time.perf_counter()
for N in grids:
    t0 = time.perf_counter()
    xc, T, info = solve_fvm(N, BETA_B)
    cpu = time.perf_counter() - t0
    Te = exact_temperature(xc, BETA_B)
    e = T - Te
    l2 = np.sqrt(np.sum(e**2) / N)
    linf = np.max(np.abs(e))
    if rows:
        p2 = np.log(rows[-1]["L2"] / l2) / np.log(2.0)
        pi_ = np.log(rows[-1]["Linf"] / linf) / np.log(2.0)
    else:
        p2 = pi_ = float("nan")
    rows.append({"N": N, "dx": info["dx"], "L2": l2, "Linf": linf,
                 "Tbar": info["T_bar"], "TL": info["T_L"],
                 "imb": info["imbalance"], "xc": xc, "T": T,
                 "iters": info["iterations"]})
    print(f"  {N:>5d} {info['dx']*1e3:>9.4f} {l2:>13.4e} {p2:>7.3f} "
          f"{linf:>13.4e} {pi_:>7.3f} {info['iterations']:>4d} "
          f"{cpu*1e3:>9.2f}")

# --- Richardson extrapolation on the mean temperature -------------------------
Tbar_exact = quad(lambda s: float(exact_temperature(np.array([s]), BETA_B)[0]),
                  0.0, L)[0] / L
f1, f2, f3 = rows[-1]["Tbar"], rows[-2]["Tbar"], rows[-3]["Tbar"]
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
GCI = 1.25 * abs((f1 - f2) / f1) / (2.0**p_obs - 1.0) * 100.0
print("\n  Richardson extrapolation on the mean plate temperature:")
print(f"    N={rows[-3]['N']:<4d} T_bar = {f3:.10f} K")
print(f"    N={rows[-2]['N']:<4d} T_bar = {f2:.10f} K")
print(f"    N={rows[-1]['N']:<4d} T_bar = {f1:.10f} K")
print(f"    observed order p             = {p_obs:.4f}")
print(f"    extrapolated                 = {f_rich:.10f} K")
print(f"    analytical (quadrature)      = {Tbar_exact:.10f} K")
print(f"    |extrapolated - analytical|  = {abs(f_rich-Tbar_exact):.3e} K")
print(f"    |finest grid - analytical|   = {abs(f1-Tbar_exact):.3e} K")
print(f"    GCI_fine                     = {GCI:.6f} %")

print(f"\n  Maximum global energy imbalance over all grids = "
      f"{max(r['imb'] for r in rows):.3e} W/m^2")
print("  The control-volume formulation conserves energy to round-off on")
print("  EVERY mesh, coarse or fine -- this is its defining property and is")
print("  independent of the discretisation error in the temperature field.")
print(f"\n  Total CPU for the sweep = {time.perf_counter()-t_start:.3f} s")
print("=" * 78)

# ==============================================================================
# 7. FIGURES
# ==============================================================================
xf_plot = np.linspace(0.0, L, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax[0].plot(xf_plot * 1e3, exact_temperature(xf_plot, BETA_A), "-", lw=2.0,
           color="#2166ac", label="Exact, Case A ($k$ const)")
rA = [r for r in rowsA if r["N"] == 20][0]
ax[0].plot(rA["xc"] * 1e3, rA["T"], "o", ms=5, mfc="none", mew=1.3,
           color="#2166ac", label="FVM, Case A ($N=20$)")
ax[0].plot(xf_plot * 1e3, exact_temperature(xf_plot, BETA_B), "-", lw=2.0,
           color="#b2182b", label="Exact, Case B ($k(T)$)")
rB = [r for r in rows if r["N"] == 20][0]
ax[0].plot(rB["xc"] * 1e3, rB["T"], "s", ms=5, mfc="none", mew=1.3,
           color="#b2182b", label="FVM, Case B ($N=20$)")
ax[0].set_xlabel(r"Position $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) FVM versus exact solution")
ax[0].legend(fontsize=8.5, loc="lower left")
ax[0].set_xlim(0, L * 1e3)

hA = infoA["history"]
_, _, infoB20 = solve_fvm(20, BETA_B)
hB = infoB20["history"]
ax[1].semilogy(hA[:, 0], np.maximum(hA[:, 1], 1e-18), "o-", lw=1.6, ms=5,
               color="#2166ac", label="Case A ($k$ const)")
ax[1].semilogy(hB[:, 0], np.maximum(hB[:, 1], 1e-18), "s-", lw=1.6, ms=5,
               color="#b2182b", label="Case B ($k(T)$)")
ax[1].axhline(1e-11, color="0.4", ls="--", lw=1.2, label="tolerance")
ax[1].set_xlabel("Picard iteration")
ax[1].set_ylabel(r"$\|R\|_\infty / q_{ref}$ [-]")
ax[1].set_title("(b) Residual history, $N = 20$")
ax[1].legend(fontsize=9)

fig.suptitle("Example 2.2 -- Finite volume verification with a source term",
             fontsize=12.5, y=1.02)
fig.savefig("fig_2_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

dxs = np.array([r["dx"] for r in rows])
ax[0].loglog(dxs * 1e3, [r["L2"] for r in rows], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$, Case B")
ax[0].loglog(dxs * 1e3, [r["Linf"] for r in rows], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$, Case B")
dxA = np.array([r["dx"] for r in rowsA])
ax[0].loglog(dxA * 1e3, [r["L2"] for r in rowsA], "^--", lw=1.5, ms=5,
             color="#1b7837", label=r"$\|e\|_2$, Case A")
ref = rows[0]["L2"] * (dxs / dxs[0])**2
ax[0].loglog(dxs * 1e3, ref, "k--", lw=1.3, label="slope 2")
ax[0].set_xlabel(r"Cell width $\Delta x$ [mm]")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title("(a) Grid convergence")
ax[0].legend(fontsize=8.5, loc="upper left")

ax[1].semilogx(dxs * 1e3, [r["Tbar"] for r in rows], "o-", lw=1.8, ms=6,
               color="#4d4d4d", label=r"FVM $\bar{T}(\Delta x)$")
ax[1].axhline(Tbar_exact, color="#b2182b", lw=1.6, label="analytical")
ax[1].axhline(f_rich, color="#2166ac", ls="--", lw=1.4,
              label="Richardson extrapolated")
ax[1].set_xlabel(r"Cell width $\Delta x$ [mm]")
ax[1].set_ylabel(r"Mean plate temperature $\bar{T}$ [K]")
ax[1].set_title(f"(b) Asymptotic behaviour ($p_{{obs}} = {p_obs:.2f}$)")
ax[1].legend(fontsize=9, loc="upper left")

fig.suptitle("Example 2.2 -- Grid convergence and Richardson extrapolation",
             fontsize=12.5, y=1.02)
fig.savefig("fig_2_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_2_2a_verification.png, fig_2_2b_convergence.png")
