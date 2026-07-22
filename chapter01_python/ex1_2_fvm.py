"""
================================================================================
 EXAMPLE 1.2 -- FINITE VOLUME SOLUTION
 The Example 1.1 wall solved by Patankar's control-volume method
================================================================================

 OBJECTIVE
 ---------
 Reproduce the Example 1.1 solution numerically, then extend it to a case the
 closed-form linear profile can no longer describe: TEMPERATURE-DEPENDENT
 conductivity.  Two configurations are solved with the SAME solver:

   Case A :  k = const                       -> exact solution is linear
   Case B :  k(T) = k0 [1 + beta (T - Tref)] -> exact solution is nonlinear,
                                                obtained by the Kirchhoff
                                                transformation (below)

 Case A is the algorithmic verification: a second-order control-volume scheme
 reproduces a linear field EXACTLY, so any error above round-off signals a bug
 in the coefficient assembly or the boundary treatment.  Case B has genuine
 truncation error and is used for the grid-convergence study.

 GOVERNING EQUATION
 ------------------
        d  /        dT \
       --- | k(T) * -- |  = 0 ,      0 < x < L
        dx \        dx /

 BOUNDARY CONDITIONS
 -------------------
   x = 0 :  T(0) = T0
   x = L : -k dT/dx|_L = h (T_L - T_inf) + eps*sigma*(T_L^4 - T_sur^4)

 INITIAL CONDITIONS
 ------------------
   None (steady).  An initial GUESS is required only to start the nonlinear
   iteration; the converged answer is independent of it (verified below).

 FINITE VOLUME DISCRETISATION  (Patankar, cell-centred, uniform mesh)
 --------------------------------------------------------------------
 The domain is divided into N control volumes of width dx = L/N.  Integrating
 the governing equation over control volume P between faces w and e:

        (k dT/dx)|_e - (k dT/dx)|_w = 0

 Using a linear profile between adjacent nodes for the face gradients:

        a_P T_P = a_W T_W + a_E T_E + b

        a_W = k_w / dx ,   a_E = k_e / dx ,   a_P = a_W + a_E - S_P

 Interface conductivities k_w, k_e use the HARMONIC mean, which is the choice
 that reproduces the exact series-resistance of two adjacent volumes and remains
 well behaved across large conductivity jumps (Patankar, Section 4.2-3):

        k_e = 2 k_P k_E / (k_P + k_E)

 BOUNDARY IMPLEMENTATION (half-control-volume, no ghost nodes)
 -------------------------------------------------------------
 West (Dirichlet).  The wall face lies dx/2 from node 1, so the link to the
 boundary has conductance 2 k_1 / dx.  It enters as a source pair:

        a_W = 0 ,  S_P = -2 k_1/dx ,  S_u = (2 k_1/dx) T0

 East (nonlinear Robin).  The radiative flux is folded into Newton form using
 the exact factorisation of the fourth-power difference,

        h_r = eps*sigma*(T_L + T_sur)(T_L^2 + T_sur^2)
        q_rad = h_r (T_L - T_sur)

 so the face sees a combined film coefficient h_s = h + h_r discharging to a
 flux-weighted effective sink temperature

        T_s = (h T_inf + h_r T_sur) / h_s

 The half-cell conduction resistance and the film resistance act in series:

        U = 1 / ( dx/(2 k_N) + 1/h_s )
        a_E = 0 ,  S_P = -U ,  S_u = U T_s

 Because h_r depends on the unknown T_L, and k depends on T in Case B, the
 system is nonlinear and is solved by Picard iteration with under-relaxation.

 LINEAR SOLVER
 -------------
 The matrix is tridiagonal, so the Thomas algorithm (TDMA) is used through
 scipy.linalg.solve_banded -- O(N) work and O(N) storage.

 RESIDUAL AND CONVERGENCE CRITERION
 ----------------------------------
 After each outer iteration the unnormalised residual of every control volume,

        R_P = a_W T_W + a_E T_E + b - a_P T_P     [W/m^2]

 is assembled and scaled by the through-wall flux to give a dimensionless
 measure.  Iteration stops when  max|R_P|/q_ref < tol  AND the temperature
 change between successive iterations falls below tol_T.

 SYMBOLS -- see Example 1.1; additionally
   N        [-]        number of control volumes
   dx       [m]        control-volume width
   a_W,a_E,a_P        [W/(m^2 K)]  discretisation coefficients
   S_P      [W/(m^2 K)] slope of the linearised source
   S_u      [W/m^2]     constant part of the linearised source
   U        [W/(m^2 K)] overall surface conductance (half-cell + film)

 OUTPUTS
 -------
   fig_1_2a_verification.png   profiles, residual history
   fig_1_2b_convergence.png    grid-convergence study for Case B

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.optimize import brentq

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. PROBLEM DATA (identical to Example 1.1)
# ==============================================================================
L = 0.05            # [m]
h = 25.0            # [W/(m^2 K)]
eps = 0.85          # [-]
SIGMA = 5.670374419e-8
T0 = 600.0          # [K]
T_inf = 300.0       # [K]
T_sur = 300.0       # [K]

K0 = 1.4            # [W/(m K)]   conductivity at the reference temperature
TREF = 300.0        # [K]         reference temperature for k(T)
BETA_A = 0.0        # [1/K]       Case A: constant conductivity
BETA_B = 1.2e-3     # [1/K]       Case B: +0.12 % per kelvin (typical ceramic)


def k_of_T(T, beta):
    """Thermal conductivity model k(T) = K0 [1 + beta (T - TREF)]  [W/(m K)]."""
    return K0 * (1.0 + beta * (T - TREF))


# ==============================================================================
# 2. EXACT SOLUTIONS
# ==============================================================================
def kirchhoff(T, beta):
    """Kirchhoff potential  psi(T) = int_TREF^T k(T') dT'   [W/m].

    Substituting psi removes the nonlinearity from the operator: the governing
    equation becomes d2(psi)/dx2 = 0, so psi is exactly LINEAR in x even when
    k depends on temperature.  This is the classical device of Carslaw & Jaeger.
    """
    u = T - TREF
    return K0 * (u + 0.5 * beta * u * u)


def kirchhoff_inverse(psi, beta):
    """Invert psi -> T.  For beta = 0 this is linear; otherwise take the
    physically admissible (positive-conductivity) root of the quadratic."""
    if beta == 0.0:
        return TREF + psi / K0
    disc = K0 * K0 + 2.0 * K0 * beta * psi
    return TREF + (-K0 + np.sqrt(disc)) / (K0 * beta)


def exact_flux(beta):
    """Solve the outer-face energy balance for the through-wall flux q'' [W/m^2].

    psi is linear:  psi(x) = psi(T0) - q'' x.  Hence T_L = psi^-1(psi(T0)-q''L),
    and q'' is the root of  q'' - h(T_L-T_inf) - eps*sigma*(T_L^4-T_sur^4) = 0.
    """
    psi0 = kirchhoff(T0, beta)

    def residual(q):
        T_L = kirchhoff_inverse(psi0 - q * L, beta)
        return q - h * (T_L - T_inf) - eps * SIGMA * (T_L**4 - T_sur**4)

    q_max = (psi0 - kirchhoff(T_inf, beta)) / L      # T_L = T_inf : upper bound
    return brentq(residual, 0.0, q_max, xtol=1e-13)


def exact_temperature(x, beta):
    """Exact field T(x) [K] for either case."""
    q = exact_flux(beta)
    return kirchhoff_inverse(kirchhoff(T0, beta) - q * np.asarray(x), beta)


# ==============================================================================
# 3. MESH GENERATION
# ==============================================================================
def make_mesh(N):
    """Uniform cell-centred mesh of N control volumes on [0, L].

    Returns
      xc : (N,)   cell-centre coordinates [m]
      xf : (N+1,) face coordinates        [m]
      dx : scalar control-volume width    [m]
    """
    xf = np.linspace(0.0, L, N + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    dx = L / N
    return xc, xf, dx


# ==============================================================================
# 4. COEFFICIENT ASSEMBLY
# ==============================================================================
def assemble(T, dx, N, beta):
    """Build the tridiagonal coefficients for the current temperature field.

    Returns aW, aE, aP, b  (each length N) in the form
        a_P T_P = a_W T_W + a_E T_E + b
    """
    k_cell = k_of_T(T, beta)                       # [W/(m K)] at each node

    # --- interface conductivities: harmonic mean between neighbouring cells ---
    k_face = np.zeros(N + 1)
    k_face[1:N] = 2.0 * k_cell[:-1] * k_cell[1:] / (k_cell[:-1] + k_cell[1:])
    k_face[0] = k_cell[0]                          # half-cell to west boundary
    k_face[N] = k_cell[-1]                         # half-cell to east boundary

    aW = np.zeros(N)
    aE = np.zeros(N)
    b = np.zeros(N)
    Sp = np.zeros(N)

    # --- interior links ------------------------------------------------------
    aW[1:] = k_face[1:N] / dx
    aE[:-1] = k_face[1:N] / dx

    # --- WEST boundary: Dirichlet T0 at the face, half-cell conductance -------
    a_bw = 2.0 * k_face[0] / dx                    # [W/(m^2 K)]
    b[0] += a_bw * T0
    Sp[0] -= a_bw

    # --- EAST boundary: convection + radiation, linearised in Newton form -----
    T_L = surface_temperature(T, dx, beta)         # current outer-face estimate
    h_r = eps * SIGMA * (T_L + T_sur) * (T_L**2 + T_sur**2)
    h_s = h + h_r                                  # combined film coefficient
    T_s = (h * T_inf + h_r * T_sur) / h_s          # effective sink temperature
    U = 1.0 / (dx / (2.0 * k_face[N]) + 1.0 / h_s)  # series: half-cell + film
    b[-1] += U * T_s
    Sp[-1] -= U

    aP = aW + aE - Sp
    return aW, aE, aP, b


def surface_temperature(T, dx, beta):
    """Extrapolate the outer-face temperature T_L from the last node.

    Balances half-cell conduction against the surface loss and solves the
    resulting nonlinear scalar equation exactly (no linearisation), so the
    reported T_L is consistent with the discrete solution to machine precision.
    """
    k_N = k_of_T(T[-1], beta)
    G = 2.0 * k_N / dx                             # half-cell conductance

    def res(Ts):
        return (G * (T[-1] - Ts)
                - h * (Ts - T_inf)
                - eps * SIGMA * (Ts**4 - T_sur**4))

    # Adaptive bracket.  res(Ts) is strictly decreasing: every term loses heat
    # faster as Ts rises.  Bracketing below every sink temperature guarantees
    # res > 0, and above every source temperature guarantees res < 0, so a sign
    # change always exists no matter how poor the current iterate T[-1] is.
    lo = min(T_inf, T_sur, T[-1]) - 1.0
    hi = max(T0, T_inf, T_sur, T[-1]) + 1.0
    return brentq(res, lo, hi, xtol=1e-13)


# ==============================================================================
# 5. LINEAR SOLVER (TDMA via banded storage)
# ==============================================================================
def solve_tridiagonal(aW, aE, aP, b):
    """Solve a_P T_P - a_W T_W - a_E T_E = b using the Thomas algorithm."""
    N = len(aP)
    ab = np.zeros((3, N))
    ab[0, 1:] = -aE[:-1]      # super-diagonal
    ab[1, :] = aP             # main diagonal
    ab[2, :-1] = -aW[1:]      # sub-diagonal
    return solve_banded((1, 1), ab, b)


# ==============================================================================
# 6. OUTER (NONLINEAR) ITERATION WITH RESIDUAL MONITORING
# ==============================================================================
def solve_fvm(N, beta, tol=1e-11, tol_T=1e-9, max_iter=200, relax=1.0,
              T_guess=None, verbose=False):
    """Full FVM solve.  Returns (xc, T, info-dict)."""
    xc, xf, dx = make_mesh(N)

    # Initial guess: linear ramp from T0 to T_inf (any guess converges)
    T = np.full(N, 0.5 * (T0 + T_inf)) if T_guess is None else T_guess.copy()

    q_ref = K0 * (T0 - T_inf) / L                  # flux scale for normalising
    history = []

    for it in range(1, max_iter + 1):
        aW, aE, aP, b = assemble(T, dx, N, beta)

        # ---- residual of the CURRENT field, before the new solve -------------
        R = np.zeros(N)
        R[1:-1] = (aW[1:-1] * T[:-2] + aE[1:-1] * T[2:] - aP[1:-1] * T[1:-1]
                   + b[1:-1])
        R[0] = aE[0] * T[1] - aP[0] * T[0] + b[0]
        R[-1] = aW[-1] * T[-2] - aP[-1] * T[-1] + b[-1]
        res_norm = np.max(np.abs(R)) / q_ref

        T_new = solve_tridiagonal(aW, aE, aP, b)
        dT = np.max(np.abs(T_new - T))
        T = T + relax * (T_new - T)
        history.append((it, res_norm, dT))

        if verbose:
            print(f"    it {it:3d}   |R|inf/q_ref = {res_norm:.3e}   "
                  f"max|dT| = {dT:.3e} K")
        if res_norm < tol and dT < tol_T:
            break

    T_L = surface_temperature(T, dx, beta)
    q_wall = 2.0 * k_of_T(T[0], beta) / dx * (T0 - T[0])   # flux at x = 0
    q_surf = h * (T_L - T_inf) + eps * SIGMA * (T_L**4 - T_sur**4)

    # Mean wall temperature: the midpoint-rule sum over the control volumes is
    # the natural second-order functional of a cell-centred FVM, and unlike the
    # wall flux it carries genuine discretisation error (see Section 9), which
    # makes it the right quantity on which to perform Richardson extrapolation.
    T_bar = dx * np.sum(T) / L

    info = {"iterations": it, "history": np.array(history), "dx": dx,
            "T_L": T_L, "q_wall": q_wall, "q_surf": q_surf, "T_bar": T_bar,
            "imbalance": abs(q_wall - q_surf), "xf": xf}
    return xc, T, info


# ==============================================================================
# 7. RUN CASE A -- CONSTANT CONDUCTIVITY (algorithmic verification)
# ==============================================================================
print("=" * 78)
print("EXAMPLE 1.2 -- FINITE VOLUME METHOD")
print("=" * 78)
print("\nCASE A: constant conductivity (exact solution is linear)")
print("-" * 78)
print("  Outer-iteration history, N = 20:")
xcA, TA, infoA = solve_fvm(20, BETA_A, verbose=True)
TA_exact = exact_temperature(xcA, BETA_A)
print("-" * 78)
print(f"  Converged in {infoA['iterations']} iterations")
print(f"  max|T_FVM - T_exact| = {np.max(np.abs(TA - TA_exact)):.3e} K   "
      "<-- round-off only")
print(f"  T_L (FVM)   = {infoA['T_L']:.10f} K")
print(f"  q'' at x=0  = {infoA['q_wall']:.10f} W/m^2")
print(f"  q'' at x=L  = {infoA['q_surf']:.10f} W/m^2")
print(f"  Global energy imbalance = {infoA['imbalance']:.3e} W/m^2")

print("\n  Grid independence for Case A (a linear field is captured exactly):")
print(f"  {'N':>6} {'Linf error [K]':>18} {'T_L [K]':>16}")
for N in [5, 10, 20, 40, 80, 160]:
    xc, T, info = solve_fvm(N, BETA_A)
    e = np.max(np.abs(T - exact_temperature(xc, BETA_A)))
    print(f"  {N:>6d} {e:>18.3e} {info['T_L']:>16.10f}")

# --- independence from the initial guess -------------------------------------
print("\n  Insensitivity to the initial guess (N = 20):")
for guess in [300.0, 450.0, 600.0, 1000.0]:
    _, T_g, info_g = solve_fvm(20, BETA_A, T_guess=np.full(20, guess))
    print(f"    T_init = {guess:7.1f} K -> T_L = {info_g['T_L']:.10f} K  "
          f"({info_g['iterations']} iters)")

# ==============================================================================
# 8. RUN CASE B -- TEMPERATURE-DEPENDENT CONDUCTIVITY (grid convergence)
# ==============================================================================
print("\n" + "=" * 78)
print("CASE B: k(T) = K0[1 + beta(T - TREF)],  beta = "
      f"{BETA_B:.1e} 1/K  (nonlinear exact solution)")
print("-" * 78)

q_exact_B = exact_flux(BETA_B)
TL_exact_B = kirchhoff_inverse(kirchhoff(T0, BETA_B) - q_exact_B * L, BETA_B)
print(f"  Exact flux        q''  = {q_exact_B:.10f} W/m^2")
print(f"  Exact face temp.  T_L  = {TL_exact_B:.10f} K")
print(f"  (Case A exact q''      = {exact_flux(BETA_A):.6f} W/m^2 -- the "
      f"{100*(q_exact_B/exact_flux(BETA_A)-1):+.2f} % shift is the k(T) effect)")

grids = [10, 20, 40, 80, 160, 320, 640]
rows = []
t_start = time.perf_counter()
for N in grids:
    t0 = time.perf_counter()
    xc, T, info = solve_fvm(N, BETA_B)
    cpu = time.perf_counter() - t0
    Te = exact_temperature(xc, BETA_B)
    err = T - Te
    l2 = np.sqrt(np.sum(err**2) / N)               # discrete L2 norm  [K]
    linf = np.max(np.abs(err))                     # Linf norm         [K]
    rows.append({"N": N, "dx": info["dx"], "L2": l2, "Linf": linf,
                 "TL": info["T_L"], "q": info["q_wall"], "cpu": cpu,
                 "Tbar": info["T_bar"],
                 "iters": info["iterations"], "imb": info["imbalance"],
                 "xc": xc, "T": T})

print(f"\n  {'N':>5} {'dx [mm]':>9} {'L2 err [K]':>13} {'p_L2':>7} "
      f"{'Linf err [K]':>14} {'p_Linf':>8} {'|q err| [W/m2]':>15} "
      f"{'its':>4} {'CPU [ms]':>9}")
print("  " + "-" * 96)
for i, r in enumerate(rows):
    if i == 0:
        p2 = pinf = float("nan")
    else:
        p2 = np.log(rows[i-1]["L2"] / r["L2"]) / np.log(2.0)
        pinf = np.log(rows[i-1]["Linf"] / r["Linf"]) / np.log(2.0)
    print(f"  {r['N']:>5d} {r['dx']*1e3:>9.4f} {r['L2']:>13.4e} "
          f"{p2:>7.3f} {r['Linf']:>14.4e} {pinf:>8.3f} "
          f"{abs(r['q']-q_exact_B):>15.4e} {r['iters']:>4d} "
          f"{r['cpu']*1e3:>9.2f}")

# --- The wall flux is exact for this conductivity law -------------------------
print("\n  NOTE on the flux column: for k LINEAR in T the computed wall flux is")
print("  exact to round-off on every grid, while the temperature field still")
print("  carries the usual second-order error.  This is a property of the")
print("  harmonic-mean interface conductivity for this particular k(T) family;")
print("  Example 1.3 shows that a quadratic k(T) restores the expected")
print("  second-order flux error.  Richardson extrapolation is therefore")
print("  applied below to the MEAN wall temperature, not to the flux.")

# --- Richardson extrapolation on the mean wall temperature -------------------
from scipy.integrate import quad
Tbar_exact = quad(lambda s: exact_temperature(s, BETA_B), 0.0, L)[0] / L

f1, f2, f3 = rows[-1]["Tbar"], rows[-2]["Tbar"], rows[-3]["Tbar"]  # fine->coarse
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
GCI = 1.25 * abs((f1 - f2) / f1) / (2.0**p_obs - 1.0) * 100.0
print("\n  Richardson extrapolation on the mean wall temperature T_bar:")
print(f"    coarse  N={rows[-3]['N']:<4d} T_bar = {f3:.10f} K")
print(f"    medium  N={rows[-2]['N']:<4d} T_bar = {f2:.10f} K")
print(f"    fine    N={rows[-1]['N']:<4d} T_bar = {f1:.10f} K")
print(f"    observed order of accuracy     p   = {p_obs:.4f}")
print(f"    extrapolated (h -> 0)      T_bar = {f_rich:.10f} K")
print(f"    analytical (quadrature)    T_bar = {Tbar_exact:.10f} K")
print(f"    |extrapolated - analytical|      = {abs(f_rich-Tbar_exact):.3e} K")
print(f"    |finest grid - analytical|       = {abs(f1-Tbar_exact):.3e} K")
print(f"    grid convergence index GCI_fine  = {GCI:.6f} %")
print(f"\n  Total CPU for the refinement sweep = "
      f"{time.perf_counter()-t_start:.3f} s")
print("=" * 78)

# ==============================================================================
# 9. FIGURES
# ==============================================================================
xfine = np.linspace(0.0, L, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

ax[0].plot(xfine * 1e3, exact_temperature(xfine, BETA_A), "-", lw=2.0,
           color="#2166ac", label="Exact, Case A ($k$ const)")
ax[0].plot(xcA * 1e3, TA, "o", ms=5, mfc="none", mew=1.3, color="#2166ac",
           label="FVM, Case A ($N=20$)")
ax[0].plot(xfine * 1e3, exact_temperature(xfine, BETA_B), "-", lw=2.0,
           color="#b2182b", label="Exact, Case B ($k(T)$)")
rB = [r for r in rows if r["N"] == 20][0]
ax[0].plot(rB["xc"] * 1e3, rB["T"], "s", ms=5, mfc="none", mew=1.3,
           color="#b2182b", label="FVM, Case B ($N=20$)")
ax[0].set_xlabel(r"Position $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) FVM versus exact solution")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].set_xlim(0, L * 1e3)

histA = infoA["history"]
_, _, infoB20 = solve_fvm(20, BETA_B)
histB = infoB20["history"]
ax[1].semilogy(histA[:, 0], np.maximum(histA[:, 1], 1e-18), "o-", lw=1.6, ms=5,
               color="#2166ac", label="Case A ($k$ const)")
ax[1].semilogy(histB[:, 0], np.maximum(histB[:, 1], 1e-18), "s-", lw=1.6, ms=5,
               color="#b2182b", label="Case B ($k(T)$)")
ax[1].axhline(1e-11, color="0.4", ls="--", lw=1.2, label="convergence tolerance")
ax[1].set_xlabel("Outer (Picard) iteration")
ax[1].set_ylabel(r"$\|R\|_\infty / q_{ref}$ [-]")
ax[1].set_title("(b) Residual history, $N = 20$")
ax[1].legend(fontsize=9)

fig.suptitle("Example 1.2 -- Finite volume verification", fontsize=12.5, y=1.02)
fig.savefig("fig_1_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

dxs = np.array([r["dx"] for r in rows])
l2s = np.array([r["L2"] for r in rows])
linfs = np.array([r["Linf"] for r in rows])
ax[0].loglog(dxs * 1e3, l2s, "o-", lw=1.8, ms=6, color="#2166ac",
             label=r"$\|e\|_2$")
ax[0].loglog(dxs * 1e3, linfs, "s-", lw=1.8, ms=6, color="#b2182b",
             label=r"$\|e\|_\infty$")
ref = l2s[0] * (dxs / dxs[0])**2
ax[0].loglog(dxs * 1e3, ref, "k--", lw=1.3, label=r"slope 2 (reference)")
ax[0].set_xlabel(r"Cell width $\Delta x$ [mm]")
ax[0].set_ylabel("Error norm [K]")
ax[0].set_title("(a) Grid convergence, Case B")
ax[0].legend(fontsize=9, loc="lower right")

Tbars = np.array([r["Tbar"] for r in rows])
ax[1].semilogx(dxs * 1e3, Tbars, "o-", lw=1.8, ms=6, color="#4d4d4d",
               label=r"FVM $\bar{T}(\Delta x)$")
ax[1].axhline(Tbar_exact, color="#b2182b", ls="-", lw=1.6,
              label="analytical")
ax[1].axhline(f_rich, color="#2166ac", ls="--", lw=1.4,
              label="Richardson extrapolated")
ax[1].set_xlabel(r"Cell width $\Delta x$ [mm]")
ax[1].set_ylabel(r"Mean wall temperature $\bar{T}$ [K]")
ax[1].set_title(f"(b) Asymptotic behaviour ($p_{{obs}} = {p_obs:.2f}$)")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 1.2 -- Grid convergence and Richardson extrapolation",
             fontsize=12.5, y=1.02)
fig.savefig("fig_1_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_1_2a_verification.png, fig_1_2b_convergence.png")
