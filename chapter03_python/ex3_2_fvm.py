"""
================================================================================
 EXAMPLE 3.2 -- FINITE VOLUME SOLUTION IN CYLINDRICAL COORDINATES
================================================================================

 OBJECTIVE
 ---------
 Discretise radial conduction by the control-volume method and compare TWO
 choices of face conductance:

   Scheme A (standard):  the face area uses the geometric face radius r_f, so
                         the conductance is  2 pi r_f k / (r_E - r_P)
   Scheme B (log-mean):  the conductance is the EXACT shell resistance between
                         the two nodes,  2 pi k / ln(r_E / r_P)

 Scheme A is the textbook default and is second-order accurate.  Scheme B is
 exact WHEN k IS CONSTANT, because the exact solution is then logarithmic in T
 and Scheme B's conductance is precisely the analytical shell resistance; it
 then reproduces the analytical solution to round-off on any mesh, however
 coarse.  That exactness is lost as soon as k depends on temperature, since the
 field that is logarithmic in r is then the Kirchhoff potential rather than the
 temperature.  Both behaviours are measured below.  The wider lesson is that in
 curvilinear geometry the natural discretisation is not the most accurate one,
 and that an exactness property must always be tested rather than assumed.

 GOVERNING EQUATION
 ------------------
        1  d  /          dT \
       --- -- | r k(T) * -- |  = 0 ,      r1 < r < r2
        r  dr \          dr /

 BOUNDARY CONDITIONS
 -------------------
   r = r1 :  T = T1
   r = r2 : -k dT/dr = h (T_2 - T_inf)

 DISCRETISATION
 --------------
 Integrating over a control volume between faces w and e (per unit length, and
 dividing out the 2 pi that multiplies every term):

        [ r k dT/dr ]_e - [ r k dT/dr ]_w = 0

 Note what has happened: the RADIUS APPEARS INSIDE the flux term.  A control
 volume in cylindrical geometry has unequal inner and outer areas, and this
 asymmetry is the entire difference from the plane wall of Chapter 2.  The
 control volume from r_w to r_e has volume 2 pi r_P dr per unit length with
 r_P the centroidal radius, and areas 2 pi r_w and 2 pi r_e.

 Writing the discrete equation as  a_P T_P = a_W T_W + a_E T_E + b, the two
 schemes differ only in how a_E (and a_W) is formed:

   Scheme A:  a_E = r_e k_e / (r_E - r_P)
   Scheme B:  a_E = k_e / ln(r_E / r_P)

 The two agree to O(dr^2) because  r_e / (r_E - r_P) -> 1 / ln(r_E/r_P)  as the
 mesh is refined; the log-mean radius r_lm = (r_E - r_P)/ln(r_E/r_P) is exactly
 the radius at which Scheme A becomes Scheme B.

 BOUNDARY IMPLEMENTATION
 -----------------------
 Half-control-volume treatment, as in Chapters 1 and 2, but with the
 appropriate radial conductance:

   inner Dirichlet:  a_bw = r1 k_1 / (r_P1 - r1)      [Scheme A]
                     a_bw = k_1 / ln(r_P1 / r1)       [Scheme B]
   outer convective: series of the half-shell and the film, the latter carrying
                     area r2:   U = 1 / ( 1/a_half + 1/(r2 h) )

 SYMBOLS -- see Example 3.1; additionally
   N       [-]    number of control volumes
   r_f     [m]    face radius
   r_lm    [m]    log-mean radius between two nodes
   a_W,a_E,a_P    discretisation coefficients [W/(m K)] per unit length /2pi

 OUTPUTS
 -------
   fig_3_2a_verification.png   profiles and residual histories
   fig_3_2b_convergence.png    grid convergence of both schemes

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
# 1. PROBLEM DATA (the Example 3.1 pipe)
# ==============================================================================
R1, R2 = 0.050, 0.100
K0 = 0.050
H = 10.0
T1 = 450.0
T_INF = 300.0
TREF = 300.0
BETA_A = 0.0            # Case A: constant conductivity
BETA_B = 1.5e-3         # Case B: k rises 0.15 % per kelvin


def k_of_T(T, beta):
    return K0 * (1.0 + beta * (T - TREF))


# ==============================================================================
# 2. EXACT SOLUTIONS
# ==============================================================================
def kirchhoff(T, beta):
    u = T - TREF
    return K0 * (u + 0.5 * beta * u * u)


def kirchhoff_inv(psi, beta):
    if beta == 0.0:
        return TREF + psi / K0
    return TREF + (-K0 + np.sqrt(K0 * K0 + 2.0 * K0 * beta * psi)) / (K0 * beta)


def exact_q(beta):
    """Heat rate per unit length divided by 2 pi, i.e. Q = q'/(2 pi) [W/(m)].

    In the Kirchhoff variable the equation is d/dr (r dpsi/dr) = 0, so
    psi(r) = psi(r1) - Q ln(r/r1).  The outer convective condition closes it.
    """
    psi1 = kirchhoff(T1, beta)

    def res(Q):
        T2 = kirchhoff_inv(psi1 - Q * np.log(R2 / R1), beta)
        return Q - R2 * H * (T2 - T_INF)

    return brentq(res, 0.0, psi1 / np.log(R2 / R1), xtol=1e-14)


def exact_T(r, beta):
    Q = exact_q(beta)
    return kirchhoff_inv(kirchhoff(T1, beta) - Q * np.log(np.asarray(r) / R1),
                         beta)


# ==============================================================================
# 3. MESH
# ==============================================================================
def make_mesh(N):
    """Uniform-in-r cell-centred mesh on [R1, R2].

    Cell centres are placed at the VOLUME centroid of each shell, which for a
    cylindrical annulus is not the arithmetic mid-radius.  Using the arithmetic
    mid-radius instead is a common and acceptable choice; the difference is
    O(dr^2) and is absorbed into the truncation error.  The arithmetic centre
    is used here so that the two schemes differ ONLY in the face conductance.
    """
    rf = np.linspace(R1, R2, N + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc, rf, (R2 - R1) / N


# ==============================================================================
# 4. ASSEMBLY -- the two schemes
# ==============================================================================
def conductance(r_lo, r_hi, k, scheme):
    """Conductance of the shell between two radii, divided by 2 pi.

    Scheme A uses the arithmetic face radius times a linear gradient;
    Scheme B uses the exact logarithmic shell resistance.
    """
    if scheme == "A":
        return 0.5 * (r_lo + r_hi) * k / (r_hi - r_lo)
    return k / np.log(r_hi / r_lo)


def assemble(T, rc, rf, N, beta, scheme):
    kc = k_of_T(T, beta)
    kf = np.zeros(N + 1)
    kf[1:N] = 2.0 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])
    kf[0], kf[N] = kc[0], kc[-1]

    aW = np.zeros(N)
    aE = np.zeros(N)
    Sp = np.zeros(N)
    b = np.zeros(N)

    # interior links: between adjacent NODE radii
    g = np.array([conductance(rc[i], rc[i + 1], kf[i + 1], scheme)
                  for i in range(N - 1)])
    aW[1:] = g
    aE[:-1] = g

    # inner Dirichlet: half shell from r1 to the first node
    a_bw = conductance(R1, rc[0], kf[0], scheme)
    b[0] += a_bw * T1
    Sp[0] -= a_bw

    # outer convective: half shell from last node to r2, in series with film
    a_be = conductance(rc[-1], R2, kf[N], scheme)
    U = 1.0 / (1.0 / a_be + 1.0 / (R2 * H))
    b[-1] += U * T_INF
    Sp[-1] -= U

    return aW, aE, aW + aE - Sp, b


def solve_fvm(N, beta, scheme, tol=1e-12, tol_T=1e-10, max_iter=200):
    rc, rf, dr = make_mesh(N)
    T = np.full(N, 0.5 * (T1 + T_INF))
    q_ref = (T1 - T_INF) / np.log(R2 / R1) * K0
    hist = []

    for it in range(1, max_iter + 1):
        aW, aE, aP, b = assemble(T, rc, rf, N, beta, scheme)
        R = np.zeros(N)
        R[1:-1] = (aW[1:-1] * T[:-2] + aE[1:-1] * T[2:]
                   - aP[1:-1] * T[1:-1] + b[1:-1])
        R[0] = aE[0] * T[1] - aP[0] * T[0] + b[0]
        R[-1] = aW[-1] * T[-2] - aP[-1] * T[-1] + b[-1]
        res = np.max(np.abs(R)) / q_ref

        ab = np.zeros((3, N))
        ab[0, 1:] = -aE[:-1]
        ab[1, :] = aP
        ab[2, :-1] = -aW[1:]
        T_new = solve_banded((1, 1), ab, b)
        dT = np.max(np.abs(T_new - T))
        T = T_new
        hist.append((it, res, dT))
        if res < tol and dT < tol_T:
            break

    # outer surface temperature from the exact half-shell balance
    a_be = conductance(rc[-1], R2, k_of_T(T[-1], beta), scheme)
    T2 = brentq(lambda Ts: a_be * (T[-1] - Ts) - R2 * H * (Ts - T_INF),
                T_INF - 10.0, T1 + 10.0, xtol=1e-13)
    Q = R2 * H * (T2 - T_INF)                 # q'/(2 pi)
    return rc, T, {"iterations": it, "history": np.array(hist), "dr": dr,
                   "T2": T2, "Q": Q, "T_bar": np.sum(T * rc) / np.sum(rc)}


# ==============================================================================
# 5. RUN
# ==============================================================================
print("=" * 78)
print("EXAMPLE 3.2 -- FVM IN CYLINDRICAL COORDINATES")
print("=" * 78)
Q_ex_A = exact_q(BETA_A)
print(f"  Exact (Case A): q' = {2*np.pi*Q_ex_A:.10f} W/m, "
      f"T2 = {exact_T(R2, BETA_A):.10f} K")

for beta, tag in [(BETA_A, "CASE A: k constant"),
                  (BETA_B, f"CASE B: k(T), beta = {BETA_B:.1e} 1/K")]:
    Q_ex = exact_q(beta)
    T2_ex = float(exact_T(np.array([R2]), beta)[0])
    print("\n" + "-" * 78)
    print(f"{tag}")
    print(f"  exact q' = {2*np.pi*Q_ex:.10f} W/m,  T2 = {T2_ex:.10f} K")
    print(f"\n  {'N':>5} | {'Scheme A (face radius)':>34} | "
          f"{'Scheme B (log-mean)':>34}")
    print(f"  {'':>5} | {'Linf [K]':>13} {'p':>7} {'q err [W/m]':>12} | "
          f"{'Linf [K]':>13} {'p':>7} {'q err [W/m]':>12}")
    print("  " + "-" * 92)
    prevA = prevB = None
    rowsA, rowsB = [], []
    for N in [10, 20, 40, 80, 160, 320]:
        out = {}
        for sch in ("A", "B"):
            rc, T, info = solve_fvm(N, beta, sch)
            e = np.max(np.abs(T - exact_T(rc, beta)))
            out[sch] = (e, abs(2 * np.pi * (info["Q"] - Q_ex)), info, rc, T)
        eA, qA = out["A"][0], out["A"][1]
        eB, qB = out["B"][0], out["B"][1]
        pA = np.log(prevA / eA) / np.log(2.0) if prevA else float("nan")
        pB = (np.log(prevB / eB) / np.log(2.0)
              if prevB and eB > 1e-13 else float("nan"))
        print(f"  {N:>5d} | {eA:>13.4e} {pA:>7.3f} {qA:>12.3e} | "
              f"{eB:>13.4e} {pB:>7.3f} {qB:>12.3e}")
        rowsA.append({"N": N, "dr": out["A"][2]["dr"], "e": eA, "q": qA,
                      "Tbar": out["A"][2]["T_bar"], "rc": out["A"][3],
                      "T": out["A"][4], "hist": out["A"][2]["history"]})
        rowsB.append({"N": N, "dr": out["B"][2]["dr"], "e": eB, "q": qB,
                      "rc": out["B"][3], "T": out["B"][4],
                      "hist": out["B"][2]["history"]})
        prevA, prevB = eA, eB
    if beta == BETA_A:
        rowsA_A, rowsB_A = rowsA, rowsB
    else:
        rowsA_B, rowsB_B = rowsA, rowsB

print("\n" + "-" * 78)
print("  READING THE TABLES.")
print("  Case A (k constant).  Scheme B is EXACT: its errors sit at 1e-13 to")
print("  1e-11 K on every mesh, which is round-off, and the apparent negative")
print("  'orders' in that column are meaningless noise about a zero error.")
print("  The reason is structural: when k is constant the exact solution is")
print("  logarithmic in T, and Scheme B's face conductance k/ln(r_E/r_P) IS")
print("  the analytical resistance of the shell between the two nodes, so the")
print("  exact solution satisfies the discrete equations identically.")
print("  Scheme A converges at the expected second order.")
print()
print("  Case B (k depends on T).  The exactness is LOST.  Both schemes are")
print("  second order and their Linf errors are nearly identical (4.78e-2 vs")
print("  4.94e-2 at N = 10).  The reason is that with k(T) the field that is")
print("  logarithmic in r is the Kirchhoff potential psi, not the temperature;")
print("  the log-mean conductance reproduces the exact resistance only when")
print("  k can be taken outside the integral.  Note also that Scheme B is")
print("  slightly WORSE than Scheme A in the heat rate (2.1e-2 vs 5.1e-3 W/m")
print("  at N = 10) even while being marginally better in Linf -- a reminder")
print("  that 'more accurate' is not a property of a scheme alone but of a")
print("  scheme together with the quantity being asked of it.")
print()
print("  The practical rule: use the log-mean conductance for constant-property")
print("  radial conduction, where it is free exactness; expect no special")
print("  benefit once the properties vary.")

# --- Richardson extrapolation on Scheme A -------------------------------------
Tbar_ex = (quad(lambda s: float(exact_T(np.array([s]), BETA_A)[0]) * s, R1, R2)[0]
           / quad(lambda s: s, R1, R2)[0])
f1, f2, f3 = rowsA_A[-1]["Tbar"], rowsA_A[-2]["Tbar"], rowsA_A[-3]["Tbar"]
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
GCI = 1.25 * abs((f1 - f2) / f1) / (2.0**p_obs - 1.0) * 100.0
print("\n  Richardson extrapolation, Scheme A, area-weighted mean temperature:")
print(f"    N={rowsA_A[-3]['N']}/{rowsA_A[-2]['N']}/{rowsA_A[-1]['N']} : "
      f"{f3:.10f} / {f2:.10f} / {f1:.10f} K")
print(f"    observed order p            = {p_obs:.4f}")
print(f"    extrapolated                = {f_rich:.10f} K")
print(f"    |extrapolated - finest|     = {abs(f_rich-f1):.3e} K")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
rp = np.linspace(R1, R2, 400)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)
ax[0].plot(rp * 1e3, exact_T(rp, BETA_A), "-", lw=2.0, color="#2166ac",
           label="Exact, Case A")
rA = [r for r in rowsA_A if r["N"] == 10][0]
rB = [r for r in rowsB_A if r["N"] == 10][0]
ax[0].plot(rA["rc"] * 1e3, rA["T"], "o", ms=6, mfc="none", mew=1.4,
           color="#b2182b", label="Scheme A, $N=10$")
ax[0].plot(rB["rc"] * 1e3, rB["T"], "s", ms=6, mfc="none", mew=1.4,
           color="#1b7837", label="Scheme B, $N=10$")
ax[0].plot(rp * 1e3, exact_T(rp, BETA_B), "--", lw=1.7, color="#762a83",
           label="Exact, Case B ($k(T)$)")
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Both schemes on a deliberately coarse mesh")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

hA = rowsA_A[1]["hist"]
hB = rowsB_B[1]["hist"]
ax[1].semilogy(hA[:, 0], np.maximum(hA[:, 1], 1e-18), "o-", lw=1.6, ms=5,
               color="#b2182b", label="Scheme A, Case A")
ax[1].semilogy(hB[:, 0], np.maximum(hB[:, 1], 1e-18), "s-", lw=1.6, ms=5,
               color="#1b7837", label="Scheme B, Case B ($k(T)$)")
ax[1].axhline(1e-12, color="0.4", ls="--", lw=1.2, label="tolerance")
ax[1].set_xlabel("Picard iteration")
ax[1].set_ylabel(r"$\|R\|_\infty / q_{ref}$ [-]")
ax[1].set_title("(b) Residual histories, $N = 20$")
ax[1].legend(fontsize=9)

fig.suptitle("Example 3.2 -- Finite volume verification in cylindrical geometry",
             fontsize=12.5, y=1.02)
fig.savefig("fig_3_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)
drs = np.array([r["dr"] for r in rowsA_A])
ax[0].loglog(drs * 1e3, [r["e"] for r in rowsA_A], "o-", lw=1.8, ms=6,
             color="#b2182b", label="Scheme A, Case A")
ax[0].loglog(drs * 1e3, [r["e"] for r in rowsA_B], "^--", lw=1.6, ms=6,
             color="#762a83", label="Scheme A, Case B")
ax[0].loglog(drs * 1e3, np.maximum([r["e"] for r in rowsB_A], 1e-16), "s-",
             lw=1.8, ms=6, color="#1b7837", label="Scheme B (both cases)")
ref = rowsA_A[0]["e"] * (drs / drs[0])**2
ax[0].loglog(drs * 1e3, ref, "k--", lw=1.3, label="slope 2")
ax[0].axhline(np.finfo(float).eps * T1, color="0.5", ls=":", lw=1.2,
              label="machine precision")
ax[0].set_xlabel(r"$\Delta r$ [mm]")
ax[0].set_ylabel(r"$\|e\|_\infty$ [K]")
ax[0].set_title("(a) Grid convergence of the two schemes")
ax[0].legend(fontsize=8, loc="center right")

ax[1].loglog(drs * 1e3, [r["q"] for r in rowsA_A], "o-", lw=1.8, ms=6,
             color="#b2182b", label="Scheme A")
ax[1].loglog(drs * 1e3, np.maximum([r["q"] for r in rowsB_A], 1e-16), "s-",
             lw=1.8, ms=6, color="#1b7837", label="Scheme B")
ax[1].loglog(drs * 1e3, [r["q"] for r in rowsA_A][0] * (drs / drs[0])**2,
             "k--", lw=1.3, label="slope 2")
ax[1].set_xlabel(r"$\Delta r$ [mm]")
ax[1].set_ylabel(r"$|q' - q'_{exact}|$ [W m$^{-1}$]")
ax[1].set_title("(b) Error in the heat rate")
ax[1].legend(fontsize=9, loc="center right")

fig.suptitle("Example 3.2 -- Grid convergence: face radius versus log-mean",
             fontsize=12.5, y=1.02)
fig.savefig("fig_3_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_3_2a_verification.png, fig_3_2b_convergence.png")
