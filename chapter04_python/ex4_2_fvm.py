"""
================================================================================
 EXAMPLE 4.2 -- FINITE VOLUME SOLUTION IN SPHERICAL COORDINATES
 and the unified geometry family  A(r) ~ r^m
================================================================================

 OBJECTIVE
 ---------
 Chapter 3 found that in cylindrical geometry the LOG-MEAN radius makes the
 face conductance exact.  The same question is asked here for the sphere, and
 the answer completes a pattern:

     geometry     m     area A(r)     exact face radius
     ---------------------------------------------------------
     plane        0     const         arithmetic mean   (r_P + r_E)/2
     cylinder     1     2 pi r        log-mean          (r_E-r_P)/ln(r_E/r_P)
     sphere       2     4 pi r^2      GEOMETRIC mean    sqrt(r_P r_E)

 The spherical result follows from the exact shell resistance.  Between two
 radii the resistance is (1/r_P - 1/r_E)/(4 pi k), so the exact conductance is

        4 pi k / (1/r_P - 1/r_E) = 4 pi k r_P r_E / (r_E - r_P)

 while the standard discretisation uses the face area 4 pi r_f^2 with a linear
 gradient, giving 4 pi k r_f^2/(r_E - r_P).  The two agree when

        r_f^2 = r_P r_E    i.e.    r_f = sqrt(r_P r_E)

 the GEOMETRIC mean.  As in Chapter 3, the correction costs nothing and the
 exactness holds only while k is constant.

 GOVERNING EQUATION
 ------------------
        1   d  /  2         dT \
       --- -- | r  k(T)  *  -- |  = 0 ,      r1 < r < r2
        r^2 dr \            dr /

 BOUNDARY CONDITIONS
 -------------------
   r = r1 :  T = T1
   r = r2 : -k dT/dr = h (T_2 - T_inf)

 DISCRETISATION
 --------------
 Integrating over a shell between faces w and e and dividing out 4 pi:

        [ r^2 k dT/dr ]_e - [ r^2 k dT/dr ]_w = 0

 giving  a_P T_P = a_W T_W + a_E T_E + b  with

   Scheme A:  a_E = r_e^2 k_e / (r_E - r_P)          face radius squared
   Scheme B:  a_E = k_e r_P r_E / (r_E - r_P)        exact shell resistance

 The control-volume "volume", divided by 4 pi, is (r_e^3 - r_w^3)/3.

 SYMBOLS -- see Example 4.1; additionally
   N       [-]   number of control volumes
   r_gm    [m]   geometric-mean radius sqrt(r_P r_E)
   m       [-]   geometry index: 0 plane, 1 cylinder, 2 sphere

 OUTPUTS
 -------
   fig_4_2a_verification.png   profiles and residual histories
   fig_4_2b_convergence.png    grid convergence of both schemes

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
# 1. DATA (the Example 4.1 vessel)
# ==============================================================================
R1, R2 = 0.050, 0.100
K0 = 0.050
H = 10.0
T1 = 450.0
T_INF = 300.0
TREF = 300.0
BETA_A = 0.0
BETA_B = 1.5e-3


def k_of_T(T, beta):
    return K0 * (1.0 + beta * (T - TREF))


# ==============================================================================
# 2. EXACT SOLUTIONS (Kirchhoff handles k(T))
# ==============================================================================
def kirchhoff(T, beta):
    u = T - TREF
    return K0 * (u + 0.5 * beta * u * u)


def kirchhoff_inv(psi, beta):
    if beta == 0.0:
        return TREF + psi / K0
    return TREF + (-K0 + np.sqrt(K0 * K0 + 2.0 * K0 * beta * psi)) / (K0 * beta)


def exact_Q(beta):
    """q/(4 pi) [W], from psi(r) = psi(r1) - Q (1/r1 - 1/r)."""
    psi1 = kirchhoff(T1, beta)

    def res(Q):
        T2 = kirchhoff_inv(psi1 - Q * (1.0 / R1 - 1.0 / R2), beta)
        return Q - R2 * R2 * H * (T2 - T_INF)

    return brentq(res, 0.0, psi1 / (1.0 / R1 - 1.0 / R2), xtol=1e-14)


def exact_T(r, beta):
    Q = exact_Q(beta)
    r = np.asarray(r, dtype=float)
    return kirchhoff_inv(kirchhoff(T1, beta) - Q * (1.0 / R1 - 1.0 / r), beta)


# ==============================================================================
# 3. MESH AND ASSEMBLY
# ==============================================================================
def make_mesh(N):
    rf = np.linspace(R1, R2, N + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc, rf, (R2 - R1) / N


def conductance(ra, rb, k, scheme):
    """Shell conductance between radii ra < rb, divided by 4 pi."""
    if scheme == "A":
        return (0.5 * (ra + rb)) ** 2 * k / (rb - ra)      # face radius squared
    return k * ra * rb / (rb - ra)                          # exact (geometric)


def assemble(T, rc, N, beta, scheme):
    kc = k_of_T(T, beta)
    kf = np.zeros(N + 1)
    kf[1:N] = 2.0 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])
    kf[0], kf[N] = kc[0], kc[-1]

    aW = np.zeros(N)
    aE = np.zeros(N)
    Sp = np.zeros(N)
    b = np.zeros(N)

    g = np.array([conductance(rc[i], rc[i + 1], kf[i + 1], scheme)
                  for i in range(N - 1)])
    aW[1:] = g
    aE[:-1] = g

    a_bw = conductance(R1, rc[0], kf[0], scheme)
    b[0] += a_bw * T1
    Sp[0] -= a_bw

    a_be = conductance(rc[-1], R2, kf[N], scheme)
    U = 1.0 / (1.0 / a_be + 1.0 / (R2 * R2 * H))
    b[-1] += U * T_INF
    Sp[-1] -= U

    return aW, aE, aW + aE - Sp, b


def solve_fvm(N, beta, scheme, tol=1e-12, tol_T=1e-10, max_iter=200):
    rc, rf, dr = make_mesh(N)
    T = np.full(N, 0.5 * (T1 + T_INF))
    q_ref = K0 * (T1 - T_INF) / (1.0 / R1 - 1.0 / R2)
    hist = []

    for it in range(1, max_iter + 1):
        aW, aE, aP, b = assemble(T, rc, N, beta, scheme)
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

    a_be = conductance(rc[-1], R2, k_of_T(T[-1], beta), scheme)
    T2 = brentq(lambda Ts: a_be * (T[-1] - Ts) - R2 * R2 * H * (Ts - T_INF),
                T_INF - 10.0, T1 + 10.0, xtol=1e-13)
    Q = R2 * R2 * H * (T2 - T_INF)
    vol = (rf[1:] ** 3 - rf[:-1] ** 3) / 3.0
    return rc, T, {"it": it, "hist": np.array(hist), "dr": dr, "T2": T2,
                   "Q": Q, "T_bar": np.sum(T * vol) / np.sum(vol)}


# ==============================================================================
# 4. RUN
# ==============================================================================
print("=" * 78)
print("EXAMPLE 4.2 -- FVM IN SPHERICAL COORDINATES")
print("=" * 78)
print("  Exact face radius for the sphere is the GEOMETRIC mean sqrt(r_P r_E).")
print("  Demonstration on the first interior pair of a 10-cell mesh:")
rc0, rf0, _ = make_mesh(10)
ra, rb = rc0[0], rc0[1]
print(f"    r_P = {ra:.8f} m,  r_E = {rb:.8f} m")
print(f"    arithmetic mean = {(ra+rb)/2:.10f} m")
print(f"    geometric  mean = {np.sqrt(ra*rb):.10f} m")
print(f"    exact conductance      = {K0*ra*rb/(rb-ra):.10f}")
print(f"    Scheme A conductance   = {((ra+rb)/2)**2*K0/(rb-ra):.10f}")
print(f"    relative difference    = "
      f"{abs(((ra+rb)/2)**2/(ra*rb)-1):.3e}")

for beta, tag in [(BETA_A, "CASE A: k constant"),
                  (BETA_B, f"CASE B: k(T), beta = {BETA_B:.1e} 1/K")]:
    Q_ex = exact_Q(beta)
    T2_ex = float(exact_T(np.array([R2]), beta)[0])
    print("\n" + "-" * 78)
    print(tag)
    print(f"  exact q = {4*np.pi*Q_ex:.10f} W,  T2 = {T2_ex:.10f} K")
    print(f"\n  {'N':>5} | {'Scheme A (face radius^2)':>32} | "
          f"{'Scheme B (geometric mean)':>32}")
    print(f"  {'':>5} | {'Linf [K]':>13} {'p':>7} {'q err [W]':>10} | "
          f"{'Linf [K]':>13} {'p':>7} {'q err [W]':>10}")
    print("  " + "-" * 88)
    prevA = prevB = None
    rowsA, rowsB = [], []
    for N in [10, 20, 40, 80, 160, 320]:
        out = {}
        for sch in ("A", "B"):
            rc, T, info = solve_fvm(N, beta, sch)
            e = np.max(np.abs(T - exact_T(rc, beta)))
            out[sch] = (e, abs(4 * np.pi * (info["Q"] - Q_ex)), info, rc, T)
        eA, qA = out["A"][0], out["A"][1]
        eB, qB = out["B"][0], out["B"][1]
        pA = np.log(prevA / eA) / np.log(2.0) if prevA else float("nan")
        pB = (np.log(prevB / eB) / np.log(2.0)
              if prevB and eB > 1e-13 else float("nan"))
        print(f"  {N:>5d} | {eA:>13.4e} {pA:>7.3f} {qA:>10.2e} | "
              f"{eB:>13.4e} {pB:>7.3f} {qB:>10.2e}")
        rowsA.append({"N": N, "dr": out["A"][2]["dr"], "e": eA, "q": qA,
                      "Tbar": out["A"][2]["T_bar"], "rc": out["A"][3],
                      "T": out["A"][4], "hist": out["A"][2]["hist"]})
        rowsB.append({"N": N, "dr": out["B"][2]["dr"], "e": eB, "q": qB,
                      "rc": out["B"][3], "T": out["B"][4],
                      "hist": out["B"][2]["hist"]})
        prevA, prevB = eA, eB
    if beta == BETA_A:
        rA_A, rB_A = rowsA, rowsB
    else:
        rA_B, rB_B = rowsA, rowsB

print("\n" + "-" * 78)
print("  Case A: Scheme B is EXACT to round-off on every mesh, exactly as the")
print("  log-mean scheme was for the cylinder.  The 'orders' printed in that")
print("  column are noise about a zero error and carry no meaning.")
print("  Case B: the exactness is lost once k depends on temperature, for the")
print("  same reason as in Chapter 3 -- with k(T) it is the Kirchhoff")
print("  potential, not the temperature, that follows the 1/r law.")

# --- Richardson on Scheme A ---------------------------------------------------
f1, f2, f3 = rA_A[-1]["Tbar"], rA_A[-2]["Tbar"], rA_A[-3]["Tbar"]
p_obs = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
f_rich = f1 + (f1 - f2) / (2.0**p_obs - 1.0)
GCI = 1.25 * abs((f1 - f2) / f1) / (2.0**p_obs - 1.0) * 100.0
print("\n  Richardson extrapolation, Scheme A, volume-weighted mean:")
print(f"    N={rA_A[-3]['N']}/{rA_A[-2]['N']}/{rA_A[-1]['N']} : "
      f"{f3:.10f} / {f2:.10f} / {f1:.10f} K")
print(f"    observed order p        = {p_obs:.4f}")
print(f"    extrapolated            = {f_rich:.10f} K")
print(f"    GCI_fine                = {GCI:.6f} %")

# --- the unified geometry family ---------------------------------------------
print("\n" + "-" * 78)
print("  THE UNIFIED PATTERN.  For area A(r) ~ r^m the exact face radius is")
print("  whatever makes r_f^m equal the harmonic-type mean of the shell:")
print(f"  {'geometry':>10} {'m':>3} {'exact face radius':>26} "
      f"{'value here [m]':>16}")
print(f"  {'plane':>10} {0:>3} {'arithmetic (r_P+r_E)/2':>26} "
      f"{(ra+rb)/2:>16.10f}")
print(f"  {'cylinder':>10} {1:>3} {'log-mean':>26} "
      f"{(rb-ra)/np.log(rb/ra):>16.10f}")
print(f"  {'sphere':>10} {2:>3} {'geometric sqrt(r_P r_E)':>26} "
      f"{np.sqrt(ra*rb):>16.10f}")
print("  The three means satisfy geometric <= log-mean <= arithmetic, which is")
print("  the classical mean inequality; the exact radius therefore DECREASES")
print("  as the geometry becomes more strongly curved.")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
rp = np.linspace(R1, R2, 400)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)
ax[0].plot(rp * 1e3, exact_T(rp, BETA_A), "-", lw=2.0, color="#2166ac",
           label="Exact, Case A")
a10 = [r for r in rA_A if r["N"] == 10][0]
b10 = [r for r in rB_A if r["N"] == 10][0]
ax[0].plot(a10["rc"] * 1e3, a10["T"], "o", ms=6, mfc="none", mew=1.4,
           color="#b2182b", label="Scheme A, $N=10$")
ax[0].plot(b10["rc"] * 1e3, b10["T"], "s", ms=6, mfc="none", mew=1.4,
           color="#1b7837", label="Scheme B, $N=10$")
ax[0].plot(rp * 1e3, exact_T(rp, BETA_B), "--", lw=1.7, color="#762a83",
           label="Exact, Case B ($k(T)$)")
ax[0].set_xlabel(r"Radius $r$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Coarse-mesh comparison")
ax[0].legend(fontsize=8.5)
ax[0].set_xlim(R1 * 1e3, R2 * 1e3)

hA = rA_A[1]["hist"]
hB = rB_B[1]["hist"]
ax[1].semilogy(hA[:, 0], np.maximum(hA[:, 1], 1e-18), "o-", lw=1.6, ms=5,
               color="#b2182b", label="Scheme A, Case A")
ax[1].semilogy(hB[:, 0], np.maximum(hB[:, 1], 1e-18), "s-", lw=1.6, ms=5,
               color="#1b7837", label="Scheme B, Case B")
ax[1].axhline(1e-12, color="0.4", ls="--", lw=1.2, label="tolerance")
ax[1].set_xlabel("Picard iteration")
ax[1].set_ylabel(r"$\|R\|_\infty / q_{ref}$ [-]")
ax[1].set_title("(b) Residual histories, $N = 20$")
ax[1].legend(fontsize=9)

fig.suptitle("Example 4.2 -- Finite volume verification in spherical geometry",
             fontsize=12.5, y=1.02)
fig.savefig("fig_4_2a_verification.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
    constrained_layout=True)
drs = np.array([r["dr"] for r in rA_A])
ax[0].loglog(drs * 1e3, [r["e"] for r in rA_A], "o-", lw=1.8, ms=6,
             color="#b2182b", label="Scheme A, Case A")
ax[0].loglog(drs * 1e3, [r["e"] for r in rA_B], "^--", lw=1.6, ms=6,
             color="#762a83", label="Scheme A, Case B")
ax[0].loglog(drs * 1e3, np.maximum([r["e"] for r in rB_A], 1e-16), "s-",
             lw=1.8, ms=6, color="#1b7837", label="Scheme B, Case A")
ax[0].loglog(drs * 1e3, np.maximum([r["e"] for r in rB_B], 1e-16), "v:",
             lw=1.6, ms=6, color="#4d4d4d", label="Scheme B, Case B")
ax[0].loglog(drs * 1e3, rA_A[0]["e"] * (drs / drs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta r$ [mm]")
ax[0].set_ylabel(r"$\|e\|_\infty$ [K]")
ax[0].set_title("(a) Grid convergence")
ax[0].legend(fontsize=7.5, loc="center right")

ax[1].loglog(drs * 1e3, [r["q"] for r in rA_A], "o-", lw=1.8, ms=6,
             color="#b2182b", label="Scheme A")
ax[1].loglog(drs * 1e3, np.maximum([r["q"] for r in rB_A], 1e-16), "s-",
             lw=1.8, ms=6, color="#1b7837", label="Scheme B")
ax[1].loglog(drs * 1e3, [r["q"] for r in rA_A][0] * (drs / drs[0])**2, "k--",
             lw=1.3, label="slope 2")
ax[1].set_xlabel(r"$\Delta r$ [mm]")
ax[1].set_ylabel(r"$|q - q_{exact}|$ [W]")
ax[1].set_title("(b) Error in the heat rate (Case A)")
ax[1].legend(fontsize=9, loc="center right")

fig.suptitle("Example 4.2 -- Face radius squared versus geometric mean",
             fontsize=12.5, y=1.02)
fig.savefig("fig_4_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_4_2a_verification.png, fig_4_2b_convergence.png")
