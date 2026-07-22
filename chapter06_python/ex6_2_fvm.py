"""
================================================================================
 EXAMPLE 6.2 -- TRANSIENT FINITE VOLUME SOLUTION IN THREE GEOMETRIES
 One solver, m = 0, 1, 2, verified against three exact series
================================================================================

 OBJECTIVE
 ---------
 Chapter 4 showed that the plane wall, the cylinder and the sphere are one
 problem with a geometry index m in the area law A(r) ~ r^m.  Here that unified
 solver is written once and verified against the EXACT transient series of all
 three geometries -- three independent analytical solutions, one code.

 The three series involve quite different special functions:

   m = 0  slab      cos(zeta x/L)         zeta tan(zeta) = Bi
   m = 1  cylinder  J0(zeta r/R)          zeta J1(zeta)/J0(zeta) = Bi
   m = 2  sphere    sin(zeta r/R)/(r/R)   1 - zeta cot(zeta) = Bi

 That a single discretisation reproduces all three to second order is a much
 stronger statement than reproducing any one of them.

 GOVERNING EQUATION
 ------------------
   rho c dT/dt = (1/r^m) d/dr ( r^m k dT/dr )        0 < r < R

 INITIAL AND BOUNDARY CONDITIONS
 -------------------------------
   T(r,0) = T_i ;  dT/dr = 0 at r = 0 ;  -k dT/dr = h (T - T_inf) at r = R

 DISCRETISATION
 --------------
 Cell-centred control volumes with the geometry-dependent measures

   volume/(geometric constant) : (rf[i+1]^(m+1) - rf[i]^(m+1))/(m+1)
   face conductance            : m = 0  k/(rb - ra)
                                 m = 1  k/ln(rb/ra)          (Chapter 3)
                                 m = 2  k ra rb/(rb - ra)    (Chapter 4)

 The exact face conductances of Chapters 3 and 4 are used, so the STEADY limit
 of each geometry is reproduced to round-off and any error seen here is purely
 the transient discretisation.

 Time integration uses the theta-family of Chapter 1: theta = 1 backward Euler
 (first order, unconditionally bounded), theta = 1/2 Crank-Nicolson (second
 order).  Both are measured.

 SYMBOLS -- see Example 6.1; additionally
   m       [-]   geometry index, 0 slab / 1 cylinder / 2 sphere
   R       [m]   half-thickness or radius
   zeta_n  [-]   n-th eigenvalue of the relevant transcendental equation
   theta   [-]   time-integration weight

 OUTPUTS
 -------
   fig_6_2a_geometries.png   the three transients, exact and computed
   fig_6_2b_convergence.png  spatial and temporal order in all three

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from scipy.special import j0, j1

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

R = 0.020
K = 15.0
RHO, CP = 7900.0, 477.0
ALPHA = K / (RHO * CP)
H = 500.0
T_I, T_INF = 800.0, 300.0
BI = H * R / K
TAU = R * R / ALPHA

NAMES = {0: "slab", 1: "cylinder", 2: "sphere"}


# ==============================================================================
# 1. EXACT SERIES FOR THE THREE GEOMETRIES
# ==============================================================================
@lru_cache(maxsize=None)
def _bessel_zeros_cached(name, count):
    """First `count` positive zeros of fn (j0 or j1), found by scanning.

    The zeros of J0 and J1 are asymptotically spaced by pi, so a scan at pi/8
    resolution cannot skip one.  Each sign change is then refined by Brent.
    """
    fn = j0 if name == "j0" else j1
    zeros = []
    x = np.arange(0.5, (count + 3) * np.pi + 1.0, np.pi / 8.0)
    v = np.array([fn(xx) for xx in x])
    for i in range(len(x) - 1):
        if v[i] == 0.0:
            zeros.append(x[i])
        elif v[i] * v[i + 1] < 0.0:
            zeros.append(brentq(fn, x[i], x[i + 1], xtol=1e-14))
        if len(zeros) >= count:
            break
    return tuple(zeros)


def _bessel_zeros(fn, count):
    return _bessel_zeros_cached("j0" if fn is j0 else "j1", count)


@lru_cache(maxsize=None)
def _eigen_cached(m, n_terms, Bi):
    """Eigenvalues for geometry m.  Each has one root per stated interval.

    Cached because the exact series is evaluated at hundreds of times and
    positions, and the cylinder branch has to locate Bessel zeros first.
    """
    d = 1e-11
    out = np.empty(n_terms)
    if m == 0:                                   # zeta tan zeta = Bi
        f = lambda z: z * np.sin(z) - Bi * np.cos(z)
        for n in range(n_terms):
            out[n] = brentq(f, n * np.pi + d, n * np.pi + 0.5 * np.pi - d,
                            xtol=1e-14)
    elif m == 1:                                 # zeta J1/J0 = Bi
        # The roots interlace the zeros of J1 and J0, which supplies a
        # guaranteed bracket for every one.  Those zeros are located here by
        # scanning rather than tabulated, so the term count is unlimited.
        f = lambda z: z * j1(z) - Bi * j0(z)
        j1z = _bessel_zeros(j1, n_terms + 1)
        j0z = _bessel_zeros(j0, n_terms + 1)
        for n in range(n_terms):
            lo = 0.0 if n == 0 else j1z[n - 1]
            out[n] = brentq(f, lo + d, j0z[n] - d, xtol=1e-14)
    else:                                        # 1 - zeta cot zeta = Bi
        f = lambda z: z * np.cos(z) - (1.0 - Bi) * np.sin(z)
        for n in range(n_terms):
            out[n] = brentq(f, n * np.pi + d, (n + 1) * np.pi - d, xtol=1e-14)
    out.flags.writeable = False
    return out


def eigen(m, n_terms, Bi):
    return _eigen_cached(int(m), int(n_terms), float(Bi))


def coeffs(m, z):
    """Expansion coefficients C_n for geometry m."""
    if m == 0:
        return 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    if m == 1:
        return (2.0 / z) * j1(z) / (j0(z) ** 2 + j1(z) ** 2)
    return 2.0 * (np.sin(z) - z * np.cos(z)) / (z - np.sin(z) * np.cos(z))


def shape(m, z, rstar):
    """Eigenfunction of geometry m at r/R = rstar (limit taken at rstar = 0)."""
    arg = np.outer(z, rstar)
    if m == 0:
        return np.cos(arg)
    if m == 1:
        return np.array([j0(a) for a in arg])
    # The spherical eigenfunction is sin(zeta r*)/(zeta r*), NOT sin/r*.
    # Omitting the zeta leaves a mode-dependent factor that does not vanish
    # under mesh refinement, which is how the error showed up: a 13 % offset
    # identical on every grid.  The limit at r* = 0 is 1.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.outer(z, np.where(rstar > 0, rstar, 1.0))
        return np.where(rstar[None, :] > 0.0, np.sin(arg) / denom, 1.0)


def theta_exact(m, rstar, Fo, Bi=BI, n_terms=40):
    rstar = np.atleast_1d(np.asarray(rstar, dtype=float))
    z = eigen(m, n_terms, Bi)
    C = coeffs(m, z)
    return np.sum(C[:, None] * np.exp(-z**2 * Fo)[:, None] * shape(m, z, rstar),
                  axis=0)


# ==============================================================================
# 2. UNIFIED FVM SOLVER
# ==============================================================================
def mesh(m, N):
    rf = np.linspace(0.0, R, N + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    vol = (rf[1:] ** (m + 1) - rf[:-1] ** (m + 1)) / (m + 1)
    return rc, rf, vol


def conductance(m, ra, rb):
    if m == 0:
        return K / (rb - ra)
    if m == 1:
        return K / np.log(rb / ra)
    return K * ra * rb / (rb - ra)


def step(T_old, dt, m, rc, rf, vol, theta=1.0):
    N = len(T_old)
    aW = np.zeros(N)
    aE = np.zeros(N)
    g = conductance(m, rc[:-1], rc[1:])
    aW[1:] = g
    aE[:-1] = g
    # r = 0: the face area vanishes, so there is simply no link.
    a_be = conductance(m, rc[-1], R)
    U = 1.0 / (1.0 / a_be + 1.0 / (R ** m * H))
    Sp = np.zeros(N)
    Su = np.zeros(N)
    Su[-1] += U * T_INF
    Sp[-1] -= U

    aPs = aW + aE - Sp
    aP0 = RHO * CP * vol / dt
    TW = np.concatenate(([0.0], T_old[:-1]))
    TE = np.concatenate((T_old[1:], [0.0]))
    F_old = aW * TW + aE * TE - aPs * T_old + Su

    ab = np.zeros((3, N))
    ab[0, 1:] = -theta * aE[:-1]
    ab[1, :] = aP0 + theta * aPs
    ab[2, :-1] = -theta * aW[1:]
    b = aP0 * T_old + theta * Su + (1.0 - theta) * F_old
    return solve_banded((1, 1), ab, b)


def integrate(m, N, dt, t_end, theta=1.0):
    rc, rf, vol = mesh(m, N)
    T = np.full(N, T_I)
    t, steps = 0.0, 0
    while t < t_end - 1e-12:
        h_ = min(dt, t_end - t)
        T = step(T, h_, m, rc, rf, vol, theta)
        t += h_
        steps += 1
    return rc, vol, T, steps


def norms(m, T, rc, vol, Fo):
    e = (T - T_INF) / (T_I - T_INF) - theta_exact(m, rc / R, Fo)
    return (np.sqrt(np.sum(e**2 * vol) / np.sum(vol)), np.max(np.abs(e)))


# ==============================================================================
# 3. VERIFY THE THREE EXACT SERIES
# ==============================================================================
print("=" * 78)
print("EXAMPLE 6.2 -- TRANSIENT FVM IN THREE GEOMETRIES")
print("=" * 78)
print(f"  R = {R} m, alpha = {ALPHA:.4e} m^2/s, Bi = {BI:.6f}, "
      f"tau = {TAU:.2f} s")
print("\n  Eigenvalue and residual check (first four of each):")
print(f"  {'geometry':>10} {'n':>3} {'zeta_n':>14} {'residual':>12} {'C_n':>11}")
for m in (0, 1, 2):
    zz = eigen(m, 4, BI)
    CC = coeffs(m, zz)
    for n, (z_, c_) in enumerate(zip(zz, CC), start=1):
        if m == 0:
            res = z_ * np.tan(z_) - BI
        elif m == 1:
            res = z_ * j1(z_) / j0(z_) - BI
        else:
            res = 1.0 - z_ / np.tan(z_) - BI
        print(f"  {NAMES[m] if n == 1 else '':>10} {n:>3d} {z_:>14.10f} "
              f"{res:>12.2e} {c_:>11.6f}")

print("\n  Series consistency at the centre as Fo -> 0 (should approach 1).")
print("  Convergence is slow because the initial and boundary data are")
print("  incompatible at the surface, as in Chapters 1 and 5.")
print(f"  {'terms':>7} {'slab':>12} {'cylinder':>12} {'sphere':>12}")
for nt in [10, 40, 120]:
    vals = [theta_exact(m, np.array([0.0]), 1e-9, BI, nt)[0] for m in (0, 1, 2)]
    print(f"  {nt:>7d} {vals[0]:>12.8f} {vals[1]:>12.8f} {vals[2]:>12.8f}")

# ==============================================================================
# 4. GRID CONVERGENCE IN ALL THREE GEOMETRIES
# ==============================================================================
FO_EVAL = 0.5
T_EVAL = FO_EVAL * TAU
print("\n" + "-" * 78)
print(f"  SPATIAL CONVERGENCE at Fo = {FO_EVAL} (Crank-Nicolson, dt = 0.02 s)")
print(f"  {'geometry':>10} {'N':>5} {'L2':>13} {'p_L2':>7} {'Linf':>13} "
      f"{'p_inf':>7}")
store = {}
for m in (0, 1, 2):
    rows = []
    for N in [10, 20, 40, 80, 160]:
        rc, vol, T, _ = integrate(m, N, 0.02, T_EVAL, theta=0.5)
        l2, li = norms(m, T, rc, vol, FO_EVAL)
        p2 = np.log(rows[-1][1] / l2) / np.log(2.0) if rows else float("nan")
        pi_ = np.log(rows[-1][2] / li) / np.log(2.0) if rows else float("nan")
        rows.append((N, l2, li, p2, pi_))
        print(f"  {NAMES[m] if N == 10 else '':>10} {N:>5d} {l2:>13.4e} "
              f"{p2:>7.3f} {li:>13.4e} {pi_:>7.3f}")
    store[m] = rows

print("\n" + "-" * 78)
print("  TEMPORAL CONVERGENCE, N = 300, measured against dt = 0.02 s (CN)")
print(f"  {'geometry':>10} {'dt [s]':>9} {'BE Linf':>12} {'p_BE':>7} "
      f"{'CN Linf':>12} {'p_CN':>7}")
tstore = {}
for m in (0, 1, 2):
    _, _, T_ref, _ = integrate(m, 300, 0.02, T_EVAL, theta=0.5)
    rows = []
    for dt in [4.0, 2.0, 1.0, 0.5]:
        _, _, T_be, _ = integrate(m, 300, dt, T_EVAL, theta=1.0)
        _, _, T_cn, _ = integrate(m, 300, dt, T_EVAL, theta=0.5)
        e_be = np.max(np.abs(T_be - T_ref)) / (T_I - T_INF)
        e_cn = np.max(np.abs(T_cn - T_ref)) / (T_I - T_INF)
        p_be = np.log(rows[-1][1] / e_be) / np.log(2.0) if rows else float("nan")
        p_cn = np.log(rows[-1][3] / e_cn) / np.log(2.0) if rows else float("nan")
        rows.append((dt, e_be, p_be, e_cn, p_cn))
        print(f"  {NAMES[m] if dt == 4.0 else '':>10} {dt:>9.2f} {e_be:>12.4e} "
              f"{p_be:>7.3f} {e_cn:>12.4e} {p_cn:>7.3f}")
    tstore[m] = rows

# ==============================================================================
# 5. THE GEOMETRY EFFECT
# ==============================================================================
print()
print("  Reading the temporal table.  Backward Euler returns p = 1.00 in all")
print("  three geometries, as it should.  The Crank-Nicolson column is NOT a")
print("  clean order measurement and should not be read as one: the reference")
print("  solution is itself Crank-Nicolson at dt = 0.02 s, so it cannot")
print("  resolve CN errors below its own, and the ratios drift as that floor")
print("  is approached.  Measuring a second-order scheme needs a reference of")
print("  higher order or a much smaller step -- the same requirement to")
print("  isolate the error being measured that Chapter 1 established.")
print()
print("  Note also the SPATIAL table.  The slab converges at 2.00 in BOTH")
print("  norms, while the cylinder and sphere reach 2.00 in L2 but only about")
print("  1.84 in Linf, rising steadily under refinement.  This is the same")
print("  behaviour left open at the end of Chapter 4, and the comparison here")
print("  is informative: it appears in BOTH curved geometries and NOT in the")
print("  plane wall, so it is tied to curvature -- to the control volume at")
print("  r = 0, where the volume weighting is most skewed -- and is not")
print("  peculiar to spheres.  The question of its precise origin remains")
print("  open; Exercise 6.C5 pursues it with this extra evidence in hand.")

print("\n" + "-" * 78)
print("  HOW FAST DOES EACH GEOMETRY COOL?  Same R, same Bi.")
print(f"  {'geometry':>10} {'zeta_1':>10} {'C_1':>9} {'Fo to theta*=0.5':>18} "
      f"{'t [s]':>9}")
for m in (0, 1, 2):
    z1 = eigen(m, 1, BI)[0]
    c1 = coeffs(m, np.array([z1]))[0]
    Fo_half = brentq(lambda F: theta_exact(m, np.array([0.0]), F)[0] - 0.5,
                     0.01, 20.0, xtol=1e-12)
    print(f"  {NAMES[m]:>10} {z1:>10.6f} {c1:>9.5f} {Fo_half:>18.6f} "
          f"{Fo_half*TAU:>9.2f}")
print("  The sphere cools fastest and the slab slowest, in the ratio of their")
print("  surface area per unit volume -- 3/R, 2/R and 1/R.  The first")
print("  eigenvalue, which sets the long-time decay rate, orders the same way.")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
Fo_g = np.linspace(0.005, 2.5, 200)
for m, col in zip((0, 1, 2), ["#4d4d4d", "#2166ac", "#b2182b"]):
    ax[0].semilogy(Fo_g, [theta_exact(m, np.array([0.0]), f)[0] for f in Fo_g],
                   "-", lw=2.0, color=col, label=f"{NAMES[m]} (exact)")
    Fos = np.array([0.1, 0.3, 0.6, 1.0, 1.6, 2.2])
    vals = []
    for f in Fos:
        rc, vol, T, _ = integrate(m, 40, 0.05, f * TAU, theta=0.5)
        vals.append((T[0] - T_INF) / (T_I - T_INF))
    ax[0].semilogy(Fos, vals, "o", ms=5, mfc="none", mew=1.3, color=col)
ax[0].set_xlabel(r"$Fo = \alpha t/R^2$")
ax[0].set_ylabel(r"$\theta^*$ at the centre")
ax[0].set_title("(a) Centre response\n(lines exact, symbols FVM $N=40$)")
ax[0].legend(fontsize=9)

rs = np.linspace(0, 1, 200)
for m, col in zip((0, 1, 2), ["#4d4d4d", "#2166ac", "#b2182b"]):
    ax[1].plot(rs, theta_exact(m, rs, 0.3), "-", lw=2.0, color=col,
               label=NAMES[m])
    rc, vol, T, _ = integrate(m, 30, 0.05, 0.3 * TAU, theta=0.5)
    ax[1].plot(rc / R, (T - T_INF) / (T_I - T_INF), "o", ms=4.5, mfc="none",
               mew=1.2, color=col)
ax[1].set_xlabel(r"$r/R$")
ax[1].set_ylabel(r"$\theta^*$")
ax[1].set_title(r"(b) Profiles at $Fo = 0.3$")
ax[1].legend(fontsize=9)

fig.suptitle("Example 6.2 -- One solver, three geometries", fontsize=12.5,
             y=1.02)
fig.savefig("fig_6_2a_geometries.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
for m, col, mk in zip((0, 1, 2), ["#4d4d4d", "#2166ac", "#b2182b"],
                      ["o", "s", "^"]):
    Ns = np.array([r[0] for r in store[m]])
    ax[0].loglog(R / Ns * 1e3, [r[1] for r in store[m]], mk + "-", lw=1.8,
                 ms=6, color=col, label=f"{NAMES[m]}, $\\|e\\|_2$")
ref = store[0][0][1] * ((R / np.array([r[0] for r in store[0]]))
                        / (R / store[0][0][0]))**2
ax[0].loglog(R / np.array([r[0] for r in store[0]]) * 1e3, ref, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta r$ [mm]")
ax[0].set_ylabel(r"$\|e\|_2$ in $\theta^*$")
ax[0].set_title("(a) Spatial convergence, all three")
ax[0].legend(fontsize=8, loc="lower right")

dts = np.array([r[0] for r in tstore[0]])
for m, col, mk in zip((0, 1, 2), ["#4d4d4d", "#2166ac", "#b2182b"],
                      ["o", "s", "^"]):
    ax[1].loglog(dts, [r[1] for r in tstore[m]], mk + "-", lw=1.7, ms=6,
                 color=col, label=f"{NAMES[m]}, BE")
    ax[1].loglog(dts, [r[3] for r in tstore[m]], mk + ":", lw=1.5, ms=5,
                 color=col, alpha=0.8)
ax[1].loglog(dts, tstore[0][0][1] * (dts / dts[0]), "k--", lw=1.2,
             label="slope 1")
ax[1].loglog(dts, tstore[0][0][3] * (dts / dts[0])**2, "k-.", lw=1.2,
             label="slope 2")
ax[1].set_xlabel(r"$\Delta t$ [s]")
ax[1].set_ylabel(r"$\|e\|_\infty$ in $\theta^*$")
ax[1].set_title("(b) Backward Euler (solid) vs Crank-Nicolson (dotted)")
ax[1].legend(fontsize=7.5, loc="lower right")

fig.suptitle("Example 6.2 -- Order of accuracy in all three geometries",
             fontsize=12.5, y=1.08)
fig.savefig("fig_6_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_6_2a_geometries.png, fig_6_2b_convergence.png")
