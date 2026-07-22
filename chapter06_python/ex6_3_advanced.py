"""
================================================================================
 EXAMPLE 6.3 -- ADVANCED TRANSIENT VERIFICATION
 The semi-infinite solid and the multidimensional product solution
================================================================================

 OBJECTIVE
 ---------
 Two classical results are developed and verified numerically:

   PART A -- the SEMI-INFINITE SOLID, whose similarity solution in terms of
             erfc governs every short-time transient before the far boundary
             is felt.  The penetration depth is quantified, and the criterion
             for treating a finite body as semi-infinite is TESTED rather than
             quoted.

   PART B -- the PRODUCT SOLUTION, in which a multidimensional transient is
             the product of one-dimensional solutions.  A rectangular bar is
             the intersection of two infinite slabs, and

                 theta*_bar(x,y,t) = theta*_slab(x,t) * theta*_slab(y,t)

             This is verified against a full two-dimensional transient finite
             volume solution built on the Chapter 5 machinery.

 GOVERNING EQUATIONS
 -------------------
   Part A:  dT/dt = alpha d2T/dx2 ,  0 < x < inf
   Part B:  dT/dt = alpha (d2T/dx2 + d2T/dy2)  on a rectangle

 EXACT SOLUTIONS, PART A
 -----------------------
 With T(x,0) = T_i and the similarity variable eta = x/(2 sqrt(alpha t)):

   Case 1, constant surface temperature T_s:
        (T - T_s)/(T_i - T_s) = erf(eta)
        q_s''(t) = k (T_s - T_i)/sqrt(pi alpha t)        -- diverges as t -> 0

   Case 2, constant surface flux q_s'':
        T - T_i = (2 q_s''/k) sqrt(alpha t/pi) exp(-eta^2)
                  - (q_s'' x/k) erfc(eta)

   Case 3, surface convection with h:
        (T - T_i)/(T_inf - T_i) = erfc(eta)
             - exp(h x/k + h^2 alpha t/k^2) erfc(eta + h sqrt(alpha t)/k)

 Case 3 contains Cases 1 and 2 as limits and is the one verified numerically.

 PENETRATION DEPTH
 -----------------
 The depth at which the temperature change has fallen to 1 % of the surface
 change follows from erf(eta) = 0.99, giving eta = 1.8214, so

        delta_p = 3.64 sqrt(alpha t)

 A finite body may be treated as semi-infinite while delta_p is smaller than
 its thickness, i.e. while Fo < (1/3.64)^2 = 0.0755.  This criterion is tested
 against the exact finite-slab series of Example 6.1.

 SYMBOLS -- see Examples 6.1 and 6.2; additionally
   eta      [-]   similarity variable x/(2 sqrt(alpha t))
   delta_p  [m]   penetration depth
   q_s''    [W/m^2] surface flux

 OUTPUTS
 -------
   fig_6_3a_semiinf.png   similarity solution and penetration depth
   fig_6_3b_product.png   product solution verified in two dimensions
   fig_6_3c_orders.png    convergence of both parts

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from scipy.special import erf, erfc

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

K = 15.0
RHO, CP = 7900.0, 477.0
ALPHA = K / (RHO * CP)
H = 500.0
T_I, T_INF = 800.0, 300.0
L = 0.020                       # half-thickness for the finite comparisons
TAU = L * L / ALPHA
BI = H * L / K


# ==============================================================================
# PART A -- SEMI-INFINITE SOLID
# ==============================================================================
def semi_inf_convection(x, t):
    """Case 3: convection at the surface of a semi-infinite solid."""
    x = np.asarray(x, dtype=float)
    s = np.sqrt(ALPHA * t)
    eta = x / (2.0 * s)
    a = H * x / K + H * H * ALPHA * t / (K * K)
    # exp(a) can overflow while erfc(...) underflows; combine them safely
    b = eta + H * s / K
    with np.errstate(over="ignore"):
        term = np.exp(np.clip(a, -700, 700)) * erfc(b)
    return T_I + (T_INF - T_I) * (erfc(eta) - term)


def semi_inf_const_Ts(x, t, Ts):
    return Ts + (T_I - Ts) * erf(np.asarray(x) / (2.0 * np.sqrt(ALPHA * t)))


# --- finite-slab exact series (Example 6.1) for the comparison ---------------
def slab_eigen(n_terms, Bi):
    f = lambda z: z * np.sin(z) - Bi * np.cos(z)
    return np.array([brentq(f, n * np.pi + 1e-12,
                            n * np.pi + 0.5 * np.pi - 1e-12, xtol=1e-14)
                     for n in range(n_terms)])


def slab_theta(xstar, Fo, Bi=BI, n_terms=60):
    z = slab_eigen(n_terms, Bi)
    C = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    xstar = np.atleast_1d(np.asarray(xstar, dtype=float))
    return np.sum(C[:, None] * np.exp(-z**2 * Fo)[:, None]
                  * np.cos(np.outer(z, xstar)), axis=0)


t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 6.3 -- SEMI-INFINITE SOLID AND THE PRODUCT SOLUTION")
print("=" * 78)
print(f"  alpha = {ALPHA:.4e} m^2/s, h = {H} W/(m^2 K), Bi = {BI:.4f}, "
      f"L^2/alpha = {TAU:.2f} s")

print("\n  PART A -- verification of the similarity solution")
print("  (i) it must satisfy the diffusion equation")
tt = 20.0
xg = np.linspace(1e-4, 0.05, 4001)
Tg = semi_inf_convection(xg, tt)
hx = xg[1] - xg[0]
d2 = (Tg[2:] - 2 * Tg[1:-1] + Tg[:-2]) / hx**2
dt_ = 1e-3
dTdt = (semi_inf_convection(xg[1:-1], tt + dt_)
        - semi_inf_convection(xg[1:-1], tt - dt_)) / (2 * dt_)
scale = np.max(np.abs(dTdt))
print(f"      max|dT/dt - alpha d2T/dx2| / max|dT/dt| = "
      f"{np.max(np.abs(dTdt - ALPHA * d2))/scale:.3e}")

print("  (ii) it must satisfy the surface energy balance")
# The solid occupies x > 0 and is HOTTER than the fluid, so heat arrives at
# the surface from within and leaves by convection:
#        k dT/dx|_0 = h (T_s - T_inf)
# Writing this with the wrong sign is easy and the check then fails by twice
# the (correct) flux, which is how the error first appeared here.
# A second-order one-sided difference is used because only x >= 0 exists.
hh = 1e-5
T0, T1_, T2_ = (semi_inf_convection(np.array([0.0]), tt)[0],
                semi_inf_convection(np.array([hh]), tt)[0],
                semi_inf_convection(np.array([2 * hh]), tt)[0])
dTdx0 = (-3.0 * T0 + 4.0 * T1_ - T2_) / (2.0 * hh)
print(f"      k dT/dx|0   = {K*dTdx0:14.6f} W/m^2")
print(f"      h(Ts-Tinf)  = {H*(T0-T_INF):14.6f} W/m^2")
print(f"      relative difference = "
      f"{abs(K*dTdx0 - H*(T0-T_INF))/abs(H*(T0-T_INF)):.3e}")

print("  (iii) far from the surface it must return to T_i")
print(f"      T at x = 0.20 m, t = 20 s : "
      f"{semi_inf_convection(np.array([0.20]), 20.0)[0]:.10f} K "
      f"(T_i = {T_I})")

# --- penetration depth and the semi-infinite criterion ------------------------
eta_99 = brentq(lambda e: erf(e) - 0.99, 0.1, 5.0, xtol=1e-14)
print(f"\n  Penetration depth: erf(eta) = 0.99 at eta = {eta_99:.6f}, so")
print(f"    delta_p = {2*eta_99:.4f} sqrt(alpha t)")
print(f"  A slab of half-thickness L stays semi-infinite while delta_p < L,")
print(f"    i.e. while Fo < {1.0/(2*eta_99)**2:.6f}")

print("\n  TESTING that criterion against the exact finite-slab series.")
print("  Both are evaluated at the CENTRE of the slab, where the finite")
print("  solution first departs from the semi-infinite one.")
print(f"  {'Fo':>8} {'finite theta*':>15} {'semi-inf theta*':>17} "
      f"{'difference':>12}")
for Fo in [0.01, 0.03, 0.0755, 0.15, 0.3]:
    t_ = Fo * TAU
    th_fin = slab_theta(np.array([0.0]), Fo)[0]
    th_semi = ((semi_inf_convection(np.array([L]), t_)[0] - T_INF)
               / (T_I - T_INF))
    print(f"  {Fo:>8.4f} {th_fin:>15.8f} {th_semi:>17.8f} "
          f"{abs(th_fin-th_semi):>12.3e}")
print(f"  At the criterion Fo = 0.0755 the two agree to about 1e-3, and the")
print("  departure grows rapidly beyond it.  The rule is sound, and it is a")
print("  statement about when the FAR BOUNDARY has been felt, not about the")
print("  accuracy of either solution.")

# ==============================================================================
# PART B -- THE PRODUCT SOLUTION
# ==============================================================================
LX, LY = 0.020, 0.012          # half-widths of the bar
BIX, BIY = H * LX / K, H * LY / K


def slab_theta_gen(xstar, Fo, Bi, n_terms=40):
    z = slab_eigen(n_terms, Bi)
    C = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    xstar = np.atleast_1d(np.asarray(xstar, dtype=float))
    return np.sum(C[:, None] * np.exp(-z**2 * Fo)[:, None]
                  * np.cos(np.outer(z, xstar)), axis=0)


def product_exact(xs, ys, t):
    """theta* for the bar as the product of two slab solutions."""
    fx = slab_theta_gen(xs / LX, ALPHA * t / LX**2, BIX)
    fy = slab_theta_gen(ys / LY, ALPHA * t / LY**2, BIY)
    return np.outer(fx, fy)


# --- 2-D transient FVM (quarter bar, symmetry at x = 0 and y = 0) ------------
def solve_2d(Nx, Ny, dt, t_end, theta=0.5, tol=1e-12):
    """Two-dimensional transient FVM on a quarter bar, solved by SOR.

    An operator-split (ADI) scheme was written first and was wrong: its error
    GREW under mesh refinement, which is the signature of an inconsistent
    scheme rather than an inaccurate one.  It is replaced here by the
    straightforward theta-scheme of Chapter 5 solved with red-black SOR at
    each step.  Slower, and obviously correct -- the right trade for a
    verification example.
    """
    dx, dy = LX / Nx, LY / Ny
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy

    aE = np.full((Nx, Ny), K * dy / dx)
    aW = np.full((Nx, Ny), K * dy / dx)
    aN = np.full((Nx, Ny), K * dx / dy)
    aS = np.full((Nx, Ny), K * dx / dy)
    Sp = np.zeros((Nx, Ny))
    Su = np.zeros((Nx, Ny))

    aW[0, :] = 0.0                       # symmetry plane x = 0
    aS[:, 0] = 0.0                       # symmetry plane y = 0
    Ux = 1.0 / (dx / (2.0 * K) + 1.0 / H)
    Uy = 1.0 / (dy / (2.0 * K) + 1.0 / H)
    aE[-1, :] = 0.0
    Sp[-1, :] -= Ux * dy
    Su[-1, :] += Ux * dy * T_INF
    aN[:, -1] = 0.0
    Sp[:, -1] -= Uy * dx
    Su[:, -1] += Uy * dx * T_INF

    aPs = aE + aW + aN + aS - Sp
    cap = RHO * CP * dx * dy

    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    colours = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]
    omega = 2.0 / (1.0 + np.sin(np.pi / max(Nx, Ny)))

    def nbr(Tc):
        s_ = np.zeros_like(Tc)
        s_[:-1, :] += aE[:-1, :] * Tc[1:, :]
        s_[1:, :] += aW[1:, :] * Tc[:-1, :]
        s_[:, :-1] += aN[:, :-1] * Tc[:, 1:]
        s_[:, 1:] += aS[:, 1:] * Tc[:, :-1]
        return s_

    T = np.full((Nx, Ny), T_I)
    t = 0.0
    while t < t_end - 1e-12:
        h_ = min(dt, t_end - t)
        aP0 = cap / h_
        F_old = nbr(T) - aPs * T + Su
        rhs = aP0 * T + theta * Su + (1.0 - theta) * F_old
        aP = aP0 + theta * aPs
        Tn = T.copy()
        for _ in range(20000):
            for c in colours:
                Tn = np.where(c, Tn + omega * ((theta * nbr(Tn) + rhs) / aP
                                               - Tn), Tn)
            r = np.max(np.abs(rhs + theta * nbr(Tn) - aP * Tn)) / np.max(
                np.abs(rhs))
            if r < tol:
                break
        T = Tn
        t += h_
    return xc, yc, T


print("\n" + "-" * 78)
print("  PART B -- THE PRODUCT SOLUTION")
print(f"  Bar half-widths {LX*1e3:.0f} x {LY*1e3:.0f} mm, "
      f"Bi_x = {BIX:.4f}, Bi_y = {BIY:.4f}")
print("  Claim: theta*_bar = theta*_slab(x) * theta*_slab(y).")
print("  This holds because the diffusion operator separates AND the initial")
print("  and boundary data are themselves products -- both are needed.")

T_EVAL = 12.0
print(f"\n  Verification against a 2-D transient FVM at t = {T_EVAL} s")
print(f"  {'Nx x Ny':>10} {'L2 in theta*':>14} {'p_L2':>7} "
      f"{'Linf in theta*':>16} {'p_inf':>7}")
rows = []
for Nx, Ny in [(10, 6), (20, 12), (40, 24), (80, 48)]:
    xc, yc, T = solve_2d(Nx, Ny, 0.02, T_EVAL)
    ex = product_exact(xc, yc, T_EVAL)
    e = (T - T_INF) / (T_I - T_INF) - ex
    l2 = np.sqrt(np.mean(e**2))
    li = np.max(np.abs(e))
    p2 = np.log(rows[-1][1] / l2) / np.log(2.0) if rows else float("nan")
    pi_ = np.log(rows[-1][2] / li) / np.log(2.0) if rows else float("nan")
    rows.append((Nx, l2, li, p2, pi_, xc, yc, T))
    print(f"  {f'{Nx} x {Ny}':>10} {l2:>14.4e} {p2:>7.3f} {li:>16.4e} "
          f"{pi_:>7.3f}")

print("\n  Centre temperature, three independent routes:")
xc, yc, T = rows[-1][5], rows[-1][6], rows[-1][7]
th_prod = (slab_theta_gen(np.array([0.0]), ALPHA * T_EVAL / LX**2, BIX)[0]
           * slab_theta_gen(np.array([0.0]), ALPHA * T_EVAL / LY**2, BIY)[0])
th_num = (T[0, 0] - T_INF) / (T_I - T_INF)
print(f"    product of two slab series : {th_prod:.10f}")
print(f"    2-D FVM (80 x 48)          : {th_num:.10f}")
print(f"    difference                 : {abs(th_prod-th_num):.3e}")
print(f"    in kelvin                  : "
      f"{abs(th_prod-th_num)*(T_I-T_INF):.6f} K")

print(f"\n  CPU time = {time.perf_counter()-t0:.3f} s")
print("=" * 78)

# ==============================================================================
# FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),constrained_layout=True)
xplot = np.linspace(0, 0.06, 300)
for t_, col in zip([2.0, 10.0, 40.0, 160.0],
                   plt.cm.viridis(np.linspace(0.05, 0.85, 4))):
    ax[0].plot(xplot * 1e3, semi_inf_convection(xplot, t_), "-", lw=1.9,
               color=col, label=rf"$t = {t_:g}$ s")
    dp = 2 * eta_99 * np.sqrt(ALPHA * t_)
    ax[0].axvline(dp * 1e3, color=col, ls=":", lw=1.2)
ax[0].axhline(T_I, color="0.5", ls="--", lw=1.0)
ax[0].annotate(r"$T_i$", xy=(52, T_I - 22), fontsize=10, color="0.4")
ax[0].set_xlabel(r"Depth $x$ [mm]")
ax[0].set_ylabel(r"Temperature $T$ [K]")
ax[0].set_title("(a) Semi-infinite solid\n(dotted: penetration depth)")
ax[0].legend(fontsize=8.5, loc="lower right")

Fo_g = np.logspace(-2.3, -0.3, 40)
diff = []
for Fo in Fo_g:
    th_f = slab_theta(np.array([0.0]), Fo)[0]
    th_s = ((semi_inf_convection(np.array([L]), Fo * TAU)[0] - T_INF)
            / (T_I - T_INF))
    diff.append(abs(th_f - th_s))
ax[1].loglog(Fo_g, diff, "-", lw=2.0, color="#b2182b")
ax[1].axvline(1.0 / (2 * eta_99) ** 2, color="#1b7837", ls="--", lw=1.6)
ax[1].annotate(rf"$Fo = {1.0/(2*eta_99)**2:.4f}$",
               xy=(1.0 / (2 * eta_99) ** 2 * 1.1, 1e-5), fontsize=9,
               color="#1b7837", rotation=90)
ax[1].set_xlabel(r"Fourier number $Fo$")
ax[1].set_ylabel(r"$|\theta^*_{finite} - \theta^*_{semi-inf}|$ at the centre")
ax[1].set_title("(b) When may a body be treated as semi-infinite?")

fig.suptitle("Example 6.3 -- The semi-infinite solid", fontsize=12.5, y=1.04)
fig.savefig("fig_6_3a_semiinf.png")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.8), constrained_layout=True)
xc, yc, T = rows[-1][5], rows[-1][6], rows[-1][7]
XG, YG = np.meshgrid(xc, yc, indexing="ij")
ex = product_exact(xc, yc, T_EVAL)
for a, F_, ttl, cm in [
        (ax[0], (T - T_INF) / (T_I - T_INF), r"(a) 2-D FVM, $\theta^*$", "inferno"),
        (ax[1], ex, r"(b) Product solution, $\theta^*$", "inferno"),
        (ax[2], ((T - T_INF) / (T_I - T_INF) - ex) * 1e5,
         r"(c) Difference $\times 10^5$", "RdBu_r")]:
    cf = a.contourf(XG * 1e3, YG * 1e3, F_, levels=22, cmap=cm)
    a.set_xlabel(r"$x$ [mm]"); a.set_ylabel(r"$y$ [mm]")
    a.set_title(ttl); a.set_aspect("equal"); a.grid(False)
    fig.colorbar(cf, ax=a, fraction=0.046, pad=0.03)
fig.suptitle("Example 6.3 -- The product solution verified in two dimensions",
             fontsize=12.5, y=1.04)
fig.savefig("fig_6_3b_product.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
Ns = np.array([r[0] for r in rows])
hs = LX / Ns
ax[0].loglog(hs * 1e3, [r[1] for r in rows], "o-", lw=1.8, ms=6,
             color="#2166ac", label=r"$\|e\|_2$")
ax[0].loglog(hs * 1e3, [r[2] for r in rows], "s-", lw=1.8, ms=6,
             color="#b2182b", label=r"$\|e\|_\infty$")
ax[0].loglog(hs * 1e3, rows[0][1] * (hs / hs[0])**2, "k--", lw=1.3,
             label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [mm]")
ax[0].set_ylabel(r"error in $\theta^*$")
ax[0].set_title("(a) 2-D transient convergence")
ax[0].legend(fontsize=9, loc="lower right")

etas = np.linspace(0, 3, 300)
ax[1].plot(etas, erf(etas), "-", lw=2.0, color="#4d4d4d",
           label=r"$\mathrm{erf}(\eta)$, constant $T_s$")
ax[1].axhline(0.99, color="0.5", ls=":", lw=1.2)
ax[1].axvline(eta_99, color="#1b7837", ls="--", lw=1.6)
ax[1].annotate(rf"$\eta = {eta_99:.4f}$", xy=(eta_99 + 0.06, 0.55),
               fontsize=9.5, color="#1b7837")
ax[1].annotate("99 %", xy=(0.1, 1.005), fontsize=9, color="0.4")
ax[1].set_xlabel(r"Similarity variable $\eta = x/(2\sqrt{\alpha t})$")
ax[1].set_ylabel(r"$(T - T_s)/(T_i - T_s)$")
ax[1].set_title("(b) The similarity solution collapses all times")
ax[1].legend(fontsize=9, loc="lower right")

fig.suptitle("Example 6.3 -- Convergence and similarity", fontsize=12.5,
             y=1.02)
fig.savefig("fig_6_3c_orders.png")
plt.close(fig)

print("Figures written: fig_6_3a_semiinf.png, fig_6_3b_product.png, "
      "fig_6_3c_orders.png")
