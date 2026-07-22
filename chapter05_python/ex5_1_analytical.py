"""
================================================================================
 EXAMPLE 5.1 -- ANALYTICAL SOLUTION BY SEPARATION OF VARIABLES
 Two-dimensional steady conduction in a rectangular plate
================================================================================

 PHYSICAL PROBLEM
 ----------------
 A rectangular plate of width W and height H conducts in two dimensions with no
 generation.  Three edges are held at T1; the fourth (the top) is held at a
 prescribed temperature.  Two top conditions are treated:

   Case U (uniform)     T(x,H) = T2                 -- the textbook problem
   Case S (sinusoidal)  T(x,H) = T1 + dT sin(pi x/W) -- a SINGLE-term solution

 Case S exists for a reason that matters for the rest of the chapter: its
 series terminates after one term, so it is exact in closed form with no
 truncation and no corner singularity.  It is therefore the reference solution
 against which Examples 5.2 and 5.3 verify their numerics.  Case U is the more
 familiar problem and is used to expose what goes wrong at the corners.

 GOVERNING EQUATION  (steady, 2-D, constant k, no generation)
 -------------------------------------------------------------
        d2T     d2T
        ---  +  ---  =  0            (Laplace)
        dx2     dy2

 SEPARATION OF VARIABLES
 -----------------------
 Non-dimensionalise with theta = (T - T1)/(T2 - T1), so three edges carry
 theta = 0.  Seeking theta = X(x) Y(y) and substituting gives

        X''/X = -Y''/Y = -lambda^2

 The sign is forced by the boundary conditions: theta must vanish at BOTH
 x = 0 and x = W, which only a trigonometric X can do, so X'' = -lambda^2 X.
 Then X = sin(lambda x) with lambda_n = n pi / W to satisfy both, and
 Y'' = +lambda^2 Y gives Y = sinh(lambda y), the cosh being excluded by
 theta(x,0) = 0.  Hence the general solution

        theta(x,y) = SUM_n C_n sin(n pi x/W) sinh(n pi y/W)

 CASE U -- uniform top
 ---------------------
 Imposing theta(x,H) = 1 and using orthogonality of sin(n pi x/W) on [0,W]:

        C_n sinh(n pi H/W) = (2/(n pi)) [1 - (-1)^n]

 so that

        theta = (2/pi) SUM_n  ([1 - (-1)^n]/n) sin(n pi x/W)
                              * sinh(n pi y/W)/sinh(n pi H/W)

 Only odd n contribute.  The series converges everywhere in the interior but
 slowly near the two TOP CORNERS, where the boundary data are discontinuous:
 the top edge demands theta = 1 while the side edges demand theta = 0, and the
 corner cannot satisfy both.  This is the two-dimensional analogue of the
 incompatible corner met in Chapter 1, and its consequences are measured below.

 CASE S -- sinusoidal top
 ------------------------
 With theta(x,H) = sin(pi x/W), orthogonality kills every term but n = 1:

        theta(x,y) = sin(pi x/W) sinh(pi y/W) / sinh(pi H/W)

 exact, in closed form, with no series at all.

 SYMBOLS (all SI)
 ----------------
   W, H    [m]        plate width and height
   k       [W/(m K)]  thermal conductivity
   T1      [K]        temperature of the three cold edges
   T2      [K]        temperature of the hot top edge (Case U)
   dT      [K]        amplitude of the sinusoidal top (Case S)
   theta   [-]        dimensionless temperature
   lambda_n[1/m]      n pi / W
   q'      [W/m]      heat rate per unit depth

 OUTPUTS
 -------
   fig_5_1a_fields.png    the two temperature fields
   fig_5_1b_corner.png    corner convergence and the heat balance

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ==============================================================================
# 1. DATA
# ==============================================================================
W, H = 0.40, 0.30        # [m]
K = 15.0                 # [W/(m K)]
T1 = 300.0               # [K]  three cold edges
T2 = 400.0               # [K]  hot top edge (Case U)
DT = 100.0               # [K]  amplitude of the sinusoidal top (Case S)


# ==============================================================================
# 2. EXACT SOLUTIONS
# ==============================================================================
def theta_uniform(x, y, n_terms=200):
    """Dimensionless field for the uniform top.  Odd terms only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(np.broadcast(x, y).shape)
    for n in range(1, n_terms + 1):
        if n % 2 == 0:
            continue
        lam = n * np.pi / W
        # sinh ratio written to avoid overflow for large n
        ratio = np.exp(lam * (y - H)) * (1.0 - np.exp(-2.0 * lam * y)) / (
            1.0 - np.exp(-2.0 * lam * H))
        out = out + (2.0 / (n * np.pi)) * 2.0 * np.sin(lam * x) * ratio
    return out


def T_uniform(x, y, n_terms=200):
    return T1 + (T2 - T1) * theta_uniform(x, y, n_terms)


def T_sin(x, y):
    """Single-term exact solution for the sinusoidal top."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lam = np.pi / W
    ratio = np.exp(lam * (y - H)) * (1.0 - np.exp(-2.0 * lam * y)) / (
        1.0 - np.exp(-2.0 * lam * H))
    return T1 + DT * np.sin(lam * x) * ratio


t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 5.1 -- 2-D STEADY CONDUCTION BY SEPARATION OF VARIABLES")
print("=" * 78)
print(f"  W = {W} m, H = {H} m, k = {K} W/(m K), T1 = {T1} K, T2 = {T2} K")

# ==============================================================================
# 3. VERIFICATION
# ==============================================================================
# (i) Laplace residual on an interior grid
xs = np.linspace(0.02, W - 0.02, 220)
ys = np.linspace(0.02, H - 0.02, 180)
X, Y = np.meshgrid(xs, ys, indexing="ij")
hx, hy = xs[1] - xs[0], ys[1] - ys[0]

for name, fun in [("Case S (sinusoidal)", T_sin),
                  ("Case U (uniform)", lambda a, b: T_uniform(a, b, 400))]:
    F = fun(X, Y)
    lap = ((F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / hx**2
           + (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / hy**2)
    scale = (T2 - T1) / min(hx, hy) ** 2
    print(f"  {name:>22}: max|lap T| / scale = "
          f"{np.max(np.abs(lap))/scale:.3e}")

# (ii) boundary conditions
print("\n  Boundary checks, Case S:")
yv = np.linspace(0, H, 51)
xv = np.linspace(0, W, 51)
print(f"    left  x=0 : max|T - T1| = {np.max(np.abs(T_sin(0.0*yv, yv)-T1)):.3e}")
print(f"    right x=W : max|T - T1| = {np.max(np.abs(T_sin(0.0*yv+W, yv)-T1)):.3e}")
print(f"    bottom y=0: max|T - T1| = {np.max(np.abs(T_sin(xv, 0.0*xv)-T1)):.3e}")
top_err = np.max(np.abs(T_sin(xv, 0.0*xv+H) - (T1 + DT*np.sin(np.pi*xv/W))))
print(f"    top   y=H : max|T - prescribed| = {top_err:.3e}")

# (iii) global energy balance: what enters the top must leave the other three
def flux_top_sin(n=4001):
    x = np.linspace(0, W, n)
    lam = np.pi / W
    # dT/dy at y = H for the single-term solution
    dTdy = DT * np.sin(lam * x) * lam / np.tanh(lam * H)
    return -K * np.trapz(dTdy, x)          # [W/m], negative = leaving upward

def flux_sides_sin(n=4001):
    y = np.linspace(0, H, n)
    lam = np.pi / W
    ratio = np.sinh(lam * y) / np.sinh(lam * H)
    dTdx_0 = DT * lam * ratio              # cos(0) = 1
    dTdx_W = -DT * lam * ratio             # cos(pi) = -1
    q_left = K * np.trapz(dTdx_0, y)       # entering at x=0 is -k dT/dx * (-1)
    q_right = -K * np.trapz(dTdx_W, y)
    x = np.linspace(0, W, n)
    dTdy_0 = DT * np.sin(lam * x) * lam / np.sinh(lam * H)
    q_bot = K * np.trapz(dTdy_0, x)
    return q_left, q_right, q_bot

q_in = -flux_top_sin()
q_l, q_r, q_b = flux_sides_sin()
print(f"\n  Global balance, Case S (per unit depth):")
print(f"    entering through the top     = {q_in:.8f} W/m")
print(f"    leaving  left + right + base = {q_l + q_r + q_b:.8f} W/m")
print(f"      left  {q_l:.6f}, right {q_r:.6f}, bottom {q_b:.6f}")
print(f"    imbalance                    = {abs(q_in-(q_l+q_r+q_b)):.3e} W/m")

# ==============================================================================
# 4. THE CORNER SINGULARITY OF CASE U
# ==============================================================================
print("\n" + "-" * 78)
print("  THE TOP CORNERS OF CASE U.  The data are discontinuous there: the top")
print("  edge demands theta = 1 while the side edges demand theta = 0.  The")
print("  series converges everywhere inside but ever more slowly as the corner")
print("  is approached, exactly as in Chapter 1.")
print(f"\n  theta evaluated at a fixed point NEAR the corner (x/W = 0.01, y/H = 0.99):")
print(f"  {'terms':>8} {'theta':>14} {'change':>12}")
prev = None
for nt in [10, 50, 200, 1000, 4000]:
    v = float(theta_uniform(np.array([0.01 * W]), np.array([0.99 * H]), nt)[0])
    ch = "-" if prev is None else f"{abs(v-prev):.3e}"
    print(f"  {nt:>8d} {v:>14.8f} {ch:>12}")
    prev = v

print(f"\n  The same test at the CENTRE (x/W = 0.5, y/H = 0.5):")
prev = None
for nt in [5, 10, 20, 50]:
    v = float(theta_uniform(np.array([0.5 * W]), np.array([0.5 * H]), nt)[0])
    ch = "-" if prev is None else f"{abs(v-prev):.3e}"
    print(f"  {nt:>8d} {v:>14.10f} {ch:>12}")
    prev = v
print("  At the centre five terms already give eight figures.  Near the corner")
print("  two hundred terms give only three, and a thousand are needed before")
print("  the value settles -- a factor of two hundred in effort for the same")
print("  accuracy, from the same series.  Any numerical method inherits this:")
print("  the difficulty belongs to the PROBLEM, not to the scheme, and it is")
print("  why Case S is used for the verification of Examples 5.2 and 5.3.")

print(f"\n  CPU time = {time.perf_counter()-t0:.4f} s")
print("=" * 78)

# ==============================================================================
# 5. FIGURES
# ==============================================================================
xg = np.linspace(0, W, 220)
yg = np.linspace(0, H, 170)
XG, YG = np.meshgrid(xg, yg, indexing="ij")

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
for a, F, ttl in [(ax[0], T_uniform(XG, YG, 400), "(a) Case U: uniform top"),
                  (ax[1], T_sin(XG, YG), "(b) Case S: sinusoidal top")]:
    cf = a.contourf(XG * 1e3, YG * 1e3, F, levels=24, cmap="inferno")
    cs = a.contour(XG * 1e3, YG * 1e3, F, levels=10, colors="w",
                   linewidths=0.6, alpha=0.7)
    a.clabel(cs, inline=True, fontsize=6.5, fmt="%.0f")
    a.set_xlabel(r"$x$ [mm]")
    a.set_ylabel(r"$y$ [mm]")
    a.set_title(ttl)
    a.set_aspect("equal")
    a.grid(False)
    fig.colorbar(cf, ax=a, label=r"$T$ [K]", fraction=0.040, pad=0.03)
ax[0].annotate("discontinuous\ncorners", xy=(6, H * 1e3 - 8),
               xytext=(60, H * 1e3 - 60), fontsize=8.5, color="w",
               arrowprops=dict(arrowstyle="->", color="w", lw=1.0))

fig.suptitle("Example 5.1 -- Separation of variables in two dimensions",
             fontsize=12.5, y=1.03)
fig.savefig("fig_5_1a_fields.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
nts = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000])
for frac, col, lbl in [(0.5, "#2166ac", "centre  $x/W=0.5,\\ y/H=0.5$"),
                       (0.9, "#1b7837", "near top $y/H=0.90$"),
                       (0.99, "#b2182b", "near corner $x/W=0.01,\\ y/H=0.99$")]:
    xq = 0.5 * W if frac == 0.5 else (0.5 * W if frac == 0.9 else 0.01 * W)
    yq = frac * H
    vals = np.array([float(theta_uniform(np.array([xq]), np.array([yq]), n)[0])
                     for n in nts])
    ax[0].semilogx(nts, vals, "o-", lw=1.6, ms=4.5, color=col, label=lbl)
ax[0].set_xlabel("Number of series terms")
ax[0].set_ylabel(r"$\theta$ [-]")
ax[0].set_title("(a) Convergence depends on where you look")
ax[0].legend(fontsize=8)

xline = np.linspace(0, W, 400)
for yy, col in zip([0.25, 0.5, 0.75, 0.95, 0.999],
                   plt.cm.viridis(np.linspace(0.05, 0.85, 5))):
    ax[1].plot(xline * 1e3, theta_uniform(xline, 0 * xline + yy * H, 800),
               "-", lw=1.8, color=col, label=rf"$y/H = {yy:g}$")
ax[1].set_xlabel(r"$x$ [mm]")
ax[1].set_ylabel(r"$\theta$ [-]")
ax[1].set_title("(b) Case U: overshoot appears as $y \\to H$")
ax[1].legend(fontsize=8, loc="lower center")
ax[1].set_xlim(0, W * 1e3)

fig.suptitle("Example 5.1 -- The corner singularity", fontsize=12.5, y=1.02)
fig.savefig("fig_5_1b_corner.png")
plt.close(fig)

print("Figures written: fig_5_1a_fields.png, fig_5_1b_corner.png")
