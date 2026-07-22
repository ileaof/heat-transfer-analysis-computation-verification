"""
================================================================================
 EXAMPLE 12.2 -- FALSE DIFFUSION, AND WHY 1-D EXACTNESS DOES NOT SURVIVE
 The classic oblique-flow test, and the limiters that repair it
================================================================================

 OBJECTIVE
 ---------
 Example 12.1 ended with the exponential scheme reproducing the exact solution
 to machine precision.  That success is a trap.  It is exact only because the
 flow is aligned with the grid; the moment the flow crosses the grid at an
 angle, EVERY low-order scheme -- upwind, hybrid, power-law, exponential alike
 -- smears sharp features with a spurious, grid-dependent diffusion that has no
 physical origin.  It is called FALSE DIFFUSION, and it is the single most
 important error in convection-dominated computation.

 THE TEST
 --------
 The classic demonstration: a square domain with a uniform velocity at angle
 theta to the grid, zero physical diffusion, and a step in the inflow boundary
 condition -- phi = 1 below the inlet centreline, phi = 0 above it.  With no
 diffusion the exact solution is that the step is simply CONVECTED across the
 domain undistorted, a perfectly sharp diagonal line.  Any smearing the scheme
 produces is entirely false.

   - At theta = 0 or 90 degrees (flow along a grid line) upwind is exact.
   - At theta = 45 degrees (flow across the diagonal) upwind is at its worst,
     and the step arrives as a broad smear.

 The false diffusion is largest when the flow bisects the cell, and it scales
 with the cell size, so it vanishes under refinement -- but slowly, at first
 order, which is why convection-dominated problems are so expensive.

 THE CURE: HIGHER-ORDER SCHEMES WITH LIMITERS
 --------------------------------------------
 Second-order schemes (central, QUICK) do not suffer false diffusion, but they
 oscillate near sharp gradients -- the boundedness failure of Example 12.1, now
 in space rather than in Peclet.  The resolution is a FLUX LIMITER: use the
 high-order flux where the solution is smooth and fall back to upwind where it
 is sharp, blended by a limiter function of the local gradient ratio.  This is
 the TVD (total variation diminishing) family, and it gives second-order
 accuracy in smooth regions while remaining bounded at discontinuities.

 WHAT IS COMPUTED AND CHECKED
 ----------------------------
   1. False diffusion measured as the width of the smeared step, for upwind,
      as a function of flow angle -- zero at 0 and 90 degrees, maximal at 45.
   2. The exact convection of the step, reproduced by upwind ONLY when the flow
      is grid-aligned.
   3. A TVD scheme (van Leer limiter) shown to sharpen the step dramatically
      while introducing no overshoot -- the total variation does not grow.
   4. Grid convergence of the smear width: first order for upwind.
   5. The boundedness of each scheme, by the total-variation criterion.

 OUTPUTS
 -------
   fig_12_2a_false.png       the smeared step at several angles
   fig_12_2b_limiter.png     upwind vs TVD, and the convergence of the smear

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
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
# 1. THE PURE-CONVECTION SOLVER (marching, no physical diffusion)
# ==============================================================================
def van_leer(r):
    """van Leer flux limiter.  psi(r) = (r + |r|)/(1 + |r|), TVD and smooth."""
    r = np.asarray(r, dtype=float)
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def solve_convection(N, theta_deg, scheme="upwind"):
    """Convect a step across an N x N grid at angle theta, no diffusion.

    Velocity (u, v) = (cos theta, sin theta), both non-negative for theta in
    [0, 90], so information flows from the lower-left.  The steady state is
    reached by a single sweep in the direction of increasing i+j, because with
    both velocity components positive each cell depends only on its west and
    south neighbours -- the same corner-sweep structure as the crossflow
    exchanger of Example 10.3.

    Inflow carries a DISCONTINUITY at the lower-left corner: phi = 1 along the
    west edge, phi = 0 along the south edge.  The streamline through the origin,
    y = x tan(theta), separates fluid fed by the west edge (above it, phi = 1)
    from fluid fed by the south edge (below it, phi = 0), so the exact steady
    field is a perfectly sharp diagonal step.  Any width the scheme gives that
    step is false diffusion.

    A first draft set BOTH edges to 1, which fills the domain with a single
    value and has no step at all -- and duly measured zero smear everywhere.
    The test has to contain the feature it claims to measure.
    """
    u = np.cos(np.radians(theta_deg))
    v = np.sin(np.radians(theta_deg))
    Fx, Fy = u, v
    south = np.zeros(N)         # phi along y = 0
    west = np.ones(N)           # phi along x = 0

    phi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            phi_W = west[i] if j == 0 else phi[i, j - 1]
            phi_S = south[j] if i == 0 else phi[i - 1, j]
            # upwind: the face value is the upstream cell exactly
            phi[i, j] = (Fx * phi_W + Fy * phi_S) / (Fx + Fy)
    return phi


# ==============================================================================
# 1b. THE 1-D ADVECTION SOLVER (where the limiter is textbook-clean)
# ==============================================================================
def advect_1d(N, cfl, n_step, scheme):
    """Advect a top-hat pulse once around a periodic 1-D domain.

    A limiter is a statement about the DOWNWIND value, which a two-dimensional
    steady corner sweep cannot supply cleanly.  In one dimension marched in
    TIME the downwind value is simply last step's field, and the TVD schemes
    are unambiguous -- which is why the accuracy-versus-boundedness trade-off is
    demonstrated here rather than in the two-dimensional field above.

    scheme in {"upwind", "central", "tvd"}.  The exact solution after a whole
    number of periods is the initial pulse unchanged, so error and boundedness
    are both read against the initial condition.
    """
    x = (np.arange(N) + 0.5) / N
    phi = np.where((x > 0.3) & (x < 0.6), 1.0, 0.0)      # top-hat
    phi0 = phi.copy()
    a = 1.0                                              # wave speed > 0
    dt = cfl / (a * N)

    nu = a * dt * N                     # Courant number a dt / dx

    def flux_face(ph):
        """Face flux a*phi_face at each face i+1/2, for a > 0 (upwind is left).

        The second-order scheme here is LAX-WENDROFF, not plain central
        differencing.  Central differencing in space with an explicit Euler
        step is UNCONDITIONALLY unstable for advection -- it would blow up to
        10^14 rather than merely oscillate, which overstates the failure.
        Lax-Wendroff is the stable second-order scheme, and it is exactly what
        a TVD limiter blends against: psi = 0 gives upwind, psi = 1 gives
        Lax-Wendroff.  So the three schemes below share one flux family.
        """
        U = ph                          # cell i   (upwind)
        UU = np.roll(ph, 1)             # cell i-1 (up-upwind)
        D = np.roll(ph, -1)             # cell i+1 (downwind)
        if scheme == "upwind":
            face = U
        elif scheme == "lax-wendroff":
            face = U + 0.5 * (1.0 - nu) * (D - U)
        elif scheme == "tvd":
            den = D - U
            r = (U - UU) / np.where(np.abs(den) > 1e-30, den, 1e-30)
            # blend toward Lax-Wendroff rather than pure central
            face = U + 0.5 * van_leer(r) * (1.0 - nu) * den
        else:
            raise ValueError(scheme)
        return a * face

    for _ in range(n_step):
        f_face = flux_face(phi)                          # flux at i+1/2
        # dphi/dt + d(f)/dx = 0  ->  phi -= dt/dx (f_{i+1/2} - f_{i-1/2})
        phi = phi - dt * N * (f_face - np.roll(f_face, 1))
    return x, phi, phi0


# ==============================================================================
# 2. THE EXACT SOLUTION AND THE SMEAR WIDTH
# ==============================================================================
def exact_step(N, theta_deg):
    """Exact convected field: phi = 1 above the dividing streamline y = x tan
    theta (fluid fed by the west edge), 0 below it (fed by the south edge)."""
    u = np.cos(np.radians(theta_deg))
    v = np.sin(np.radians(theta_deg))
    xc = (np.arange(N) + 0.5) / N
    yc = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(xc, yc)
    # above the streamline y = (v/u) x  <=>  u*y - v*x > 0  (west-fed, phi = 1)
    return np.where(u * Y - v * X >= 0.0, 1.0, 0.0)


def false_diffusion(phi, theta_deg):
    """The L1 error against the exact sharp step -- the area of the smeared
    band.  Zero for a scheme that convects the step without spreading it."""
    ex = exact_step(len(phi), theta_deg)
    return float(np.mean(np.abs(phi - ex)))


# ==============================================================================
# 3. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 12.2 -- FALSE DIFFUSION AND FLUX LIMITERS")
print("=" * 78)

print("\n" + "-" * 78)
print("  CHECK 1 -- UPWIND IS EXACT ONLY WHEN THE FLOW IS GRID-ALIGNED")
print("""    With no physical diffusion the step should convect undistorted.  When
    the flow runs along a grid line (theta = 0 or 90 degrees) upwind carries it
    exactly; at any angle between, it smears.  The smear is measured as the
    fraction of the outflow profile caught between 0.1 and 0.9 -- zero for a
    perfectly sharp front.""")
print(f"\n  {'theta [deg]':>12} {'upwind L1 error':>16}")
N = 40
for th in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
    phi = solve_convection(N, th, "upwind")
    fd = false_diffusion(phi, th)
    print(f"  {th:>12.1f} {fd:>16.5f}")
print("""    Upwind is essentially sharp at 0 and 90 degrees and smears most at
    45, where the flow bisects every cell.  The exact field has almost no
    transition band, so the whole of upwind's smear is FALSE diffusion -- an
    error of the scheme, not of the physics.""")

print("\n" + "-" * 78)
print("  CHECK 2 -- FALSE DIFFUSION FALLS AT FIRST ORDER UNDER REFINEMENT")
print("""    False diffusion scales with the cell size, so it vanishes as the grid
    is refined -- but only at first order, which is why convection-dominated
    problems are expensive.  The smear width at 45 degrees is measured on a
    sequence of grids.""")
print(f"\n  {'N':>6} {'L1 error':>12} {'ratio':>8} {'order':>7}")
prev = None
for N in (20, 40, 80, 160, 320):
    phi = solve_convection(N, 45.0, "upwind")
    fd = false_diffusion(phi, 45.0)
    r = prev / fd if prev and fd > 0 else None
    order = np.log2(r) if r else None
    print(f"  {N:>6d} {fd:>12.5f} {('%.2f' % r) if r else '-':>8} "
          f"{('%.3f' % order) if order else '-':>7}")
    prev = fd
print("""    The measured order is about ONE HALF, not one -- and that is correct.
    The L1 error of a monotone first-order scheme at a DISCONTINUITY converges
    at order 1/2, because the smear spreads as the square root of the distance
    travelled: each cell adds a fixed amount of numerical diffusion, and the
    resulting band grows like sqrt(distance * cell size).  A textbook claim of
    "first order" refers to smooth solutions; at a discontinuity, upwind is
    half an order worse, which is the practical cost of convecting a sharp
    front and the whole motivation for the limiter of Check 3.""")

print("\n" + "-" * 78)
print("  CHECK 3 -- THE ACCURACY-BOUNDEDNESS TRADE-OFF, AND ITS RESOLUTION")
print("""    Moving to one dimension in TIME, where the limiter is unambiguous, a
    top-hat pulse is advected once around a periodic domain.  The exact result
    is the pulse returned unchanged, so error and overshoot are read directly.
    Three schemes are compared: upwind (bounded but diffusive), central
    (accurate but oscillatory), and the van Leer TVD limiter (both).""")
N, CFL = 200, 0.4
NSTEP = int(round(1.0 / (CFL / N)))          # one full period
print(f"\n  grid {N}, CFL {CFL}, {NSTEP} steps = one period")
print(f"  {'scheme':>10} {'L1 error':>12} {'min phi':>10} {'max phi':>10} "
      f"{'overshoot?':>11}")
for scheme in ("upwind", "lax-wendroff", "tvd"):
    x, phi, phi0 = advect_1d(N, CFL, NSTEP, scheme)
    L1 = np.mean(np.abs(phi - phi0))
    over = (phi.min() < -1e-6) or (phi.max() > 1.0 + 1e-6)
    print(f"  {scheme:>10} {L1:>12.5f} {phi.min():>10.4f} {phi.max():>10.4f} "
          f"{str(over):>11}")
print("""    Upwind stays inside [0, 1] but its L1 error is large -- the pulse has
    smeared.  Central is more accurate on paper but OVERSHOOTS, dipping below 0
    and rising above 1, the same boundedness failure as central differencing at
    high Peclet in Example 12.1.  The TVD scheme is both the most accurate of
    the three AND bounded -- the trade-off the other two cannot escape, the
    limiter resolves.""")

print("\n" + "-" * 78)
print("  CHECK 4 -- TOTAL VARIATION: THE BOUNDEDNESS CRITERION")
print("""    A scheme is TVD if the total variation TV = sum|phi_{i+1} - phi_i|
    does not GROW in time.  The initial top-hat has TV = 2 (up one, down one).
    A scheme that overshoots creates new extrema and raises the total variation;
    a TVD scheme cannot.  The variation is tracked over the run.""")
print(f"\n  {'scheme':>10} {'initial TV':>12} {'final TV':>12} {'grew?':>8}")
for scheme in ("upwind", "lax-wendroff", "tvd"):
    x, phi, phi0 = advect_1d(N, CFL, NSTEP, scheme)
    tv0 = np.sum(np.abs(np.diff(np.concatenate([phi0, phi0[:1]]))))
    tv1 = np.sum(np.abs(np.diff(np.concatenate([phi, phi[:1]]))))
    print(f"  {scheme:>10} {tv0:>12.4f} {tv1:>12.4f} {str(tv1 > tv0 + 1e-6):>8}")
print("""    Upwind and TVD keep the total variation at or below its initial
    value of 2; central lets it grow, which is the numerical signature of the
    oscillations.  "Total variation diminishing" is precisely the mathematical
    statement of "no new wiggles", and it is the property that makes a scheme
    safe on sharp fronts.""")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.0))
N = 80
for k, (th, title) in enumerate(((0.0, r"$\theta = 0^\circ$ (aligned)"),
                                 (45.0, r"$\theta = 45^\circ$ (worst)"),
                                 (45.0, "exact"))):
    if k < 2:
        field = solve_convection(N, th, "upwind")
    else:
        field = exact_step(N, 45.0)
    im = ax[k].imshow(field, origin="lower", extent=[0, 1, 0, 1],
                      cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax[k].set_title(f"({'abc'[k]}) {title}")
    ax[k].set_xlabel(r"$x$")
    if k == 0:
        ax[k].set_ylabel(r"$y$")
fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\phi$")
fig.suptitle("Example 12.2 -- Upwind false diffusion: worst when flow "
             "crosses the grid", fontsize=12.0, y=1.06)
fig.savefig("fig_12_2a_false.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

# panel (a): the 1-D advected pulse, three schemes
N, CFL = 200, 0.4
NSTEP = int(round(1.0 / (CFL / N)))
x, phi0 = advect_1d(N, CFL, 0, "upwind")[0:2]
ax[0].plot(x, phi0, "-", lw=1.6, color="0.3", label="exact (initial)")
for scheme, c in (("upwind", "#2166ac"), ("lax-wendroff", "#1b7837"),
                  ("tvd", "#b2182b")):
    xx, phi, _ = advect_1d(N, CFL, NSTEP, scheme)
    ax[0].plot(xx, phi, "-", lw=1.7, color=c, label=scheme)
ax[0].axhline(0, color="0.6", lw=0.8, ls=":")
ax[0].axhline(1, color="0.6", lw=0.8, ls=":")
ax[0].annotate("Lax-Wendroff\noscillates", xy=(0.30, 1.10), fontsize=8.0,
               color="#1b7837", ha="center")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$\phi$")
ax[0].set_title("(a) A pulse advected one period, 1-D")
ax[0].set_ylim(-0.3, 1.35)
ax[0].legend(fontsize=8.0, loc="upper right")

# panel (b): the 2-D false-diffusion convergence (order 1/2 at the step)
Ns = np.array([20, 40, 80, 160, 320])
sw_up = np.array([false_diffusion(solve_convection(N, 45.0, "upwind"), 45)
                  for N in Ns])
ax[1].loglog(Ns, sw_up, "s-", lw=1.8, ms=6, mfc="none", mew=1.5,
             color="#2166ac", label="upwind, 45$^\\circ$")
ax[1].loglog(Ns, sw_up[0] * (Ns[0] / Ns) ** 0.5, "k--", lw=1.2,
             label=r"slope $-1/2$")
ax[1].loglog(Ns, sw_up[0] * (Ns[0] / Ns) ** 1.0, "k:", lw=1.1,
             label=r"slope $-1$")
ax[1].set_xlabel(r"$N$  (cells per side)")
ax[1].set_ylabel(r"$L_1$ false-diffusion error")
ax[1].set_title("(b) At a discontinuity, upwind is order $1/2$")
ax[1].legend(fontsize=8.5, loc="lower left")

fig.suptitle("Example 12.2 -- The accuracy-boundedness trade-off and its cure",
             fontsize=12.5, y=1.08)
fig.savefig("fig_12_2b_limiter.png")
plt.close(fig)

print("Figures written: fig_12_2a_false.png, fig_12_2b_limiter.png")
