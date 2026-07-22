"""
================================================================================
 EXAMPLE 8.2 -- FINITE VOLUME SOLUTION OF THE BOUNDARY LAYER EQUATIONS
 Downstream marching, with the similarity solution as the reference
================================================================================

 OBJECTIVE
 ---------
 Example 8.1 obtained h for the flat plate by exploiting a similarity variable.
 That trick works only because the flat plate has no length scale.  Change the
 wall temperature along the plate, or the free-stream velocity, and similarity
 is gone while the physics is not.  This example therefore solves the boundary
 layer equations DIRECTLY by the finite volume method, using the similarity
 solution purely as a reference against which the discretisation is measured.

 GOVERNING EQUATIONS (steady, laminar, constant property, no dissipation)
 -----------------------------------------------------------------------
   continuity      d(u)/dx + d(v)/dy = 0
   momentum        u du/dx + v du/dy = nu  d2u/dy2
   energy          u dT/dx + v dT/dy = alpha d2T/dy2

 with u(x,0) = v(x,0) = 0, T(x,0) = T_s, and u -> U, T -> T_inf as y -> inf.

 WHY THIS IS A MARCHING PROBLEM
 ------------------------------
 These equations are PARABOLIC in x.  There is no d2u/dx2 term, because the
 boundary layer approximation discarded it as small; and since u > 0 everywhere
 in the layer, information travels downstream only.  Nothing at station i+1 can
 influence station i.  So x plays exactly the role that TIME played in Chapter
 6, and the whole apparatus built there -- implicit stepping, the theta scheme,
 order of accuracy in the marching variable -- transfers unchanged.  The only
 novelty is that the "time step" is a distance and the "initial condition" is
 the profile at the leading edge.

 THE DISCRETE CONTINUITY TRICK
 -----------------------------
 The transverse velocity v is not an independent unknown; it follows from
 continuity.  It would be possible to integrate continuity analytically and
 evaluate v from a formula.  Doing so is a mistake, and an instructive one.
 Patankar's coefficient sum rule for this equation reads

        a_P = a_N + a_S + a_P^0 + [ (v_n - v_s) + (u_P - u_P^old) dy/dx ]

 and the bracket is precisely the DISCRETE continuity residual.  If v is
 obtained from the discrete continuity equation, the bracket vanishes to
 machine precision, a_P equals the sum of its neighbours plus a_P^0, the scheme
 is exactly conservative, and boundedness is guaranteed.  If v is obtained from
 an analytic formula the bracket is merely small, the sum rule is violated by
 the continuity error, and the scheme leaks momentum at a rate that no grid
 refinement study will cleanly expose.  So v is computed here from the discrete
 equation, cell by cell, marching away from the wall.

 SCHEME
 ------
   y direction : second-order central diffusion; convection by v treated with
                 the upwind form max(+/-F, 0), Patankar's rule, so the
                 coefficients cannot go negative at any cell Peclet number
   x direction : theta scheme.  theta = 1 is implicit Euler (first order),
                 theta = 1/2 is Crank-Nicolson (second order).  Both are run,
                 and the observed orders are measured.
   coupling    : Picard iteration at each station -- solve momentum, update v
                 from discrete continuity, repeat -- then solve energy once,
                 since energy does not feed back into momentum for constant
                 properties.
   linear solve: tridiagonal (Thomas), because each station is a 1-D problem

 GRIDS
 -----
 The x grid is uniform in sqrt(x), because the layer grows as sqrt(x) and the
 leading edge is singular.  This concentrates stations where the solution
 changes fastest, at no cost in complexity.

 SYMBOLS (all SI)
 ----------------
   U       [m/s]      free-stream velocity
   L       [m]        plate length
   nu      [m^2/s]    kinematic viscosity
   alpha   [m^2/s]    thermal diffusivity
   T_s     [K]        wall temperature (uniform)
   T_inf   [K]        free-stream temperature
   u, v    [m/s]      streamwise and transverse velocity
   Cf_x    [-]        local skin friction coefficient
   Nu_x    [-]        local Nusselt number

 OUTPUTS
 -------
   fig_8_2a_field.png       computed field and profile collapse onto similarity
   fig_8_2b_convergence.png grid convergence in y and in x

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

# ==============================================================================
# 1. PHYSICAL DATA -- air at about 325 K, 1 atm
# ==============================================================================
U_INF = 5.0             # m/s
L_PLATE = 0.30          # m
NU = 1.589e-5           # m^2/s
PR = 0.707              # -
ALPHA = NU / PR         # m^2/s
K_AIR = 0.02735         # W/m.K
T_S = 350.0             # K
T_INF = 300.0           # K

RE_L = U_INF * L_PLATE / NU

# reference values from the similarity solution (Example 8.1)
FPP0 = 0.3320573362      # f''(0)
ETA99 = 4.9099888        # eta at u/U = 0.99
DELTA_STAR_C = 1.7207876  # displacement constant
# theta'(0) at Pr = 0.707.  This number is COMPUTED by Example 8.1 (both by
# shooting and by quadrature, agreeing to seven figures) and copied here.  An
# earlier draft carried a plausible-looking value typed from memory, 0.2929864,
# which is wrong in the fourth decimal.  The consequence was not a wrong
# answer but something harder to spot: every grid convergence study flattened
# out at an error of about 7e-4 and the observed orders decayed towards zero,
# because the solver was converging correctly towards a target that was off by
# exactly that much.  A convergence study measures the distance to whatever
# reference it is given, and cannot tell a discretisation error from a typo.
THETA_P0_PR = 0.2937211


# ==============================================================================
# 2. GRIDS
# ==============================================================================
def x_grid(Nx, L=L_PLATE):
    """Stations uniform in sqrt(x): fine at the singular leading edge."""
    s = np.linspace(0.0, 1.0, Nx + 1)
    return L * s ** 2


def y_grid(Ny, Ymax):
    """Uniform cell-centred grid with half cells at both boundaries.

    Nodes sit at cell centres except at y = 0 and y = Ymax, where the node is
    ON the boundary -- the half-control-volume treatment used throughout this
    book.  No ghost nodes are introduced anywhere.
    """
    faces = np.linspace(0.0, Ymax, Ny + 1)
    yc = 0.5 * (faces[1:] + faces[:-1])
    y = np.concatenate(([0.0], yc, [Ymax]))
    return y, faces


def domain_height(L=L_PLATE):
    """Far field placed at five times the thickest boundary layer on the plate.

    Both layers must fit.  For Pr < 1 the THERMAL layer is the thicker one, so
    the height is set by whichever of the two is larger -- the mistake caught
    in Example 8.1, avoided here by construction.
    """
    delta_v = ETA99 * np.sqrt(NU * L / U_INF)
    delta_t = delta_v / PR ** (1.0 / 3.0)
    return 5.0 * max(delta_v, delta_t)


# ==============================================================================
# 3. THE STATION SOLVER
# ==============================================================================
def tdma(a_S, a_P, a_N, b):
    """Thomas algorithm via the banded solver.  a_S, a_N are the sub/super
    diagonals already sign-flipped into standard form."""
    n = len(a_P)
    ab = np.zeros((3, n))
    ab[0, 1:] = -a_N[:-1]
    ab[1, :] = a_P
    ab[2, :-1] = -a_S[1:]
    return solve_banded((1, 1), ab, b)


def transport_step(phi_old, u_old, u_now, v_now, y, faces, dx, diff, theta,
                   bc_wall, bc_inf):
    """Advance one scalar (u or T) one station downstream.

    phi_old  profile at the upstream station
    u_old    streamwise velocity at the upstream station (the a_P^0 carrier)
    u_now    streamwise velocity at the current station (for the theta blend)
    v_now    transverse velocity at the current station, face values
    diff     nu for momentum, alpha for energy
    theta    1 = implicit Euler, 0.5 = Crank-Nicolson
    """
    n = len(y)
    dy_cv = np.diff(faces)                 # interior control volume widths
    dy_nd = np.diff(y)                     # node-to-node distances
    a_P = np.zeros(n)
    a_N = np.zeros(n)
    a_S = np.zeros(n)
    b = np.zeros(n)
    I = slice(1, n - 1)                    # the interior nodes

    # --- interior cells, assembled as arrays ------------------------------
    # The cell-by-cell form of this block would read more like the algorithm
    # in the text, but it is one Python loop deep inside two others (Picard
    # inside marching) and it dominated the run time by an order of magnitude.
    # The arithmetic below is identical, index for index.
    D_n = diff / dy_nd[1:]                 # conductance to the north NODE
    D_s = diff / dy_nd[:-1]                # conductance to the south NODE
    F_n = v_now[1:n - 1]                   # flux through the north FACE
    F_s = v_now[0:n - 2]                   # flux through the south FACE

    # Patankar upwind: coefficients stay non-negative at any cell Peclet number
    aN = D_n + np.maximum(-F_n, 0.0)
    aS = D_s + np.maximum(F_s, 0.0)
    aP0 = u_old[I] * dy_cv / dx

    # theta blend: the explicit part uses the same spatial operator
    rhs_expl = (1.0 - theta) * (aN * (phi_old[2:] - phi_old[I]) +
                                aS * (phi_old[:-2] - phi_old[I]))

    a_N[I] = theta * aN
    a_S[I] = theta * aS
    # The term below is the DISCRETE CONTINUITY RESIDUAL.  It is retained
    # explicitly rather than dropped, so that the sum rule can be checked at
    # run time instead of assumed.
    cont = (F_n - F_s) + (u_now[I] - u_old[I]) * dy_cv / dx
    a_P[I] = theta * (aN + aS) + aP0 + theta * cont
    b[I] = aP0 * phi_old[I] + rhs_expl

    # --- boundaries: Dirichlet at the wall and in the free stream ---------
    a_P[0] = 1.0
    b[0] = bc_wall
    a_P[-1] = 1.0
    b[-1] = bc_inf
    return tdma(a_S, a_P, a_N, b), (a_P, a_N, a_S)


def march(Nx, Ny, theta=1.0, n_picard=30, tol=1e-12, Ymax=None, L=L_PLATE,
          probe=(), save_field=False):
    """March the coupled system from the leading edge to x = L.

    The initial condition is the UNIFORM stream, u = U for y > 0 and u = 0 at
    the wall, which is the physically correct statement at x = 0 and is also
    deliberately NOT the similarity profile.  Starting from the answer would
    make the verification circular.  The parabolic system forgets its initial
    condition within a few stations, and the reported quantities are taken at
    x = L, far downstream of any starting transient.
    """
    if Ymax is None:
        Ymax = domain_height(L)
    xs = x_grid(Nx, L)
    y, faces = y_grid(Ny, Ymax)
    n = len(y)
    dy_cv = np.diff(faces)

    u = np.full(n, U_INF)
    u[0] = 0.0
    T = np.full(n, T_INF)
    T[0] = T_S
    v = np.zeros(n - 1)                 # face values, v[j] = v at face j

    hist = {"x": [], "cf": [], "nu": [], "delta": [], "cont": [],
            "sumrule": [], "vinf": []}
    # Stations at which to save a full profile.  Probing ONE marched solution
    # at several x is a stronger similarity test than running several marches:
    # the profiles then share a grid, a domain height and a history, so any
    # spread between them is genuine non-similarity and not a grid artefact.
    probe_at = [int(round(p * Nx)) for p in probe]
    saved = []
    # Saving the whole field costs one array write per station.  Reconstructing
    # it instead by re-marching to each x in turn -- which an earlier version
    # of this script did -- costs O(Nx^2) station solves for no extra accuracy.
    field = np.zeros((n, Nx + 1)) if save_field else None
    fieldT = np.zeros((n, Nx + 1)) if save_field else None
    if save_field:
        field[:, 0], fieldT[:, 0] = u, T

    for i in range(1, Nx + 1):
        dx = xs[i] - xs[i - 1]
        if dx <= 0.0:
            continue
        u_old, T_old = u.copy(), T.copy()
        u_now = u_old.copy()
        v_now = v.copy()

        # ---- Picard loop: momentum and continuity are coupled through v ----
        for it in range(n_picard):
            u_new, coeffs = transport_step(u_old, u_old, u_now, v_now, y,
                                           faces, dx, NU, theta, 0.0, U_INF)
            # --- v from the DISCRETE continuity equation, wall outwards -----
            # v_face(j) = v_face(j-1) - (u_P - u_P^old) dy / dx, started from
            # v = 0 at the wall.  As a running sum this is a cumulative
            # integral, so numpy does it in one call.
            v_new = np.zeros(n - 1)
            v_new[1:n - 1] = -np.cumsum(
                (u_new[1:n - 1] - u_old[1:n - 1]) * dy_cv / dx)
            v_new[-1] = v_new[-2]
            picard_res = np.max(np.abs(u_new - u_now)) / U_INF
            u_now, v_now = u_new, v_new
            if picard_res < tol and it > 0:
                break

        u, v = u_now, v_now
        T, _ = transport_step(T_old, u_old, u, v, y, faces, dx, ALPHA, theta,
                              T_S, T_INF)

        # ---- diagnostics ---------------------------------------------------
        x = xs[i]
        dudy_w = (u[1] - u[0]) / (y[1] - y[0])
        dTdy_w = (T[1] - T[0]) / (y[1] - y[0])
        Re_x = U_INF * x / NU
        cf = 2.0 * NU * dudy_w / U_INF ** 2
        h = -K_AIR * dTdy_w / (T_S - T_INF)
        hist["x"].append(x)
        hist["cf"].append(cf * np.sqrt(Re_x))
        hist["nu"].append(h * x / K_AIR / np.sqrt(Re_x))
        hist["delta"].append(np.interp(0.99 * U_INF, u[:-1], y[:-1]))
        # Picard residual: how far u moved on the LAST coupling iteration.
        # (The discrete continuity residual itself is identically zero by
        # construction -- v is defined by that equation -- so reporting it
        # would be a tautology dressed up as a check.  What can be checked is
        # whether the u used to build v is the u that came back out.)
        hist["cont"].append(picard_res)
        aP, aN, aS = coeffs
        sr = np.max(np.abs(aP[1:-1] - aN[1:-1] - aS[1:-1] -
                           u_old[1:-1] * dy_cv / dx))
        hist["sumrule"].append(sr)
        # v in the free stream: an INDEPENDENT check, since the exact
        # Blasius entrainment velocity is v_inf sqrt(Re_x)/U = beta/2 = 0.8604
        hist["vinf"].append(v[-1] * np.sqrt(Re_x) / U_INF)
        if save_field:
            field[:, i], fieldT[:, i] = u, T
        if i in probe_at:
            saved.append((x / L, y * np.sqrt(U_INF / (NU * x)), u / U_INF,
                          (T - T_S) / (T_INF - T_S)))

    for k in hist:
        hist[k] = np.array(hist[k])
    return xs, y, u, T, v, hist, saved, field, fieldT


# ==============================================================================
# 4. RUN AND VERIFY
# ==============================================================================
t0 = time.perf_counter()
print("=" * 78)
print("EXAMPLE 8.2 -- FVM SOLUTION OF THE BOUNDARY LAYER EQUATIONS")
print("=" * 78)
print(f"  Air at 1 atm:  U = {U_INF} m/s, L = {L_PLATE} m, Pr = {PR}")
print(f"  Re_L = {RE_L:,.0f}  (laminar; transition is near 5e5)")
print(f"  Far field placed at y = {domain_height():.5f} m "
      f"= {domain_height()/np.sqrt(NU*L_PLATE/U_INF):.2f} similarity units")

NX, NY = 400, 200
xs, y, u, T, v, H, prof, _, _ = march(NX, NY, theta=0.5,
                               probe=(0.5, 0.7071, 1.0))

print(f"\n  Grid: {NX} stations x {NY} cells.  "
      f"Marching with theta = 0.5 (Crank-Nicolson)")
print("\n  CONSISTENCY AND CONSERVATION CHECKS")
print(f"    max Picard residual over all stations = {H['cont'].max():.3e}")
print(f"    max |a_P - a_N - a_S - a_P^0|         = {H['sumrule'].max():.3e}")
print("    The second is the Patankar sum rule.  It holds to round-off only")
print("    because v satisfies the DISCRETE continuity equation; had v been")
print("    evaluated from an analytic formula the residual would sit at the")
print("    continuity truncation error instead, and the scheme would leak")
print("    momentum at a rate no grid study would cleanly separate.")
print(f"\n    entrainment velocity  v_inf sqrt(Re_x)/U = {H['vinf'][-1]:.6f}")
print(f"    exact Blasius value   beta/2             = "
      f"{DELTA_STAR_C/2:.6f}   ({100*(H['vinf'][-1]/(DELTA_STAR_C/2)-1):+.3f} %)")
print("    This one is independent: v never appears in the wall gradients")
print("    that produced C_f and Nu, so it tests a different part of the code.")

print("\n  COMPARISON WITH THE SIMILARITY SOLUTION AT x = L")
cf_num, nu_num = H["cf"][-1], H["nu"][-1]
print(f"    C_f sqrt(Re_x):  FVM {cf_num:.6f}   similarity "
      f"{2*FPP0:.6f}   ({100*(cf_num/(2*FPP0)-1):+.3f} %)")
print(f"    Nu_x/sqrt(Re_x): FVM {nu_num:.6f}   similarity "
      f"{THETA_P0_PR:.6f}   ({100*(nu_num/THETA_P0_PR-1):+.3f} %)")
d_num = H["delta"][-1] / np.sqrt(NU * L_PLATE / U_INF)
print(f"    delta_99 (sim units): FVM {d_num:.5f}   similarity "
      f"{ETA99:.5f}   ({100*(d_num/ETA99-1):+.3f} %)")
print("\n    The first two agree to about a tenth of a per cent; delta_99 is")
print("    thirty times worse, and that is a property of the QUANTITY, not of")
print("    the solution.  delta_99 is located by asking where u/U reaches")
print("    0.99, and at that height du/dy has nearly vanished, so a tiny error")
print("    in u displaces the crossing point a long way.  C_f and Nu are read")
print("    at the wall where the gradient is steepest and the inversion is")
print("    best conditioned.  Judging a boundary layer code by delta_99 is")
print("    measuring the sharpness of a knife by how well it cuts water.")

# ---- profile collapse -------------------------------------------------------
print("\n  SELF-SIMILARITY IS AN OUTPUT, NOT AN INPUT.  The solver knows")
print("  nothing about eta; it marches in x.  Plotting u/U against eta at")
print("  several stations should nevertheless collapse the profiles onto one")
print("  curve.  Spread across x/L = " +
      ", ".join(f"{p[0]:.2f}" for p in prof) + " within ONE marched solution:")
grid_eta = np.linspace(0.05, 6.0, 60)
stack = np.array([np.interp(grid_eta, p[1], p[2]) for p in prof])
print(f"    velocity    max spread = {np.max(np.ptp(stack, axis=0)):.3e}")
stackT = np.array([np.interp(grid_eta, p[1], p[3]) for p in prof])
print(f"    temperature max spread = {np.max(np.ptp(stackT, axis=0)):.3e}")
print("\n  A NONZERO SPREAD PROVES NOTHING BY ITSELF.  The upstream stations")
print("  sit in a thinner layer covered by fewer cells, so they carry more")
print("  discretisation error than the downstream ones, and some spread is")
print("  expected.  What distinguishes discretisation error from a genuine")
print("  failure of similarity is whether the spread VANISHES under")
print("  refinement.  It does:")
print(f"  {'Ny':>6} {'velocity spread':>17}")
for Ny_s in (50, 100, 200):
    _, _, _, _, _, _, pr_s, _, _ = march(400, Ny_s, theta=0.5,
                                   probe=(0.5, 0.7071, 1.0))
    st = np.array([np.interp(grid_eta, p[1], p[2]) for p in pr_s])
    print(f"  {Ny_s:>6d} {np.max(np.ptp(st, axis=0)):>17.4e}")
print("  Had the spread settled onto a fixed nonzero value, the marched")
print("  solution would not have been self-similar and the solver would have")
print("  been wrong -- the same diagnostic used on the spherical eigenfunction")
print("  bug in Chapter 6.")

# ==============================================================================
# 5. GRID CONVERGENCE
# ==============================================================================
print("\n" + "-" * 78)
print("  MEASURING ORDER WHEN TWO GRIDS ARE IN PLAY")
print("""
  The discrete solution carries error from BOTH grids,

        f(Nx, Ny) = f_exact + C_x/Nx^q + C_y/Ny^p + ...

  so refining y with x frozen drives the y term down until it disappears
  beneath the frozen x term, after which the measured error stops falling and
  the apparent order collapses.  A first draft of this study did exactly that,
  and at Ny = 400 the error even grew, because the two terms had opposite sign
  and passed through each other.

  The remedy is to take the observed order from SUCCESSIVE DIFFERENCES rather
  than from the error itself:

        f(Ny) - f(2Ny) = C_y/Ny^p (1 - 2^-p)

  The frozen-grid term C_x/Nx^q is identical in both solutions and cancels
  exactly in the subtraction, so the difference isolates the y error no matter
  how large the x error happens to be.  The raw error is still reported, but
  the order is read from the differences.""")

def order_from(vals, i):
    """Observed order from three successive values, via their differences."""
    if i + 2 >= len(vals):
        return None
    return np.log2(abs((vals[i] - vals[i + 1]) / (vals[i + 1] - vals[i + 2])))


print("\n  CONVERGENCE IN y (x frozen at 800 stations, Crank-Nicolson)")
print(f"  {'Ny':>6} {'Nu_x/sqrt(Re)':>15} {'error':>12} {'diff':>12} {'p':>7}")
NYS = [50, 100, 200, 400]
VAL_Y, ERR_Y = [], []
for Ny in NYS:
    _, _, _, _, _, h2, _, _, _ = march(800, Ny, theta=0.5)
    VAL_Y.append(h2["nu"][-1])
    ERR_Y.append(abs(h2["nu"][-1] - THETA_P0_PR))
for i, Ny in enumerate(NYS):
    d = VAL_Y[i] - VAL_Y[i + 1] if i + 1 < len(NYS) else None
    p = order_from(VAL_Y, i)
    print(f"  {Ny:>6d} {VAL_Y[i]:>15.7f} {ERR_Y[i]:>12.3e} "
          f"{('%.3e' % d) if d is not None else '-':>12} "
          f"{('%.3f' % p) if p is not None else '-':>7}")

print("\n  CONVERGENCE IN x (y frozen at 400 cells)")
print(f"  {'Nx':>6} {'Euler':>13} {'p':>7} {'Crank-Nicolson':>16} {'p':>7}")
NXS = [50, 100, 200, 400]
VE, VC, ERR_E, ERR_C = [], [], [], []
for Nx in NXS:
    _, _, _, _, _, he, _, _, _ = march(Nx, 400, theta=1.0)
    _, _, _, _, _, hc, _, _, _ = march(Nx, 400, theta=0.5)
    VE.append(he["nu"][-1])
    VC.append(hc["nu"][-1])
    ERR_E.append(abs(he["nu"][-1] - THETA_P0_PR))
    ERR_C.append(abs(hc["nu"][-1] - THETA_P0_PR))


for i, Nx in enumerate(NXS):
    pe = order_from(VE, i)
    pc = order_from(VC, i)
    print(f"  {Nx:>6d} {VE[i]:>13.7f} "
          f"{('%.3f' % pe) if pe is not None else '-':>7} "
          f"{VC[i]:>16.7f} "
          f"{('%.3f' % pc) if pc is not None else '-':>7}")

print("\n  Implicit Euler marching is first order in x; Crank-Nicolson is")
print("  better but does not reach a clean two, because with y frozen its")
print("  contribution soon falls below the y error.  Neither single-direction")
print("  study can therefore settle the order of the scheme as a whole.")

# ---- the decisive test: refine BOTH directions together ---------------------
print("\n  COMBINED REFINEMENT (Nx = 100m, Ny = 50m, Crank-Nicolson)")
print("  Refining both grids at once is the honest way to state the order of")
print("  a two-dimensional scheme.  Neither error can hide behind the other,")
print("  and the leading-edge region -- where the layer is thinner than one")
print("  cell and no y grid resolves it -- improves in both directions at the")
print("  same time.")
print(f"  {'m':>4} {'Nx':>6} {'Ny':>6} {'Nu_x/sqrt(Re)':>15} {'error':>12} {'p':>7}")
MS = [1, 2, 4, 8, 16]
VAL_C = []
for m in MS:
    _, _, _, _, _, hm, _, _, _ = march(100 * m, 50 * m, theta=0.5)
    VAL_C.append(hm["nu"][-1])
    hf = hm                     # keep the finest for the summary below
for i, m in enumerate(MS):
    p = order_from(VAL_C, i)
    print(f"  {m:>4d} {100*m:>6d} {50*m:>6d} {VAL_C[i]:>15.7f} "
          f"{abs(VAL_C[i]-THETA_P0_PR):>12.3e} "
          f"{('%.3f' % p) if p is not None else '-':>7}")

print("\n  The observed order rises through the table and reaches about two at")
print("  the finest levels, while the error falls by a factor of roughly 280")
print("  across a sixteen-fold refinement.  The coarse grids were simply not")
print("  in the asymptotic range: on them the boundary layer near the leading")
print("  edge is thinner than a single cell over an appreciable length of")
print("  plate, and an unresolved region does not obey the order of the")
print("  scheme applied to it.  Reporting the coarse-grid orders as though")
print("  they measured the discretisation would have understated it.")
print(f"\n  Finest grid, Nx = {100*MS[-1]}, Ny = {50*MS[-1]}, Crank-Nicolson:")
print(f"    Nu_x/sqrt(Re_x) = {hf['nu'][-1]:.7f}   similarity "
      f"{THETA_P0_PR:.7f}   ({100*(hf['nu'][-1]/THETA_P0_PR-1):+.4f} %)")
print(f"    C_f sqrt(Re_x)  = {hf['cf'][-1]:.7f}   similarity "
      f"{2*FPP0:.7f}   ({100*(hf['cf'][-1]/(2*FPP0)-1):+.4f} %)")

print(f"\n  CPU time = {time.perf_counter()-t0:.2f} s")
print("=" * 78)

# ==============================================================================
# 6. FIGURES
# ==============================================================================
# ONE march, with the field saved at every station.
Nxp, Nyp = 300, 160
Ymax = domain_height()
xs_p, yy, _, _, _, _, _, UU, TT = march(Nxp, Nyp, theta=0.5, save_field=True)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
X, Y = np.meshgrid(xs_p, yy)
# pcolormesh, not contourf.  Above the layer the field is uniform to round-off,
# and a contour routine asked to place two dozen levels through constant data
# tiles that whole region with spurious islands.  A mesh plot simply shows the
# values.  The view is also cropped to the region where anything happens: the
# computational domain is five layer thicknesses tall because the far-field
# condition must be applied where the flow is genuinely undisturbed, but
# plotting all of it wastes four fifths of the panel on empty free stream.
cs = ax[0].pcolormesh(X * 1e3, Y * 1e3, TT, cmap="inferno", shading="gouraud",
                      vmin=T_INF, vmax=T_S)
cb = fig.colorbar(cs, ax=ax[0])
cb.set_label(r"$T$  [K]")
dl = ETA99 * np.sqrt(NU * xs_p / U_INF)
ax[0].plot(xs_p * 1e3, dl * 1e3, "w--", lw=1.8,
           label=r"$\delta_{99}$, similarity")
ax[0].plot(xs_p * 1e3, dl * 1e3 / PR ** (1 / 3), "w:", lw=2.0,
           label=r"$\delta_t$, similarity")
ax[0].set_xlabel(r"$x$  [mm]")
ax[0].set_ylabel(r"$y$  [mm]")
ax[0].set_title("(a) Computed thermal field")
leg = ax[0].legend(fontsize=8.5, loc="upper left", framealpha=0.85)
ax[0].set_ylim(0, 1.6 * ETA99 * np.sqrt(NU * L_PLATE / U_INF) * 1e3
               / PR ** (1 / 3))
ax[0].grid(False)

for (frac, eta_s, uu, tt), col in zip(prof,
                                      ["#2166ac", "#1b7837", "#b2182b"]):
    ax[1].plot(uu, eta_s, "-", lw=1.6, color=col,
               label=rf"$u/U$, $x/L = {frac:.2f}$")
    ax[1].plot(tt, eta_s, "--", lw=1.6, color=col, alpha=0.75,
               label=rf"$\theta$,   $x/L = {frac:.2f}$")
ax[1].set_xlabel(r"$u/U$   and   $\theta$")
ax[1].set_ylabel(r"$\eta = y\sqrt{U/\nu x}$")
ax[1].set_title("(b) Profiles collapse without being told to")
ax[1].set_ylim(0, 7)
ax[1].set_xlim(0, 1.04)
ax[1].legend(fontsize=7.5, loc="upper left", ncol=2)
ax[1].annotate("three stations, one curve:\n"
               rf"spread $< {np.max(np.ptp(stack, axis=0)):.1e}$",
               xy=(0.04, 0.52), xycoords="axes fraction", fontsize=8.5,
               color="0.25", ha="left",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))

fig.suptitle("Example 8.2 -- Finite volume boundary layer, marched in $x$",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_2a_field.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
# reuse the numbers already computed in section 5 -- recomputing them would
# double the run time and could not, by construction, give a different answer
Nxs = np.array(NXS)
ee_l, ec_l = np.array(ERR_E), np.array(ERR_C)
ax[0].loglog(Nxs, ee_l, "s-", lw=1.8, ms=7, mfc="none", mew=1.6,
             color="#2166ac", label=r"implicit Euler, $\theta = 1$")
ax[0].loglog(Nxs, ec_l, "o-", lw=1.8, ms=7, mfc="none", mew=1.6,
             color="#b2182b", label=r"Crank-Nicolson, $\theta = 1/2$")
ax[0].loglog(Nxs, ee_l[0] * (Nxs[0] / Nxs) ** 1.0, "k--", lw=1.2,
             label=r"slope $-1$")
ax[0].set_xlabel(r"$N_x$  (marching stations, $N_y$ frozen)")
ax[0].set_ylabel(r"$|Nu_x/\sqrt{Re_x} - $ similarity$|$")
ax[0].set_title(r"(a) The $\theta$ scheme, now in $x$")
ax[0].legend(fontsize=8, loc="lower left")

MSa = np.array(MS)
err_c = np.array([abs(v - THETA_P0_PR) for v in VAL_C])
ndof = 100 * MSa
ax[1].loglog(ndof, err_c, "o-", lw=2.0, ms=8, mfc="none", mew=1.8,
             color="#762a83", label="both grids refined together")
ax[1].loglog(ndof, err_c[0] * (ndof[0] / ndof) ** 2.0, "k--", lw=1.4,
             label=r"slope $-2$")
ax[1].set_xlabel(r"$N_x$   (with $N_y = N_x/2$)")
ax[1].set_ylabel(r"$|Nu_x/\sqrt{Re_x} - $ similarity$|$")
ax[1].set_title("(b) Combined refinement recovers second order")
ax[1].legend(fontsize=9, loc="lower left")
p_last = order_from(VAL_C, len(MS) - 3)
ax[1].annotate(rf"observed $p = {p_last:.2f}$" "\n"
               rf"final error $= {err_c[-1]:.1e}$",
               xy=(0.40, 0.72), xycoords="axes fraction", fontsize=8.5,
               color="0.25",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))

fig.suptitle("Example 8.2 -- Order of accuracy, measured two ways",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_2b_convergence.png")
plt.close(fig)

print("Figures written: fig_8_2a_field.png, fig_8_2b_convergence.png")
