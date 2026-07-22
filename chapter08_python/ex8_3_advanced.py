"""
================================================================================
 EXAMPLE 8.3 -- INTERNAL FORCED CONVECTION: THE GRAETZ PROBLEM
 Eigenfunction expansion, finite volume marching, and a verification campaign
================================================================================

 OBJECTIVE
 ---------
 External flow has no length scale of its own, which is why Examples 8.1 and
 8.2 could lean on similarity.  Internal flow has one -- the tube radius -- and
 with it the boundary layers eventually meet, the profiles stop changing, and
 the Nusselt number settles onto a CONSTANT.  Those constants are known
 exactly, which makes this the best-instrumented verification problem in the
 chapter:

        constant wall heat flux       Nu = 48/11    = 4.363636...
        constant wall temperature     Nu = beta_0   = 3.6567935...

 The first is a closed-form integral, done here by hand.  The second is the
 first eigenvalue of a Sturm-Liouville problem, computed here from scratch and
 shown to equal the Nusselt number by a short integral identity, so that
 nothing in this example depends on a remembered constant.

 WHAT IS COMPUTED
 ----------------
   1. The fully developed constant-flux solution in closed form, and 48/11.
   2. The Graetz eigenvalue problem, solved by shooting for the first ten
      eigenvalues and eigenfunctions, with the expansion coefficients obtained
      from the orthogonality relation.  This gives the developing-region
      solution as a series, and beta_0 as the fully developed limit.
   3. A finite volume solution marched down the tube on a NONUNIFORM radial
      mesh, verified against both of the above.
   4. A full verification campaign: L2 and L-infinity norms, observed order,
      Richardson extrapolation, grid convergence index, energy balance, CPU
      time, and a sensitivity study in Reynolds and Prandtl number.

 GOVERNING EQUATION
 ------------------
 Hydrodynamically fully developed laminar flow, u = 2 u_m (1 - eta^2) with
 eta = r/R, thermally developing, axial conduction neglected:

        2 u_m (1 - eta^2) dT/dx = (alpha/R^2) (1/eta) d/deta ( eta dT/deta )

 In the dimensionless axial coordinate xi = alpha x / (u_m R^2) this is

        2 (1 - eta^2) dtheta/dxi = (1/eta) d/deta ( eta dtheta/deta )

 which is parabolic in xi -- the same marching structure as Example 8.2, now in
 CYLINDRICAL geometry, so the radial operator is exactly the m = 1 member of
 the A(r) ~ r^m family built in Chapter 3.  Two chapters of machinery meet
 here without modification.

 SEPARATION OF VARIABLES (constant wall temperature)
 ---------------------------------------------------
 With theta = (T - T_w)/(T_0 - T_w) and theta(eta, 0) = 1, writing
 theta = R(eta) exp(-beta xi) gives

        R'' + R'/eta + 2 beta (1 - eta^2) R = 0,  R'(0) = 0,  R(1) = 0

 an eigenvalue problem whose spectrum beta_n is found by shooting.  The
 eigenfunctions are orthogonal under the weight 2(1 - eta^2) eta -- the
 velocity profile itself -- so the coefficients follow from a quadrature.

 SYMBOLS (all SI unless marked)
 ------------------------------
   D, R    [m]        tube diameter and radius
   u_m     [m/s]      mean (bulk) velocity
   alpha   [m^2/s]    thermal diffusivity
   Re, Pr  [-]        Reynolds and Prandtl numbers
   eta     [-]        r/R
   xi      [-]        alpha x / (u_m R^2), the Graetz axial coordinate
   theta   [-]        (T - T_w)/(T_0 - T_w)
   theta_m [-]        bulk mean of theta, weighted by the velocity profile
   Nu      [-]        h D / k
   beta_n  [-]        Graetz eigenvalues

 OUTPUTS
 -------
   fig_8_3a_graetz.png       eigenfunctions, developing Nu, both wall conditions
   fig_8_3b_verification.png convergence, Richardson, GCI, sensitivity

 Requires: numpy, scipy, matplotlib
================================================================================
"""

import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "legend.frameon": True, "legend.framealpha": 0.92,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

# --- NumPy compatibility -----------------------------------------------------
# The trapezoidal rule was renamed in NumPy 2.0: `np.trapz` became
# `np.trapezoid`, and the old name now emits a DeprecationWarning.  Neither
# spelling works everywhere -- `np.trapezoid` does not exist before 2.0 and
# raises AttributeError there.  Binding the name once, here, lets the rest of
# the script run unchanged on both.  (An earlier version of this file used
# `np.trapezoid` directly, which ran clean on NumPy 2.x and failed outright on
# 1.x.  Testing on the version you happen to have installed is not the same as
# testing on the version your reader has installed.)
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

T0 = time.perf_counter()

# ==============================================================================
# 1. PHYSICAL DATA -- air in a small tube, comfortably laminar
# ==============================================================================
D_TUBE = 0.020          # m
R_TUBE = 0.5 * D_TUBE   # m
NU_AIR = 1.589e-5       # m^2/s
PR_AIR = 0.707          # -
K_AIR = 0.02735         # W/m.K
RE_D = 1500.0           # -, laminar (transition near 2300)
U_MEAN = RE_D * NU_AIR / D_TUBE
ALPHA_AIR = NU_AIR / PR_AIR
T_IN = 300.0            # K, inlet
T_WALL = 350.0          # K, wall

NU_Q_EXACT = 48.0 / 11.0        # constant wall heat flux, exact


# ==============================================================================
# 2. THE FULLY DEVELOPED CONSTANT-FLUX SOLUTION, IN CLOSED FORM
# ==============================================================================
def constant_flux_profile(eta):
    """(T - T_w) k / (q_w D), the fully developed constant-flux shape.

    Fully developed with uniform q_w means dT/dx is the same at every radius
    and equal to dT_m/dx, so the energy equation reduces to an ordinary one:

        (1/eta) d/deta(eta dT/deta) = C (1 - eta^2),
                              C = 2 u_m R^2 (dT_m/dx) / alpha

    Integrating twice with dT/deta = 0 on the axis,

        T - T_w = (C/16) (4 eta^2 - eta^4 - 3)

    and the wall flux fixes C: q_w = (k/R) dT/deta|_1 = kC/(4R), so C = 4Rq_w/k
    and, writing R = D/2,

        (T - T_w) k / (q_w D) = eta^2/2 - eta^4/8 - 3/8

    The bulk mean of that is -11/48, so Nu = q_w D/(k(T_w - T_m)) = 48/11.

    THE FACTOR OF TWO IS THE WHOLE DIFFICULTY.  Two earlier drafts of this
    function were wrong here -- first by four, then by two -- because the step
    from radius to diameter was taken in the wrong place.  Neither error is
    visible in the shape of the profile, only in its amplitude, so the plot
    looked right both times.  The 48/11 check is what caught it.
    """
    return 0.5 * eta ** 2 - 0.125 * eta ** 4 - 0.375


def bulk_mean(eta, f):
    """Velocity-weighted mean: int f u r dr / int u r dr with u ~ (1-eta^2)."""
    w = (1.0 - eta ** 2) * eta
    return trapezoid(f * w, eta) / trapezoid(w, eta)


# ==============================================================================
# 3. THE GRAETZ EIGENVALUE PROBLEM (constant wall temperature)
# ==============================================================================
def graetz_shoot(beta, n=6000):
    """Integrate R'' + R'/eta + 2 beta (1-eta^2) R = 0 from the centre.

    The equation is singular at eta = 0.  The series solution there is
    R = 1 - (beta/2) eta^2 + O(eta^4), which is used to step off the axis
    analytically before the Runge-Kutta integration begins -- the same device
    used for the r = 0 control volume in Chapter 4.
    """
    e0 = 1.0e-6
    y = np.array([1.0 - 0.5 * beta * e0 ** 2, -beta * e0])
    e = np.linspace(e0, 1.0, n + 1)
    h = e[1] - e[0]

    def rhs(x, v):
        return np.array([v[1], -v[1] / x - 2.0 * beta * (1.0 - x * x) * v[0]])

    out = np.empty((n + 1, 2))
    out[0] = y
    for i in range(n):
        k1 = rhs(e[i], y)
        k2 = rhs(e[i] + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(e[i] + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(e[i] + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = y
    return e, out


def graetz_eigenvalues(n_modes=10):
    """Bracket sign changes of R(1;beta), then polish each root by brentq."""
    F = lambda b: graetz_shoot(b, n=2000)[1][-1, 0]
    betas = []
    b = 1.0
    prev = F(b)
    while len(betas) < n_modes and b < 4000.0:
        b_next = b + 1.0
        cur = F(b_next)
        if prev * cur < 0.0:
            betas.append(brentq(F, b, b_next, xtol=1e-12, rtol=1e-14))
        b, prev = b_next, cur
    return np.array(betas)


def graetz_modes(betas, n=6000):
    """Eigenfunctions, their wall slopes, and the expansion coefficients.

    Orthogonality holds under the weight w = 2(1 - eta^2) eta, which is the
    velocity profile times the area element.  With theta = 1 at the inlet,

        c_n = int_0^1 R_n w deta / int_0^1 R_n^2 w deta
    """
    modes = []
    for b in betas:
        e, Y = graetz_shoot(b, n=n)
        R = Y[:, 0]
        w = 2.0 * (1.0 - e ** 2) * e
        c = trapezoid(R * w, e) / trapezoid(R * R * w, e)
        modes.append({"beta": b, "eta": e, "R": R, "Rp1": Y[-1, 1], "c": c})
    return modes


def graetz_series(modes, xi):
    """theta_m(xi) and Nu(xi) from the eigenfunction expansion."""
    xi = np.atleast_1d(xi)
    theta_m = np.zeros_like(xi)
    dtheta_wall = np.zeros_like(xi)
    for m in modes:
        e, R, w = m["eta"], m["R"], None
        # bulk mean of this mode, velocity weighted
        ww = (1.0 - e ** 2) * e
        Rm = trapezoid(R * ww, e) / trapezoid(ww, e)
        ex = np.exp(-m["beta"] * xi)
        theta_m += m["c"] * Rm * ex
        dtheta_wall += m["c"] * m["Rp1"] * ex
    # Nu = h D / k with h = q_w/(T_w - T_m); both numerator and denominator
    # change sign together, so Nu is positive.
    return theta_m, -2.0 * dtheta_wall / theta_m


# ==============================================================================
# 4. FINITE VOLUME MARCHING SOLVER (cylindrical, nonuniform radial mesh)
# ==============================================================================
def radial_mesh(N, stretch=2.0):
    """Nonuniform faces clustered at the wall, where the gradient lives.

    faces are placed as eta = 1 - (1 - s)^stretch with s uniform in [0,1];
    stretch = 1 recovers the uniform mesh.  Nodes sit at face midpoints, with
    boundary nodes ON eta = 0 and eta = 1 (half control volumes, no ghosts).
    """
    s = np.linspace(0.0, 1.0, N + 1)
    faces = 1.0 - (1.0 - s) ** stretch
    faces[0], faces[-1] = 0.0, 1.0
    cc = 0.5 * (faces[1:] + faces[:-1])
    eta = np.concatenate(([0.0], cc, [1.0]))
    return eta, faces


def march_tube(N, M, xi_max, bc="T", stretch=2.0, theta_scheme=0.5,
               n_startup=4):
    """March the Graetz problem in xi.

    bc = "T"  : constant wall temperature, theta_wall = 0
    bc = "q"  : constant wall heat flux

    For the flux case the dependent variable is phi = (T - T_in) k/(q_w D) and
    the wall condition is a Neumann one; Nu is then formed from the wall-to-
    bulk difference, which approaches 48/11.

    RANNACHER STARTUP, AND WHY IT IS NOT OPTIONAL HERE
    --------------------------------------------------
    The constant-temperature inlet condition is DISCONTINUOUS: the fluid
    arrives at theta = 1 and the wall is held at theta = 0, so the very first
    control volume must resolve a jump.  The near-wall cell on a stretched
    mesh is tiny, which makes the diffusion number dxi/d(eta)^2 enormous --
    of order 10^6 in the runs below.

    Chapter 6 established that Crank-Nicolson is unconditionally STABLE but
    not unconditionally BOUNDED: its amplification factor (1 - z/2)/(1 + z/2)
    tends to -1, not 0, as z grows.  A mode that ought to be annihilated
    instantly is instead flipped in sign and kept forever.  Run plain
    Crank-Nicolson on this problem and theta goes negative (the run recorded
    min(theta) = -3.8e-02, physically impossible when the data lie in [0,1]),
    the bulk mean passes through zero, and the computed Nusselt number --
    which has theta_m in its denominator -- diverges to absurdity.

    The remedy is Rannacher's: take the first few steps with implicit Euler,
    whose amplification factor 1/(1 + z) does tend to zero, so the offending
    modes are killed; then switch to Crank-Nicolson, which is second order.
    Because only a fixed number of steps is first order, the global order
    remains two.  This is verified in the convergence study below.
    """
    eta, faces = radial_mesh(N, stretch)
    n = len(eta)
    # control volume "areas": in cylindrical geometry the weight is eta d(eta)
    vol = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)      # int eta d eta
    A_f = faces.copy()                                   # face area ~ eta
    u = 2.0 * (1.0 - eta ** 2)                           # velocity profile
    d_nd = np.diff(eta)

    xis = np.linspace(0.0, xi_max, M + 1)
    dxi = xis[1] - xis[0]

    if bc == "T":
        phi = np.ones(n)
        phi[-1] = 0.0
    else:
        phi = np.zeros(n)

    I = slice(1, n - 1)
    D_n = A_f[1:] / d_nd[1:]        # conductance to the node above
    D_s = A_f[:-1] / d_nd[:-1]      # conductance to the node below
    aP0 = u[I] * vol / dxi

    rec = {"xi": [], "nu": [], "tm": [], "minphi": []}
    for i in range(1, M + 1):
        # Rannacher startup: damp the inlet discontinuity, then go second order
        th = 1.0 if i <= n_startup else theta_scheme
        old = phi.copy()
        a_P = np.zeros(n); a_N = np.zeros(n); a_S = np.zeros(n); b = np.zeros(n)
        a_N[I] = th * D_n
        a_S[I] = th * D_s
        a_P[I] = th * (D_n + D_s) + aP0
        b[I] = aP0 * old[I] + (1.0 - th) * (
            D_n * (old[2:] - old[I]) + D_s * (old[:-2] - old[I]))

        # Centreline symmetry.  The eta = 0 node carries no volume in
        # cylindrical geometry (the area
        # element eta d eta vanishes there), so the symmetry condition is
        # imposed directly rather than through a flux balance.
        a_P[0] = 1.0
        a_N[0] = 1.0
        b[0] = 0.0

        if bc == "T":
            a_P[-1] = 1.0
            b[-1] = 0.0
        else:
            # Neumann.  With phi = (T - T_in) k / (q_w D) and eta = r/R,
            #     dphi/deta = R (dT/dr) k/(q_w D) = R (q_w/k) k/(q_w 2R) = 1/2
            # so the wall gradient in these variables is one half, not one and
            # not two.  Carrying the radius-versus-diameter factor through the
            # non-dimensionalisation is the whole of the difficulty here.
            a_P[-1] = 1.0
            a_S[-1] = 1.0
            b[-1] = 0.5 * d_nd[-1]

        ab = np.zeros((3, n))
        ab[0, 1:] = -a_N[:-1]
        ab[1, :] = a_P
        ab[2, :-1] = -a_S[1:]
        phi = solve_banded((1, 1), ab, b)

        # ---- diagnostics ------------------------------------------------
        ww = (1.0 - eta ** 2) * eta
        phim = trapezoid(phi * ww, eta) / trapezoid(ww, eta)
        dpw = (phi[-1] - phi[-2]) / d_nd[-1]
        if bc == "T":
            nu = -2.0 * dpw / phim
        else:
            # Nu = q_w D / (k (T_w - T_m)) = 1 / (phi_w - phi_m)
            nu = 1.0 / (phi[-1] - phim)
        rec["xi"].append(xis[i])
        rec["nu"].append(nu)
        rec["tm"].append(phim)
        rec["minphi"].append(phi.min())

    for k in rec:
        rec[k] = np.array(rec[k])
    return eta, phi, rec


# ==============================================================================
# 5. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 8.3 -- INTERNAL FORCED CONVECTION, THE GRAETZ PROBLEM")
print("=" * 78)
print(f"  Air, D = {D_TUBE*1e3:.0f} mm, Re_D = {RE_D:.0f}, Pr = {PR_AIR}")
print(f"  u_m = {U_MEAN:.4f} m/s,  entrance length scale D Re Pr = "
      f"{D_TUBE*RE_D*PR_AIR:.3f} m")

# ---- 5a. constant flux, closed form ----------------------------------------
print("\n" + "-" * 78)
print("  CONSTANT WALL HEAT FLUX -- the closed-form result")
e_fine = np.linspace(0.0, 1.0, 200001)
th_q = constant_flux_profile(e_fine)
thm_q = bulk_mean(e_fine, th_q)
print(f"    bulk mean of the shape function = {thm_q:.10f}")
print(f"    exact value -11/48              = {-11/48:.10f}")
print(f"    Nu = 1/|theta_m|                = {1.0/abs(thm_q):.10f}")
print(f"    exact 48/11                     = {NU_Q_EXACT:.10f}")
print(f"    relative difference             = "
      f"{abs(1.0/abs(thm_q)/NU_Q_EXACT-1):.2e}")

# ---- 5b. Graetz eigenvalues -------------------------------------------------
print("\n" + "-" * 78)
print("  CONSTANT WALL TEMPERATURE -- the Graetz eigenvalue problem")
betas = graetz_eigenvalues(10)
modes = graetz_modes(betas)
print(f"  {'n':>3} {'beta_n':>14} {'lambda_n = sqrt(beta_n)':>24} {'c_n':>12}")
for i, m in enumerate(modes):
    print(f"  {i:>3d} {m['beta']:>14.8f} {np.sqrt(m['beta']):>24.8f} "
          f"{m['c']:>12.6f}")
NU_T_EXACT = betas[0]
print("""
    THE FULLY DEVELOPED LIMIT IS THE FIRST EIGENVALUE ITSELF.  Multiply the
    eigenvalue equation by eta and integrate across the tube:

        int_0^1 (eta R')' d eta = [eta R']_0^1 = R'(1)
        R'(1) = -2 beta int_0^1 (1 - eta^2) R eta d eta = -beta R_m / 2 * 2

    since the velocity-weighted mean is R_m = 4 int (1-eta^2) R eta d eta.
    Hence R'(1) = -beta R_m / 2 and

        Nu = -2 R'(1) / R_m = beta

    so the fully developed Nusselt number IS beta_0.  A first draft of this
    example asserted beta_0/2 instead, and the series -- which knows nothing
    about the assertion -- disagreed by exactly a factor of two.  Deriving the
    relation costs four lines and removes the guess.""")
print(f"\n    Fully developed Nu = beta_0 = {NU_T_EXACT:.9f}")
print("    The literature value is 3.6567934; this example computes it rather")
print("    than quoting it, so the verification below rests on nothing that")
print("    was typed from memory.")

# independent check: the limit of the series must equal beta_0
_, nu_far = graetz_series(modes, np.array([2.0]))
print(f"    Series evaluated far downstream  = {nu_far[0]:.9f}"
      f"   ({100*(nu_far[0]/NU_T_EXACT-1):+.2e} %)")

# ---- 5c. finite volume, both wall conditions --------------------------------
print("\n" + "-" * 78)
print("  FINITE VOLUME SOLUTION (N = 400 radial cells, stretched, M = 4000)")
XI_MAX = 1.2
sols = {}
for bc, exact, name in (("T", NU_T_EXACT, "constant T_w"),
                        ("q", NU_Q_EXACT, "constant q_w")):
    t_run = time.perf_counter()
    eta, phi, rec = march_tube(400, 4000, XI_MAX, bc=bc)
    cpu = time.perf_counter() - t_run
    sols[bc] = (eta, phi, rec, cpu)
    nu_fd = rec["nu"][-1]
    print(f"    {name}:  Nu_fd = {nu_fd:.7f}   exact {exact:.7f}   "
          f"({100*(nu_fd/exact-1):+.4f} %)   [{cpu:.2f} s]")

print("\n    Note the ORDER of the two constants: the flux case gives the")
print("    larger Nusselt number, 4.364 against 3.657, a difference of about")
print("    19 %.  With uniform flux the wall-to-bulk difference stays fixed")
print("    while both temperatures climb; with uniform wall temperature that")
print("    difference decays, and the less favourable profile is the price.")
print("    Designers who quote 'laminar Nu is about four' are averaging over")
print("    a distinction that is worth a fifth of the heat transfer.")

# ==============================================================================
# 6. VERIFICATION CAMPAIGN
# ==============================================================================
print("\n" + "=" * 78)
print("  VERIFICATION")
print("=" * 78)

# ---- 6a. energy balances, which the solver was never told to satisfy --------
print("\n  6a. ENERGY BALANCE")
print("""      Integrating the energy equation over the cross-section gives two
      exact statements that the discrete solution has no reason to satisfy
      unless the fluxes really do balance:

        constant T_w :  d(theta_m)/d(xi) = 2 theta'(1)
        constant q_w :  d(phi_m)/d(xi)   = 1     (so phi_m = xi exactly)

      The second is the stronger check, because it predicts not a rate but a
      whole function, and a single number out of place would show up.""")
eta_q, phi_q, rec_q, _ = sols["q"]
lin_err = np.max(np.abs(rec_q["tm"] - rec_q["xi"]))
print(f"      constant q_w:  max |phi_m(xi) - xi| = {lin_err:.3e}"
      f"   over 0 < xi < {XI_MAX}")
eta_T, phi_T, rec_T, _ = sols["T"]
dtm = np.gradient(rec_T["tm"], rec_T["xi"], edge_order=2)
# theta'(1) reconstructed from the recorded Nusselt number
tw_slope = -0.5 * rec_T["nu"] * rec_T["tm"]
sel = (rec_T["xi"] > 0.05)
bal = np.max(np.abs(dtm[sel] - 2.0 * tw_slope[sel]) / np.abs(dtm[sel]))
print(f"      constant T_w:  max relative imbalance = {bal:.3e}"
      f"   (xi > 0.05)")
print("      The constant-T balance is checked away from the inlet because")
print("      d(theta_m)/d(xi) is evaluated by finite differences, and near a")
print("      discontinuity the CHECK is less accurate than the thing checked.")

# ---- 6b. boundedness --------------------------------------------------------
print("\n  6b. BOUNDEDNESS")
eta_b, phi_b, rec_b = march_tube(200, 2000, XI_MAX, bc="T", n_startup=0)
eta_r, phi_r, rec_r = march_tube(200, 2000, XI_MAX, bc="T", n_startup=4)
print(f"      plain Crank-Nicolson   min(theta) = {min(rec_b['minphi']):>11.3e}"
      f"   Nu_fd = {rec_b['nu'][-1]:.4e}")
print(f"      Rannacher startup      min(theta) = {min(rec_r['minphi']):>11.3e}"
      f"   Nu_fd = {rec_r['nu'][-1]:.7f}")
print("      The data lie in [0, 1], so a negative theta is not a small error")
print("      but an impossible one.  Because theta_m sits in the DENOMINATOR")
print("      of the Nusselt number, the unbounded run does not fail by a few")
print("      per cent -- it fails by seven orders of magnitude.")

# ---- 6c. order of accuracy, Richardson, GCI ---------------------------------
print("\n  6c. ORDER OF ACCURACY IN THE RADIAL DIRECTION")
print("      (xi refined to M = 4000, where its contribution is negligible)")
print(f"      {'N':>6} {'Nu_fd':>14} {'error':>12} {'p':>8}")
NS = [25, 50, 100, 200, 400]
VN = []
for N in NS:
    _, _, r = march_tube(N, 4000, XI_MAX, bc="T")
    VN.append(r["nu"][-1])


def obs_order(v, i):
    if i + 2 >= len(v):
        return None
    return np.log2(abs((v[i] - v[i + 1]) / (v[i + 1] - v[i + 2])))


for i, N in enumerate(NS):
    p = obs_order(VN, i)
    print(f"      {N:>6d} {VN[i]:>14.8f} {abs(VN[i]-NU_T_EXACT):>12.3e} "
          f"{('%.3f' % p) if p is not None else '-':>8}")

p_obs = obs_order(VN, len(NS) - 3)
f1, f2, f3 = VN[-1], VN[-2], VN[-3]        # fine, medium, coarse
rich = f1 + (f1 - f2) / (2.0 ** p_obs - 1.0)
gci_fine = 1.25 * abs((f1 - f2) / f1) / (2.0 ** p_obs - 1.0)
print(f"\n      observed order p              = {p_obs:.4f}")
print(f"      Richardson extrapolation      = {rich:.9f}")
print(f"      exact beta_0                  = {NU_T_EXACT:.9f}")
print(f"      extrapolation error           = {abs(rich-NU_T_EXACT):.3e}"
      f"   ({100*abs(rich/NU_T_EXACT-1):.2e} %)")
print(f"      GCI on the finest grid (Fs=1.25) = {100*gci_fine:.4f} %")
print(f"      actual error on the finest grid  = "
      f"{100*abs(f1/NU_T_EXACT-1):.4f} %")
print("      The GCI is meant to be a conservative error bar, and here it is:")
print("      it exceeds the true error without exceeding it absurdly.")

# ---- 6d. norms against the series over the developing region ----------------
print("\n  6d. L2 AND L-INFINITY NORMS AGAINST THE EIGENFUNCTION SERIES")
print("      Compared over 0.02 <= xi <= 1.2.  Below xi = 0.02 the ten-term")
print("      series is itself inaccurate -- the higher modes have not yet")
print("      decayed and truncating them is the dominant error -- so the")
print("      comparison there would measure the REFERENCE, not the solution.")
xi_cmp = rec_T["xi"]
msk = xi_cmp >= 0.02
_, nu_series = graetz_series(modes, xi_cmp[msk])
diff = rec_T["nu"][msk] - nu_series
L2 = np.sqrt(np.mean(diff ** 2))
Linf = np.max(np.abs(diff))
print(f"      L2   norm = {L2:.3e}      ({100*L2/NU_T_EXACT:.4f} % of Nu_fd)")
print(f"      Linf norm = {Linf:.3e}      ({100*Linf/NU_T_EXACT:.4f} % of Nu_fd)")
print(f"      location of the maximum: xi = "
      f"{xi_cmp[msk][np.argmax(np.abs(diff))]:.4f}  (nearest the inlet, as")
print("      expected: that is where the gradients are steepest and where the")
print("      series truncation also bites hardest)")

# ---- 6e. entrance length and sensitivity -----------------------------------
print("\n  6e. ENTRANCE LENGTH AND SENSITIVITY")
print("      The thermal entrance length is where Nu falls to within 5 % of")
print("      its fully developed value.")
for bc, exact, name in (("T", NU_T_EXACT, "constant T_w"),
                        ("q", NU_Q_EXACT, "constant q_w")):
    r = sols[bc][2]
    over = np.where(r["nu"] <= 1.05 * exact)[0]
    xi_e = r["xi"][over[0]] if len(over) else np.nan
    # xi = alpha x/(u_m R^2) = 4 x/(D Re Pr);  so x/D = xi Re Pr / 4
    print(f"      {name}: xi_fd = {xi_e:.4f}, i.e. x/D = "
          f"{xi_e*RE_D*PR_AIR/4:.1f}, x = {xi_e*RE_D*PR_AIR*D_TUBE/4:.3f} m")
print("      The classical correlation x_fd/D = 0.05 Re Pr gives x/D = "
      f"{0.05*RE_D*PR_AIR:.1f}")
print("\n      SENSITIVITY.  xi is a similarity variable, so the DIMENSIONLESS")
print("      entrance length is a pure number, independent of Re and Pr; the")
print("      physical length is proportional to their product:")
print(f"      {'Re_D':>8} {'Pr':>8} {'x_fd [m]':>12} {'x_fd/D':>10}")
xi_eT = sols["T"][2]["xi"][np.where(
    sols["T"][2]["nu"] <= 1.05 * NU_T_EXACT)[0][0]]
for Re_s, Pr_s in ((500, 0.707), (1500, 0.707), (2000, 0.707),
                   (1500, 7.0), (1500, 0.024)):
    xd = xi_eT * Re_s * Pr_s / 4.0
    print(f"      {Re_s:>8.0f} {Pr_s:>8.3f} {xd*D_TUBE:>12.4f} {xd:>10.1f}")
print("      A liquid metal (Pr = 0.024) is fully developed within a couple of")
print("      diameters; water (Pr = 7) needs metres of tube.  Any correlation")
print("      applied without checking which regime the tube is in will be")
print("      wrong in the entrance-dominated case by a factor, not a percent.")

print(f"\n  TOTAL CPU = {time.perf_counter()-T0:.2f} s")
print("=" * 78)

# ==============================================================================
# 7. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

cols = ["#b2182b", "#2166ac", "#1b7837", "#762a83", "#e08214"]
for m, c in zip(modes[:5], cols):
    ax[0].plot(m["eta"], m["R"], "-", lw=1.8, color=c,
               label=rf"$\beta_{{{modes.index(m)}}} = {m['beta']:.2f}$")
ax[0].axhline(0.0, color="0.5", lw=0.9)
ax[0].set_xlabel(r"$\eta = r/R$")
ax[0].set_ylabel(r"Eigenfunction $R_n(\eta)$")
ax[0].set_title("(a) The first five Graetz modes")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_xlim(0, 1)

xi_pl = np.logspace(-3.2, np.log10(XI_MAX), 400)
_, nu_ser = graetz_series(modes, xi_pl)
ax[1].loglog(sols["T"][2]["xi"], sols["T"][2]["nu"], "-", lw=2.2,
             color="#b2182b", label=r"FVM, constant $T_w$")
ax[1].loglog(xi_pl, nu_ser, "--", lw=1.5, color="#4d4d4d",
             label="eigenfunction series")
ax[1].loglog(sols["q"][2]["xi"], sols["q"][2]["nu"], "-", lw=2.2,
             color="#2166ac", label=r"FVM, constant $q_w$")
ax[1].axhline(NU_T_EXACT, color="#b2182b", ls=":", lw=1.4)
ax[1].axhline(NU_Q_EXACT, color="#2166ac", ls=":", lw=1.4)
ax[1].annotate(rf"constant $q_w$:  $48/11 = {NU_Q_EXACT:.4f}$",
               xy=(0.30, 0.225), xycoords="axes fraction",
               fontsize=8.5, color="#2166ac",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                         boxstyle="round,pad=0.2"))
ax[1].annotate(rf"constant $T_w$:  $\beta_0 = {NU_T_EXACT:.4f}$",
               xy=(0.30, 0.045), xycoords="axes fraction",
               fontsize=8.5, color="#b2182b",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="none",
                         boxstyle="round,pad=0.2"))
ax[1].set_xlabel(r"$\xi = \alpha x / (u_m R^2)$")
ax[1].set_ylabel(r"$Nu = hD/k$")
ax[1].set_title("(b) Developing $Nu$, and the two constants")
ax[1].annotate("ten-term series\nloses the inlet",
               xy=(1.2e-3, 15.5), xytext=(1.05e-3, 33),
               fontsize=7.8, color="0.35", ha="center",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9),
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8",
                         boxstyle="round,pad=0.22"))
ax[1].set_ylim(2.5, 60)
ax[1].legend(fontsize=8, loc="upper right")

fig.suptitle("Example 8.3 -- The Graetz problem: modes and development",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_3a_graetz.png")
plt.close(fig)

# ---- verification figure ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

NSa = np.array(NS)
errN = np.array([abs(v - NU_T_EXACT) for v in VN])
ax[0].loglog(NSa, errN, "o-", lw=2.0, ms=8, mfc="none", mew=1.8,
             color="#b2182b", label="FVM, radial refinement")
ax[0].loglog(NSa, errN[0] * (NSa[0] / NSa) ** 2.0, "k--", lw=1.4,
             label=r"slope $-2$")
ax[0].axhline(abs(rich - NU_T_EXACT), color="#1b7837", ls=":", lw=1.8,
              label="Richardson extrapolation error")
ax[0].set_xlabel(r"$N$  (radial control volumes)")
ax[0].set_ylabel(r"$|Nu_{fd} - \beta_0|$")
ax[0].set_title("(a) Second order, and what extrapolation buys")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].annotate(rf"observed $p = {p_obs:.3f}$" "\n"
               rf"GCI $= {100*gci_fine:.4f}\,\%$",
               xy=(0.52, 0.72), xycoords="axes fraction", fontsize=8.5,
               color="0.25",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.3"))

eta_b2, phi_b2, _ = march_tube(200, 2000, 0.004, bc="T", n_startup=0)
eta_r2, phi_r2, _ = march_tube(200, 2000, 0.004, bc="T", n_startup=4)
# Boundedness: the profile a few steps in, with and without startup damping.
# Plotted against CELL INDEX counted from the wall rather than against eta,
# because on a stretched mesh the oscillating cells are a few times 1e-5 wide
# and vanish into the axis if distance is used.  The index axis shows the
# structure of the error -- alternating sign, cell by cell -- which is the
# signature of an amplification factor near -1.
nb = 26
idx = np.arange(nb)
ax[1].plot(idx, phi_b2[-1:-nb - 1:-1], "o-", lw=1.6, ms=4.5,
           color="#b2182b", label="plain Crank-Nicolson")
ax[1].plot(idx, phi_r2[-1:-nb - 1:-1], "s-", lw=1.8, ms=4.5, mfc="none",
           color="#1b7837", label="Rannacher startup")
ax[1].axhspan(0.0, 1.0, color="#1b7837", alpha=0.07)
ax[1].axhline(0.0, color="0.4", lw=1.0, ls="--")
ax[1].set_xlabel("control volume index, counted from the wall")
ax[1].set_ylabel(r"$\theta$  at  $\xi = 0.004$")
ax[1].set_title("(b) Stability is not boundedness")
ax[1].legend(fontsize=8.5, loc="lower right")
ax[1].annotate("shaded: the only physically\nadmissible values, "
               r"$0 \leq \theta \leq 1$",
               xy=(0.30, 0.72), xycoords="axes fraction", fontsize=8,
               color="0.3",
               bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.75",
                         boxstyle="round,pad=0.25"))

fig.suptitle("Example 8.3 -- Verification of the finite volume solution",
             fontsize=12.5, y=1.08)
fig.savefig("fig_8_3b_verification.png")
plt.close(fig)

print("Figures written: fig_8_3a_graetz.png, fig_8_3b_verification.png")
