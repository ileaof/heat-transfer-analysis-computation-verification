"""
================================================================================
 EXAMPLE 13.3 -- UNCERTAINTY QUANTIFICATION: THE COMPLETE ERROR BUDGET
 Where every kind of error the book has met comes together
================================================================================

 OBJECTIVE
 ---------
 Example 13.1 bounded NUMERICAL error (the discretisation).  Example 13.2
 exposed MODEL error (the wrong equation).  A third kind remains, and in
 practice it usually dominates both: INPUT error, the uncertainty in the data
 fed to the model -- the conductivity, the heat transfer coefficient, the
 boundary temperatures, none of which is ever known exactly.

 This final example assembles the complete error budget of a computed result.
 It takes the verified fin model of Example 13.2, assigns realistic
 uncertainties to its inputs, propagates them to the output two independent
 ways, and combines the input uncertainty with the numerical uncertainty into a
 single honest error bar on the answer.  This is uncertainty quantification, and
 it is the last thing that stands between a number and a decision.

 THE QUANTITY OF INTEREST
 ------------------------
 The heat dissipated by the pin fin of Example 13.2,

        q_fin = sqrt(h P k A) theta_b tanh(mL)        (Chapter 7)

 a function of four uncertain inputs: k, h, the base excess theta_b, and the
 geometry (through P and A).  Each carries an uncertainty; the question is how
 large the resulting uncertainty in q_fin is, and which input dominates it.

 TWO METHODS, CROSS-CHECKED
 --------------------------
   1. LINEAR PROPAGATION (the "delta method").  To first order,

          U_q^2 = sum_i ( dq/dx_i )^2 U_{x_i}^2

      the sensitivities dq/dx_i computed by finite differences on the verified
      model.  Fast, and exact for a linear response; approximate otherwise.

   2. MONTE CARLO.  Sample each input from its distribution, evaluate q_fin for
      each sample, and take the standard deviation of the outputs.  Exact in the
      limit of many samples, for any nonlinearity, and it delivers the whole
      output DISTRIBUTION, not just its width.

 The two must agree when the response is nearly linear, and their agreement is
 the verification of the UQ itself.  Where they disagree, the disagreement
 measures the nonlinearity -- information the linear method cannot provide.

 THE COMPLETE BUDGET
 -------------------
 The total uncertainty combines the input-driven and numerical parts in
 quadrature (they are independent):

        U_total = sqrt(U_input^2 + U_num^2)

 and the example shows which term dominates -- almost always the input, which is
 the quiet lesson of the whole exercise: the grid convergence studies that
 occupy most of a verification effort often refine an answer far past the
 precision its inputs can justify.

 OUTPUTS
 -------
   fig_13_3a_uq.png          the output distribution and the sensitivity budget
   fig_13_3b_budget.png      the complete error budget, input vs numerical

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
# 1. NOMINAL INPUTS AND THEIR UNCERTAINTIES (1-sigma)
# ==============================================================================
#                    nominal       1-sigma      meaning
K0, K_U = 180.0, 9.0            # W/m.K       conductivity, 5 %
H0, H_U = 50.0, 7.5            # W/m^2 K     convection coeff, 15 % (the worst-known)
TB0, TB_U = 100.0, 2.0        # K           base excess, +-2 K
D0, D_U = 0.005, 0.00005      # m           diameter, +-50 um (machining)
L_FIN = 0.05                   # m           length (taken exact)

INPUTS = ("k", "h", "theta_b", "d")
NOMINAL = np.array([K0, H0, TB0, D0])
SIGMA = np.array([K_U, H_U, TB_U, D_U])


# ==============================================================================
# 2. THE QUANTITY OF INTEREST (the verified fin model)
# ==============================================================================
def q_fin(k, h, theta_b, d, Lf=L_FIN):
    """Heat dissipated by an insulated-tip pin fin (Chapter 7), the verified
    model of Example 13.2 evaluated at the fin base."""
    A = np.pi * d ** 2 / 4.0
    P = np.pi * d
    m = np.sqrt(h * P / (k * A))
    M = np.sqrt(h * P * k * A) * theta_b
    return M * np.tanh(m * Lf)


def q_of_vector(v):
    return q_fin(v[0], v[1], v[2], v[3])


# ==============================================================================
# 3. RUN AND VERIFY
# ==============================================================================
print("=" * 78)
print("EXAMPLE 13.3 -- UNCERTAINTY QUANTIFICATION AND THE ERROR BUDGET")
print("=" * 78)
q_nom = q_of_vector(NOMINAL)
print(f"  Nominal fin heat rate q = {q_nom:.6f} W")
print(f"  Inputs (nominal +/- 1 sigma):")
for name, x0, u in zip(INPUTS, NOMINAL, SIGMA):
    print(f"    {name:>8} = {x0:.5g} +/- {u:.3g}  ({100*u/x0:.1f} %)")

print("\n" + "-" * 78)
print("  CHECK 1 -- LINEAR PROPAGATION AND THE SENSITIVITY BUDGET")
print("""    Each input's contribution to the output variance is (dq/dx_i * U_i)^2.
    The sensitivities come from central differences on the model.  The
    contributions must sum (in quadrature) to the total, and their relative
    sizes say which input to measure better.""")
sens = np.zeros(4)
for i in range(4):
    dx = 1e-6 * NOMINAL[i]
    vp = NOMINAL.copy(); vp[i] += dx
    vm = NOMINAL.copy(); vm[i] -= dx
    sens[i] = (q_of_vector(vp) - q_of_vector(vm)) / (2 * dx)
contrib = (sens * SIGMA) ** 2
U_lin = np.sqrt(contrib.sum())
print(f"\n  {'input':>10} {'dq/dx':>13} {'(dq/dx * U)^2':>16} "
      f"{'% of variance':>15}")
for name, s, c in zip(INPUTS, sens, contrib):
    print(f"  {name:>10} {s:>13.4e} {c:>16.4e} {100*c/contrib.sum():>14.1f}%")
print(f"\n    linear-propagation output uncertainty U_lin = {U_lin:.5f} W "
      f"({100*U_lin/q_nom:.2f} %)")
dom = INPUTS[np.argmax(contrib)]
print(f"    the dominant contributor is '{dom}' -- "
      f"{100*contrib.max()/contrib.sum():.0f}% of the variance.")
print("    Reducing any other input's uncertainty barely moves the answer;")
print("    this is where a measurement campaign should spend its money.")

print("\n" + "-" * 78)
print("  CHECK 2 -- MONTE CARLO, AND ITS AGREEMENT WITH LINEAR PROPAGATION")
print("""    Sampling the inputs from independent normals and evaluating the model
    gives the full output distribution.  Its standard deviation must match the
    linear estimate when the response is nearly linear -- the agreement of two
    independent methods is the verification of the UQ.""")
rng = np.random.default_rng(20240711)
print(f"\n  {'samples':>9} {'mean q [W]':>13} {'std q [W]':>12} "
      f"{'vs linear':>11}")
for Nmc in (1000, 10000, 100000):
    samples = rng.normal(NOMINAL, SIGMA, size=(Nmc, 4))
    samples[:, 3] = np.abs(samples[:, 3])       # diameter must stay positive
    qs = np.array([q_of_vector(v) for v in samples])
    print(f"  {Nmc:>9d} {qs.mean():>13.5f} {qs.std(ddof=1):>12.5f} "
          f"{100*(qs.std(ddof=1)/U_lin - 1):>10.2f}%")
# a large final run for the distribution
Nmc = 200000
samples = rng.normal(NOMINAL, SIGMA, size=(Nmc, 4))
samples[:, 3] = np.abs(samples[:, 3])
qs = np.array([q_of_vector(v) for v in samples])
U_mc = qs.std(ddof=1)
print(f"\n    Monte Carlo U_mc = {U_mc:.5f} W, linear U_lin = {U_lin:.5f} W")
print(f"    difference = {100*abs(U_mc/U_lin - 1):.2f} % -- the response is")
print("    nearly linear over this input range, so the cheap linear method is")
print("    trustworthy here.  The Monte Carlo confirms it AND supplies the")
print("    distribution's shape, which the linear method cannot.")

# skewness as a measure of nonlinearity
skew = np.mean(((qs - qs.mean()) / U_mc) ** 3)
print(f"    output skewness = {skew:+.4f} (0 for a linear-Gaussian response)")

print("\n" + "-" * 78)
print("  CHECK 3 -- THE COMPLETE ERROR BUDGET")
print("""    The numerical uncertainty from a grid study (Example 13.1's GCI on
    this model) is combined in quadrature with the input uncertainty.  The point
    of the exercise is the RATIO of the two.""")
# numerical uncertainty: a representative GCI for the fin heat rate.  The tanh
# model is analytic, so the "numerical" part comes from the DISCRETE fin solver
# of Example 13.2; a second-order solve at N = 40 has a GCI of order 0.1 %.
U_num = 0.001 * q_nom            # 0.1 %, a typical converged-grid GCI
U_input = U_mc
U_total = np.sqrt(U_input ** 2 + U_num ** 2)
print(f"\n    input uncertainty     U_input = {U_input:.5f} W "
      f"({100*U_input/q_nom:.2f} %)")
print(f"    numerical uncertainty U_num   = {U_num:.5f} W "
      f"({100*U_num/q_nom:.2f} %)")
print(f"    total uncertainty     U_total = {U_total:.5f} W "
      f"({100*U_total/q_nom:.2f} %)")
print(f"\n    input variance is {100*U_input**2/U_total**2:.1f}% of the total;"
      f" numerical is {100*U_num**2/U_total**2:.1f}%.")
print("""    The input uncertainty dwarfs the numerical one.  This is the quiet
    lesson of the whole chapter: after a code is verified and the grid is
    converged, the answer's honest error bar is set almost entirely by how well
    the INPUTS are known -- and refining the grid further is polishing a number
    whose leading digits are already uncertain for a different reason.  The
    reported result is""")
print(f"      q_fin = {q_nom:.3f} +/- {2*U_total:.3f} W  (95 %, k = 2)")

print(f"\n  CPU time = {time.perf_counter()-T_START:.2f} s")
print("=" * 78)

# ==============================================================================
# 4. FIGURES
# ==============================================================================
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

ax[0].hist(qs, bins=70, density=True, color="#2166ac", alpha=0.55,
           edgecolor="none")
xg = np.linspace(qs.min(), qs.max(), 300)
gauss = np.exp(-0.5 * ((xg - qs.mean()) / U_mc) ** 2) / (U_mc * np.sqrt(2 * np.pi))
ax[0].plot(xg, gauss, "-", lw=2.0, color="#b2182b",
           label="Gaussian fit")
ax[0].axvline(q_nom, color="#1b7837", lw=1.8, ls="--", label="nominal")
ax[0].axvspan(q_nom - 2 * U_total, q_nom + 2 * U_total, color="0.6",
              alpha=0.15)
ax[0].set_xlabel(r"$q_{fin}$  [W]")
ax[0].set_ylabel("probability density")
ax[0].set_title("(a) Output distribution (200k Monte Carlo)")
ax[0].legend(fontsize=8.5, loc="upper right")

# sensitivity budget
order = np.argsort(contrib)[::-1]
labels = [INPUTS[i] for i in order]
vals = [100 * contrib[i] / contrib.sum() for i in order]
colors = ["#b2182b", "#2166ac", "#1b7837", "#e08214"]
ax[1].bar(range(4), vals, color=colors, alpha=0.85)
ax[1].set_xticks(range(4))
ax[1].set_xticklabels([{"k": r"$k$", "h": r"$h$", "theta_b": r"$\theta_b$",
                        "d": r"$d$"}[l] for l in labels])
ax[1].set_ylabel("% of output variance")
ax[1].set_title("(b) Which input to measure better")
for i, v in enumerate(vals):
    ax[1].annotate(f"{v:.0f}%", xy=(i, v), xytext=(0, 4),
                   textcoords="offset points", ha="center", fontsize=9)
ax[1].set_ylim(0, max(vals) * 1.2)

fig.suptitle("Example 13.3 -- Propagating input uncertainty",
             fontsize=12.5, y=1.08)
fig.savefig("fig_13_3a_uq.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

# the budget: input vs numerical, as variances
ax[0].bar([0], [U_input ** 2], 0.6, color="#2166ac", alpha=0.85,
          label="input variance")
ax[0].bar([0], [U_num ** 2], 0.6, bottom=[U_input ** 2], color="#b2182b",
          alpha=0.85, label="numerical variance")
ax[0].set_xlim(-1, 1)
ax[0].set_xticks([])
ax[0].set_ylabel(r"variance  [W$^2$]")
ax[0].set_title("(a) The variance budget: input dominates")
ax[0].legend(fontsize=8.5, loc="upper right")
ax[0].annotate(f"input:\n{100*U_input**2/U_total**2:.1f}%",
               xy=(0, U_input ** 2 * 0.5), ha="center", fontsize=9,
               color="white")

# convergence of the reported uncertainty with MC samples
mc_ns = np.array([100, 300, 1000, 3000, 10000, 30000, 100000])
mc_us = []
for Nm in mc_ns:
    sm = rng.normal(NOMINAL, SIGMA, size=(Nm, 4))
    sm[:, 3] = np.abs(sm[:, 3])
    mc_us.append(np.array([q_of_vector(v) for v in sm]).std(ddof=1))
ax[1].semilogx(mc_ns, mc_us, "o-", lw=1.8, ms=6, mfc="none", mew=1.5,
               color="#2166ac", label="Monte Carlo std")
ax[1].axhline(U_lin, color="#b2182b", ls="--", lw=1.8,
              label="linear propagation")
ax[1].fill_between(mc_ns, U_lin * (1 - 1.4 / np.sqrt(mc_ns)),
                   U_lin * (1 + 1.4 / np.sqrt(mc_ns)), color="0.6",
                   alpha=0.2, label=r"$\pm 1/\sqrt{N}$ band")
ax[1].set_xlabel("Monte Carlo samples")
ax[1].set_ylabel(r"output uncertainty  [W]")
ax[1].set_title("(b) MC converges to the linear estimate")
ax[1].legend(fontsize=8.0, loc="upper right")

fig.suptitle("Example 13.3 -- The complete error budget",
             fontsize=12.5, y=1.08)
fig.savefig("fig_13_3b_budget.png")
plt.close(fig)

print("Figures written: fig_13_3a_uq.png, fig_13_3b_budget.png")
