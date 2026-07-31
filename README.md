# Heat Transfer: Analysis, Computation, and Verification

### *A Finite-Volume Approach with Python*

**Author:** I. L. Ferreira

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/NumPy-required-013243.svg?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-required-8CAAE6.svg?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Matplotlib-required-11557C.svg?logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/status-stable-brightgreen.svg" alt="Status: Stable">
  <img src="https://img.shields.io/badge/method-Finite%20Volume-orange.svg" alt="Finite Volume Method">
  <img src="https://img.shields.io/badge/verification-V%26V%20%26%20UQ-purple.svg" alt="Verification & Validation">
  <img src="https://img.shields.io/badge/chapters-13-informational.svg" alt="13 chapters">
  <img src="https://img.shields.io/badge/programs-39-informational.svg" alt="39 programs">
  <img src="https://img.shields.io/badge/PRs-welcome-informational.svg" alt="PRs Welcome">
</p>

> **Official code companion** to the textbook *Heat Transfer: Analysis, Computation, and Verification — A Finite-Volume Approach with Python*.
> This repository contains **every Python program** developed throughout the book — one analytical solution, one finite-volume solution, and one research-grade verification study for each of the thirteen chapters (**39 self-contained programs** in all).

The programs were written for **educational purposes** — to be read, run, and modified by students — while maintaining **research-grade numerical accuracy**. Each one was executed to produce the exact numbers and figures printed in the book; they depend only on `numpy`, `scipy`, and `matplotlib`, contain no random elements, and are reproducible to the last digit.

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🗂️ Repository Structure](#️-repository-structure)
- [📚 Chapter-by-Chapter Description](#-chapter-by-chapter-description)
- [🧾 Program Index](#-program-index)
- [🛠️ Software Requirements](#️-software-requirements)
- [▶️ Running the Examples](#️-running-the-examples)
- [🧭 Repository Philosophy](#-repository-philosophy)
- [🎓 Educational Objectives](#-educational-objectives)
- [👥 Intended Audience](#-intended-audience)
- [📖 Citation](#-citation)
- [⚖️ License](#️-license)
- [✉️ Contact](#️-contact)
- [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Features

Each chapter ships **three complete, executable programs** of increasing depth, plus the machinery to verify every one of them. Across the repository you will find:

| Category | What is included |
|---|---|
| 🧮 **Analytical solutions** | Exact, closed-form solutions derived and coded directly from the governing equations |
| 🧊 **Finite Volume Method** | Conservative FVM discretizations following the Patankar control-volume philosophy |
| ✅ **Verification against exact solutions** | Every numerical field is compared, point by point, with its analytical counterpart |
| 📉 **Grid convergence studies** | Systematic mesh refinement with measured (observed) order of accuracy |
| 🔬 **Richardson extrapolation** | Estimation of the exact solution and discretization error from a mesh sequence |
| 📐 **Error norm calculations** | Quantitative $L_2$ and $L_\infty$ error measures on every mesh |
| 🖼️ **Publication-quality figures** | Clean, serif, high-DPI Matplotlib figures ready for print |
| 📊 **Scientific visualization** | Temperature fields, convergence plots, boundary layers, radiosity maps, and more |
| 🧾 **Numerical verification** | Method of Manufactured Solutions, Grid Convergence Index (GCI), asymptotic-range checks |
| ⚙️ **Engineering applications** | Every example is anchored in a real device or design decision |

---

## 🗂️ Repository Structure

The repository is organized **one directory per chapter**. Each chapter directory contains the same three programs: an analytical solution, a finite-volume solution of the *same* problem, and an advanced verification study.

```text
heat-transfer-analysis-computation-verification/
│
├── chapter01_python/                 # Introduction
│   ├── ex1_1_analytical.py           # composite-mode furnace wall (exact)
│   ├── ex1_2_fvm.py                  # the same wall by the finite-volume method
│   └── ex1_3_advanced.py             # research-grade verification study
│
├── chapter02_python/                 # Steady One-Dimensional Conduction
│   ├── ex2_1_analytical.py
│   ├── ex2_2_fvm.py
│   └── ex2_3_advanced.py
│
├── chapter03_python/                 # Radial Conduction in Cylinders
├── chapter04_python/                 # Radial Conduction in Spheres
├── chapter05_python/                 # Two-Dimensional Heat Conduction
├── chapter06_python/                 # Transient Heat Conduction
├── chapter07_python/                 # Extended Surfaces (Fins)
├── chapter08_python/                 # Convective Heat Transfer
├── chapter09_python/                 # Thermal Radiation
├── chapter10_python/                 # Heat Exchangers
├── chapter11_python/                 # Phase Change Heat Transfer
├── chapter12_python/                 # Numerical Heat Transfer
│
├── chapter13_python/                 # Verification and Validation
│   ├── ex13_1_analytical.py
│   ├── ex13_2_fvm.py
│   └── ex13_3_advanced.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

> 💡 **Naming convention.** In every `chapterNN_python/` directory, the file `ex<N>_1_analytical.py` is the **analytical** solution, `ex<N>_2_fvm.py` is the **finite-volume** solution of the same problem, and `ex<N>_3_advanced.py` is the **advanced verification** study (order of accuracy, Richardson extrapolation, GCI, and — where relevant — uncertainty quantification). Here `N` is the chapter number, so Chapter 6 contains `ex6_1_analytical.py`, `ex6_2_fvm.py`, and `ex6_3_advanced.py`.

---

## 📚 Chapter-by-Chapter Description

<details open>
<summary><strong>Click to expand / collapse the full chapter guide</strong></summary>

### Chapter 1 — Introduction

- **Topics covered:** Conservation of energy; the heat diffusion equation; the three modes of heat transfer (Fourier conduction, Newton convection, Stefan–Boltzmann radiation); thermophysical properties; dimensionless groups; the finite-volume method in outline.
- **Numerical techniques:** Control-volume integration; first FVM discretization of the 1-D steady equation; scalar root-finding (Brent) for a nonlinear radiative boundary.
- **Analytical methods:** Direct integration of the diffusion equation; linearized-radiation coefficient; thermal-resistance network.
- **Python examples:** `ex1_1_analytical.py` (composite-mode furnace wall), `ex1_2_fvm.py` (same wall by the control-volume method), `ex1_3_advanced.py` (research-grade verification study).
- **Learning objectives:** Understand what heat transfer adds to thermodynamics; read the governing equation term by term; see how a computed result is verified.
- **Engineering applications:** Furnace walls, electronics cooling, energy systems — the framing problems of the whole book.

### Chapter 2 — Steady One-Dimensional Conduction

- **Topics covered:** The plane wall; thermal resistances; internal heat generation; temperature-dependent conductivity; startup transients.
- **Numerical techniques:** FVM with harmonic-mean face conductivity; source-term linearization (Patankar's $S = S_u + S_P T_P$, $S_P \le 0$); tridiagonal (Thomas) solution; Picard iteration.
- **Analytical methods:** Exact parabolic profile; the resistance network; the Kirchhoff transformation for variable conductivity.
- **Python examples:** `ex2_1_analytical.py` (plane wall with uniform generation), `ex2_2_fvm.py` (FVM with a volumetric source), `ex2_3_advanced.py` (advanced verification — startup of the heated plate).
- **Learning objectives:** Build and verify the first complete FVM solver; connect resistances to the discrete equations; measure the observed order of accuracy.
- **Engineering applications:** Building envelopes, furnace linings, fuel plates, insulated pipe walls.

### Chapter 3 — Radial Conduction in Cylinders

- **Topics covered:** The cylindrical shell; critical radius of insulation; generation in a rod; composite cylinders.
- **Numerical techniques:** FVM on a radial mesh with area-weighted faces; the Method of Manufactured Solutions for a transient shell.
- **Analytical methods:** Logarithmic temperature profile; cylindrical resistance; critical-radius analysis.
- **Python examples:** `ex3_1_analytical.py` (hollow and composite cylinders; critical insulation radius), `ex3_2_fvm.py` (FVM in cylindrical coordinates), `ex3_3_advanced.py` (MMS verification of transient radial conduction).
- **Learning objectives:** Extend the plane-wall method to curvilinear control volumes without losing conservation.
- **Engineering applications:** Pipe insulation, current-carrying wires, nuclear fuel pins, annular heat guards.

### Chapter 4 — Radial Conduction in Spheres

- **Topics covered:** The spherical shell; generation in a sphere; critical radius for spheres; solid vs. hollow vs. composite spheres.
- **Numerical techniques:** Spherical FVM reusing the unified geometry family $A(r)\propto r^{m}$ ($m = 2$); transient startup solve.
- **Analytical methods:** $1/r$ temperature profile; spherical resistance and its limits.
- **Python examples:** `ex4_1_analytical.py` (hollow, composite and solid spheres), `ex4_2_fvm.py` (FVM in spherical coordinates and the $A(r)\sim r^m$ family), `ex4_3_advanced.py` (transient startup of a generating solid sphere).
- **Learning objectives:** Show that one control-volume framework spans plane, cylinder, and sphere.
- **Engineering applications:** Pressure vessels, nuclear fuel particles, cryogenic tanks, catalyst pellets.

### Chapter 5 — Two-Dimensional Heat Conduction

- **Topics covered:** The 2-D steady conduction (Poisson) equation; shape factors; mixed boundary conditions on a rectangle.
- **Numerical techniques:** Five-point FVM stencil; sparse linear systems; conjugate-gradient solution; mesh-anisotropy study.
- **Analytical methods:** Separation of variables and Fourier series.
- **Python examples:** `ex5_1_analytical.py` (series solution by separation of variables), `ex5_2_fvm.py` (2-D FVM, and why the tridiagonal solver no longer works), `ex5_3_advanced.py` (MMS, conjugate gradients, and mesh anisotropy).
- **Learning objectives:** Move from tridiagonal to sparse two-dimensional systems; verify a *field*, not just a profile.
- **Engineering applications:** Chip carriers, cooling fins on a base, buried pipes, structural thermal bridges.

### Chapter 6 — Transient Heat Conduction

- **Topics covered:** Lumped capacitance; the Biot and Fourier numbers; one-term and multi-term series; semi-infinite solids; multidimensional product solutions.
- **Numerical techniques:** Time-marching FVM in three geometries ($m = 0, 1, 2$ from one solver); the $\theta$-family of time integrators; temporal order of accuracy.
- **Analytical methods:** Separation of variables in time; error-function solutions; the product (Heisler) solution.
- **Python examples:** `ex6_1_analytical.py` (lumped capacitance, exact series, one-term approximation), `ex6_2_fvm.py` (one solver for $m = 0,1,2$ vs. three exact series), `ex6_3_advanced.py` (semi-infinite solid and the multidimensional product solution).
- **Learning objectives:** Distinguish stability from accuracy; measure temporal order of convergence.
- **Engineering applications:** Quenching and heat treatment, thermal transients in electronics, ground-temperature response.

### Chapter 7 — Extended Surfaces (Fins)

- **Topics covered:** The fin equation; fin efficiency and effectiveness; the four classical fin cases; the corrected-length rule; radiating fins.
- **Numerical techniques:** FVM for the fin equation with a convective sink term; source linearization; a validity study of 1-D fin theory.
- **Analytical methods:** Hyperbolic-function solutions for uniform fins; efficiency correlations.
- **Python examples:** `ex7_1_analytical.py` (four classical fin cases, efficiency, corrected length), `ex7_2_fvm.py` (lateral convection as a linearized sink, and a radiating fin), `ex7_3_advanced.py` (when is one-dimensional fin theory actually valid?).
- **Learning objectives:** Model distributed convective loss; verify a model *and* test its physical assumptions.
- **Engineering applications:** Heat sinks, finned tubes, air-cooled cylinders, compact heat exchangers.

### Chapter 8 — Convective Heat Transfer

- **Topics covered:** Boundary layers; the Blasius and Pohlhausen similarity solutions; the Nusselt/Prandtl/Reynolds groups; internal forced convection and the Graetz problem.
- **Numerical techniques:** Similarity-equation integration (Runge–Kutta with shooting); downstream boundary-layer marching; eigenfunction expansion for the Graetz problem.
- **Analytical methods:** Self-similar boundary-layer theory; eigenfunction (Sturm–Liouville) expansions.
- **Python examples:** `ex8_1_analytical.py` (Blasius/Pohlhausen similarity), `ex8_2_fvm.py` (marching solution of the boundary-layer equations), `ex8_3_advanced.py` (the Graetz problem — expansion, FVM marching, verification campaign).
- **Learning objectives:** Connect wall gradients to heat-transfer coefficients; learn where correlations hold and where they fail.
- **Engineering applications:** Flow over plates and tubes, ducted flow, thermal management of surfaces.

### Chapter 9 — Thermal Radiation

- **Topics covered:** The Planck distribution; Stefan–Boltzmann and Wien laws (from first principles); view factors; the radiosity method; combined conduction–radiation.
- **Numerical techniques:** Numerical integration of the Planck function; the radiosity linear system for enclosures; the $T^4$ nonlinearity handled by Patankar's source rule.
- **Analytical methods:** Blackbody laws derived from Planck; view-factor algebra; closed-form enclosure exchange.
- **Python examples:** `ex9_1_analytical.py` (blackbody radiation from first principles), `ex9_2_fvm.py` (view factors and the radiosity enclosure), `ex9_3_advanced.py` (combined conduction and radiation, verified).
- **Learning objectives:** Treat radiation as a global (dense-matrix) exchange problem, unlike sparse conduction.
- **Engineering applications:** Furnaces, spacecraft thermal control, solar collectors, high-temperature enclosures.

### Chapter 10 — Heat Exchangers

- **Topics covered:** The LMTD method; the effectiveness–NTU method and their identity; parallel-, counter-, and cross-flow arrangements.
- **Numerical techniques:** Boundary-value solution of coupled stream equations; two-dimensional discretization of a crossflow exchanger; robust log-mean evaluation.
- **Analytical methods:** Closed-form effectiveness relations for standard configurations.
- **Python examples:** `ex10_1_analytical.py` (LMTD and ε-NTU, derived, compared, pushed to limits), `ex10_2_fvm.py` (two-stream exchanger as a boundary-value problem), `ex10_3_advanced.py` (crossflow with both fluids unmixed — the correlation put on trial).
- **Learning objectives:** Prove the LMTD ≡ ε-NTU identity numerically; test correlations at their limits.
- **Engineering applications:** Shell-and-tube and plate exchangers, radiators, condensers, evaporators.

### Chapter 11 — Phase Change Heat Transfer

- **Topics covered:** The Stefan problem; latent heat; the enthalpy method; the mushy zone; two-dimensional solidification.
- **Numerical techniques:** Source-based enthalpy method (no front tracking); nonlinear latent-heat source; 2-D solidification where no analytic solution exists.
- **Analytical methods:** The Neumann similarity solution for a moving interface, and its limiting cases.
- **Python examples:** `ex11_1_analytical.py` (Neumann solution of the Stefan problem), `ex11_2_fvm.py` (the enthalpy method without tracking the front), `ex11_3_advanced.py` (two-dimensional solidification in a corner, and how to trust it).
- **Learning objectives:** Handle a moving boundary and a nonlinear latent-heat source conservatively.
- **Engineering applications:** Solidification and casting, thermal energy storage, welding, freezing processes.

### Chapter 12 — Numerical Heat Transfer

- **Topics covered:** Convection–diffusion; the cell Péclet number; central/upwind/hybrid/power-law schemes; boundedness; false diffusion; flux limiters.
- **Numerical techniques:** Full convection–diffusion FVM; the classic oblique-flow test for false diffusion; flux-limiter comparisons; the Method of Manufactured Solutions.
- **Analytical methods:** Exact convection–diffusion profile; manufactured exact solutions.
- **Python examples:** `ex12_1_analytical.py` (convection–diffusion and the schemes that discretize it), `ex12_2_fvm.py` (false diffusion and the limiters that repair it), `ex12_3_advanced.py` (MMS — verifying a code where no exact solution exists).
- **Learning objectives:** Understand why naive schemes oscillate and how boundedness is restored.
- **Engineering applications:** Flow-coupled heat transport, CFD thermal modeling, advection-dominated transport.

### Chapter 13 — Verification and Validation

- **Topics covered:** Verification vs. validation vs. uncertainty quantification; observed order of accuracy; the Grid Convergence Index; the complete error budget.
- **Numerical techniques:** Richardson extrapolation; GCI with a factor of safety; asymptotic-range index; linear and Monte-Carlo uncertainty propagation.
- **Analytical methods:** Manufactured and exact reference solutions used as verification benchmarks.
- **Python examples:** `ex13_1_analytical.py` (solution verification — Richardson, the GCI, and honesty), `ex13_2_fvm.py` (validation — a verified code can solve the wrong problem), `ex13_3_advanced.py` (uncertainty quantification — the complete error budget).
- **Learning objectives:** Assemble the book's methodological thread into an explicit, defensible discipline.
- **Engineering applications:** Any computation whose result will inform a real engineering decision.

</details>

---

## 🧾 Program Index

Every program is self-contained and runs with no command-line arguments. Column headers below map to the three files present in each `chapterNN_python/` directory.

| # | Chapter | `ex<N>_1_analytical.py` | `ex<N>_2_fvm.py` | `ex<N>_3_advanced.py` |
|:--:|---|---|---|---|
| 1 | Introduction | Composite-mode furnace wall | Wall by the control-volume method | Research-grade verification study |
| 2 | Steady 1-D Conduction | Plane wall with generation | FVM with a volumetric source | Startup of the heated plate |
| 3 | Radial Conduction — Cylinders | Hollow/composite cylinders; critical radius | FVM in cylindrical coordinates | MMS: transient radial conduction |
| 4 | Radial Conduction — Spheres | Hollow, composite & solid spheres | FVM in spherical coords; $A(r)\sim r^m$ | Transient startup of a solid sphere |
| 5 | Two-Dimensional Conduction | Separation of variables | 2-D FVM (sparse systems) | MMS, CG solver, mesh anisotropy |
| 6 | Transient Conduction | Lumped, exact series, one-term | One solver for $m = 0,1,2$ | Semi-infinite & product solutions |
| 7 | Extended Surfaces (Fins) | Four classical fin cases | Fin equation with convective sink | Validity of 1-D fin theory |
| 8 | Convective Heat Transfer | Blasius / Pohlhausen similarity | Boundary-layer marching | The Graetz problem, verified |
| 9 | Thermal Radiation | Blackbody from first principles | View factors & radiosity enclosure | Combined conduction–radiation |
| 10 | Heat Exchangers | LMTD and ε-NTU | Two-stream exchanger (BVP) | Crossflow, both fluids unmixed |
| 11 | Phase Change | Neumann Stefan solution | Enthalpy method (no front tracking) | 2-D solidification in a corner |
| 12 | Numerical Heat Transfer | Convection–diffusion & schemes | False diffusion & flux limiters | Method of Manufactured Solutions |
| 13 | Verification & Validation | Richardson & the GCI | Validation vs. verification | Full uncertainty / error budget |

---

## 🛠️ Software Requirements

| Requirement | Version | Purpose |
|---|---|---|
| 🐍 **Python** | 3.9 or newer | Language runtime |
| 🔢 **NumPy** | ≥ 1.20 | Arrays and vectorized numerics |
| 🧪 **SciPy** | ≥ 1.7 | Linear/banded solvers, special functions, integration, root finding |
| 📈 **Matplotlib** | ≥ 3.4 | Publication-quality figures |

Install the dependencies with `pip`:

```bash
pip install numpy scipy matplotlib
```

Or, using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```text
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
```

---

## ▶️ Running the Examples

**1. Clone the repository**

```bash
git clone https://github.com/ileaof/heat-transfer-analysis-computation-verification.git
cd heat-transfer-analysis-computation-verification
```

**2. Create and activate a virtual environment**

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Run any example**

Each program writes its figures to the current working directory, so the tidiest way to run one is from inside its chapter folder:

```bash
# Example: the advanced verification study from Chapter 6
cd chapter06_python
python ex6_3_advanced.py
```

You can also run any program directly from the repository root:

```bash
python chapter02_python/ex2_2_fvm.py
```

Each program is **self-contained**: running it prints its verification report to the terminal (error norms, observed order of accuracy, Richardson/GCI results) and saves its figures as `.png` files. No command-line arguments are required.

> ✅ **Reproducibility check.** Because the programs are deterministic — no random seeds, no hidden state — the numbers you obtain should match those printed in the book exactly.

---

## 🧭 Repository Philosophy

This repository is built on one principle: **a computed number is an opinion until it has been verified.**

- 🔁 **Verification first.** Every numerical solution is checked against an analytical solution whenever one is available, and against a manufactured exact solution (MMS) when one is not.
- 📊 **Measured, not assumed, accuracy.** Convergence is demonstrated by refining the mesh and *measuring* the order of accuracy, then comparing it with the scheme's theoretical order.
- 🧱 **Conservation by construction.** The finite-volume discretizations conserve energy to machine precision on any mesh, coarse or fine.
- 🔬 **Reproducibility.** No random seeds, no hidden state — the printed code produces the printed results.
- 🛡️ **Engineering reliability.** The advanced examples carry the full apparatus — Richardson extrapolation, the Grid Convergence Index, and uncertainty budgets — that a result needs before it can support a real decision.

---

## 🎓 Educational Objectives

The repository was designed to help readers:

- 🌡️ **Understand heat transfer theory** — from the conservation of energy to convection, radiation, and phase change.
- 💻 **Learn scientific programming** — clean, readable, vectorized Python for engineering computation.
- 🧮 **Develop numerical methods** — build finite-volume solvers from first principles.
- 🔎 **Verify computational models** — quantify error, measure convergence, and separate verification from validation.
- 🖼️ **Produce publication-quality simulations** — generate figures and reports suitable for reports, theses, and papers.

---

## 👥 Intended Audience

| Audience | How this repository helps |
|---|---|
| 🔧 **Mechanical Engineers** | Conduction, convection, exchangers, and thermal design |
| ⚗️ **Chemical Engineers** | Transport phenomena, reactors, and process heat transfer |
| ✈️ **Aerospace Engineers** | Radiation, transient heating, and thermal protection |
| 🧱 **Materials Engineers** | Solidification, phase change, and heat treatment |
| ⚡ **Energy Engineers** | Exchangers, storage, and energy-system thermal analysis |
| 🎓 **Graduate students** | A verified, from-scratch numerical-methods foundation |
| 🔬 **Researchers** | A reproducible V&V template for computational heat transfer |
| 👩‍🏫 **Educators** | Ready-to-teach, runnable examples aligned with the textbook |

---

## 📖 Citation

If you use this code in your teaching, research, or published work, please cite the textbook:

```bibtex
@book{Ferreira_HeatTransfer_2026,
  author    = {Ferreira, I. L.},
  title     = {Heat Transfer: Analysis, Computation, and Verification},
  subtitle  = {A Finite-Volume Approach with Python},
  year      = {2026},
  publisher = {<PUBLISHER>},
  address   = {<CITY, COUNTRY>},
  edition   = {1st},
  isbn      = {979-8188453312},
  url       = {https://github.com/ileaof/heat-transfer-analysis-computation-verification}
}
```

---

## ⚖️ License

This project is released under the **MIT License**. You are free to use, modify, and distribute the code, including for commercial purposes, provided the copyright notice and permission notice are preserved. The full text is in the [`LICENSE`](LICENSE) file.

```text
MIT License

Copyright (c) 2026 I. L. Ferreira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ✉️ Contact

| | |
|---|---|
| 👤 **Author** | I. L. Ferreira |
| 🐙 **GitHub** | [@ileaof](https://github.com/ileaof) |
| 📧 **Email** | `ileao@ufpa.br` |

Questions, corrections, and contributions are welcome — please open an [issue](https://github.com/ileaof/heat-transfer-analysis-computation-verification/issues) or submit a pull request.

---

## 🙏 Acknowledgments

This repository is offered in the spirit of **open scientific computing** and **reproducible engineering research**. It reflects the conviction that computational results should be transparent, independently verifiable, and freely available to those who wish to learn from them, build on them, or test them.

Gratitude is due to the pioneers of numerical heat transfer and verification — whose control-volume methods and error-estimation frameworks make disciplined computation possible — and to the open-source scientific Python community, whose tools make that discipline accessible to every student and engineer.

<p align="center"><em>"A number is worth what the evidence for it is worth. Everything else is decoration."</em></p>
