# Physics Guild: Structural Dynamics & Analysis Domain

## Overview
The Physics Guild encompasses the core mathematical engines and physical models that drive the DeVana framework. This domain is responsible for simulating the dynamic behavior of mechanical systems, performing high-fidelity frequency response analysis, and conducting sensitivity studies to understand parameter impact.

## Knowledge Map

### 1. [FRF Engine](FRF_Engine.md)
The central Frequency Response Function (FRF) solver for Multi-Degree-of-Freedom (MDOF) systems.
- **Key Concepts:** Complex matrix inversion, selective DOF elimination, peak/slope characterization.
- **Core Files:** `codes/modules/FRF.py`, `devana/physics/frf.py`.

### 2. [Continuous Beam FEA](Continuous_Beam.md)
Specialized Finite Element Analysis (FEA) for Euler-Bernoulli beams.
- **Key Concepts:** Hermite elements, Transformed Section Method for composite layers, Rayleigh damping.
- **Core Files:** `codes/Continues_beam/backend/model.py`, `devana/physics/beam.py`.

### 3. [Sobol Sensitivity Analysis](Sobol_Sensitivity.md)
Global sensitivity analysis to quantify the influence of design variables on system performance.
- **Key Concepts:** Variance-based decomposition, Saltelli sampling, $S_1$ and $S_T$ indices.
- **Core Files:** `codes/modules/sobol_sensitivity.py`, `devana/sensitivity/sobol.py`.

## Core Mandates
1. **Mathematical Rigor:** All physical laws are expressed in exact LaTeX formulations.
2. **Computational Efficiency:** Use of vectorized operations (NumPy) and parallel processing (Joblib) for high-performance analysis.
3. **Robustness:** Selective DOF elimination and regularized solvers ensure stability even for ill-conditioned systems.
4. **Insight Extraction:** Automated peak detection, bandwidth calculation, and slope analysis transform raw FRF data into actionable engineering metrics.

---
**Maintained by:** Dolores Guild
**Last Updated:** Sunday, June 7, 2026
