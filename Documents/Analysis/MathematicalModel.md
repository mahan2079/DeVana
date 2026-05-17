# Mathematical Model (2DOF-3DOF)

## Overview
DeVana's primary benchmark model is a fully-coupled **2DOF-3DOF** system. It consists of a 2-Degree-of-Freedom main structure attached to three 1-Degree-of-Freedom Dynamic Vibration Absorbers (DVAs). This model captures complex modal interactions and energy transfer phenomena.

## System Components
- **Main Structure**: Two main masses ($M_1, M_2$) connected by springs and dampers.
- **DVAs**: Three absorber masses ($m_1, m_2, m_3$).
- **Coupling Elements**: The system features complete coupling, meaning every mass can be connected to any other mass and to the ground via:
  - **Springs** ($k_i$) for elastic coupling.
  - **Dampers** ($c_i$) for viscous energy dissipation.
  - **Inerters** ($b_i$) for inertial coupling.

## Equations of Motion
The system's behavior is governed by the matrix differential equation:

$$ \mathbf{M} \ddot{\mathbf{q}} + \mathbf{C} \dot{\mathbf{q}} + \mathbf{K} \mathbf{q} = \mathbf{F}(t) $$

Where $\mathbf{q} = [U_1, U_2, u_1, u_2, u_3]^\top$ is the vector of generalized coordinates.

### Dimensionless Formulation
To generalize the analysis, the system is transformed into a dimensionless form:

$$ \mathbf{\bar{M}} \ddot{\mathbf{q}} + 2 \zeta_{dc} \omega_{dc} \mathbf{\bar{C}} \dot{\mathbf{q}} + \omega_{dc}^2 \mathbf{\bar{K}} \mathbf{q} = \mathbf{\bar{F}}(t) $$

**Dimensionless Parameters:**
- Mass Ratios: $\Gamma = M_2/M_1$, $\mu_i = m_i/M_1$
- Inerter Ratios: $\beta_i = b_i/M_1$
- Damping Ratios: $\mathcal{N}_i = C_i/C_1$, $\nu_i = c_i/C_1$
- Stiffness Ratios: $\Lambda_i = K_i/K_1$, $\lambda_i = k_i/K_1$

### Harmonic Response
Assuming harmonic excitation $\mathbf{F}(t) = \mathbf{F} e^{j\omega t}$, the steady-state response $\mathbf{q}(t) = \mathbf{X} e^{j\omega t}$ is solved in the frequency domain:

$$ \mathbf{X} = \omega_{dc}^2 \left( -\Omega^2 \mathbf{\bar{M}} + j 2 \zeta_{dc} \Omega \mathbf{\bar{C}} + \mathbf{\bar{K}} \right)^{-1} \mathbf{\bar{F}} $$

## Benchmark System Parameters
For standard evaluation and optimization testing, DeVana utilizes a reference parameter set where all DVA design variables are bounded within $[0, 1]$.

**Example Baseline Setup:**
- Mass ratio ($\mu$): 1.0
- Stiffness bounds ($\lambda_{\text{LOW}}, \lambda_{\text{UPP}}$): [0.05, 0.95]
- Base Natural Frequency ($\Omega_{\text{DC}}$): 100.0 rad/s
- Base Damping Ratio ($\zeta_{\text{DC}}$): 0.01

**Cost Structure for Optimization:**
- **Springs ($\lambda$)**: Material $3, Manufacturing $12, Operation $1.
- **Dampers ($\nu$)**: Material $7, Manufacturing $25, Maintenance $100, Operation $2.
- **Inerters ($\beta$)**: Material $10, Manufacturing $100, Maintenance $100, Operation $2.
- **Masses ($\mu$)**: Material $10, Manufacturing $20, Operation $5.
