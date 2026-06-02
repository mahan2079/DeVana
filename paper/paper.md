---
title: 'DeVana: An Open-Source Framework for Algorithmic Optimization and Robust Design of Dynamic Vibration Absorbers'
tags:
  - Python
  - structural dynamics
  - vibration control
  - optimization
  - machine learning
  - reinforcement learning
  - evolutionary algorithms
authors:
  - name: Mahan Dashti Gohari
    orcid: 0000-0000-0000-0000 # Placeholder: Replace with actual ORCID if available
    affiliation: 1
affiliations:
 - name: Affiliation Name # Placeholder: Replace with actual affiliation
   index: 1
date: 21 May 2026
bibliography: paper.bib
---

# Summary

Mechanical systems in aerospace, civil, and robotic engineering are frequently subjected to parasitic vibrations that can lead to structural fatigue, noise, and operational failure. Dynamic Vibration Absorbers (DVAs)—auxiliary systems consisting of masses, springs, dampers, and inerters—are widely used to suppress these vibrations. However, the design of multi-component, fully-coupled DVA systems involves navigating high-dimensional, non-convex optimization landscapes that are traditionally explored through manual trial-and-error or engineering intuition.

`DeVana` is an open-source Python framework designed to bridge the gap between theoretical vibration analysis and practical, algorithmic design. It provides a modular suite of state-of-the-art metaheuristics, including Genetic Algorithms (GA), Particle Swarm Optimization (PSO), and CMA-ES [@cma], alongside Reinforcement Learning (RL) agents [@pytorch] to autonomously optimize DVA topologies. A key innovation of `DeVana` is its shift from finding single "optimal points" to defining **manufacturable parameter ranges** through a strict statistical extraction protocol, ensuring that design solutions remain robust under real-world manufacturing tolerances.

# Statement of need

While specialized commercial software exists for Finite Element Analysis (FEA), these tools typically focus on the *analysis* of existing designs rather than the *automated synthesis* of optimal configurations. Engineers designing vibration control systems often face three primary challenges:
1. **Dimensionality**: The number of potential connections and parameter combinations (mass, stiffness, damping, inertance) grows exponentially with system complexity.
2. **Multi-criteria Trade-offs**: Balancing peak vibration suppression against manufacturing cost, total weight, and design sparsity (minimizing the number of active components).
3. **Robustness**: Theoretical optimal values (e.g., an exact stiffness of 145.32 N/m) are often impossible to manufacture. There is a need for tools that identify "safe ranges" where performance is guaranteed.

`DeVana` addresses these needs by providing a comprehensive environment for discrete vibrational models. It leverages `NumPy` [@numpy] and `SciPy` [@scipy] for high-fidelity Frequency Response Function (FRF) computation, incorporating robust linear solvers and automatic degree-of-freedom pruning to handle ill-conditioned systems. The framework also integrates global sensitivity analysis via `SALib` [@salib] to identify the most influential design variables.

# State of the Field

In the domain of structural dynamics, many researchers rely on private, non-standardized scripts in MATLAB or C++. While some libraries offer general-purpose optimization (e.g., `PyGMO`, `SciPy.optimize`), they lack the domain-specific integration required for DVA design, such as automated matrix assembly for coupled systems, specialized objective functions for vibration suppression, and topological prominence filtering for peak detection. `DeVana` fills this gap as a dedicated, extensible framework specifically tailored for the structural dynamics community.

# Mathematics and Objective Function

The behavior of the structural systems in `DeVana` is governed by the matrix differential equation:

$$\mathbf{M} \ddot{\mathbf{q}} + \mathbf{C} \dot{\mathbf{q}} + \mathbf{K} \mathbf{q} = \mathbf{F}(t)$$

where $\mathbf{M}$, $\mathbf{C}$, and $\mathbf{K}$ are the mass, damping, and stiffness matrices, respectively. The framework evaluates candidate designs using a multi-criteria objective function $f(\mathbf{x})$:

$$ f(\mathbf{x}) = w_{\mathrm{FRF}} f_{\mathrm{FRF}}(\mathbf{x}) + w_{\mathrm{sparsity}} f_{\mathrm{sparsity}}(\mathbf{x}) + w_{\mathrm{cost}} f_{\mathrm{cost}}(\mathbf{x}) $$

where $f_{\mathrm{FRF}}$ measures dynamic performance, $f_{\mathrm{sparsity}}$ encourages simpler designs using L1 regularization, and $f_{\mathrm{cost}}$ evaluates economic viability.

# System Identification and Inverse Modeling

A unique capability of `DeVana` is the **PINN Discretisizer**, a hybrid engine that utilizes Physics-Informed Neural Networks (PINNs) to solve the inverse vibration problem. Unlike black-box machine learning, the PINN architecture embeds Newton's Second Law directly into its loss function. By leveraging automatic differentiation to compute exact velocity and acceleration gradients, the framework can "reverse-engineer" a physical system from raw time-domain sensor data (displacement, velocity, and acceleration) to identify the equivalent lumped **M**, **C**, and **K** matrices. This allows for the automated synthesis of discrete models from continuous structures, such as gearboxes or complex mechanical housings.

# Research Impact Statement

`DeVana` has been benchmarked on fully-coupled 2-Degree-of-Freedom systems with multiple 1-Degree-of-Freedom absorbers (the 2DOF-3DOF model). It allows researchers to perform deep statistical extraction—analyzing the top 10% and median 5% of independent optimization runs—to derive reliable parameter bounds. This protocol has demonstrated the ability to significantly reduce resonant peaks while identifying robust regions in the design space that are less sensitive to parameter perturbations.

# AI Usage Disclosure

Generative AI (Gemini CLI) was used in the preparation of this manuscript to assist in synthesizing project documentation and drafting the initial structure of the paper. All technical claims and mathematical formulations have been manually verified by the authors.

# Acknowledgements

The authors would like to thank the structural dynamics community for their feedback during the development of this framework. [Placeholder for specific funding acknowledgements].

# References
