# DeVana: Dynamic Vibration Absorber Optimization Framework

![DeVana Logo](Logo.png)

**Version:** 1.0.0 
**Release Date:** December 2024  
**Author:** Mahan Dashti Gohari  

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Equations of Motion](#equations-of-motion)
  - [Dimensional Form](#dimensional-form)
  - [Dimensionless Form](#dimensionless-form)
- [Performance Criteria](#performance-criteria)
- [Optimization Objectives and Parameters](#optimization-objectives-and-parameters)
  - [Primary System Parameters](#primary-system-parameters)
  - [DVA Parameters](#dva-parameters)
  - [Frequency Range](#frequency-range)
  - [Optimization Objectives](#optimization-objectives)
    - [Mass 1 Objectives](#mass-1-objectives)
    - [Mass 2 Objectives](#mass-2-objectives)
- [Sobol Sensitivity Analysis](#sobol-sensitivity-analysis)
  - [Overview of the Sobol Method](#overview-of-the-sobol-method)
  - [Implementation in DeVana](#implementation-in-devana)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining the System](#defining-the-system)
  - [Configuring DVAs](#configuring-dvas)
  - [Setting Performance Targets](#setting-performance-targets)
  - [Running FRF Analysis](#running-frf-analysis)
  - [Conducting Sobol Sensitivity Analysis](#conducting-sobol-sensitivity-analysis)
  - [Optimizing with Genetic Algorithms](#optimizing-with-genetic-algorithms)
  - [Reviewing and Implementing Optimized Parameters](#reviewing-and-implementing-optimized-parameters)
- [Example Use Case](#example-use-case)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

**DeVana** is an advanced optimization framework designed to model, analyze, and optimize Dynamic Vibration Absorbers (DVAs) within multi-degree-of-freedom (MDOF) mechanical systems. By leveraging robust mathematical models and sophisticated optimization algorithms, DeVana facilitates the design of DVAs that effectively mitigate undesired vibrations, enhance system stability, and prolong the operational lifespan of mechanical structures.

Key applications of DeVana include:

- **Structural Engineering**: Designing DVAs for buildings and bridges to withstand environmental vibrations.
- **Aerospace Engineering**: Optimizing vibration absorbers in aircraft and spacecraft components.
- **Automotive Engineering**: Enhancing vehicle stability and comfort through optimized DVAs.
- **Industrial Machinery**: Reducing mechanical stress and operational noise in heavy machinery.

---

## Key Features

- **Comprehensive Equation of Motion Modeling**:
  - **Dimensional and Dimensionless Forms**: Facilitates both intuitive and scalable analyses.
  - **Coupled System Analysis**: Models interactions between primary and DVA masses for accurate vibration predictions.

- **Performance Optimization**:
  - **Multi-Objective Optimization**: Simultaneously addresses multiple performance criteria for balanced DVA designs.
  - **Genetic Algorithms (GA)**: Utilizes GA for efficient exploration of complex design spaces.

- **Sobol Sensitivity Analysis**:
  - **Variance-Based Global Sensitivity**: Identifies and ranks parameters based on their influence on system performance.
  - **Parameter Prioritization**: Focuses optimization efforts on the most impactful parameters.

- **Customization and Flexibility**:
  - **Customizable Parameters**: Full control over mass, damping, stiffness, and inertial coupling ratios.
  - **Configuration Files**: Easily editable JSON files for system parameters and optimization objectives.

- **Visualization Tools**:
  - **Dynamic Frequency Response Function (FRF) Plots**: Visualizes vibrational responses across frequencies.
  - **Optimization Progress Tracking**: Monitors GA convergence and optimization metrics in real-time.

- **User-Friendly Interface**:
  - **Configuration Tabs**: Organized sections for system parameters, DVA settings, performance targets, and more.
  - **Interactive Plots and Dashboards**: Enhances data interpretation and decision-making.

- **Robust Documentation and Support**:
  - **Comprehensive User Manual**: Detailed guides on installation, configuration, and usage.
  - **Example Use Cases**: Practical scenarios demonstrating DeVana's capabilities.
  - **Community Support**: Open for contributions and collaborative improvements.

---

## Equations of Motion

DeVana models the mechanical system comprising a primary structure and a DVA subsystem through a set of coupled equations of motion. These equations capture the dynamic interactions and responses under external excitations.

### Dimensional Form

The dimensional equations of motion are expressed as:

\[
\mathbf{M} \ddot{\mathbf{q}} + \mathbf{C} \dot{\mathbf{q}} + \mathbf{K} \mathbf{q} = \mathbf{F}(t)
\]

Where:

- **\(\mathbf{q}\)**: Generalized displacement vector, capturing displacements of the primary and DVA masses.
- **\(\mathbf{M}\)**: Mass matrix.
- **\(\mathbf{C}\)**: Damping matrix.
- **\(\mathbf{K}\)**: Stiffness matrix.
- **\(\mathbf{F}(t)\)**: External force vector, including external loads and base motion effects.

#### Generalized Coordinate Vector

\[
\mathbf{q} = 
\begin{bmatrix}
U_1 \\ U_2 \\ u_1 \\ u_2 \\ u_3
\end{bmatrix}
\]

#### Mass Matrix

\[
[M] = 
\begin{bmatrix}
1 + b_1 + b_2 + b_3 & 0 & -b_1 & -b_2 & -b_3 \\
0 & 1 + b_4 + b_5 + b_6 & -b_4 & -b_5 & -b_6 \\
-b_1 & -b_4 & m_1 + b_1 + b_4 + b_7 + b_8 + b_9 + b_{10} & -b_9 & -b_{10} \\
-b_2 & -b_5 & -b_9 & m_2 + b_2 + b_5 + b_9 + b_{11} + b_{12} & -b_{15} \\
-b_3 & -b_6 & -b_{10} & -b_{15} & m_3 + b_3 + b_6 + b_{10} + b_{13} + b_{14} + b_{15}
\end{bmatrix}
\]

#### Damping Matrix

\[
[C] = 
\begin{bmatrix}
C_1 + C_2 + C_3 & -C_3 & -c_1 & -c_2 & -c_3 \\
-C_3 & C_3 + C_4 + C_5 + c_4 + c_5 + c_6 & -c_4 & -c_5 & -c_6 \\
-\nu_1 & -\nu_2 & \nu_1 + \nu_4 + \nu_7 + \nu_8 + \nu_9 + \nu_{10} & -\nu_9 & -\nu_{10} \\
-\nu_2 & -\nu_5 & -\nu_9 & \nu_2 + \nu_5 + \nu_9 + \nu_{11} + \nu_{12} + \nu_{15} & -\nu_{15} \\
-\nu_3 & -\nu_6 & -\nu_{10} & -\nu_{15} & \nu_3 + \nu_6 + \nu_{10} + \nu_{13} + \nu_{14} + \nu_{15}
\end{bmatrix}
\]

#### Stiffness Matrix

\[
[K] = 
\begin{bmatrix}
K_1 + K_2 + K_3 & -K_3 & -k_1 & -k_2 & -k_3 \\
-K_3 & K_3 + K_4 + K_5 + k_4 + k_5 + k_6 & -k_4 & -k_5 & -k_6 \\
-k_1 & -k_2 & k_1 + k_4 + k_7 + k_8 + k_9 + k_{10} & -k_9 & -k_{10} \\
-k_2 & -k_5 & -k_9 & k_2 + k_5 + k_9 + k_{11} + k_{12} + k_{15} & -k_{15} \\
-k_3 & -k_6 & -k_{10} & -k_{15} & k_3 + k_6 + k_{10} + k_{13} + k_{14} + k_{15}
\end{bmatrix}
\]

#### Force Vector

\[
[F] = 
\begin{bmatrix}
F_1(t) + C_1 \dot{U}_{low} + C_2 \dot{U}_{upp} + K_1 U_{low} + K_2 U_{upp} \\
F_2(t) + C_4 \dot{U}_{low} + C_5 \dot{U}_{upp} + K_4 U_{low} + K_5 U_{upp} \\
\beta_7 \ddot{U}_{low} + \beta_8 \ddot{U}_{upp} + 2 \zeta_{dc} \omega_{dc} (\nu_7 \dot{U}_{low} + \nu_8 \dot{U}_{upp}) + \omega_{dc}^2 (\lambda_7 U_{low} + \lambda_8 U_{upp}) \\
\beta_{11} \ddot{U}_{low} + \beta_{12} \ddot{U}_{upp} + 2 \zeta_{dc} \omega_{dc} (\nu_{11} \dot{U}_{low} + \nu_{12} \dot{U}_{upp}) + \omega_{dc}^2 (\lambda_{11} U_{low} + \lambda_{12} U_{upp}) \\
\beta_{13} \ddot{U}_{low} + \beta_{14} \ddot{U}_{upp} + 2 \zeta_{dc} \omega_{dc} (\nu_{13} \dot{U}_{low} + \nu_{14} \dot{U}_{upp}) + \omega_{dc}^2 (\lambda_{13} U_{low} + \lambda_{14} U_{upp})
\end{bmatrix}
\]

### Dimensionless Form

To simplify analysis and improve numerical stability, the system is normalized using dimensionless parameters.

#### Dimensionless Parameters

| **Parameter Group**          | **Parameter** | **Definition**                              |
|------------------------------|---------------|----------------------------------------------|
| **Mass Ratios**              | \( \Gamma \)  | \( \Gamma = \frac{M_2}{M_1} \)              |
|                              | \( \mu_i \)    | \( \mu_i = \frac{m_i}{M_1} \)              |
| **Inertial Coupling Ratios** | \( \beta_i \)  | \( \beta_i = \frac{b_i}{M_1} \)              |
| **Damping Ratios**           | \( \mathcal{N}_i \) | \( \mathcal{N}_i = \frac{C_i}{C_1} \)  |
|                              | \( \nu_i \)    | \( \nu_i = \frac{c_i}{C_1} \)              |
| **Stiffness Ratios**         | \( \Lambda_i \) | \( \Lambda_i = \frac{K_i}{K_1} \)          |
|                              | \( \lambda_i \) | \( \lambda_i = \frac{k_i}{K_1} \)          |
| **Decoupled Primary System** | \( \omega_{dc} \) | \( \omega_{dc} = \sqrt{\frac{K_1}{M_1}} \) |
|                              | \( \zeta_{dc} \)   | \( \zeta_{dc} = \frac{C_1}{2 M_1 \omega_{dc}} \) |

#### Dimensionless Equations of Motion

\[
\mathbf{M} \ddot{\mathbf{q}} + 2 \zeta_{dc} \omega_{dc} \mathbf{C} \dot{\mathbf{q}} + \omega_{dc}^2 \mathbf{K} \mathbf{q} = \mathbf{F}(t)
\]

The dimensionless mass, damping, and stiffness matrices, along with the force vector, are defined as per Equations \eqref{Eq.mass_matrix_dimensionless_combined}, \eqref{Eq.damping_matrix_dimensionless_combined}, \eqref{Eq.stiffness_matrix_dimensionless_combined}, and \eqref{Eq.force_vector_dimensionless_combined} respectively.

---

## Performance Criteria

DeVana optimizes the DVA system based on the following performance criteria:

1. **Minimization of the Area Under the Frequency Response Function (FRF) Curve**:
   
   \[
   A = \int_{\omega_{\min}}^{\omega_{\max}} |H(\omega)| \, d\omega
   \]
   
   - **Objective**: Reduce the total vibrational energy transmitted through the structure.
   - **Benefit**: Decreases mechanical stress and enhances operational efficiency and longevity.

2. **Optimization of Bandwidth Between Resonance Peaks**:
   
   \[
   BW_{i,j} = \omega_j - \omega_i
   \]
   
   - **Objective**: Maximize the separation between consecutive resonance frequencies.
   - **Benefit**: Mitigates resonant amplification within the operational frequency range, increasing system robustness.

3. **FRF Peak Positions and Values**:
   
   \[
   \frac{d|H(\omega)|}{d\omega}\Bigg|_{\omega = \omega_p} = 0, \quad \frac{d^2|H(\omega)|}{d\omega^2}\Bigg|_{\omega = \omega_p} < 0
   \]
   
   - **Objective**: Control the locations and magnitudes of resonance peaks.
   - **Benefit**: Shifts resonances away from undesirable frequencies and prevents excessive amplification.

4. **Optimization of the Slope Between Resonance Peaks**:
   
   \[
   S_{i,j} = \frac{H_{(\omega_j)} - H_{(\omega_i)}}{\omega_j - \omega_i}
   \]
   
   - **Objective**: Ensure uniform peak heights across the FRF curve.
   - **Benefit**: Facilitates balanced energy dissipation, enhancing system stability and robustness.

---

## Optimization Objectives and Parameters

### Primary System Parameters

| **Parameter**               | **Value** |
|-----------------------------|-----------|
| Mass Ratio (\(\Gamma\))     | 2.0       |
| Stiffness Ratios (\(\Lambda_1\), \(\Lambda_2\)) | 0.4, 0.6 |
| Damping Ratios (\(\mathcal{N}_1\), \(\mathcal{N}_2\)) | 0.8, 0.7 |
| External Forcing Amplitudes (\(A_{\text{low}}\), \(A_{\text{upp}}\)) | 0.02 |
| Forcing Function Amplitudes (\(F_1\), \(F_2\)) | 150 |
| Natural Frequency (\(\omega_{\text{dc}}\)) | 8000 rad/s |
| Damping Ratio (\(\zeta_{\text{dc}}\)) | 0.02 |

### DVA Parameters

| **Parameter**               | **Range** | **Description**           |
|-----------------------------|-----------|---------------------------|
| \(\beta_i\) (for \(i=1\) to \(15\)) | 0 to 2.5   | Inertial coupling ratios  |
| \(\lambda_i\) (for \(i=1\) to \(15\)) | 0 to 2.5   | Stiffness ratios          |
| \(\mu_i\) (for \(i=1\) to \(3\))    | 0 to 0.75  | Mass ratios               |
| \(\nu_i\) (for \(i=1\) to \(15\))   | 0 to 2.5   | Damping ratios            |

### Frequency Range

- **Start Frequency (\(\omega_{\text{start}}\))**: 0 rad/s
- **End Frequency (\(\omega_{\text{end}}\))**: 12000 rad/s
- **Number of Frequency Points**: 1500

### Optimization Objectives

The optimization process targets different objectives for each mass within the primary system, with normalized weights ensuring balanced importance.

#### Mass 1 Objectives

1. **Minimize the Maximum Resonant Peaks**:
   - **Target Value**: Peak amplitude at Peak 1 (\(P_1\)) = 0.05
   - **Weight**: 0.5

2. **Clear a Frequency Band Between the Second and Third Peaks**:
   - **Target Value**: Bandwidth (\(BW\)) = 100 rad/s
   - **Weight**: 0.5

| **Objective**                   | **Target Value** | **Weight** |
|---------------------------------|-------------------|------------|
| Peak Amplitude at Peak 1 (\(P_1\)) | 0.05              | 0.5        |
| Bandwidth between Peaks 2 and 3   | 100 rad/s         | 0.5        |

#### Mass 2 Objectives

1. **Minimize the Maximum Resonant Response of the FRF**:
   - **Target Value**: FRF Peak Value = 0.05
   - **Weight**: 0.5

| **Objective**           | **Target Value** | **Weight** |
|-------------------------|-------------------|------------|
| FRF Peak Value          | 0.05              | 0.5        |

#### Rationalization

- **Mass 1**: Emphasizes critical bandwidths and peak amplitudes to ensure operational stability.
- **Mass 2**: Focuses on overall FRF peak minimization to maintain system-wide robustness.

---

## Sobol Sensitivity Analysis

To identify the most influential parameters affecting the DVA's performance across multiple criteria, DeVana employs Sobol sensitivity analysis—a variance-based global sensitivity method.

### Overview of the Sobol Method

Sobol sensitivity analysis decomposes the variance of an output \( Y \) into contributions from each input parameter \( X_i \) and their interactions. For each performance criterion \( C_k \), the total Sobol sensitivity index \( S_{i,k}^T \) for parameter \( X_i \) is calculated as:

\[
S_{i,k}^T = 1 - \frac{\operatorname{Var}_{\mathbf{x}_{\sim i}} \left( \mathbb{E}_{x_i}[C_k \mid \mathbf{x}_{\sim i}] \right)}{\operatorname{Var}(C_k)}
\]

Where:

- \( \operatorname{Var}(C_k) \): Total variance of criterion \( C_k \).
- \( \mathbb{E}_{x_i}[C_k \mid \mathbf{x}_{\sim i}] \): Expected value of \( C_k \) over \( x_i \), conditional on other parameters \( \mathbf{x}_{\sim i} \).
- \( \operatorname{Var}_{\mathbf{x}_{\sim i}} \): Variance over all parameters except \( X_i \).

### Implementation in DeVana

DeVana's Sobol sensitivity analysis involves the following steps:

1. **Sampling**:
   - Generate input parameter combinations within defined ranges using quasi-random sequences (e.g., Sobol sequences) for uniform coverage of the parameter space.

2. **Model Evaluation**:
   - For each sampled set, compute the corresponding output responses, including all optimization objectives.

3. **Variance Decomposition**:
   - Apply Sobol variance decomposition to partition the output variance into contributions from individual parameters and their interactions.

4. **Calculation of Sobol Indices**:
   - Compute the first-order and total-order Sobol indices for each parameter to quantify their individual and collective influences on the output responses.

### Advantages of the Sobol Method

- **Comprehensive Analysis**: Captures both individual parameter effects and their interactions.
- **Global Sensitivity**: Evaluates sensitivity across the entire parameter space, ensuring robust findings.
- **Uncorrelated Inputs**: Assumes input parameters are independent, aligning with DeVana's parameter definitions.
- **Quantitative Metrics**: Provides clear, quantitative measures of sensitivity, facilitating objective decision-making in optimization.

---

## Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)
- **Git**

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/DeVana.git
   cd DeVana
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**

   Ensure that all dependencies are installed correctly by running:

   ```bash
   python main.py --help
   ```

   This should display the help message with available commands and options.

---

## Usage

DeVana offers a structured workflow for defining system parameters, configuring DVAs, setting performance targets, running analyses, and optimizing DVA configurations.

### Defining the System

1. **Navigate to the Configuration File**

   DeVana uses a JSON configuration file located at `config/system_config.json`.

2. **Edit System Parameters**

   Define the primary system parameters:

   ```json
   {
       "mass_ratio": 2.0,
       "stiffness_ratios": [0.4, 0.6],
       "damping_ratios": [0.8, 0.7],
       "external_forcing_amplitudes": {
           "low": 0.02,
           "upper": 0.02
       },
       "forcing_function_amplitudes": {
           "F1": 150,
           "F2": 150
       },
       "natural_frequency": 8000,
       "damping_ratio": 0.02
   }
   ```

### Configuring DVAs

1. **Navigate to the DVA Parameters File**

   Located at `config/dva_config.json`.

2. **Define DVA Parameters**

   ```json
   {
       "dvas": [
           {
               "beta": 0.5,
               "lambda": 0.3,
               "mu": 0.8,
               "nu": 1.2
           },
           {
               "beta": 0.4,
               "lambda": 0.4,
               "mu": 0.7,
               "nu": 1.0
           }
       ]
   }
   ```

   - **beta**: Inertial coupling ratio (\(\beta_i\))
   - **lambda**: Stiffness ratio (\(\lambda_i\))
   - **mu**: Mass ratio (\(\mu_i\))
   - **nu**: Damping ratio (\(\nu_i\))

### Setting Performance Targets

1. **Navigate to the Performance Targets File**

   Located at `config/performance_targets.json`.

2. **Define Objectives and Weights**

   ```json
   {
       "mass1": {
           "objectives": [
               {
                   "name": "peak_amplitude",
                   "target": 0.05,
                   "weight": 0.5
               },
               {
                   "name": "bandwidth",
                   "target": 100,
                   "weight": 0.5
               }
           ]
       },
       "mass2": {
           "objectives": [
               {
                   "name": "frf_peak_value",
                   "target": 0.05,
                   "weight": 0.5
               }
           ]
       }
   }
   ```

### Running FRF Analysis

1. **Execute FRF Analysis**

   ```bash
   python main.py run_frf
   ```

2. **View Results**

   The Frequency Response Function (FRF) plots will be available in the `output/frf_plots/` directory.

### Conducting Sobol Sensitivity Analysis

1. **Configure Sobol Parameters**

   Edit `config/sobol_config.json` if necessary (e.g., sample sizes, number of jobs).

2. **Run Sobol Analysis**

   ```bash
   python main.py run_sobol
   ```

3. **Review Sensitivity Indices**

   Results are saved in `output/sobol_results/`.

### Optimizing with Genetic Algorithms

1. **Configure GA Parameters**

   Edit `config/ga_config.json`:

   ```json
   {
       "population_size": 1000,
       "number_of_generations": 200,
       "crossover_probability": 0.8,
       "mutation_probability": 0.15,
       "tolerance": 0.005,
       "sparsity_penalty": 0.02
   }
   ```

2. **Run GA Optimization**

   ```bash
   python main.py run_ga
   ```

3. **Monitor Optimization Progress**

   Progress logs are displayed in the console and saved in `output/ga_logs/`.

### Reviewing and Implementing Optimized Parameters

1. **Access GA Results**

   Optimized parameters are available in `output/ga_results/`.

2. **Update Configuration Files**

   Replace parameters in `config/dva_config.json` with optimized values.

3. **Validate Optimized System**

   Re-run FRF Analysis to ensure performance improvements.

   ```bash
   python main.py run_frf
   ```

---

## Example Use Case

### Scenario: Optimizing a Multi-DVA System for a Building Structure

**Objective**: Design a multi-DVA system to mitigate vibrations in a high-rise building subjected to wind-induced excitations. The goal is to reduce peak amplitudes at critical frequencies and broaden the effective damping bandwidth.

**Steps**:

1. **Define Main System Parameters**:

   Edit `config/system_config.json`:

   ```json
   {
       "mass_ratio": 2.0,
       "stiffness_ratios": [0.4, 0.6],
       "damping_ratios": [0.8, 0.7],
       "external_forcing_amplitudes": {
           "low": 0.02,
           "upper": 0.02
       },
       "forcing_function_amplitudes": {
           "F1": 150,
           "F2": 150
       },
       "natural_frequency": 8000,
       "damping_ratio": 0.02
   }
   ```

2. **Configure DVAs**:

   Edit `config/dva_config.json`:

   ```json
   {
       "dvas": [
           {
               "beta": 0.5,
               "lambda": 0.3,
               "mu": 0.8,
               "nu": 1.2
           },
           {
               "beta": 0.4,
               "lambda": 0.4,
               "mu": 0.7,
               "nu": 1.0
           }
       ]
   }
   ```

3. **Set Performance Targets**:

   Edit `config/performance_targets.json`:

   ```json
   {
       "mass1": {
           "objectives": [
               {
                   "name": "peak_amplitude",
                   "target": 0.05,
                   "weight": 0.5
               },
               {
                   "name": "bandwidth",
                   "target": 100,
                   "weight": 0.5
               }
           ]
       },
       "mass2": {
           "objectives": [
               {
                   "name": "frf_peak_value",
                   "target": 0.05,
                   "weight": 0.5
               }
           ]
       }
   }
   ```

4. **Define Frequency Range and Plot Settings**:

   Edit `config/frequency_config.json`:

   ```json
   {
       "start_frequency": 0,
       "end_frequency": 12000,
       "number_of_points": 1500,
       "plot_peaks": true,
       "plot_slopes": true
   }
   ```

5. **Run FRF Analysis**:

   ```bash
   python main.py run_frf
   ```

6. **Conduct Sobol Sensitivity Analysis**:

   ```bash
   python main.py run_sobol
   ```

7. **Prioritize Parameters for Optimization**:

   Based on Sobol results, adjust `config/parameter_prioritization.json` to focus on influential parameters.

8. **Configure and Run GA Optimization**:

   ```bash
   python main.py run_ga
   ```

9. **Review and Implement Optimized Parameters**:

   - Access optimized parameters in `output/ga_results/`.
   - Update `config/dva_config.json` with these values.
   - Re-run FRF Analysis to validate improvements.

---

## File Structure

```
DeVana/
├── config/
│   ├── system_config.json
│   ├── dva_config.json
│   ├── performance_targets.json
│   ├── frequency_config.json
│   ├── sobol_config.json
│   ├── ga_config.json
│   └── parameter_prioritization.json
├── data/
│   └── example_configuration.json
├── docs/
│   └── user_manual.pdf
├── output/
│   ├── frf_plots/
│   ├── sobol_results/
│   ├── ga_logs/
│   └── ga_results/
├── tests/
│   ├── test_equations.py
│   ├── test_optimization.py
│   └── test_sobol.py
├── main.py
├── requirements.txt
├── LICENSE
├── README.md
└── DeVana.png
```

- **`config/`**: Contains all configuration files for system parameters, DVA settings, performance targets, and optimization configurations.
- **`data/`**: Stores input data and example configurations for testing and demonstration purposes.
- **`docs/`**: Comprehensive documentation including user manuals and technical guides.
- **`output/`**: Houses results from analyses and optimizations, organized into subdirectories.
- **`tests/`**: Unit tests ensuring the reliability and accuracy of framework components.
- **`main.py`**: The primary script to run analyses and optimizations.
- **`requirements.txt`**: Lists all Python dependencies required to run DeVana.
- **`LICENSE`**: Project licensing information.
- **`README.md`**: This file, providing an overview and guidance on using DeVana.
- **`DeVana.png`**: Project logo.

---

## Contributing

Contributions to **DeVana** are highly encouraged! Whether you're looking to add new features, fix bugs, improve documentation, or enhance existing functionalities, your efforts are welcome.

### How to Contribute

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page to create a personal copy.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/DeVana.git
   cd DeVana
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature or fix within the appropriate files and directories.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add [feature/bugfix]: Description of your changes"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   - Navigate to your fork on GitHub.
   - Click the "Compare & pull request" button.
   - Provide a clear description of your changes and submit the pull request.

### Guidelines

- **Code Quality**: Ensure your code adheres to the project's coding standards and passes all existing tests.
- **Documentation**: Update or add documentation as necessary to reflect your changes.
- **Testing**: Include unit tests for new features or bug fixes to maintain framework reliability.

---

## Contact

For questions, suggestions, or support, please reach out to:

**Mahan Dashti Gohari**  
Email: [mahan.dashti@example.com](mailto:mahan.dashti@example.com)  
GitHub: [https://github.com/yourusername](https://github.com/yourusername)

---

## Acknowledgments

Special thanks to the researchers and practitioners in the field of vibration analysis whose work inspired the development of DeVana, including:

- Crandall, S. H., & Mark, S. A. (2014). Random number generation for quasi-Monte Carlo integration in engineering. *Journal of Computational and Applied Mathematics*, 265, 260-269.
- [Any other relevant works or contributors]

---
```

---

**Notes for Customization:**

1. **Replace Placeholders**:
   - **`yourusername`**: Update with your actual GitHub username.
   - **`mahan.dashti@example.com`**: Replace with your actual contact email.
   - **`DeVana.png`**: Ensure the logo image is placed in the root directory or adjust the path accordingly.

2. **Configuration Files**:
   - Ensure all JSON configuration files (`system_config.json`, `dva_config.json`, etc.) are properly set up within the `config/` directory as per your project's requirements.

3. **Dependencies**:
   - Populate `requirements.txt` with all necessary Python packages. Example:

     ```txt
     numpy
     scipy
     matplotlib
     pandas
     seaborn
     deap
     SALib
     ```

4. **Scripts**:
   - **`main.py`** should contain the logic for running FRF Analysis, Sobol Sensitivity Analysis, and GA Optimization. Ensure it can parse the configuration files and execute the required processes.

5. **Documentation**:
   - Populate the `docs/` directory with detailed user manuals and technical guides to complement the README.

6. **Testing**:
   - Implement comprehensive unit tests within the `tests/` directory to ensure all components function as intended.

7. **Examples**:
   - Provide example configurations and data within the `data/` directory to help users get started quickly.

8. **Visualization**:
   - Ensure that output directories like `output/frf_plots/`, `output/sobol_results/`, etc., are created automatically by the scripts or include instructions for manual creation.

By following this comprehensive README structure, users and contributors will have a clear understanding of DeVana's purpose, capabilities, and how to effectively utilize and contribute to the framework.

## License
       Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2025 Mahan Dashti Gohari

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
