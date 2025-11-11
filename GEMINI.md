# Project: DeVana: Dynamic Vibration Absorber Optimization Framework

## Project Overview

DeVana is a state-of-the-art framework designed for the modeling, analysis, and optimization of Dynamic Vibration Absorbers (DVAs) within complex mechanical systems. It caters to engineers and researchers by providing a comprehensive workflow that spans from discrete system modeling to continuous system analysis. The framework's core functionality revolves around optimizing DVA parameters to create efficient, robust, and stable systems across various industrial applications.

Key capabilities include:
-   **Comprehensive Modeling**: Supports multi-degree-of-freedom systems with integrated DVAs.
-   **Advanced Optimization Suite**: Incorporates a variety of algorithms such as Genetic Algorithms (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE), Simulated Annealing (SA), CMA-ES, and Bayesian Optimization.
-   **Sensitivity Analysis**: Utilizes the Sobol method to identify and prioritize influential parameters.
-   **Statistical Analysis**: Aggregates results from multiple optimization runs to determine robust parameter ranges and analyze solution stability.
-   **Real-Time Visualization**: Offers interactive dashboards for Frequency Response Functions (FRF) and optimization progress.
-   **Continuous Beam Analysis**: Features advanced composite beam modeling with temperature-dependent material properties and finite element analysis.
-   **User-Friendly Configuration**: Employs intuitive JSON-based configuration files for flexible customization.

## Building and Running

To set up and run the DeVana application, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mahan2079/DeVana.git
    cd DeVana
    ```

2.  **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the application**:
    ```bash
    python codes/run.py
    ```

## Development Conventions

The project adheres to the following development guidelines:

*   **Code Quality**: Maintain high code quality standards.
*   **Documentation**: Keep documentation updated with any changes or new features.
*   **Unit Tests**: Include unit tests for all new features and bug fixes to ensure reliability.
*   **Code Style and Architecture**: Follow the existing code style and architectural patterns established within the project.

## Key Files for Interaction

Based on the project structure and the `README.md`, the following files are likely important for development and understanding the codebase:

*   `codes/run.py`: The main entry point for launching the application.
*   `codes/mainwindow.py`: Defines the main window structure and integrates various mixins.
*   `codes/gui/main_window/stochastic_mixin.py`: Manages the "Stochastic Design" page, including the main tab groups like "Input", "Sensitivity Analysis", and "Optimization". This file was recently modified to include "Multi-Objective Optimizations".
*   `codes/gui/main_window/ga_mixin.py`: Contains logic and UI for Genetic Algorithm optimization.
*   `codes/gui/main_window/pso_mixin.py`: Contains logic and UI for Particle Swarm Optimization.
*   `codes/gui/main_window/de_mixin.py`: Contains logic and UI for Differential Evolution optimization.
*   `codes/gui/main_window/nsga2_mixin.py`: (Newly created) Placeholder for NSGA-II Multi-Objective Optimization.
*   `codes/gui/main_window/adavea_mixin.py`: (Newly created) Placeholder for AdaVEA Multi-Objective Optimization.
*   `requirements.txt`: Lists all Python dependencies required for the project.
*   `README.md`: Provides a comprehensive overview, installation, and usage instructions.
