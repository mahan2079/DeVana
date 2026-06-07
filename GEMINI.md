# Project: DeVana: Dynamic Vibration Absorber Optimization Framework

## Project Overview
DeVana is an advanced engineering suite for the analysis and multi-objective optimization of Dynamic Vibration Absorbers (DVAs). It bridges the gap between theoretical vibration analysis and practical, manufacturable DVA design. The framework uses metaheuristic algorithms and machine learning to find optimal DVA configurations (topology) and safe parameter ranges for discrete vibrational models.

### Key Features
- **Advanced Optimization Suite**: GA, PSO, CMA-ES, NSGA-II, AdaVEA, DE, SA, and RL-based optimization.
- **Intelligent Seeding**: Neural Seeder (PyTorch-based) and Memory Seeder to accelerate convergence.
- **High-Fidelity Analysis**: Robust Frequency Response Function (FRF) solver and Sobol sensitivity analysis.
- **Statistical Design**: Extraction of safe, reliable parameter ranges from optimization results.
- **Modular Architecture**: Mixin-based GUI development for extensibility.

## Tech Stack
- **Language**: Python 3.8+
- **GUI Framework**: PyQt5
- **Numerical & Analysis**: NumPy, SciPy, Pandas, Matplotlib, Seaborn
- **Optimization Libraries**: DEAP (GA, NSGA-II), CMA (CMA-ES), SALib (Sobol)
- **Machine Learning**: PyTorch (NeuralSeeder), Scikit-learn (KDE, ML tools)
- **Performance**: Joblib, Psutil

## Directory Structure
- `codes/`: Main source code directory.
    - `gui/`: Modular UI architecture using Mixins.
        - `main_window/`: Specific Mixins for optimization algorithms and analysis tools.
    - `workers/`: Multi-threaded implementations of optimization and analysis algorithms.
    - `modules/`: Core physics engines (FRF), plotting, and sensitivity analysis logic.
    - `RL/`: Reinforcement Learning agents and workers.
    - `Continues_beam/`: Specialized module for continuous beam analysis.
- `tests/`: Comprehensive test suite for algorithms, workers, and integration.
- `Documents/`: Technical documentation, mathematical models, and flowcharts.
- `paper/`: JOSS-style paper and bibtex files.

## Building and Running

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository and navigate to the root.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the main application from the project root:
```bash
python codes/run.py
```

### Running the REST API
Launch the headless API server:
```bash
source venv_api/bin/activate
python codes/api/main.py
```

### Running Tests
Execute all tests from the project root:
```bash
python tests/run_all.py
```
Or run specific tests using `pytest`:
```bash
pytest tests/test_frf.py
```

### Development Conventions

### Documentation Standards (Dolores RAG)
All technical documentation must adhere to the **[Dolores Documentation Standards](Documents/DOLORES_DOC_STANDARDS.md)**:
- **Framework:** Diátaxis (Tutorials, How-To, Reference, Explanation).
- **Visuals:** Extensive Mermaid.js flowcharts and sequence diagrams.
- **Math:** Exact LaTeX formulations for all physics and optimization logic.
- **Coverage:** No missed lines of code; granular reference for all workers and modules.
- **Index:** The central entry point for all project knowledge is **[Documents/INDEX.md](Documents/INDEX.md)**.

### Architecture: The Mixin Pattern
The `MainWindow` (in `codes/mainwindow.py`) is composed of numerous Mixins located in `codes/gui/`. 
- When adding a new feature or algorithm, create a new Mixin in the appropriate `gui/` subdirectory.
- Logic for long-running tasks should be encapsulated in a `Worker` class (inheriting from `QThread`) in `codes/workers/` to keep the UI responsive.

### Coding Style
- Follow PEP 8 guidelines.
- Use type hints where possible to improve maintainability.
- Maintain the extensive documentation within the code, especially for complex mathematical implementations.

### Testing
- Always add or update tests in the `tests/` directory for any new functionality or bug fix.
- Ensure `tests/run_all.py` passes before proposing significant changes.

## Key Files for Interaction
- `codes/run.py`: Application entry point.
- `codes/mainwindow.py`: Main window assembly and coordination.
- `codes/workers/`: Directory containing the core execution logic for algorithms.
- `codes/modules/FRF.py`: Central physics logic for vibration analysis.
- `codes/gui/main_window/stochastic_mixin.py`: Orchestrates the main optimization workflow.
- `requirements.txt`: Project dependencies.
