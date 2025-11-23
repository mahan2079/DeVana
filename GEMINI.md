# GEMINI.md

## Project Overview

This project, named "DeVana", is a desktop application for performing design, analysis, and optimization tasks on vibrating continuous systems. It is written in Python and uses the PyQt5 framework for its graphical user interface.

The application is designed for mechanical engineers and researchers to model, analyze, and optimize Dynamic Vibration Absorbers (DVAs) in complex mechanical systems.

**Key Technologies:**

*   **Backend:** Python
*   **GUI:** PyQt5
*   **Numerical/Scientific Libraries:** NumPy, SciPy, Pandas
*   **Plotting:** Matplotlib, Seaborn
*   **Optimization:** DEAP (for Genetic Algorithms), and implementations of Particle Swarm Optimization (PSO), Differential Evolution (DE), Simulated Annealing (SA), and CMA-ES.
*   **Sensitivity Analysis:** SALib

**Architecture:**

The application follows a typical GUI/backend architecture:

*   The main application window is defined in `codes/mainwindow.py`, which is the central hub of the GUI.
*   The GUI is built using a modular approach with mixins for different functionalities (e.g., `ga_mixin.py`, `pso_mixin.py`). These mixins are located in the `codes/gui/` directory.
*   Long-running computations like optimizations are performed in background threads using `QThread`. The worker implementations are in the `codes/workers/` directory (e.g., `GAWorker.py`, `PSOWorker.py`).
*   The workers communicate with the main GUI thread using Qt's signal and slot mechanism to provide progress updates and display results without freezing the UI.
*   The core scientific computations, like the Frequency Response Function (FRF), are in the `codes/modules/` directory.

## Building and Running

### Prerequisites

*   Python 3.7+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mahan2079/DeVana.git
    cd DeVana
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the DeVana application, run the following command from the project's root directory:

```bash
python codes/run.py
```

### Testing

There are no dedicated tests in the provided file structure.
TODO: Add instructions for running tests once they are available.

## Development Conventions

*   **Code Style:** The code generally follows PEP 8 style guidelines, but it's not strictly enforced.
*   **Modularity:** The code is organized into modules and packages based on functionality (e.g., `gui`, `workers`, `modules`).
*   **GUI Development:** The GUI is developed using PyQt5. The main window class `MainWindow` is extended with mixins for different optimization algorithms and features. This allows for a clean separation of concerns in the UI code.
*   **Concurrency:** Long-running tasks are offloaded to `QThread` workers to keep the GUI responsive. Communication between the worker threads and the main GUI thread is handled via signals and slots.
*   **Documentation:** The code contains a mix of docstrings and comments. The `GAWorker.py` file, in particular, has extensive comments and explanations.
*   **Dependencies:** Project dependencies are managed in the `requirements.txt` file.
*   **Configuration:** The `README.md` mentions JSON-based configuration files in a `config/` directory, but this directory is not present in the file listing.
*   **Versioning:** The application name and version are managed in `codes/app_info.py`.
