# Step-by-Step Software Implementation Roadmap
## Comprehensive Python DVA Multi-Objective Optimization System
### CLI + GUI Architecture with Full Backend and Frontend

---

## TABLE OF CONTENTS

1. **Project Overview and Architecture**
2. **Development Environment Setup**
3. **Phase 1: Core Backend Foundation (Weeks 1-3)**
4. **Phase 2: Algorithm Implementation (Weeks 4-7)**
5. **Phase 3: CLI Interface Development (Week 8)**
6. **Phase 4: GUI Development (Weeks 9-12)**
7. **Phase 5: Integration and Testing (Weeks 13-14)**
8. **Phase 6: Performance Optimization (Week 15)**
9. **Phase 7: Documentation and Deployment (Week 16)**
10. **Project Structure and File Organization**
11. **Technology Stack and Dependencies**
12. **Best Practices and Design Patterns**

---

## 1. PROJECT OVERVIEW AND ARCHITECTURE

### 1.1 System Requirements

**Functional Requirements:**
- Support 5 multi-objective evolutionary algorithms (NSGA-II, NSGA-III, MOEA/D, SPEA2, AdaVEA-MOO)
- Handle 48-parameter DVA optimization with 3 objectives
- Run multiple independent trials (30 runs for statistical analysis)
- Real-time monitoring and visualization
- Comprehensive performance metrics calculation
- Both CLI and GUI interfaces
- Checkpoint/resume capability
- Export results in multiple formats (CSV, JSON, HDF5, PDF reports)

**Non-Functional Requirements:**
- Scalable to larger problems (up to 100 parameters, 10 objectives)
- Parallel execution support (multi-core)
- Memory-efficient (handle populations of 1000+)
- Cross-platform (Windows, macOS, Linux)
- Professional-grade code quality
- Comprehensive documentation

### 1.2 Architecture Pattern: MVC with Layered Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   CLI Interface      │    │   GUI Interface (PyQt6)      │  │
│  │   (Typer + Rich)     │    │   - Main Window              │  │
│  │   - Commands         │    │   - Experiment Manager       │  │
│  │   - Progress bars    │    │   - Real-time Plots          │  │
│  │   - Color output     │    │   - Configuration Editor     │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROLLER LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   ExperimentController                                    │  │
│  │   - Manages experiment lifecycle                         │  │
│  │   - Coordinates between Model and View                   │  │
│  │   - Handles user input validation                        │  │
│  │   - Triggers computations                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE LAYER (Business Logic)                │
│  ┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Optimization  │  │ Metrics         │  │ Visualization   │  │
│  │ Service       │  │ Service         │  │ Service         │  │
│  │ - Run algos   │  │ - Calculate HV  │  │ - Plot Pareto   │  │
│  │ - Parallel    │  │ - Compute IGD+  │  │ - Convergence   │  │
│  │ - Checkpoint  │  │ - Statistics    │  │ - Heatmaps      │  │
│  └───────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DOMAIN MODEL LAYER                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Algorithm       │  │ Problem      │  │ Solution         │  │
│  │ - NSGA-II       │  │ - DVA        │  │ - Population     │  │
│  │ - MOEA/D        │  │ - Objectives │  │ - Archive        │  │
│  │ - AdaVEA-MOO    │  │ - Constraints│  │ - Fitness        │  │
│  └─────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   Repository Pattern                                      │  │
│  │   - ExperimentRepository (CRUD operations)               │  │
│  │   - ResultRepository (Save/Load results)                 │  │
│  │   - ConfigurationRepository (Manage configs)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   Storage Backends                                        │  │
│  │   - HDF5 (large arrays)                                  │  │
│  │   - SQLite (metadata, experiments)                       │  │
│  │   - JSON/YAML (configurations)                           │  │
│  │   - CSV (tabular results)                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Design Patterns Used

1. **MVC (Model-View-Controller)**: Separate UI from business logic
2. **Repository Pattern**: Abstract data access
3. **Strategy Pattern**: Interchangeable algorithms
4. **Observer Pattern**: Real-time updates to GUI
5. **Factory Pattern**: Create algorithm instances
6. **Singleton Pattern**: Configuration manager
7. **Dependency Injection**: Testable components

---

## 2. DEVELOPMENT ENVIRONMENT SETUP

### 2.1 Install Python and Tools

**Requirements:**
- Python 3.10+ (recommended 3.11 for performance)
- Git for version control
- VS Code or PyCharm IDE

**Installation (Windows/macOS/Linux):**

```bash
# Verify Python version
python --version  # Should be 3.10+

# Create project directory
mkdir dva_optimization
cd dva_optimization

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2.2 Install Core Dependencies

Create `requirements.txt`:

```
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Optimization frameworks
pymoo>=0.6.0
deap>=1.4.0

# Performance metrics
pygmo>=2.19.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# CLI framework
typer[all]>=0.9.0
rich>=13.0.0
click>=8.1.0

# GUI framework
PyQt6>=6.5.0
PyQt6-Charts>=6.5.0
pyqtgraph>=0.13.0

# Data storage
h5py>=3.8.0
tables>=3.8.0

# Parallel computing
joblib>=1.3.0
dask[complete]>=2023.5.0

# Statistical analysis
scikit-learn>=1.3.0
statsmodels>=0.14.0
pingouin>=0.5.3

# Sensitivity analysis
SALib>=1.4.7

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Code quality
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
pylint>=2.17.0

# Documentation
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

### 2.3 Project Initialization

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Project structure"

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/experiments/
*.h5
*.hdf5
*.db

# OS
.DS_Store
Thumbs.db

# Coverage
.coverage
htmlcov/

# Build
build/
dist/
*.egg-info/
EOF
```

---

## 3. PHASE 1: CORE BACKEND FOUNDATION (Weeks 1-3)

### 3.1 Week 1: Project Structure and Base Classes

**Step 1.1: Create directory structure**

```bash
dva_optimization/
├── src/
│   ├── __init__.py
│   ├── core/                    # Core domain models
│   │   ├── __init__.py
│   │   ├── problem.py          # Base Problem class
│   │   ├── solution.py         # Solution representation
│   │   ├── population.py       # Population management
│   │   └── algorithm.py        # Base Algorithm class
│   ├── problems/               # Problem-specific implementations
│   │   ├── __init__.py
│   │   ├── dva_problem.py      # DVA optimization problem
│   │   └── test_functions.py  # ZDT, DTLZ test problems
│   ├── algorithms/             # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── nsga2.py
│   │   ├── nsga3.py
│   │   ├── moead.py
│   │   ├── spea2.py
│   │   └── adavea_moo.py
│   ├── operators/              # Genetic operators
│   │   ├── __init__.py
│   │   ├── crossover.py       # SBX, etc.
│   │   ├── mutation.py        # Polynomial, Gaussian, etc.
│   │   └── selection.py       # Tournament, etc.
│   ├── metrics/                # Performance metrics
│   │   ├── __init__.py
│   │   ├── hypervolume.py
│   │   ├── igd_plus.py
│   │   ├── diversity.py
│   │   └── statistical.py
│   ├── services/               # Business logic layer
│   │   ├── __init__.py
│   │   ├── optimization_service.py
│   │   ├── metrics_service.py
│   │   ├── visualization_service.py
│   │   └── experiment_service.py
│   ├── repositories/           # Data access layer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── experiment_repo.py
│   │   ├── result_repo.py
│   │   └── config_repo.py
│   ├── cli/                    # Command-line interface
│   │   ├── __init__.py
│   │   ├── app.py             # Main CLI app
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── run.py
│   │   │   ├── analyze.py
│   │   │   └── visualize.py
│   │   └── utils.py
│   ├── gui/                    # Graphical user interface
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── controllers/
│   │   │   ├── __init__.py
│   │   │   └── experiment_controller.py
│   │   ├── views/
│   │   │   ├── __init__.py
│   │   │   ├── experiment_view.py
│   │   │   ├── config_view.py
│   │   │   ├── results_view.py
│   │   │   └── plot_view.py
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── progress_widget.py
│   │   │   └── plot_widget.py
│   │   └── resources/
│   │       ├── ui/             # Qt Designer files
│   │       └── icons/
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── logging_config.py  # Logging setup
│   │   └── validators.py     # Input validation
│   └── config/                 # Configuration files
│       ├── default_config.yaml
│       └── algorithms/
│           ├── nsga2.yaml
│           └── adavea_moo.yaml
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/                       # Data directory (gitignored)
│   ├── experiments/
│   ├── results/
│   └── cache/
├── docs/                       # Documentation
│   ├── api/
│   ├── tutorials/
│   └── user_guide/
├── scripts/                    # Utility scripts
│   ├── setup_env.sh
│   └── run_benchmark.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

**Create this structure:**

```bash
# Create all directories
mkdir -p src/{core,problems,algorithms,operators,metrics,services,repositories,cli/commands,gui/{controllers,views,widgets,resources/{ui,icons}},utils,config/algorithms}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p data/{experiments,results,cache}
mkdir -p docs/{api,tutorials,user_guide}
mkdir -p scripts

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
touch tests/__init__.py
```

**Step 1.2: Implement base Problem class**

File: `src/core/problem.py`

```python
"""
Base Problem class for multi-objective optimization.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ProblemMetadata:
    """Metadata about the optimization problem."""
    name: str
    n_var: int  # Number of decision variables
    n_obj: int  # Number of objectives
    n_constr: int = 0  # Number of constraints
    xl: Optional[np.ndarray] = None  # Lower bounds
    xu: Optional[np.ndarray] = None  # Upper bounds
    description: str = ""
    
    def __post_init__(self):
        """Validate and initialize bounds."""
        if self.xl is None:
            self.xl = np.zeros(self.n_var)
        if self.xu is None:
            self.xu = np.ones(self.n_var)
        
        # Ensure bounds are numpy arrays
        self.xl = np.asarray(self.xl)
        self.xu = np.asarray(self.xu)
        
        # Validate
        assert len(self.xl) == self.n_var, "Lower bounds dimension mismatch"
        assert len(self.xu) == self.n_var, "Upper bounds dimension mismatch"
        assert np.all(self.xl <= self.xu), "Invalid bounds: xl must be <= xu"


class Problem(ABC):
    """
    Abstract base class for multi-objective optimization problems.
    
    All problem implementations must inherit from this class and
    implement the _evaluate method.
    """
    
    def __init__(self, metadata: ProblemMetadata):
        """
        Initialize problem.
        
        Args:
            metadata: Problem metadata (name, dimensions, bounds)
        """
        self.metadata = metadata
        self._evaluations = 0
        self._evaluation_cache: Dict[Tuple, np.ndarray] = {}
    
    @abstractmethod
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate objective functions.
        
        Args:
            X: Decision variables, shape (n_samples, n_var)
        
        Returns:
            Objective values, shape (n_samples, n_obj)
        """
        pass
    
    def evaluate(self, X: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Public interface to evaluate objectives with caching.
        
        Args:
            X: Decision variables
            use_cache: Whether to use cached evaluations
        
        Returns:
            Objective values
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        # Validate dimensions
        if X.shape[1] != self.metadata.n_var:
            raise ValueError(
                f"Expected {self.metadata.n_var} variables, "
                f"got {X.shape[1]}"
            )
        
        # Clip to bounds
        X = np.clip(X, self.metadata.xl, self.metadata.xu)
        
        if use_cache:
            # Check cache for each solution
            results = np.zeros((n_samples, self.metadata.n_obj))
            uncached_idx = []
            uncached_X = []
            
            for i, x in enumerate(X):
                key = tuple(x)
                if key in self._evaluation_cache:
                    results[i] = self._evaluation_cache[key]
                else:
                    uncached_idx.append(i)
                    uncached_X.append(x)
            
            # Evaluate uncached solutions
            if uncached_X:
                uncached_X = np.array(uncached_X)
                uncached_F = self._evaluate(uncached_X)
                
                for i, idx in enumerate(uncached_idx):
                    key = tuple(X[idx])
                    self._evaluation_cache[key] = uncached_F[i]
                    results[idx] = uncached_F[i]
                
                self._evaluations += len(uncached_idx)
        else:
            results = self._evaluate(X)
            self._evaluations += n_samples
        
        return results
    
    def evaluate_constraints(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate constraint violations.
        
        Args:
            X: Decision variables
        
        Returns:
            Constraint violations (0 if satisfied, >0 if violated)
            or None if no constraints
        """
        if self.metadata.n_constr == 0:
            return None
        return self._evaluate_constraints(X)
    
    def _evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """Override this if problem has constraints."""
        raise NotImplementedError(
            "Problem has constraints but _evaluate_constraints not implemented"
        )
    
    @property
    def n_evaluations(self) -> int:
        """Total number of function evaluations performed."""
        return self._evaluations
    
    def reset_cache(self):
        """Clear evaluation cache."""
        self._evaluation_cache.clear()
        self._evaluations = 0
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_var={self.metadata.n_var}, "
            f"n_obj={self.metadata.n_obj}, "
            f"evaluations={self._evaluations})"
        )
```

**Step 1.3: Implement Solution class**

File: `src/core/solution.py`

```python
"""
Solution representation for multi-objective optimization.
"""
from typing import Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Solution:
    """
    Represents a single solution in the population.
    
    Attributes:
        X: Decision variables (genotype)
        F: Objective values (fitness)
        rank: Non-dominated rank (1 = best)
        crowding_distance: Crowding distance for diversity
        constraint_violation: Constraint violation (0 = feasible)
        dominated_solutions: List of solutions this dominates
        domination_count: Number of solutions dominating this
    """
    X: np.ndarray
    F: Optional[np.ndarray] = None
    rank: int = 0
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0
    dominated_solutions: list = field(default_factory=list)
    domination_count: int = 0
    
    def dominates(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another (Pareto dominance).
        
        For minimization problems:
        - All objectives <= other's objectives
        - At least one objective < other's objective
        
        Args:
            other: Another solution
        
        Returns:
            True if this solution dominates other
        """
        if self.F is None or other.F is None:
            raise ValueError("Cannot compare solutions without fitness values")
        
        # Handle constraints: feasible solutions dominate infeasible ones
        if self.constraint_violation <= 0 and other.constraint_violation > 0:
            return True
        if self.constraint_violation > 0 and other.constraint_violation <= 0:
            return False
        
        # Both feasible or both infeasible: compare objectives
        at_least_as_good = np.all(self.F <= other.F)
        strictly_better = np.any(self.F < other.F)
        
        return at_least_as_good and strictly_better
    
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        return Solution(
            X=self.X.copy(),
            F=self.F.copy() if self.F is not None else None,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            constraint_violation=self.constraint_violation,
            dominated_solutions=self.dominated_solutions.copy(),
            domination_count=self.domination_count
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on decision variables."""
        if not isinstance(other, Solution):
            return False
        return np.allclose(self.X, other.X)
    
    def __hash__(self) -> int:
        """Hash based on decision variables (for sets/dicts)."""
        return hash(tuple(self.X))
    
    def __repr__(self) -> str:
        f_str = (
            f"[{', '.join(f'{x:.3f}' for x in self.F)}]" 
            if self.F is not None else "None"
        )
        return (
            f"Solution(X={len(self.X)} vars, F={f_str}, "
            f"rank={self.rank}, cd={self.crowding_distance:.3f})"
        )
```

**Step 1.4: Implement Population class**

File: `src/core/population.py`

```python
"""
Population management for evolutionary algorithms.
"""
from typing import List, Optional, Tuple
import numpy as np
from .solution import Solution


class Population:
    """
    Container for a population of solutions.
    
    Provides methods for common population operations like
    non-dominated sorting, crowding distance calculation, etc.
    """
    
    def __init__(self, solutions: Optional[List[Solution]] = None):
        """
        Initialize population.
        
        Args:
            solutions: List of Solution objects
        """
        self.solutions = solutions if solutions is not None else []
    
    def __len__(self) -> int:
        return len(self.solutions)
    
    def __getitem__(self, idx) -> Solution:
        return self.solutions[idx]
    
    def __iter__(self):
        return iter(self.solutions)
    
    def add(self, solution: Solution):
        """Add a solution to the population."""
        self.solutions.append(solution)
    
    def extend(self, solutions: List[Solution]):
        """Add multiple solutions."""
        self.solutions.extend(solutions)
    
    def get_decision_variables(self) -> np.ndarray:
        """
        Get decision variables as array.
        
        Returns:
            Array of shape (pop_size, n_var)
        """
        return np.array([sol.X for sol in self.solutions])
    
    def get_objective_values(self) -> np.ndarray:
        """
        Get objective values as array.
        
        Returns:
            Array of shape (pop_size, n_obj)
        """
        return np.array([sol.F for sol in self.solutions])
    
    def get_rank_counts(self) -> dict:
        """Get count of solutions in each rank."""
        ranks = [sol.rank for sol in self.solutions if sol.rank > 0]
        return {r: ranks.count(r) for r in set(ranks)}
    
    def get_pareto_front(self, rank: int = 1) -> 'Population':
        """
        Extract solutions of a specific rank.
        
        Args:
            rank: Rank to extract (default: 1 = first front)
        
        Returns:
            New Population containing only solutions of specified rank
        """
        front_solutions = [s for s in self.solutions if s.rank == rank]
        return Population(front_solutions)
    
    def merge(self, other: 'Population') -> 'Population':
        """
        Merge with another population.
        
        Args:
            other: Another Population object
        
        Returns:
            New Population containing solutions from both
        """
        return Population(self.solutions + other.solutions)
    
    def sort_by_objective(self, obj_idx: int, ascending: bool = True):
        """
        Sort population by a specific objective.
        
        Args:
            obj_idx: Index of objective to sort by
            ascending: Sort order
        """
        self.solutions.sort(
            key=lambda s: s.F[obj_idx],
            reverse=not ascending
        )
    
    def truncate(self, size: int):
        """
        Truncate population to specified size.
        
        Args:
            size: Target population size
        """
        self.solutions = self.solutions[:size]
    
    def __repr__(self) -> str:
        rank_counts = self.get_rank_counts()
        return (
            f"Population(size={len(self)}, "
            f"fronts={len(rank_counts)}, "
            f"rank_1={rank_counts.get(1, 0)})"
        )
```

### 3.2 Week 2: Configuration and Logging System

**Step 2.1: Configuration Management**

File: `src/utils/config.py`

```python
"""
Configuration management using YAML files.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, asdict
import json


@dataclass
class AlgorithmConfig:
    """Configuration for an optimization algorithm."""
    name: str
    population_size: int
    max_generations: int
    crossover_prob: float
    mutation_prob: float
    crossover_eta: float = 20.0
    mutation_eta: float = 20.0
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AlgorithmConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProblemConfig:
    """Configuration for optimization problem."""
    name: str
    n_var: int
    n_obj: int
    xl: list
    xu: list
    problem_specific_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.problem_specific_params is None:
            self.problem_specific_params = {}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    algorithm_config: AlgorithmConfig
    problem_config: ProblemConfig
    n_runs: int = 30
    random_seed: Optional[int] = None
    parallel_workers: int = 1
    checkpoint_frequency: int = 50
    save_directory: str = "data/experiments"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'algorithm_config': self.algorithm_config.to_dict(),
            'problem_config': asdict(self.problem_config),
            'n_runs': self.n_runs,
            'random_seed': self.random_seed,
            'parallel_workers': self.parallel_workers,
            'checkpoint_frequency': self.checkpoint_frequency,
            'save_directory': self.save_directory
        }
    
    def save(self, filepath: Path):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        data['algorithm_config'] = AlgorithmConfig.from_dict(
            data['algorithm_config']
        )
        data['problem_config'] = ProblemConfig(**data['problem_config'])
        
        return cls(**data)


class ConfigurationManager:
    """
    Singleton configuration manager.
    
    Manages loading, saving, and accessing configurations.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config_dir = Path("src/config")
        self.default_config_path = self.config_dir / "default_config.yaml"
        self.current_config: Optional[ExperimentConfig] = None
        self._initialized = True
    
    def load_default(self) -> ExperimentConfig:
        """Load default configuration."""
        if not self.default_config_path.exists():
            raise FileNotFoundError(
                f"Default config not found: {self.default_config_path}"
            )
        
        self.current_config = ExperimentConfig.load(self.default_config_path)
        return self.current_config
    
    def load(self, config_path: Path) -> ExperimentConfig:
        """Load configuration from file."""
        self.current_config = ExperimentConfig.load(config_path)
        return self.current_config
    
    def save(self, config: ExperimentConfig, filepath: Path):
        """Save configuration to file."""
        config.save(filepath)
        self.current_config = config
    
    def get(self) -> ExperimentConfig:
        """Get current configuration."""
        if self.current_config is None:
            return self.load_default()
        return self.current_config
```

**Step 2.2: Logging Configuration**

File: `src/utils/logging_config.py`

```python
"""
Centralized logging configuration.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
        console_output: Whether to output to console
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("DVAOptimization")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logging()
```

### 3.3 Week 3: Repository Layer (Data Access)

**Step 3.1: Base Repository**

File: `src/repositories/base.py`

```python
"""
Base repository pattern implementation.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from pathlib import Path

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository for data access.
    
    Implements CRUD operations:
    - Create
    - Read
    - Update
    - Delete
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize repository.
        
        Args:
            storage_path: Root directory for data storage
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def create(self, entity: T) -> str:
        """
        Create new entity and return its ID.
        
        Args:
            entity: Entity to create
        
        Returns:
            Unique identifier for created entity
        """
        pass
    
    @abstractmethod
    def read(self, entity_id: str) -> Optional[T]:
        """
        Read entity by ID.
        
        Args:
            entity_id: Unique identifier
        
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, entity_id: str, entity: T) -> bool:
        """
        Update existing entity.
        
        Args:
            entity_id: Unique identifier
            entity: Updated entity
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Unique identifier
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_all(self) -> List[str]:
        """
        List all entity IDs.
        
        Returns:
            List of entity identifiers
        """
        pass
    
    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Unique identifier
        
        Returns:
            True if exists
        """
        return entity_id in self.list_all()
```

**Step 3.2: Result Repository (HDF5 Storage)**

File: `src/repositories/result_repo.py`

```python
"""
Repository for storing optimization results using HDF5.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import h5py
import numpy as np
import json
from datetime import datetime
from .base import BaseRepository


class ResultRepository(BaseRepository[Dict[str, Any]]):
    """
    Store and retrieve optimization results using HDF5.
    
    Structure:
    experiment_id/
        ├── run_001/
        │   ├── population_X (dataset)
        │   ├── population_F (dataset)
        │   ├── archive_X (dataset)
        │   ├── archive_F (dataset)
        │   ├── metrics_per_gen (dataset)
        │   └── metadata (attributes)
        ├── run_002/
        └── ...
    """
    
    def __init__(self, storage_path: Path):
        super().__init__(storage_path)
        self.h5_file = self.storage_path / "results.h5"
    
    def create(self, entity: Dict[str, Any]) -> str:
        """
        Save optimization result.
        
        Args:
            entity: Dictionary with keys:
                - experiment_id: str
                - run_id: str
                - population_X: np.ndarray
                - population_F: np.ndarray
                - archive_X: np.ndarray (optional)
                - archive_F: np.ndarray (optional)
                - metrics: Dict[str, np.ndarray]
                - metadata: Dict[str, Any]
        
        Returns:
            Compound ID: "experiment_id/run_id"
        """
        exp_id = entity['experiment_id']
        run_id = entity['run_id']
        compound_id = f"{exp_id}/{run_id}"
        
        with h5py.File(self.h5_file, 'a') as f:
            # Create group for this run
            if compound_id in f:
                del f[compound_id]  # Overwrite
            
            grp = f.create_group(compound_id)
            
            # Save datasets
            grp.create_dataset('population_X', data=entity['population_X'])
            grp.create_dataset('population_F', data=entity['population_F'])
            
            if 'archive_X' in entity and entity['archive_X'] is not None:
                grp.create_dataset('archive_X', data=entity['archive_X'])
                grp.create_dataset('archive_F', data=entity['archive_F'])
            
            # Save metrics (each metric as separate dataset)
            metrics_grp = grp.create_group('metrics')
            for key, value in entity.get('metrics', {}).items():
                metrics_grp.create_dataset(key, data=value)
            
            # Save metadata as JSON attribute
            metadata = entity.get('metadata', {})
            metadata['timestamp'] = datetime.now().isoformat()
            grp.attrs['metadata'] = json.dumps(metadata)
        
        return compound_id
    
    def read(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Load optimization result.
        
        Args:
            entity_id: "experiment_id/run_id"
        
        Returns:
            Dictionary with result data
        """
        if not self.h5_file.exists():
            return None
        
        with h5py.File(self.h5_file, 'r') as f:
            if entity_id not in f:
                return None
            
            grp = f[entity_id]
            
            result = {
                'population_X': grp['population_X'][:],
                'population_F': grp['population_F'][:],
                'metrics': {},
                'metadata': json.loads(grp.attrs['metadata'])
            }
            
            if 'archive_X' in grp:
                result['archive_X'] = grp['archive_X'][:]
                result['archive_F'] = grp['archive_F'][:]
            
            # Load metrics
            if 'metrics' in grp:
                for key in grp['metrics'].keys():
                    result['metrics'][key] = grp['metrics'][key][:]
            
            return result
    
    def update(self, entity_id: str, entity: Dict[str, Any]) -> bool:
        """Update is same as create (overwrite)."""
        self.create(entity)
        return True
    
    def delete(self, entity_id: str) -> bool:
        """Delete a result."""
        if not self.h5_file.exists():
            return False
        
        with h5py.File(self.h5_file, 'a') as f:
            if entity_id in f:
                del f[entity_id]
                return True
        return False
    
    def list_all(self) -> list:
        """List all result IDs."""
        if not self.h5_file.exists():
            return []
        
        ids = []
        with h5py.File(self.h5_file, 'r') as f:
            def collect_ids(name, obj):
                if isinstance(obj, h5py.Group) and '/' in name:
                    ids.append(name)
            
            f.visititems(collect_ids)
        
        return ids
```

*Continue to next message for remaining phases...*