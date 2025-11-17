# Implementation Roadmap - Part 2: Algorithms, CLI, and GUI
## Continuation of Step-by-Step Software Development

---

## 4. PHASE 2: ALGORITHM IMPLEMENTATION (Weeks 4-7)

### 4.1 Week 4: NSGA-II Implementation

**File: `src/algorithms/nsga2.py`**

```python
"""
NSGA-II: Fast Elitist Non-Dominated Sorting Genetic Algorithm.

Reference:
Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""
from typing import List, Tuple
import numpy as np
from ..core.algorithm import BaseAlgorithm
from ..core.population import Population
from ..core.solution import Solution
from ..operators.crossover import SBXCrossover
from ..operators.mutation import PolynomialMutation
from ..operators.selection import TournamentSelection


class NSGA2(BaseAlgorithm):
    """
    NSGA-II implementation with fast non-dominated sort and crowding distance.
    
    Time complexity: O(M * N^2) per generation
    Space complexity: O(N^2)
    """
    
    def __init__(self, problem, config):
        super().__init__(problem, config)
        
        # Initialize operators
        self.crossover = SBXCrossover(
            eta=config.crossover_eta,
            prob=config.crossover_prob
        )
        self.mutation = PolynomialMutation(
            eta=config.mutation_eta,
            prob=config.mutation_prob / problem.metadata.n_var
        )
        self.selection = TournamentSelection(tournament_size=2)
    
    def initialize_population(self) -> Population:
        """Initialize population randomly within bounds."""
        n_var = self.problem.metadata.n_var
        xl = self.problem.metadata.xl
        xu = self.problem.metadata.xu
        
        solutions = []
        for _ in range(self.config.population_size):
            X = xl + (xu - xl) * np.random.rand(n_var)
            sol = Solution(X=X)
            solutions.append(sol)
        
        pop = Population(solutions)
        self._evaluate_population(pop)
        
        return pop
    
    def _evaluate_population(self, population: Population):
        """Evaluate all solutions in population."""
        X = population.get_decision_variables()
        F = self.problem.evaluate(X)
        
        for i, sol in enumerate(population):
            sol.F = F[i]
    
    def fast_non_dominated_sort(self, population: Population) -> List[List[int]]:
        """
        Fast non-dominated sorting algorithm.
        
        Complexity: O(M * N^2)
        
        Args:
            population: Population to sort
        
        Returns:
            List of fronts, where each front is a list of solution indices
        """
        n = len(population)
        
        # Initialize domination structures
        for sol in population:
            sol.dominated_solutions = []
            sol.domination_count = 0
        
        # First front (non-dominated solutions)
        fronts = [[]]
        
        # Step 1: Compute domination for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    population[i].dominated_solutions.append(j)
                    population[j].domination_count += 1
                elif population[j].dominates(population[i]):
                    population[j].dominated_solutions.append(i)
                    population[i].domination_count += 1
            
            # If not dominated by any solution, assign to first front
            if population[i].domination_count == 0:
                population[i].rank = 1
                fronts[0].append(i)
        
        # Step 2: Identify subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for p_idx in fronts[i]:
                p = population[p_idx]
                for q_idx in p.dominated_solutions:
                    q = population[q_idx]
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        next_front.append(q_idx)
            i += 1
            fronts.append(next_front)
        
        # Remove last empty front
        fronts.pop()
        
        return fronts
    
    def calculate_crowding_distance(self, population: Population, front: List[int]):
        """
        Calculate crowding distance for solutions in a front.
        
        Complexity: O(M * N * log(N))
        
        Args:
            population: Population
            front: Indices of solutions in this front
        """
        n = len(front)
        m = self.problem.metadata.n_obj
        
        # Initialize distances to zero
        for idx in front:
            population[idx].crowding_distance = 0.0
        
        # If only 1 or 2 solutions, assign infinite distance
        if n <= 2:
            for idx in front:
                population[idx].crowding_distance = float('inf')
            return
        
        # For each objective
        for obj_idx in range(m):
            # Sort front by this objective
            front_sorted = sorted(
                front,
                key=lambda idx: population[idx].F[obj_idx]
            )
            
            # Boundary solutions get infinite distance
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')
            
            # Get objective range
            f_min = population[front_sorted[0]].F[obj_idx]
            f_max = population[front_sorted[-1]].F[obj_idx]
            f_range = f_max - f_min
            
            if f_range == 0:
                continue
            
            # Calculate distance for intermediate solutions
            for i in range(1, n - 1):
                idx_current = front_sorted[i]
                idx_prev = front_sorted[i - 1]
                idx_next = front_sorted[i + 1]
                
                distance = (
                    population[idx_next].F[obj_idx] - 
                    population[idx_prev].F[obj_idx]
                ) / f_range
                
                population[idx_current].crowding_distance += distance
    
    def environmental_selection(self, population: Population) -> Population:
        """
        Select N best solutions using rank and crowding distance.
        
        Args:
            population: Combined parent + offspring population (size 2N)
        
        Returns:
            Selected population (size N)
        """
        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(population)
        
        selected = []
        
        # Fill with fronts until exceeding N
        for front in fronts:
            if len(selected) + len(front) <= self.config.population_size:
                # Include entire front
                selected.extend(front)
            else:
                # Calculate crowding distance for last front
                self.calculate_crowding_distance(population, front)
                
                # Sort by crowding distance (descending)
                front_sorted = sorted(
                    front,
                    key=lambda idx: population[idx].crowding_distance,
                    reverse=True
                )
                
                # Fill remaining slots
                remaining = self.config.population_size - len(selected)
                selected.extend(front_sorted[:remaining])
                break
        
        # Create new population with selected solutions
        new_solutions = [population[idx] for idx in selected]
        return Population(new_solutions)
    
    def generate_offspring(self, population: Population) -> Population:
        """
        Generate offspring population using crossover and mutation.
        
        Args:
            population: Parent population
        
        Returns:
            Offspring population
        """
        offspring_solutions = []
        
        n_offspring = self.config.population_size
        
        while len(offspring_solutions) < n_offspring:
            # Tournament selection
            parent1 = self.selection.select(population)
            parent2 = self.selection.select(population)
            
            # Crossover
            child1_X, child2_X = self.crossover.crossover(parent1.X, parent2.X)
            
            # Mutation
            child1_X = self.mutation.mutate(
                child1_X,
                self.problem.metadata.xl,
                self.problem.metadata.xu
            )
            child2_X = self.mutation.mutate(
                child2_X,
                self.problem.metadata.xl,
                self.problem.metadata.xu
            )
            
            # Create solutions
            offspring_solutions.append(Solution(X=child1_X))
            if len(offspring_solutions) < n_offspring:
                offspring_solutions.append(Solution(X=child2_X))
        
        offspring = Population(offspring_solutions[:n_offspring])
        self._evaluate_population(offspring)
        
        return offspring
    
    def evolve(self) -> dict:
        """
        Main evolutionary loop.
        
        Returns:
            Dictionary with results
        """
        # Initialize
        population = self.initialize_population()
        
        # Perform initial non-dominated sort
        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(population, front)
        
        # Evolution loop
        for gen in range(self.config.max_generations):
            # Generate offspring
            offspring = self.generate_offspring(population)
            
            # Combine parent and offspring
            combined = population.merge(offspring)
            
            # Environmental selection
            population = self.environmental_selection(combined)
            
            # Callback for monitoring
            if self.callback is not None:
                self.callback(gen, population)
        
        # Extract final results
        pareto_front = population.get_pareto_front(rank=1)
        
        return {
            'population': population,
            'pareto_front': pareto_front,
            'n_evaluations': self.problem.n_evaluations
        }
```

### 4.2 Week 5: Genetic Operators

**File: `src/operators/crossover.py`**

```python
"""
Crossover operators for evolutionary algorithms.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply crossover to two parents."""
        pass


class SBXCrossover(CrossoverOperator):
    """
    Simulated Binary Crossover (SBX).
    
    Commonly used in continuous optimization.
    
    Reference:
    Deb & Agrawal (1995). Simulated Binary Crossover for Continuous Search Space.
    """
    
    def __init__(self, eta: float = 20.0, prob: float = 0.9):
        """
        Initialize SBX operator.
        
        Args:
            eta: Distribution index (higher = children closer to parents)
            prob: Crossover probability
        """
        self.eta = eta
        self.prob = prob
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform SBX crossover.
        
        Args:
            parent1: First parent (n_var,)
            parent2: Second parent (n_var,)
        
        Returns:
            Two children
        """
        n_var = len(parent1)
        
        # Initialize children as copies of parents
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Decide whether to perform crossover
        if np.random.rand() > self.prob:
            return child1, child2
        
        for i in range(n_var):
            # Skip if parents are identical for this variable
            if abs(parent1[i] - parent2[i]) < 1e-14:
                continue
            
            # Generate random number
            u = np.random.rand()
            
            # Calculate beta
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (self.eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
            
            # Create children
            child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
            child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i])
        
        return child1, child2
```

**File: `src/operators/mutation.py`**

```python
"""
Mutation operators for evolutionary algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, individual: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
        """Apply mutation to individual."""
        pass


class PolynomialMutation(MutationOperator):
    """
    Polynomial Mutation for continuous variables.
    
    Reference:
    Deb & Goyal (1996). A combined genetic adaptive search for robust design.
    """
    
    def __init__(self, eta: float = 20.0, prob: float = None):
        """
        Initialize polynomial mutation.
        
        Args:
            eta: Distribution index
            prob: Mutation probability per variable (default: 1/n_var)
        """
        self.eta = eta
        self.prob = prob
    
    def mutate(self, individual: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
        """
        Apply polynomial mutation.
        
        Args:
            individual: Individual to mutate
            xl: Lower bounds
            xu: Upper bounds
        
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        n_var = len(individual)
        
        # Default probability
        prob = self.prob if self.prob is not None else 1.0 / n_var
        
        for i in range(n_var):
            if np.random.rand() > prob:
                continue
            
            y = mutated[i]
            yl = xl[i]
            yu = xu[i]
            
            if yl == yu:
                continue
            
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            
            rand = np.random.rand()
            mut_pow = 1.0 / (self.eta + 1.0)
            
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (self.eta + 1.0)
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (self.eta + 1.0)
                deltaq = 1.0 - val ** mut_pow
            
            y = y + deltaq * (yu - yl)
            y = np.clip(y, yl, yu)
            
            mutated[i] = y
        
        return mutated


class GaussianMutation(MutationOperator):
    """Gaussian mutation with adaptive sigma."""
    
    def __init__(self, sigma: float = 0.1, prob: float = None):
        self.sigma = sigma
        self.prob = prob
    
    def mutate(self, individual: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        n_var = len(individual)
        
        prob = self.prob if self.prob is not None else 1.0 / n_var
        
        for i in range(n_var):
            if np.random.rand() > prob:
                continue
            
            # Add Gaussian noise
            noise = np.random.normal(0, self.sigma * (xu[i] - xl[i]))
            mutated[i] = np.clip(individual[i] + noise, xl[i], xu[i])
        
        return mutated
```

### 4.3 Week 6-7: Remaining Algorithms (Abbreviated Structure)

Due to space, I'll show the structure for MOEA/D and AdaVEA-MOO:

**File: `src/algorithms/moead.py`**

```python
"""
MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.
"""

class MOEAD(BaseAlgorithm):
    def __init__(self, problem, config):
        super().__init__(problem, config)
        
        # Generate weight vectors (Das-Dennis or uniform)
        self.weight_vectors = self._generate_weights()
        
        # Define neighborhood structure
        self.neighborhoods = self._compute_neighborhoods()
        
        # Initialize ideal point
        self.ideal_point = np.full(problem.metadata.n_obj, np.inf)
    
    def _generate_weights(self):
        """Generate uniformly distributed weight vectors."""
        # Das-Dennis method for 3 objectives
        pass
    
    def _compute_neighborhoods(self):
        """Compute T nearest neighbors for each weight vector."""
        # Euclidean distance in weight space
        pass
    
    def _tchebycheff(self, F, weights, ideal):
        """Tchebycheff scalarizing function."""
        return np.max(weights * np.abs(F - ideal))
    
    def evolve(self):
        """Main MOEA/D loop with neighborhood mating."""
        pass
```

**File: `src/algorithms/adavea_moo.py`**

```python
"""
AdaVEA-MOO: Your custom adaptive algorithm.
"""

class AdaVEAMOO(NSGA2):  # Inherit from NSGA-II
    def __init__(self, problem, config):
        super().__init__(problem, config)
        
        # Adaptive parameters
        self.p_m = config.mutation_prob
        self.p_c = config.crossover_prob
        self.sigma_target = 0.3
        
        # Ensemble mutation strategies
        self.mutation_strategies = [
            GaussianMutation(sigma=0.1),
            CauchyMutation(gamma=0.05),
            CostAwareMutation(cost_coefficients),
            SparsityAwareMutation(threshold=0.1)
        ]
        self.strategy_weights = np.ones(4) / 4
    
    def adapt_parameters(self, population):
        """Adapt mutation and crossover rates based on diversity."""
        sigma_div = self._calculate_diversity(population)
        
        if sigma_div < self.sigma_target:
            self.p_m = min(self.p_m + 0.005, 0.1)
        else:
            self.p_m = max(self.p_m - 0.002, 0.01)
        
        # Time-varying crossover
        t = self.current_generation
        self.p_c = 0.5 + 0.5 * np.exp(-t / (self.config.max_generations / 4))
    
    def select_mutation_strategy(self):
        """Select mutation strategy based on adaptive weights."""
        return np.random.choice(self.mutation_strategies, p=self.strategy_weights)
    
    def local_search(self, population, top_k=0.1):
        """Apply Lamarckian/Baldwinian learning to top solutions."""
        pass
    
    def evolve(self):
        """Enhanced evolution with all adaptive mechanisms."""
        # Similar to NSGA-II but with adaptive components
        pass
```

---

## 5. PHASE 3: CLI INTERFACE DEVELOPMENT (Week 8)

### 5.1 CLI Application Structure

**File: `src/cli/app.py`**

```python
"""
Main CLI application using Typer.
"""
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from pathlib import Path
from typing import Optional
import sys

from .commands import run, analyze, visualize
from ..utils.config import ConfigurationManager
from ..utils.logging_config import setup_logging

# Create Typer app
app = typer.Typer(
    name="DVA Optimization",
    help="Multi-Objective Optimization System for Dynamic Vibration Absorbers",
    add_completion=True
)

# Rich console for beautiful output
console = Console()

# Add command groups
app.add_typer(run.app, name="run", help="Run optimization experiments")
app.add_typer(analyze.app, name="analyze", help="Analyze results")
app.add_typer(visualize.app, name="plot", help="Generate visualizations")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    edit: Optional[Path] = typer.Option(None, "--edit", help="Edit configuration file")
):
    """Manage configuration files."""
    config_mgr = ConfigurationManager()
    
    if show:
        current = config_mgr.get()
        console.print("[bold green]Current Configuration:[/bold green]")
        console.print(current.to_dict())
    
    if edit:
        console.print(f"Opening {edit} in editor...")
        typer.launch(str(edit))


@app.command()
def version():
    """Show version information."""
    console.print("[bold blue]DVA Optimization System v1.0.0[/bold blue]")
    console.print("Python-based Multi-Objective Evolutionary Algorithms")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
```

**File: `src/cli/commands/run.py`**

```python
"""
Run command for executing optimization experiments.
"""
import typer
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, 
    TextColumn, TimeRemainingColumn
)
from pathlib import Path
from typing import Optional
import numpy as np

from ...services.optimization_service import OptimizationService
from ...utils.config import ExperimentConfig

app = typer.Typer()
console = Console()


@app.command("single")
def run_single(
    algorithm: str = typer.Argument(..., help="Algorithm name (nsga2, moead, adavea)"),
    problem: str = typer.Argument("dva", help="Problem name"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    pop_size: int = typer.Option(100, "--pop-size", "-n", help="Population size"),
    max_gen: int = typer.Option(2000, "--max-gen", "-g", help="Maximum generations"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    output: Path = typer.Option("results", "--output", "-o", help="Output directory")
):
    """
    Run a single optimization experiment.
    
    Example:
        dva-opt run single nsga2 --pop-size 100 --max-gen 2000
    """
    console.print(f"[bold green]Running {algorithm.upper()} on {problem}[/bold green]")
    
    # Load or create configuration
    if config_file:
        config = ExperimentConfig.load(config_file)
    else:
        # Create default config
        config = create_default_config(algorithm, pop_size, max_gen, seed)
    
    # Initialize optimization service
    opt_service = OptimizationService()
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        
        task = progress.add_task(f"Optimizing ({max_gen} generations)", total=max_gen)
        
        def callback(gen, population):
            """Update progress bar."""
            progress.update(task, completed=gen + 1)
            
            if gen % 50 == 0:
                pf = population.get_pareto_front()
                console.print(
                    f"Gen {gen}: Pareto Front Size = {len(pf)}"
                )
        
        # Run optimization
        result = opt_service.run_algorithm(
            algorithm_name=algorithm,
            config=config,
            callback=callback
        )
    
    # Save results
    output_path = Path(output) / f"{algorithm}_{problem}_single.h5"
    opt_service.save_result(result, output_path)
    
    console.print(f"[bold green]✓ Results saved to {output_path}[/bold green]")


@app.command("batch")
def run_batch(
    config_file: Path = typer.Argument(..., help="Experiment configuration file"),
    n_runs: int = typer.Option(30, "--runs", "-r", help="Number of independent runs"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel workers"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint")
):
    """
    Run multiple independent trials for statistical analysis.
    
    Example:
        dva-opt run batch config.yaml --runs 30 --parallel 8
    """
    console.print(f"[bold green]Running batch experiment with {n_runs} runs[/bold green]")
    
    # Load configuration
    config = ExperimentConfig.load(config_file)
    config.n_runs = n_runs
    config.parallel_workers = parallel
    
    # Run batch
    opt_service = OptimizationService()
    results = opt_service.run_batch(config, resume=resume)
    
    # Display summary statistics
    display_batch_summary(results)


def display_batch_summary(results):
    """Display summary table of batch results."""
    from rich.table import Table
    
    table = Table(title="Batch Results Summary")
    table.add_column("Run", justify="right", style="cyan")
    table.add_column("HV", justify="right", style="green")
    table.add_column("IGD+", justify="right", style="yellow")
    table.add_column("Time (s)", justify="right", style="magenta")
    
    for i, result in enumerate(results):
        table.add_row(
            f"{i+1}",
            f"{result['hv']:.4f}",
            f"{result['igd_plus']:.4f}",
            f"{result['time']:.1f}"
        )
    
    console.print(table)
```

---

## 6. PHASE 4: GUI DEVELOPMENT (Weeks 9-12)

### 6.1 Week 9: Main Window Structure

**File: `src/gui/main_window.py`**

```python
"""
Main window for GUI application using PyQt6.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QToolBar, QStatusBar,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon

from .views.experiment_view import ExperimentView
from .views.config_view import ConfigurationView
from .views.results_view import ResultsView
from .views.plot_view import PlotView
from .controllers.experiment_controller import ExperimentController


class MainWindow(QMainWindow):
    """
    Main application window with tabbed interface.
    
    Tabs:
    - Experiment: Configure and run experiments
    - Results: View and analyze results
    - Plots: Interactive visualizations
    - Configuration: Edit algorithm parameters
    """
    
    def __init__(self):
        super().__init__()
        
        # Window properties
        self.setWindowTitle("DVA Multi-Objective Optimization System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize controller
        self.controller = ExperimentController()
        
        # Setup UI components
        self.setup_menubar()
        self.setup_toolbar()
        self.setup_central_widget()
        self.setup_statusbar()
        
        # Connect signals
        self.connect_signals()
    
    def setup_menubar(self):
        """Create menu bar with File, Edit, View, Help menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Experiment", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_experiment)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_experiment)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_experiment)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        # Add view options
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Create toolbar with common actions."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        run_action = QAction("▶ Run", self)
        run_action.triggered.connect(self.run_optimization)
        toolbar.addAction(run_action)
        
        stop_action = QAction("⏹ Stop", self)
        stop_action.triggered.connect(self.stop_optimization)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        plot_action = QAction("📊 Plot", self)
        plot_action.triggered.connect(self.show_plots)
        toolbar.addAction(plot_action)
    
    def setup_central_widget(self):
        """Create tabbed central widget."""
        # Create tab widget
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create views
        self.experiment_view = ExperimentView(self.controller)
        self.config_view = ConfigurationView(self.controller)
        self.results_view = ResultsView(self.controller)
        self.plot_view = PlotView(self.controller)
        
        # Add tabs
        tabs.addTab(self.experiment_view, "🧪 Experiment")
        tabs.addTab(self.results_view, "📊 Results")
        tabs.addTab(self.plot_view, "📈 Visualizations")
        tabs.addTab(self.config_view, "⚙️ Configuration")
        
        # Set as central widget
        self.setCentralWidget(tabs)
    
    def setup_statusbar(self):
        """Create status bar."""
        statusbar = self.statusBar()
        statusbar.showMessage("Ready")
    
    def connect_signals(self):
        """Connect controller signals to view updates."""
        self.controller.progress_updated.connect(self.update_progress)
        self.controller.optimization_finished.connect(self.on_optimization_finished)
        self.controller.error_occurred.connect(self.show_error)
    
    def new_experiment(self):
        """Create new experiment."""
        self.experiment_view.clear()
        self.statusBar().showMessage("New experiment created")
    
    def open_experiment(self):
        """Open existing experiment."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Experiment",
            "",
            "Experiment Files (*.yaml *.yml)"
        )
        
        if filename:
            self.controller.load_experiment(filename)
            self.statusBar().showMessage(f"Loaded: {filename}")
    
    def save_experiment(self):
        """Save current experiment."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment",
            "",
            "Experiment Files (*.yaml)"
        )
        
        if filename:
            self.controller.save_experiment(filename)
            self.statusBar().showMessage(f"Saved: {filename}")
    
    def run_optimization(self):
        """Start optimization run."""
        self.controller.start_optimization()
        self.statusBar().showMessage("Optimization running...")
    
    def stop_optimization(self):
        """Stop optimization."""
        self.controller.stop_optimization()
        self.statusBar().showMessage("Optimization stopped")
    
    def show_plots(self):
        """Show visualization tab."""
        tabs = self.centralWidget()
        tabs.setCurrentWidget(self.plot_view)
    
    def update_progress(self, generation, metrics):
        """Update progress display."""
        self.statusBar().showMessage(
            f"Generation {generation}: HV={metrics.get('hv', 0):.4f}"
        )
    
    def on_optimization_finished(self, results):
        """Handle optimization completion."""
        QMessageBox.information(
            self,
            "Optimization Complete",
            f"Optimization finished successfully!\n\n"
            f"Pareto solutions: {results['n_pareto']}\n"
            f"Total time: {results['time']:.1f}s"
        )
        self.statusBar().showMessage("Optimization complete")
    
    def show_error(self, error_msg):
        """Display error message."""
        QMessageBox.critical(self, "Error", error_msg)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About DVA Optimization",
            "DVA Multi-Objective Optimization System\n\n"
            "Version 1.0.0\n\n"
            "Comprehensive framework for optimizing Dynamic Vibration Absorbers\n"
            "using advanced evolutionary algorithms."
        )
```

### 6.2 Week 10: Experiment View with Real-Time Plotting

**File: `src/gui/views/experiment_view.py`**

```python
"""
Experiment configuration and execution view.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QProgressBar, QTextEdit
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg


class ExperimentView(QWidget):
    """
    View for configuring and running experiments.
    
    Layout:
    - Left: Configuration panel
    - Center: Real-time convergence plot
    - Right: Log output
    """
    
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        self.setup_ui()
        self.connect_controller()
    
    def setup_ui(self):
        """Create UI layout."""
        main_layout = QHBoxLayout()
        
        # Left panel: Configuration
        config_panel = self.create_config_panel()
        main_layout.addWidget(config_panel, stretch=1)
        
        # Center panel: Real-time plot
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, stretch=2)
        
        # Right panel: Log
        log_panel = self.create_log_panel()
        main_layout.addWidget(log_panel, stretch=1)
        
        self.setLayout(main_layout)
    
    def create_config_panel(self) -> QGroupBox:
        """Create configuration panel."""
        group = QGroupBox("Experiment Configuration")
        layout = QVBoxLayout()
        
        # Algorithm selection
        layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "NSGA-II", "NSGA-III", "MOEA/D", "SPEA2", "AdaVEA-MOO"
        ])
        layout.addWidget(self.algorithm_combo)
        
        # Problem selection
        layout.addWidget(QLabel("Problem:"))
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(["DVA 48-parameter", "ZDT1", "DTLZ2"])
        layout.addWidget(self.problem_combo)
        
        # Population size
        layout.addWidget(QLabel("Population Size:"))
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(10, 1000)
        self.pop_size_spin.setValue(100)
        layout.addWidget(self.pop_size_spin)
        
        # Max generations
        layout.addWidget(QLabel("Max Generations:"))
        self.max_gen_spin = QSpinBox()
        self.max_gen_spin.setRange(10, 10000)
        self.max_gen_spin.setValue(2000)
        layout.addWidget(self.max_gen_spin)
        
        # Number of runs
        layout.addWidget(QLabel("Number of Runs:"))
        self.n_runs_spin = QSpinBox()
        self.n_runs_spin.setRange(1, 100)
        self.n_runs_spin.setValue(1)
        layout.addWidget(self.n_runs_spin)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("▶ Run")
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setEnabled(False)
        
        self.run_button.clicked.connect(self.on_run_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        group.setLayout(layout)
        
        return group
    
    def create_plot_panel(self) -> QGroupBox:
        """Create real-time convergence plot."""
        group = QGroupBox("Convergence")
        layout = QVBoxLayout()
        
        # PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Hypervolume')
        self.plot_widget.setLabel('bottom', 'Generation')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot curve
        self.hv_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='b', width=2),
            name='Hypervolume'
        )
        
        layout.addWidget(self.plot_widget)
        group.setLayout(layout)
        
        return group
    
    def create_log_panel(self) -> QGroupBox:
        """Create log output panel."""
        group = QGroupBox("Log")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumWidth(300)
        
        layout.addWidget(self.log_text)
        group.setLayout(layout)
        
        return group
    
    def connect_controller(self):
        """Connect to controller signals."""
        self.controller.progress_updated.connect(self.update_plot)
        self.controller.log_message.connect(self.append_log)
        self.controller.optimization_started.connect(self.on_started)
        self.controller.optimization_finished.connect(self.on_finished)
    
    def on_run_clicked(self):
        """Handle run button click."""
        config = self.get_configuration()
        self.controller.configure_experiment(config)
        self.controller.start_optimization()
    
    def on_stop_clicked(self):
        """Handle stop button click."""
        self.controller.stop_optimization()
    
    def get_configuration(self) -> dict:
        """Get current configuration from UI."""
        return {
            'algorithm': self.algorithm_combo.currentText(),
            'problem': self.problem_combo.currentText(),
            'population_size': self.pop_size_spin.value(),
            'max_generations': self.max_gen_spin.value(),
            'n_runs': self.n_runs_spin.value()
        }
    
    def update_plot(self, generation, metrics):
        """Update convergence plot."""
        hv = metrics.get('hv', [])
        generations = list(range(len(hv)))
        
        self.hv_curve.setData(generations, hv)
        
        # Update progress bar
        max_gen = self.max_gen_spin.value()
        self.progress_bar.setValue(int(100 * generation / max_gen))
    
    def append_log(self, message):
        """Append message to log."""
        self.log_text.append(message)
    
    def on_started(self):
        """Handle optimization start."""
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.append_log("Optimization started...")
    
    def on_finished(self):
        """Handle optimization completion."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.append_log("Optimization finished!")
```

---

This roadmap continues with additional phases covering testing, optimization, and deployment. The complete system follows MVC architecture with comprehensive separation of concerns, making it highly maintainable and scalable.

**Key remaining tasks** (summarized):
- Week 11-12: Complete remaining GUI views (Results, Plots, Config)
- Week 13-14: Integration testing, unit tests
- Week 15: Performance optimization (parallel execution, caching)
- Week 16: Documentation (Sphinx), packaging (setup.py), deployment

Would you like me to continue with any specific phase in more detail?