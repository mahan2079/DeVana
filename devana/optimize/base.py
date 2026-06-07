from abc import ABC, abstractmethod

class Solver(ABC):
    """
    Base class for all optimization solvers in DeVana.
    
    This class provides a common interface for optimization algorithms,
    decoupling them from GUI frameworks (like PyQt5) and specific physics implementations.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        """
        Initialize the solver.
        
        Args:
            config (dict): Configuration parameters for the solver.
            evaluate_fn (callable, optional): A function that evaluates an individual.
                If not provided, the solver should implement its own evaluation logic
                or expect it to be passed to the solve() method.
            callback (callable, optional): A function called periodically with 
                progress updates. signature: (generation, best_fitness, best_individual, metrics)
        """
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.callback = callback
        self.stop_requested = False
        
        # Common configuration parameters (with defaults)
        self.pop_size = config.get('pop_size', 50)
        self.num_generations = config.get('num_generations', 100)
        self.parameter_data = config.get('parameter_data', [])
        self.tolerance = config.get('tolerance', 1e-6)
        
        # Parse parameter bounds
        self.parameter_names = []
        self.parameter_bounds = []
        self.fixed_parameters = {}
        
        for idx, (name, low, high, fixed) in enumerate(self.parameter_data):
            self.parameter_names.append(name)
            if fixed:
                self.parameter_bounds.append((low, low))
                self.fixed_parameters[idx] = low
            else:
                self.parameter_bounds.append((low, high))
        
        self.num_parameters = len(self.parameter_bounds)

    def request_stop(self):
        """Request the solver to stop gracefully."""
        self.stop_requested = True

    def _report_progress(self, generation, best_fitness, best_individual, metrics=None):
        """Report progress via the callback function."""
        if self.callback:
            self.callback(generation, best_fitness, best_individual, metrics)

    def evaluate(self, individual):
        """
        Evaluate an individual using the provided evaluate_fn.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            The fitness of the individual.
        """
        if self.evaluate_fn:
            return self.evaluate_fn(individual)
        raise NotImplementedError("evaluate_fn not provided and evaluate() not overridden.")

    @abstractmethod
    def solve(self):
        """
        Perform the optimization.
        
        Returns:
            dict: A dictionary containing the results of the optimization,
                including the best solution found and performance metrics.
        """
        pass
