"""
DeVana: Dynamic Vibration Absorber Optimization Framework
Standalone Python Library
"""

__version__ = "1.0.0"

# Import core physics and continuous beam modeling
from .physics.frf import (
    process_mass,
    remove_zero_mass_dofs,
    # Export other critical FRF functions if needed
)
from .physics.beam import BeamModel
from .physics.beam_optimize import optimize_values_at_locations, optimize_placement_and_values
from .systems.dva_system import DVASystem

# Import Data Models
from .models import (
    DVAConfiguration,
    OptimizationRequest,
)

# Import Optimization Solvers
from .optimize import (
    GASolver,
    PSOSolver,
    CMAESSolver,
    NSGA2Solver,
    DESolver,
    SASolver,
    AdaVEASolver,
    MOGASolver,
    RLSolver,
)

# Import Machine Learning Seeders and Surrogate
from .ml.seeding import MemorySeeder, NeuralSeeder
from .ml.surrogate import NeuralSurrogate
from .ml.pinn import PINNSolver, PhysicsInformedFRF

# Import Sensitivity Analysis
from .sensitivity.sobol import perform_sobol_analysis

# Import Utils
from .utils.metrics import get_hardware_profile, get_resource_usage

__all__ = [
    # Physics
    "process_mass",
    "remove_zero_mass_dofs",
    "BeamModel",
    "DVASystem",
    "optimize_values_at_locations",
    "optimize_placement_and_values",
    
    # Models
    "DVAConfiguration",
    "OptimizationRequest",
    
    # Optimization
    "GASolver",
    "PSOSolver",
    "CMAESSolver",
    "NSGA2Solver",
    "DESolver",
    "SASolver",
    "AdaVEASolver",
    "MOGASolver",
    "RLSolver",
    
    # ML
    "MemorySeeder",
    "NeuralSeeder",
    "NeuralSurrogate",
    "PINNSolver",
    "PhysicsInformedFRF",
    
    # Sensitivity
    "perform_sobol_analysis",

    # Utils
    "get_hardware_profile",
    "get_resource_usage"
]
