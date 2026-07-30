from dataclasses import dataclass
from typing import List

@dataclass
class TargetWeight:
    values: List[float]
    weights: List[float]

@dataclass
class FrequencyRange:
    start: float
    end: float
    points: int

@dataclass
class PhysicsConfig:
    main_system_parameters: List[float]
    target_values_weights: List[TargetWeight]
    frequency_range: FrequencyRange

@dataclass
class SparsityConfig:
    """
    Sparsity configuration containing penalties for the layout.
    - tau (\\tau): Sparsity threshold penalty.
    - alpha (\\alpha): Coefficient for the primary sparsity metric.
    - beta (\\beta): Coefficient for the secondary sparsity metric.
    """
    tau: float
    alpha: float
    beta: float

@dataclass
class NSGAIIHyperparameters:
    """
    NSGA-II specific parameters.
    - eta_c (\\eta_c): Crowding distance distribution index for crossover.
    - eta_m (\\eta_m): Distribution index for polynomial mutation.
    """
    pop_size: int
    generations: int
    cxpb: float
    mutpb: float
    eta_c: float
    eta_m: float
    indpb: float
    num_runs: int

@dataclass
class OptimizationConfig:
    physics: PhysicsConfig
    sparsity: SparsityConfig
    hyperparameters: NSGAIIHyperparameters
