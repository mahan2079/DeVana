"""
Continuous Beam Optimization Module for DeVana

Provides two optimization modes for an Euler–Bernoulli beam with ground-connected
linear springs and viscous dampers:

1) Values-only optimization at user-selected locations
2) Placement + values optimization

Public API:
- create_beam_optimization_interface(parent=None): returns the main PyQt widget

Backend modules:
- backend.model: BeamModel, TargetSpecification
- backend.optimizers: optimize_values_only, optimize_placement_and_values
"""

from .backend.model import BeamModel, TargetSpecification
from .backend.optimizers import optimize_values_only, optimize_placement_and_values
from .ui.interface import BeamOptimizationInterface

__all__ = [
    'BeamModel',
    'TargetSpecification',
    'optimize_values_only',
    'optimize_placement_and_values',
    'BeamOptimizationInterface',
    'create_beam_optimization_interface',
]


def create_beam_optimization_interface(parent=None):
    return BeamOptimizationInterface(parent=parent)



