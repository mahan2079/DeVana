"""
Continuous Beam Vibration Analysis Module
"""

# Import main components for convenience
from .utils import ForceRegionManager, get_force_generators, parse_expression
from .beam.solver import solve_beam_vibration 