"""
Beam module for finite element analysis of beams.
"""

from .solver import solve_beam_vibration, BeamVibrationSolver
from .fem import BeamAssembler
from .properties import calc_composite_properties 