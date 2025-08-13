"""
Composite Beam Analysis Module for DeVana

This module provides advanced composite beam analysis capabilities including:
- Multi-layer composite beam modeling
- Temperature-dependent material properties
- Finite element analysis with Euler-Bernoulli beam theory
- Dynamic analysis and modal analysis
- Interactive visualization tools
- Professional GUI interface for beam design
"""

# Core analysis functions
from .beam.solver import solve_beam_vibration, BeamVibrationSolver
from .beam.properties import calc_composite_properties
from .beam.fem import BeamAssembler

# Utility functions for force management and expression parsing
from .utils import (
    ForceRegionManager, 
    ForceRegion,
    get_force_generators, 
    parse_expression,
    create_harmonic_force,
    create_step_force,
    create_impulse_force,
    create_custom_force
)

# Main UI components for integration
from .ui.composite_beam_interface import CompositeBeamInterface
from .ui.enhanced_cross_section_visualizer import EnhancedCrossSectionVisualizer
from .ui.beam_side_view import BeamSideViewWidget
from .ui.force_visualization import ForceVisualizationWidget
from .ui.material_database import MaterialDatabase

# Animation adapters for easy integration
from .beam_animation_adapter import BeamAnimationAdapter
from .mode_shape_adapter import ModeShapeAdapter

# Additional UI components
from .ui.layer_dialog import LayerDialog
from .ui.results_dashboard import ResultsDashboard
from .ui.cross_section_visualizer import CrossSectionVisualizer

# Version information
__version__ = "V0.4.1"
__author__ = "DeVana Development Team"
__description__ = "Advanced composite beam analysis with temperature-dependent material properties"

# Export the main public API
__all__ = [
    # Core analysis
    'solve_beam_vibration',
    'BeamVibrationSolver', 
    'calc_composite_properties',
    'BeamAssembler',
    
    # Force management
    'ForceRegionManager',
    'ForceRegion',
    'get_force_generators',
    'parse_expression',
    'create_harmonic_force',
    'create_step_force', 
    'create_impulse_force',
    'create_custom_force',
    
    # Main UI components
    'CompositeBeamInterface',
    'EnhancedCrossSectionVisualizer',
    'BeamSideViewWidget',
    'ForceVisualizationWidget',
    'MaterialDatabase',
    
    # Animation adapters
    'BeamAnimationAdapter',
    'ModeShapeAdapter',
    
    # Additional UI components
    'LayerDialog',
    'ResultsDashboard',
    'CrossSectionVisualizer',
    
    # Module info
    '__version__',
    '__author__',
    '__description__'
]

# Convenience function for creating the main composite beam interface
def create_composite_beam_interface(parent=None):
    """
    Create a fully configured composite beam interface.
    
    This function provides a convenient way to create the main composite beam
    analysis interface with all features enabled.
    
    Parameters:
    -----------
    parent : QWidget, optional
        Parent widget for the interface
        
    Returns:
    --------
    CompositeBeamInterface
        Configured composite beam interface ready for use
    """
    return CompositeBeamInterface(parent)

def get_available_materials():
    """
    Get a list of all available materials in the database.
    
    Returns:
    --------
    list
        List of available material names
    """
    db = MaterialDatabase()
    return db.get_material_names()

def create_example_composite():
    """
    Create an example composite beam configuration for demonstration.
    
    Returns:
    --------
    dict
        Example composite configuration with layers and parameters
    """
    return {
        'name': 'Example Steel-Aluminum Composite',
        'layers': [
            {
                'name': 'Steel Bottom',
                'thickness': 0.003,  # 3mm
                'material_type': 'Steel',
                'E_func': lambda T: 210e9 * (1 - 0.0001 * T),
                'rho_func': lambda T: 7800 * (1 + 0.00001 * T)
            },
            {
                'name': 'Aluminum Core',
                'thickness': 0.004,  # 4mm
                'material_type': 'Aluminum',
                'E_func': lambda T: 70e9 * (1 - 0.0002 * T),
                'rho_func': lambda T: 2700 * (1 + 0.00002 * T)
            },
            {
                'name': 'Steel Top',
                'thickness': 0.003,  # 3mm
                'material_type': 'Steel',
                'E_func': lambda T: 210e9 * (1 - 0.0001 * T),
                'rho_func': lambda T: 7800 * (1 + 0.00001 * T)
            }
        ],
        'beam_width': 0.05,  # 50mm
        'beam_length': 1.0,  # 1m
        'temperature': 20.0  # 20°C
    }

# Add convenience functions to __all__
__all__.extend([
    'create_composite_beam_interface',
    'get_available_materials', 
    'create_example_composite'
]) 