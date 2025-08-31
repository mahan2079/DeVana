from .interface import BeamOptimizationInterface

__all__ = ['BeamOptimizationInterface']

"""
Composite Beam Analysis UI Components

This package provides comprehensive UI components for composite beam analysis including:
- Main composite beam interface with layer management
- Enhanced cross-section visualization with material properties
- Professional beam side view with engineering drawing style
- Advanced force visualization with time-varying forces
- Material database for common engineering materials
- Results dashboard and animation widgets
"""

# Main composite beam interface
from .composite_beam_interface import CompositeBeamInterface

# Enhanced visualization widgets
from .enhanced_cross_section_visualizer import EnhancedCrossSectionVisualizer
from .beam_side_view import BeamSideViewWidget
from .force_visualization import ForceVisualizationWidget

# Material and data management
from .material_database import MaterialDatabase

# Existing UI components
from .cross_section_visualizer import CrossSectionVisualizer
from .layer_dialog import LayerDialog
from .force_region_dialog import ForceRegionDialog
from .force_regions_panel import ForceRegionsPanel
from .force_widgets import (
    createDistributedSpatialWidget,
    createPointSpatialWidget,
    createForceWidget
)
from .results_dashboard import ResultsDashboard
from .beam_animation import BeamAnimationWidget
from .mode_shape_animation import ModeShapeAnimationWidget
from .plot_canvas import PlotCanvas
from .scrollable_form_widget import ScrollableFormWidget

# Export all public components
__all__ = [
    # Main interface
    'CompositeBeamInterface',
    
    # Enhanced visualization
    'EnhancedCrossSectionVisualizer',
    'BeamSideViewWidget', 
    'ForceVisualizationWidget',
    
    # Material database
    'MaterialDatabase',
    
    # Existing components
    'CrossSectionVisualizer',
    'LayerDialog',
    'ForceRegionDialog',
    'ForceRegionsPanel',
    'createDistributedSpatialWidget',
    'createPointSpatialWidget',
    'createForceWidget',
    'ResultsDashboard',
    'BeamAnimationWidget',
    'ModeShapeAnimationWidget',
    'PlotCanvas',
    'ScrollableFormWidget',
] 