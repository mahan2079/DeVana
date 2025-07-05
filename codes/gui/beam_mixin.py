from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import pandas as pd

from gui.widgets import ModernQTabWidget
from Continues_beam.ui.cross_section_visualizer import CrossSectionVisualizer
from Continues_beam.ui.force_regions_panel import ForceRegionsPanel

# Check if beam module imports are successful
BEAM_IMPORTS_SUCCESSFUL = True
try:
    from Continues_beam.utils import ForceRegionManager
except ImportError:
    BEAM_IMPORTS_SUCCESSFUL = False

class ContinuousBeamMixin:
    def create_continuous_beam_page(self):
        """Create the continuous beam analysis page"""
        if not BEAM_IMPORTS_SUCCESSFUL:
            # Create placeholder page if imports failed
            beam_page = QWidget()
            layout = QVBoxLayout(beam_page)
            
            # Centered content
            center_widget = QWidget()
            center_layout = QVBoxLayout(center_widget)
            center_layout.setAlignment(Qt.AlignCenter)
            
            # Error message
            error_label = QLabel("Continuous Beam Module Not Available")
            error_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
            error_label.setAlignment(Qt.AlignCenter)
            center_layout.addWidget(error_label)
            
            description = QLabel("Please make sure the 'Continues beam' module is correctly installed.")
            description.setFont(QFont("Segoe UI", 12))
            description.setAlignment(Qt.AlignCenter)
            center_layout.addWidget(description)
            
            layout.addWidget(center_widget)
            self.content_stack.addWidget(beam_page)
            return
