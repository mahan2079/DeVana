
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import computational_metrics_new  # Added import for computational metrics visualization

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup,
    QStackedWidget, QFrame, QListWidget, QListWidgetItem, QApplication, QGraphicsDropShadowEffect,
    QRadioButton, QButtonGroup, QGridLayout, QDialogButtonBox, QDialog, QGraphicsOpacityEffect,
    QToolButton, QStyle, QStyledItemDelegate, QProgressBar, QInputDialog, QColorDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint, QPropertyAnimation, QEasingCurve, QRect, QTimer, QDateTime
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont, QPixmap, QCursor, QPainter, QBrush, QLinearGradient, QMovie

# Matplotlib backends
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Local imports from "modules" subfolder
from modules.FRF import frf
from modules.plotwindow import PlotWindow

# Local imports from "workers" subfolder
from workers.FRFWorker import FRFWorker
from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker, TopologyType
from workers.DEWorker import DEWorker
from workers.SAWorker import SAWorker
from workers.CMAESWorker import CMAESWorker
# RL module import removed

from gui.menu_mixin import MenuMixin
from gui.beam_mixin import ContinuousBeamMixin
from gui.microchip_mixin import MicrochipPageMixin
from gui.main_window.theme_mixin import ThemeMixin
from gui.main_window.ga_mixin import GAOptimizationMixin
from gui.main_window.frf_mixin import FRFMixin
from gui.main_window.pso_mixin import PSOMixin
from gui.main_window.input_mixin import InputTabsMixin
from gui.main_window.extra_opt_mixin import ExtraOptimizationMixin
from gui.main_window.sidebar_mixin import SidebarMixin
from gui.main_window.stochastic_mixin import StochasticDesignMixin
from gui.main_window.sobol_mixin import SobolAnalysisMixin


# Additional libraries used
import random
from deap import base, creator, tools

# Seaborn style and LaTeX rendering
sns.set(style="whitegrid")
plt.rc('text', usetex=True)

# Import Continuous Beam functionality
try:
    from Continues_beam.beam_animation_adapter import BeamAnimationAdapter
    from Continues_beam.mode_shape_adapter import ModeShapeAdapter
    from Continues_beam.beam.solver import solve_beam_vibration
    from Continues_beam.utils import parse_expression, ForceRegionManager, get_force_generators
    from Continues_beam.ui.force_region_dialog import ForceRegionDialog
    from Continues_beam.ui.force_widgets import createDistributedSpatialWidget, createPointSpatialWidget, createForceWidget
    from Continues_beam.ui.force_regions_panel import ForceRegionsPanel
    from Continues_beam.ui.cross_section_visualizer import CrossSectionVisualizer
    from Continues_beam.ui.layer_dialog import LayerDialog
    from Continues_beam.ui.plot_canvas import PlotCanvas
    BEAM_IMPORTS_SUCCESSFUL = True
except ImportError:
    BEAM_IMPORTS_SUCCESSFUL = False
    print("Warning: Continuous Beam module imports failed. This functionality will be disabled.")


        

class MainWindow(QMainWindow, MenuMixin, ContinuousBeamMixin, MicrochipPageMixin,
                 ThemeMixin, FRFMixin, PSOMixin, GAOptimizationMixin,
                 InputTabsMixin, ExtraOptimizationMixin,
                 SidebarMixin, StochasticDesignMixin, SobolAnalysisMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeVana")
        self.resize(1600, 900)
        
        # Disable LaTeX rendering in matplotlib to prevent Unicode errors with Greek characters
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        
        # Initialize theme
        self.current_theme = 'Dark'  # Changed to default dark theme
        
        # Initialize variables for comparative visualization
        self.available_plots_list = None
        self.legend_table = None
        self.legend_map = {}
        self.frf_plots = {}
        self.comp_fig = None
        self.comp_canvas = None
        
        # Create central widget with main layout
        central_widget = QWidget()
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(central_widget)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create stacked widget for main content
        self.content_stack = QStackedWidget()
        self.main_layout.addWidget(self.content_stack, 1)
        
        # Create the various content pages
        self.create_stochastic_design_page()
        self.create_microchip_controller_page()
        self.create_continuous_beam_page()
        
        # Set default active page
        self.content_stack.setCurrentIndex(0)
        
        # Apply theme
        self.apply_dark_theme()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create menubar and toolbar
        self.create_menubar()
        self.create_toolbar()
        
        self.current_ga_best_params = None
        self.current_ga_best_fitness = None
        self.current_ga_full_results = None
        self.current_ga_settings = None

        # Initialize other optimization results holders if they follow a similar pattern
        self.current_pso_best_params = None
    
    def set_default_values(self):
        """Reset all inputs to their default values"""
        self.status_bar.showMessage("Reset to default values")
        # Reset logic for all parameters would be implemented here


