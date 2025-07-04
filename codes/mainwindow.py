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
        self.create_sidebar(BEAM_IMPORTS_SUCCESSFUL)
        
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
        self.status_bar.showMessage("Resetting to default values...")
        
        try:
            # Reset PSO parameters
            self.pso_swarm_size_box.setValue(40)
            self.pso_num_iterations_box.setValue(100)
            self.pso_inertia_box.setValue(0.729)
            self.pso_cognitive_box.setValue(1.49445)
            self.pso_social_box.setValue(1.49445)
            self.pso_tol_box.setValue(1e-6)
            self.pso_alpha_box.setValue(0.01)
            self.pso_benchmark_runs_box.setValue(1)
            
            # Reset advanced PSO parameters
            self.pso_adaptive_params_checkbox.setChecked(True)
            self.pso_topology_combo.setCurrentText("Global")
            self.pso_w_damping_box.setValue(1.0)
            self.pso_mutation_rate_box.setValue(0.1)
            self.pso_max_velocity_factor_box.setValue(0.1)
            self.pso_stagnation_limit_box.setValue(10)
            self.pso_boundary_handling_combo.setCurrentText("absorbing")
            self.pso_diversity_threshold_box.setValue(0.01)
            self.pso_early_stopping_checkbox.setChecked(True)
            self.pso_early_stopping_iters_box.setValue(15)
            self.pso_early_stopping_tol_box.setValue(1e-5)
            self.pso_quasi_random_init_checkbox.setChecked(True)
            
            # Reset DVA parameters in PSO table
            for row in range(self.pso_param_table.rowCount()):
                param_name = self.pso_param_table.item(row, 0).text()
                # Uncheck fixed checkbox
                fixed_checkbox = self.pso_param_table.cellWidget(row, 1)
                fixed_checkbox.setChecked(False)
                
                # Reset fixed value
                fixed_value_spin = self.pso_param_table.cellWidget(row, 2)
                fixed_value_spin.setValue(0.0)
                
                # Reset bounds based on parameter type
                lower_bound_spin = self.pso_param_table.cellWidget(row, 3)
                upper_bound_spin = self.pso_param_table.cellWidget(row, 4)
                
                if param_name.startswith(("beta_", "lambda_", "nu_")):
                    lower_bound_spin.setValue(0.0001)
                    upper_bound_spin.setValue(10.0)
                elif param_name.startswith("mu_"):
                    lower_bound_spin.setValue(0.0)
                    upper_bound_spin.setValue(1.0)
            
            # Reset FRF parameters
            self.omega_start_box.setValue(0.1)
            self.omega_end_box.setValue(10.0)
            self.omega_points_box.setValue(1000)
            
            # Reset comparative visualization options
            self.plot_title_edit.clear()
            self.title_font_size.setValue(14)
            self.fig_width_spin.setValue(10)
            self.fig_height_spin.setValue(6)
            self.x_norm_check.setChecked(False)
            self.y_norm_check.setChecked(False)
            self.x_norm_value.setValue(1.0)
            self.y_norm_value.setValue(1.0)
            self.show_grid_check.setChecked(True)
            self.legend_position_combo.setCurrentText("best")
            
            # Clear plots and results
            self.available_plots_list.clear()
            self.legend_table.setRowCount(0)
            self.legend_map.clear()
            self.frf_plots.clear()
            
            # Reset optimization results
            self.current_ga_best_params = None
            self.current_ga_best_fitness = None
            self.current_ga_full_results = None
            self.current_ga_settings = None
            self.current_pso_best_params = None
            
            self.status_bar.showMessage("All values reset to defaults", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Reset Error", 
                              f"Error resetting some values: {str(e)}\nSome values may not have been reset.")
            self.status_bar.showMessage("Error during reset", 3000)


