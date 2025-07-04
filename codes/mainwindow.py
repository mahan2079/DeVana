
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
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name,
    save_results
)
from modules.plotwindow import PlotWindow

# Local imports from "workers" subfolder
from workers.FRFWorker import FRFWorker
from workers.SobolWorker import SobolWorker
from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker, TopologyType
from workers.DEWorker import DEWorker
from workers.SAWorker import SAWorker
from workers.CMAESWorker import CMAESWorker
# RL module import removed
from gui.widgets import ModernQTabWidget, SidebarButton
from gui.menu_mixin import MenuMixin
from gui.beam_mixin import ContinuousBeamMixin
from gui.microchip_mixin import MicrochipPageMixin
from gui.main_window.theme_mixin import ThemeMixin
from gui.main_window.ga_mixin import GAOptimizationMixin
from gui.main_window.frf_mixin import FRFMixin
from gui.main_window.pso_mixin import PSOMixin
from gui.main_window.input_mixin import InputTabsMixin
from gui.main_window.extra_opt_mixin import ExtraOptimizationMixin


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
                 InputTabsMixin, ExtraOptimizationMixin):
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
    
    def debug_array(self, array, name):
        """Print info about an array for debugging"""
        print(f"DEBUG {name}: type={type(array)}, shape={getattr(array, 'shape', 'N/A')}, min={np.min(array) if hasattr(array, '__len__') else array}, max={np.max(array) if hasattr(array, '__len__') else array}")
    
    def create_sidebar(self):
        """Create the sidebar with navigation buttons"""
        # Create sidebar container
        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar")
        sidebar_container.setFixedWidth(250)
        
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Add logo/title at the top
        logo_container = QWidget()
        logo_container.setObjectName("logo-container")
        logo_container.setMinimumHeight(100)
        
        logo_layout = QVBoxLayout(logo_container)
        title = QLabel("DeVana")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        logo_layout.addWidget(title)
        
        version = QLabel("v2.0")
        version.setAlignment(Qt.AlignCenter)
        version.setFont(QFont("Segoe UI", 10))
        logo_layout.addWidget(version)
        
        sidebar_layout.addWidget(logo_container)
        
        # Add separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line)
        
        # Main navigation buttons
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(10)
        
        # Stochastic Design button
        self.stochastic_btn = SidebarButton(None, "Stochastic Design")
        self.stochastic_btn.setObjectName("active-nav-btn")
        self.stochastic_btn.mousePressEvent = lambda event: self.change_page(0)
        nav_layout.addWidget(self.stochastic_btn)
        
        # Microchip Controller button
        self.microchip_btn = SidebarButton(None, "Microchip Controller")
        self.microchip_btn.mousePressEvent = lambda event: self.change_page(1)
        nav_layout.addWidget(self.microchip_btn)
        
        # Continuous Beam button
        self.beam_btn = SidebarButton(None, "Continuous Beam")
        self.beam_btn.mousePressEvent = lambda event: self.change_page(2)
        if not BEAM_IMPORTS_SUCCESSFUL:
            self.beam_btn.setEnabled(False)
            self.beam_btn.setToolTip("Continuous Beam module not available")
        nav_layout.addWidget(self.beam_btn)
        
        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_container)
        
        # Theme toggle and settings at bottom
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        
        # Theme toggle button
        self.theme_toggle = QPushButton("Toggle Theme")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        bottom_layout.addWidget(self.theme_toggle)
        
        sidebar_layout.addWidget(bottom_container)
        sidebar_layout.addSpacing(20)
        
        # Add sidebar to main layout
        self.main_layout.addWidget(sidebar_container)
    
    def change_page(self, index):
        """Change the active page in the content stack"""
        self.content_stack.setCurrentIndex(index)
        
        # Update active button styling
        for btn in [self.stochastic_btn, self.microchip_btn, self.beam_btn]:
            btn.setObjectName("")
            btn.setStyleSheet("")
        
        if index == 0:
            self.stochastic_btn.setObjectName("active-nav-btn")
        elif index == 1:
            self.microchip_btn.setObjectName("active-nav-btn")
        elif index == 2:
            self.beam_btn.setObjectName("active-nav-btn")
        
        self.apply_current_theme()  # Reapply theme to update active button styling
    

    def create_stochastic_design_page(self):
        """Create the stochastic design page with all existing tabs"""
        stochastic_page = QWidget()
        page_layout = QVBoxLayout(stochastic_page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title and description
        header = QWidget()
        header_layout = QVBoxLayout(header)
        title = QLabel("Stochastic Design")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header_layout.addWidget(title)
        
        description = QLabel("Design and optimize stochastic vibration systems with advanced algorithms")
        description.setFont(QFont("Segoe UI", 11))
        header_layout.addWidget(description)
        
        # Add header to page layout
        page_layout.addWidget(header)
        
        # Create a horizontal split for content and results
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Tabs and main controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create all original tabs
        self.create_main_system_tab()
        self.create_dva_parameters_tab()
        self.create_target_weights_tab()
        self.create_frequency_tab()
        self.create_omega_sensitivity_tab()  # Create the new Omega Sensitivity tab
        self.create_sobol_analysis_tab()
        self.create_ga_tab()
        self.create_pso_tab()
        self.create_de_tab()
        self.create_sa_tab()
        self.create_cmaes_tab()
        # RL tab creation removed
        
        # Create the tab widget with modern styling
        self.design_tabs = ModernQTabWidget()
        
        # Create mother tab for Input (contains Main System, DVA Parameters, Targets & Weights, Frequency Plot)
        self.input_tabs = ModernQTabWidget()
        self.input_tabs.addTab(self.main_system_tab, "Main System")
        self.input_tabs.addTab(self.dva_tab, "DVA Parameters")
        self.input_tabs.addTab(self.tw_tab, "Targets & Weights")
        self.input_tabs.addTab(self.freq_tab, "Frequency & Plot")
        self.input_tabs.addTab(self.omega_sensitivity_tab, "Ω Sensitivity")

        # Create mother tab for Sensitivity Analysis (contains Sobol Analysis)
        self.sensitivity_tabs = ModernQTabWidget()
        self.sensitivity_tabs.addTab(self.sobol_tab, "Sobol Analysis")

        # Create mother tab for Optimization (contains GA, PSO, DE, SA, and CMA-ES)
        self.optimization_tabs = ModernQTabWidget()
        self.optimization_tabs.addTab(self.ga_tab, "GA Optimization")
        self.optimization_tabs.addTab(self.pso_tab, "PSO Optimization")
        self.optimization_tabs.addTab(self.de_tab, "DE Optimization")
        self.optimization_tabs.addTab(self.sa_tab, "SA Optimization")
        self.optimization_tabs.addTab(self.cmaes_tab, "CMA-ES Optimization")

        # Comprehensive Analysis tab has been removed
        # Here we add the RL tab (with integrated Sobol settings, epsilon decay, and reward settings sub-tabs)
        # Comprehensive analysis tab creation removed
        # RL tab addition removed
        
        # Add all tabs to the main tab widget
        self.design_tabs.addTab(self.input_tabs, "Input")
        self.design_tabs.addTab(self.sensitivity_tabs, "Sensitivity Analysis")
        self.design_tabs.addTab(self.optimization_tabs, "Optimization")
        # Comprehensive tab addition removed
        
        left_layout.addWidget(self.design_tabs)
        
        # Run buttons in a card-like container
        run_card = QFrame()
        run_card.setObjectName("run-card")
        run_card.setMinimumHeight(120)
        run_card_layout = QVBoxLayout(run_card)
        
        run_title = QLabel("Actions")
        run_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        run_card_layout.addWidget(run_title)
        
        run_buttons_layout = QHBoxLayout()
        
        self.run_frf_button = QPushButton("Run FRF")
        self.run_frf_button.setObjectName("primary-button")
        self.run_frf_button.setMinimumHeight(40)
        self.run_frf_button.clicked.connect(self.run_frf)
        self.run_frf_button.setVisible(False)  # Hide button
        
        self.run_sobol_button = QPushButton("Run Sobol")
        self.run_sobol_button.setObjectName("primary-button")
        self.run_sobol_button.setMinimumHeight(40)
        self.run_sobol_button.clicked.connect(self.run_sobol)
        self.run_sobol_button.setVisible(False)  # Hide button
        
        self.run_ga_button = QPushButton("Run GA")
        self.run_ga_button.setObjectName("primary-button")
        self.run_ga_button.setMinimumHeight(40)
        self.run_ga_button.clicked.connect(self.run_ga)
        self.run_ga_button.setVisible(False)  # Hide button
        
        self.run_pso_button = QPushButton("Run PSO")
        self.run_pso_button.setObjectName("primary-button")
        self.run_pso_button.setMinimumHeight(40)
        self.run_pso_button.clicked.connect(self.run_pso)
        self.run_pso_button.setVisible(False)  # Hide button
        
        self.run_de_button = QPushButton("Run DE")
        self.run_de_button.setObjectName("primary-button")
        self.run_de_button.setMinimumHeight(40)
        self.run_de_button.clicked.connect(self.run_de)
        self.run_de_button.setVisible(False)  # Hide button
        
        self.run_sa_button = QPushButton("Run SA")
        self.run_sa_button.setObjectName("primary-button")
        self.run_sa_button.setMinimumHeight(40)
        self.run_sa_button.clicked.connect(self.run_sa)
        self.run_sa_button.setVisible(False)  # Hide button
        
        self.run_cmaes_button = QPushButton("Run CMA-ES")
        self.run_cmaes_button.setObjectName("primary-button")
        self.run_cmaes_button.setMinimumHeight(40)
        self.run_cmaes_button.clicked.connect(self.run_cmaes)
        self.run_cmaes_button.setVisible(False)  # Hide button
        
        run_buttons_layout.addWidget(self.run_frf_button)
        run_buttons_layout.addWidget(self.run_sobol_button)
        run_buttons_layout.addWidget(self.run_ga_button)
        run_buttons_layout.addWidget(self.run_pso_button)
        run_buttons_layout.addWidget(self.run_de_button)
        run_buttons_layout.addWidget(self.run_sa_button)
        run_buttons_layout.addWidget(self.run_cmaes_button)
        
        run_card_layout.addLayout(run_buttons_layout)
        run_card.setVisible(False)  # Hide entire run card
        left_layout.addWidget(run_card)
        
        # Add left panel to splitter
        content_splitter.addWidget(left_panel)
        
        # Right panel - Results area with tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        results_tabs = ModernQTabWidget()
        
        # Results text panel
        results_panel = QWidget()
        results_panel_layout = QVBoxLayout(results_panel)
        
        results_title = QLabel("Results")
        results_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_panel_layout.addWidget(results_title)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_panel_layout.addWidget(self.results_text)
        
        # FRF plots panel
        frf_panel = QWidget()
        frf_layout = QVBoxLayout(frf_panel)
        
        frf_header = QWidget()
        frf_header_layout = QHBoxLayout(frf_header)
        
        frf_title = QLabel("FRF Plots")
        frf_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        frf_header_layout.addWidget(frf_title)
        
        self.frf_combo = QComboBox()
        self.frf_combo.currentIndexChanged.connect(self.update_frf_plot)
        frf_header_layout.addWidget(self.frf_combo)
        
        self.frf_save_plot_button = QPushButton("Save Plot")
        self.frf_save_plot_button.setObjectName("secondary-button")
        self.frf_save_plot_button.clicked.connect(lambda: self.save_plot(self.frf_fig, "FRF"))
        frf_header_layout.addWidget(self.frf_save_plot_button)
        
        frf_layout.addWidget(frf_header)
        
        self.frf_fig = Figure(figsize=(6, 4))
        self.frf_canvas = FigureCanvas(self.frf_fig)
        self.frf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.frf_toolbar = NavigationToolbar(self.frf_canvas, frf_panel)
        frf_layout.addWidget(self.frf_toolbar)
        frf_layout.addWidget(self.frf_canvas)
        
        # Comparative FRF plots panel
        comp_panel = QWidget()
        comp_layout = QVBoxLayout(comp_panel)
        
        comp_header = QWidget()
        comp_header_layout = QHBoxLayout(comp_header)
        
        comp_title = QLabel("Comparative FRF Plots")
        comp_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        comp_header_layout.addWidget(comp_title)
        
        self.comp_save_plot_button = QPushButton("Save Plot")
        self.comp_save_plot_button.setObjectName("secondary-button")
        self.comp_save_plot_button.clicked.connect(lambda: self.save_plot(self.comp_fig, "Comparative FRF"))
        comp_header_layout.addWidget(self.comp_save_plot_button)
        
        comp_layout.addWidget(comp_header)
        
        self.comp_fig = Figure(figsize=(6, 4))
        self.comp_canvas = FigureCanvas(self.comp_fig)
        self.comp_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, comp_panel)
        comp_layout.addWidget(self.comp_toolbar)
        comp_layout.addWidget(self.comp_canvas)
        
        # Sobol plots panel
        sobol_panel = QWidget()
        sobol_layout = QVBoxLayout(sobol_panel)
        
        sobol_header = QWidget()
        sobol_header_layout = QHBoxLayout(sobol_header)
        
        sobol_title = QLabel("Sobol Analysis")
        sobol_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        sobol_header_layout.addWidget(sobol_title)
        
        self.sobol_combo = QComboBox()
        self.sobol_combo.currentIndexChanged.connect(self.update_sobol_plot)
        sobol_header_layout.addWidget(self.sobol_combo)
        
        self.sobol_save_plot_button = QPushButton("Save Plot")
        self.sobol_save_plot_button.setObjectName("secondary-button")
        self.sobol_save_plot_button.clicked.connect(lambda: self.save_plot(self.sobol_fig, "Sobol"))
        sobol_header_layout.addWidget(self.sobol_save_plot_button)
        
        sobol_layout.addWidget(sobol_header)
        
        self.sobol_fig = Figure(figsize=(6, 4))
        self.sobol_canvas = FigureCanvas(self.sobol_fig)
        self.sobol_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.sobol_toolbar = NavigationToolbar(self.sobol_canvas, sobol_panel)
        sobol_layout.addWidget(self.sobol_toolbar)
        sobol_layout.addWidget(self.sobol_canvas)
        
        sobol_results_container = QWidget()
        sobol_results_layout = QVBoxLayout(sobol_results_container)
        sobol_results_layout.setContentsMargins(0, 10, 0, 0)
        
        sobol_results_header = QHBoxLayout()
        sobol_results_title = QLabel("Sobol Results")
        sobol_results_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sobol_results_header.addWidget(sobol_results_title)
        
        self.sobol_save_results_button = QPushButton("Save Results")
        self.sobol_save_results_button.setObjectName("secondary-button")
        self.sobol_save_results_button.clicked.connect(self.save_sobol_results)
        sobol_results_header.addWidget(self.sobol_save_results_button)
        
        sobol_results_layout.addLayout(sobol_results_header)
        
        self.sobol_results_text = QTextEdit()
        self.sobol_results_text.setReadOnly(True)
        self.sobol_results_text.setStyleSheet("font-family: monospace;")
        sobol_results_layout.addWidget(self.sobol_results_text)
        
        sobol_layout.addWidget(sobol_results_container)
        
        # Add tabs to results panel
        results_tabs.addTab(results_panel, "Text Results")
        results_tabs.addTab(frf_panel, "FRF Visualization")
        results_tabs.addTab(comp_panel, "Comparative FRF")
        results_tabs.addTab(sobol_panel, "Sobol Visualization")
        
        right_layout.addWidget(results_tabs)
        
        # Add right panel to splitter
        content_splitter.addWidget(right_panel)
        
        # Set the initial sizes of the splitter
        content_splitter.setSizes([800, 800])
        
        # Add the splitter to the page layout
        page_layout.addWidget(content_splitter)
        
        # Add page to content stack
        self.content_stack.addWidget(stochastic_page)

    def _run_sobol_implementation(self):
        """Run Sobol sensitivity analysis - main implementation"""
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        # Get required parameters
        target_values, weights = self.get_target_values_weights()
        num_samples_list = self.get_num_samples_list()
        n_jobs = self.n_jobs_spin.value()

        # Update UI to show analysis is running
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.hyper_run_sobol_button.setEnabled(False)
        
        # Clear and update results text area
        self.sobol_results_text.clear()
        self.sobol_results_text.append("--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        # Get main system parameters
        main_params = self.get_main_system_params()

        # Get DVA bounds from parameter table
        dva_bounds = {}
        EPSILON = 1e-6
        
        for row in range(self.dva_param_table.rowCount()):
            param_item = self.dva_param_table.item(row, 0)
            param_name = param_item.text()

            fixed_widget = self.dva_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()

            if fixed:
                fixed_value_widget = self.dva_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.dva_param_table.cellWidget(row, 3)
                upper_bound_widget = self.dva_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                
                if lower > upper:
                    QMessageBox.warning(self, "Input Error", 
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    self.run_frf_button.setEnabled(True)
                    self.run_sobol_button.setEnabled(True)
                    self.run_ga_button.setEnabled(True)
                    self.hyper_run_sobol_button.setEnabled(True)
                    return
                    
                dva_bounds[param_name] = (lower, upper)

        # Define parameter order
        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]

        # Print sample size
        self.sobol_results_text.append(f"Sample sizes: {num_samples_list}")
        
        # Create and start worker
        try:
            self.sobol_worker = SobolWorker(
            main_params=main_params,
            dva_bounds=dva_bounds,
            dva_order=original_dva_parameter_order,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            num_samples_list=num_samples_list,
            target_values_dict=target_values,
            weights_dict=weights,
            n_jobs=n_jobs
        )
            
            # Connect signals
            self.sobol_worker.finished.connect(self.display_sobol_results)
            self.sobol_worker.error.connect(self.handle_sobol_error)
            
            # Start the worker thread
            self.sobol_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start Sobol analysis: {str(e)}")
            self.run_frf_button.setEnabled(True)
            self.run_sobol_button.setEnabled(True)
            self.run_ga_button.setEnabled(True)
            self.hyper_run_sobol_button.setEnabled(True)
            self.status_bar.showMessage("Sobol analysis failed to start")

    def run_sobol(self):
        """Run the Sobol sensitivity analysis - delegate to main implementation"""
        # Call the main implementation with a different name to avoid recursion
        self._run_sobol_implementation()
        
    def get_num_samples_list(self):
        """Get the list of sample sizes for Sobol analysis"""
        num_samples_text = self.num_samples_line.text().strip()
        if not num_samples_text:
            return [32, 64, 128]  # Default values
        
        try:
            # Parse comma-separated values
            samples = [int(n.strip()) for n in num_samples_text.split(',') if n.strip()]
            if not samples:
                return [32, 64, 128]  # Default if parsing yields empty list
            return samples
        except ValueError:
            QMessageBox.warning(self, "Input Error", 
                               "Invalid num_samples format. Using default values: 32, 64, 128")
            return [32, 64, 128]
            
    def handle_sobol_error(self, err):
        """Handle errors from the Sobol worker"""
        QMessageBox.critical(self, "Error in Sobol Analysis", str(err))
        self.sobol_results_text.append(f"\nError running Sobol analysis: {err}")
        self.status_bar.showMessage("Sobol analysis failed")
        
        # Re-enable buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        self.hyper_run_sobol_button.setEnabled(True)

    def display_sobol_results(self, all_results, warnings=None):
        """
        Called when SobolWorker finishes. This includes the full
        Sobol analysis results in `all_results` and any warnings in `warnings`.
        """
        self.sobol_results = all_results
        self.sobol_warnings = warnings
        self.sobol_results_text.append("\n--- Sobol Sensitivity Analysis Results ---")

        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        param_names = original_dva_parameter_order

        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6f}"
            return str(val)

        for run_idx, num_samples in enumerate(all_results['samples']):
            self.sobol_results_text.append(f"\nSample Size: {num_samples}")
            S1 = all_results['S1'][run_idx]
            ST = all_results['ST'][run_idx]
            self.sobol_results_text.append(f"  Length of S1: {len(S1)}, Length of ST: {len(ST)}")

            for param_index, param_name in enumerate(param_names):
                if param_index < len(S1) and param_index < len(ST):
                    s1_val = S1[param_index]
                    st_val = ST[param_index]
                    self.sobol_results_text.append(f"Parameter {param_name}: S1 = {s1_val:.6f}, ST = {st_val:.6f}")
                else:
                    self.sobol_results_text.append(f"IndexError: Parameter {param_name} out of range")

        if warnings:
            self.sobol_results_text.append("\nWarnings:")
            for w in warnings:
                self.sobol_results_text.append(w)
        else:
            self.sobol_results_text.append("\nNo warnings encountered.")

        self.status_bar.showMessage("Sobol Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.sobol_combo.clear()
        self.sobol_plots.clear()

        # Build the various Sobol plots
        self.generate_sobol_plots(all_results, param_names)
        self.update_sobol_plot()
        self.sobol_canvas.draw()

    def generate_sobol_plots(self, all_results, param_names):
        """
        This method prepares all the standard plots
        and adds them to self.sobol_plots so the user can pick them in the combo box.
        """
        fig_last_run = self.visualize_last_run(all_results, param_names)
        self.sobol_combo.addItem("Last Run Results")
        self.sobol_plots["Last Run Results"] = fig_last_run

        fig_grouped_ST = self.visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names)
        self.sobol_combo.addItem("Grouped Bar (Sorted by ST)")
        self.sobol_plots["Grouped Bar (Sorted by ST)"] = fig_grouped_ST

        conv_figs = self.visualize_convergence_plots(all_results, param_names)
        for i, cf in enumerate(conv_figs, start=1):
            name = f"Convergence Plots Fig {i}"
            self.sobol_combo.addItem(name)
            self.sobol_plots[name] = cf

        fig_heat = self.visualize_combined_heatmap(all_results, param_names)
        self.sobol_combo.addItem("Combined Heatmap")
        self.sobol_plots["Combined Heatmap"] = fig_heat

        fig_comp_radar = self.visualize_comprehensive_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Comprehensive Radar Plot")
        self.sobol_plots["Comprehensive Radar Plot"] = fig_comp_radar

        fig_s1_radar, fig_st_radar = self.visualize_separate_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Radar Plot S1")
        self.sobol_plots["Radar Plot S1"] = fig_s1_radar
        self.sobol_combo.addItem("Radar Plot ST")
        self.sobol_plots["Radar Plot ST"] = fig_st_radar

        fig_box = self.visualize_box_plots(all_results)
        self.sobol_combo.addItem("Box Plots")
        self.sobol_plots["Box Plots"] = fig_box

        fig_violin = self.visualize_violin_plots(all_results)
        self.sobol_combo.addItem("Violin Plots")
        self.sobol_plots["Violin Plots"] = fig_violin

        fig_scatter = self.visualize_scatter_S1_ST(all_results, param_names)
        self.sobol_combo.addItem("Scatter S1 vs ST")
        self.sobol_plots["Scatter S1 vs ST"] = fig_scatter

        fig_parallel = self.visualize_parallel_coordinates(all_results, param_names)
        self.sobol_combo.addItem("Parallel Coordinates")
        self.sobol_plots["Parallel Coordinates"] = fig_parallel

        fig_s1_hist, fig_st_hist = self.visualize_histograms(all_results)
        self.sobol_combo.addItem("S1 Histogram")
        self.sobol_plots["S1 Histogram"] = fig_s1_hist
        self.sobol_combo.addItem("ST Histogram")
        self.sobol_plots["ST Histogram"] = fig_st_hist

    ########################################################################
    # -------------- Sobol Visualization Methods --------------
    ########################################################################

    def visualize_last_run(self, all_results, param_names):
        # Basic example: bar chart sorted by S1
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
        sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
        S1_sorted = S1_last_run[sorted_indices_S1]
        ST_sorted = ST_last_run[sorted_indices_S1]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_S1)) - 0.175, S1_sorted, 0.35, label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_S1)) + 0.175, ST_sorted, 0.35, label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        ax.set_title('First-order ($S_1$) & Total-order ($S_T$)', fontsize=16)
        ax.set_xticks(np.arange(len(sorted_param_names_S1)))
        ax.set_xticklabels([self.format_parameter_name(p) for p in sorted_param_names_S1], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        # Similar bar chart, sorted by ST
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        sorted_indices_ST = np.argsort(ST_last_run)[::-1]
        sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
        S1_sorted = S1_last_run[sorted_indices_ST]
        ST_sorted = ST_last_run[sorted_indices_ST]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_ST)) - 0.175, S1_sorted, 0.35, label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_ST)) + 0.175, ST_sorted, 0.35, label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        ax.set_title('Sorted by $S_T$', fontsize=16)
        ax.set_xticks(np.arange(len(sorted_param_names_ST)))
        ax.set_xticklabels([self.format_parameter_name(p) for p in sorted_param_names_ST], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        # Plot S1 and ST vs sample size, for each parameter
        sample_sizes = all_results['samples']
        S1_matrix = np.array(all_results['S1'])
        ST_matrix = np.array(all_results['ST'])

        plots_per_fig = 12
        total_params = len(param_names)
        num_figs = int(np.ceil(total_params / plots_per_fig))
        figs = []

        for fig_idx in range(num_figs):
            fig = Figure(figsize=(20,15))
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, total_params)
            for subplot_idx, param_idx in enumerate(range(start_idx, end_idx)):
                param = param_names[param_idx]
                ax = fig.add_subplot(3,4,subplot_idx+1)
                S1_values = S1_matrix[:, param_idx]
                ST_values = ST_matrix[:, param_idx]
                ax.plot(sample_sizes, S1_values, 'o-', color='blue', label=r'$S_1$')
                ax.plot(sample_sizes, ST_values, 's-', color='red', label=r'$S_T$')
                ax.set_title(f"Convergence: {self.format_parameter_name(param)}", fontsize=10)
                ax.set_xlabel("Sample Size", fontsize=8)
                ax.set_ylabel("Index", fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(True)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        # 2D Heatmap (S1, ST) for the last run
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])

        import pandas as pd
        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last})
        df = df.set_index('Parameter')

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Sensitivity'}, ax=ax)
        ax.set_title("Combined Heatmap (S1 & ST)")
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        # Radar plot combining S1 and ST in single chart
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig = Figure(figsize=(10,10))
        ax = fig.add_subplot(111, polar=True)
        max_val = max(np.max(S1), np.max(ST)) * 1.1
        ax.set_ylim(0, max_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)

        S1_vals = list(S1) + [S1[0]]
        ST_vals = list(ST) + [ST[0]]
        ax.plot(angles, S1_vals, label=r"$S_1$", color='blue', linewidth=2)
        ax.fill(angles, S1_vals, color='blue', alpha=0.2)
        ax.plot(angles, ST_vals, label=r"$S_T$", color='red', linewidth=2)
        ax.fill(angles, ST_vals, color='red', alpha=0.2)

        ax.legend(loc='best')
        ax.set_title("Comprehensive Radar Plot")
        return fig

    def visualize_separate_radar_plots(self, all_results, param_names):
        # One radar for S1, one for ST
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Radar for S1
        fig_s1 = Figure(figsize=(10,10))
        ax_s1 = fig_s1.add_subplot(111, polar=True)
        max_val_s1 = np.max(S1)*1.1
        ax_s1.set_ylim(0, max_val_s1)
        ax_s1.set_xticks(angles[:-1])
        ax_s1.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)
        s1_vals = list(S1) + [S1[0]]
        ax_s1.plot(angles, s1_vals, color='blue', linewidth=2, label=r"$S_1$")
        ax_s1.fill(angles, s1_vals, color='blue', alpha=0.2)
        ax_s1.set_title("Radar - First-order S1")
        ax_s1.legend()

        # Radar for ST
        fig_st = Figure(figsize=(10,10))
        ax_st = fig_st.add_subplot(111, polar=True)
        max_val_st = np.max(ST)*1.1
        ax_st.set_ylim(0, max_val_st)
        ax_st.set_xticks(angles[:-1])
        ax_st.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)
        st_vals = list(ST) + [ST[0]]
        ax_st.plot(angles, st_vals, color='red', linewidth=2, label=r"$S_T$")
        ax_st.fill(angles, st_vals, color='red', alpha=0.2)
        ax_st.set_title("Radar - Total-order ST")
        ax_st.legend()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        # Box plot of all S1 and ST from all runs
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.boxplot(data=df, palette=['skyblue', 'salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title("Box Plots of S1 & ST")
        return fig

    def visualize_violin_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_title("Violin Plots of S1 & ST")
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)), cmap='tab20', edgecolor='k')
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001, self.format_parameter_name(param), fontsize=8)

        ax.set_xlabel("S1")
        ax.set_ylabel("ST")
        ax.set_title("Scatter: S1 vs ST")
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
        data = []
        for run_idx, num_samples in enumerate(all_results['samples']):
            row = {"Sample Size": num_samples}
            for param_idx, param in enumerate(param_names):
                row[f"S1_{param}"] = all_results['S1'][run_idx][param_idx]
                row[f"ST_{param}"] = all_results['ST'][run_idx][param_idx]
            data.append(row)
        df = pd.DataFrame(data)

        fig = Figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        for param in param_names:
            ax.plot(df["Sample Size"], df[f"S1_{param}"], marker='o', label=f"S1 {param}", alpha=0.4)
            ax.plot(df["Sample Size"], df[f"ST_{param}"], marker='s', label=f"ST {param}", alpha=0.4)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Sensitivity Index")
        ax.set_title("Parallel Coordinates of S1 & ST vs Sample Size")
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=6)
        fig.tight_layout()
        return fig

    def visualize_histograms(self, all_results):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(6,4))
        ax_s1 = fig_s1.add_subplot(111)
        import seaborn as sns
        sns.histplot(S1_last_run, bins=20, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_title("Histogram of S1")

        fig_st = Figure(figsize=(6,4))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=20, kde=True, color='salmon', ax=ax_st)
        ax_st.set_title("Histogram of ST")

        return fig_s1, fig_st
        
    def get_main_system_params(self):
        """Get the main system parameters in a tuple format"""
        return (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(), 
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )
        
        
    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                  "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self, "Success", f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save results: {e}")
            
    def set_default_values(self):
        """Reset all inputs to their default values"""
        self.status_bar.showMessage("Reset to default values")
        # Reset logic for all parameters would be implemented here


