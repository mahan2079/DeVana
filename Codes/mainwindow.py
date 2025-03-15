import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup,QStackedWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont

# Matplotlib backends
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Local imports from "modules" subfolder
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)
from modules.plotwindow import PlotWindow

# Local imports from "workers" subfolder
from workers.FRFWorker import FRFWorker
from workers.SobolWorker import SobolWorker
from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker
from workers.DEWorker import DEWorker
from workers.SAWorker import SAWorker
from workers.CMAESWorker import CMAESWorker
from RL.RLWorker import RLWorker

# Additional libraries used
import random
from deap import base, creator, tools

# Seaborn style and LaTeX rendering
sns.set(style="whitegrid")
plt.rc('text', usetex=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeVana")
        self.resize(1600, 900)

        # Initialize theme
        self.current_theme = 'Light'
        self.apply_light_theme()

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # -------------- TABS --------------
        # Create individual tabs (assuming these functions are defined elsewhere)
        self.create_main_system_tab()
        self.create_dva_parameters_tab()
        self.create_target_weights_tab()
        self.create_frequency_tab()
        self.create_sobol_analysis_tab()
        self.create_ga_tab()
        self.create_pso_tab()    # PSO Optimization tab
        self.create_de_tab()     # DE Optimization tab
        self.create_sa_tab()     # SA Optimization tab
        self.create_cmaes_tab()  # CMA-ES Optimization tab
        self.create_rl_tab()     # RL Optimization tab (now with sub-tabs)

        # Create mother tab for Input (contains Main System, DVA Parameters, Targets & Weights, Frequency Plot)
        self.input_tabs = QTabWidget()
        self.input_tabs.addTab(self.main_system_tab, "Main System")
        self.input_tabs.addTab(self.dva_tab, "DVA Parameters")
        self.input_tabs.addTab(self.tw_tab, "Targets & Weights")
        self.input_tabs.addTab(self.freq_tab, "Frequency & Plot")

        # Create mother tab for Sensitivity Analysis (contains Sobol Analysis)
        self.sensitivity_tabs = QTabWidget()
        self.sensitivity_tabs.addTab(self.sobol_tab, "Sobol Analysis")

        # Create mother tab for Optimization (contains GA, PSO, DE, SA, and CMA-ES)
        self.optimization_tabs = QTabWidget()
        self.optimization_tabs.addTab(self.ga_tab, "GA Optimization")
        self.optimization_tabs.addTab(self.pso_tab, "PSO Optimization")
        self.optimization_tabs.addTab(self.de_tab, "DE Optimization")
        self.optimization_tabs.addTab(self.sa_tab, "SA Optimization")
        self.optimization_tabs.addTab(self.cmaes_tab, "CMA-ES Optimization")

        # Create a new mother tab for Comprehensive Analysis.
        # Here we add the RL tab (with integrated Sobol settings, epsilon decay, and reward settings sub-tabs)
        self.comprehensive_analysis_tabs = QTabWidget()
        self.comprehensive_analysis_tabs.addTab(self.rl_tab, "RL Optimization")
        # (Additional comprehensive analysis tabs can be added here if needed.)

        # Create the main top-level tab widget and add the mother tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.input_tabs, "Input")
        self.tabs.addTab(self.sensitivity_tabs, "Sensitivity Analysis")
        self.tabs.addTab(self.optimization_tabs, "Optimization")
        self.tabs.addTab(self.comprehensive_analysis_tabs, "Comprehensive Analysis")

        # Global results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)

        # -------------- DOCK: FRF --------------
        self.frf_fig = Figure(figsize=(6, 4))
        self.frf_canvas = FigureCanvas(self.frf_fig)
        self.frf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frf_combo = QComboBox()
        self.frf_combo.currentIndexChanged.connect(self.update_frf_plot)

        frf_widget = QWidget()
        frf_layout = QVBoxLayout(frf_widget)
        frf_layout.addWidget(QLabel("Select FRF Plot:"))
        frf_layout.addWidget(self.frf_combo)
        self.frf_toolbar = NavigationToolbar(self.frf_canvas, frf_widget)
        frf_layout.addWidget(self.frf_toolbar)
        frf_layout.addWidget(self.frf_canvas)
        self.frf_save_plot_button = QPushButton("Save FRF Plot")
        self.frf_save_plot_button.clicked.connect(lambda: self.save_plot(self.frf_fig, "FRF"))
        frf_layout.addWidget(self.frf_save_plot_button)
        frf_layout.setStretchFactor(self.frf_canvas, 1)
        frf_widget.setLayout(frf_layout)

        self.frf_dock = QDockWidget("FRF Plots", self)
        self.frf_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.frf_dock.setWidget(frf_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.frf_dock)
        self.add_maximize_action(self.frf_dock, self.frf_canvas, is_sobol=False)

        # -------------- DOCK: Sobol --------------
        self.sobol_fig = Figure(figsize=(6, 4))
        self.sobol_canvas = FigureCanvas(self.sobol_fig)
        self.sobol_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sobol_combo = QComboBox()
        self.sobol_combo.currentIndexChanged.connect(self.update_sobol_plot)

        sobol_widget = QWidget()
        sobol_layout = QVBoxLayout(sobol_widget)
        sobol_layout.addWidget(QLabel("Select Sobol Plot:"))
        sobol_layout.addWidget(self.sobol_combo)
        self.sobol_results_text = QTextEdit()
        self.sobol_results_text.setReadOnly(True)
        sobol_layout.addWidget(QLabel("Sobol Analysis Results:"))
        sobol_layout.addWidget(self.sobol_results_text)
        self.sobol_save_results_button = QPushButton("Save Sobol Results")
        self.sobol_save_results_button.clicked.connect(self.save_sobol_results)
        sobol_layout.addWidget(self.sobol_save_results_button)
        self.sobol_toolbar = NavigationToolbar(self.sobol_canvas, sobol_widget)
        sobol_layout.addWidget(self.sobol_toolbar)
        sobol_layout.addWidget(self.sobol_canvas)
        self.sobol_save_plot_button = QPushButton("Save Sobol Plot")
        self.sobol_save_plot_button.clicked.connect(lambda: self.save_plot(self.sobol_fig, "Sobol"))
        sobol_layout.addWidget(self.sobol_save_plot_button)
        sobol_layout.setStretchFactor(self.sobol_canvas, 1)
        sobol_widget.setLayout(sobol_layout)

        self.sobol_dock = QDockWidget("Sobol Plots & Results", self)
        self.sobol_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.sobol_dock.setWidget(sobol_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sobol_dock)
        self.add_maximize_action(self.sobol_dock, self.sobol_canvas, is_sobol=True)

        # -------------- SPLITTER --------------
        splitter_main = QSplitter(Qt.Vertical)
        splitter_main.addWidget(self.tabs)
        splitter_main.addWidget(self.results_text)
        splitter_main.setStretchFactor(0, 5)
        splitter_main.setStretchFactor(1, 1)
        central_layout.addWidget(splitter_main)

        # -------------- MENUBAR / TOOLBAR / STATUS --------------
        self.create_menubar()
        self.create_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # -------------- RUN BUTTONS (Organized in 2 Rows) --------------
        run_buttons_layout = QVBoxLayout()
        run_layout_row1 = QHBoxLayout()
        run_layout_row1.addStretch()
        self.run_frf_button = QPushButton("Run FRF")
        self.run_frf_button.setFixedWidth(150)
        self.run_frf_button.clicked.connect(self.run_frf)
        run_layout_row1.addWidget(self.run_frf_button)
        self.run_sobol_button = QPushButton("Run Sobol")
        self.run_sobol_button.setFixedWidth(150)
        self.run_sobol_button.clicked.connect(self.run_sobol)
        run_layout_row1.addWidget(self.run_sobol_button)
        self.run_ga_button = QPushButton("Run GA")
        self.run_ga_button.setFixedWidth(150)
        self.run_ga_button.clicked.connect(self.run_ga)
        run_layout_row1.addWidget(self.run_ga_button)
        self.run_pso_button = QPushButton("Run PSO")
        self.run_pso_button.setFixedWidth(150)
        self.run_pso_button.clicked.connect(self.run_pso)
        run_layout_row1.addWidget(self.run_pso_button)
        run_layout_row1.addStretch()
        run_layout_row2 = QHBoxLayout()
        run_layout_row2.addStretch()
        self.run_de_button = QPushButton("Run DE")
        self.run_de_button.setFixedWidth(150)
        self.run_de_button.clicked.connect(self.run_de)
        run_layout_row2.addWidget(self.run_de_button)
        self.run_sa_button = QPushButton("Run SA")
        self.run_sa_button.setFixedWidth(150)
        self.run_sa_button.clicked.connect(self.run_sa)
        run_layout_row2.addWidget(self.run_sa_button)
        self.run_cmaes_button = QPushButton("Run CMA-ES")
        self.run_cmaes_button.setFixedWidth(150)
        self.run_cmaes_button.clicked.connect(self.run_cmaes)
        run_layout_row2.addWidget(self.run_cmaes_button)
        run_layout_row2.addStretch()
        run_buttons_layout.addLayout(run_layout_row1)
        run_buttons_layout.addLayout(run_layout_row2)
        central_layout.addLayout(run_buttons_layout)

        # -------------- INITIAL STATE --------------
        self.frf_results = None
        self.sobol_results = None
        self.sobol_warnings = None
        self.frf_plots = {}
        self.sobol_plots = {}
        self.ga_plots = {}
        self.ga_results = None
        self.pso_results = None
        self.pso_plots = {}
        self.de_results = None
        self.de_plots = {}
        self.sa_results = None
        self.sa_plots = {}
        self.cmaes_results = None
        self.cmaes_plots = {}

        self.set_default_values()
        
        # Redraw canvases to ensure toolbar interactivity
        self.frf_canvas.draw()
        self.sobol_canvas.draw()
 
    ########################################################################
    # -------------- Helper Methods for Setup --------------
    ########################################################################

    def add_maximize_action(self, dock, canvas, is_sobol=False):
        """
        Adds a 'Maximize' action to the dock's toolbar so the user can pop out
        the plot into a bigger window or float it maximized.
        """
        toolbar = QToolBar("Plot Controls", dock)
        dock_layout = dock.widget().layout()
        dock_layout.insertWidget(0, toolbar)

        maximize_action = QAction("Maximize", dock)
        maximize_action.setIcon(QIcon.fromTheme("zoom-in"))
        if is_sobol:
            # For Sobol Dock, maximize only the plot by opening it in a new window
            maximize_action.triggered.connect(lambda: self.maximize_plot(dock, canvas, is_sobol=True))
        else:
            # For FRF or other docks, float and resize
            maximize_action.triggered.connect(lambda: self.maximize_dock(dock))
        toolbar.addAction(maximize_action)

    def maximize_plot(self, dock, canvas, is_sobol=False):
        """
        For the Sobol dock, we open the chosen plot in a new PlotWindow.
        For FRF, you could adapt similarly if desired.
        """
        current_plot_key = ""
        if is_sobol:
            current_plot_key = self.sobol_combo.currentText()
            fig = self.sobol_plots.get(current_plot_key, None)
            title = f"Sobol Plot: {current_plot_key}"
        else:
            current_plot_key = self.frf_combo.currentText()
            fig = self.frf_plots.get(current_plot_key, None)
            title = f"FRF Plot: {current_plot_key}"

        if fig:
            self.plot_window = PlotWindow(fig, title=title)
            self.plot_window.show()
        else:
            QMessageBox.warning(self, "Maximize Plot", "No plot available to maximize.")

    def maximize_dock(self, dock):
        """
        Converts a docked widget into a floating dialog, then resizes to full screen.
        """
        if not dock.isFloating():
            dock.setFloating(True)
        screen_geometry = self.screen().geometry()
        dock.resize(screen_geometry.width(), screen_geometry.height())
        dock.show()

    def create_menubar(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        theme_menu = menubar.addMenu("Theme")
        run_menu = menubar.addMenu("Run")
        help_menu = menubar.addMenu("Help")

        # File -> Exit
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu -> show/hide docks
        self.show_frf_dock_act = QAction("Show FRF Dock", self, checkable=True, checked=True)
        self.show_frf_dock_act.triggered.connect(lambda checked: self.frf_dock.setVisible(checked))
        view_menu.addAction(self.show_frf_dock_act)

        self.show_sobol_dock_act = QAction("Show Sobol Dock", self, checkable=True, checked=True)
        self.show_sobol_dock_act.triggered.connect(lambda checked: self.sobol_dock.setVisible(checked))
        view_menu.addAction(self.show_sobol_dock_act)

        # Theme menu
        light_theme_act = QAction("Light Theme", self, checkable=True, checked=True)
        dark_theme_act = QAction("Dark Theme", self, checkable=True)
        light_theme_act.triggered.connect(lambda: self.switch_theme('Light'))
        dark_theme_act.triggered.connect(lambda: self.switch_theme('Dark'))
        theme_menu.addAction(light_theme_act)
        theme_menu.addAction(dark_theme_act)
        theme_group = QActionGroup(self)
        theme_group.addAction(light_theme_act)
        theme_group.addAction(dark_theme_act)

        # Run menu
        run_frf_act = QAction("Run FRF Analysis", self)
        run_frf_act.triggered.connect(self.run_frf)
        run_sobol_act = QAction("Run Sobol Analysis", self)
        run_sobol_act.triggered.connect(self.run_sobol)
        run_ga_act = QAction("Run GA Optimization", self)
        run_ga_act.triggered.connect(self.run_ga)
        run_menu.addAction(run_frf_act)
        run_menu.addAction(run_sobol_act)
        run_menu.addAction(run_ga_act)

        # Help -> About
        about_act = QAction("About", self)
        about_act.triggered.connect(lambda: QMessageBox.information(self, "About", "VIBRAOPT Program\nVersion 1.0"))
        help_menu.addAction(about_act)

        self.setMenuBar(menubar)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(True)
        run_frf_act = QAction(QIcon.fromTheme("system-run"), "Run FRF", self)
        run_frf_act.triggered.connect(self.run_frf)
        run_sobol_act = QAction(QIcon.fromTheme("system-run"), "Run Sobol", self)
        run_sobol_act.triggered.connect(self.run_sobol)
        run_ga_act = QAction(QIcon.fromTheme("system-run"), "Run GA", self)
        run_ga_act.triggered.connect(self.run_ga)

        toolbar.addAction(run_frf_act)
        toolbar.addAction(run_sobol_act)
        toolbar.addAction(run_ga_act)

        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def switch_theme(self, theme):
        if theme == 'Dark':
            self.apply_dark_theme()
            self.current_theme = 'Dark'
        else:
            self.apply_light_theme()
            self.current_theme = 'Light'

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53,53,53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25,25,25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53,53,53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53,53,53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Highlight, QColor(142,45,197).lighter())
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.window().setPalette(dark_palette)

        dark_stylesheet = """
            QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: "Roboto", "Segoe UI", "Helvetica", sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background: #555;
                color: white;
                padding: 10px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6A0DAD, stop:1 #FF8C00);
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                margin-top: 15px;
                background-color: #2E2E2E;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #FF8C00;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 11pt;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #63B8FF;
            }
            QPushButton:pressed {
                background-color: #104E8B;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2E2E2E;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                color: #FFFFFF;
                height: 25px;
            }
            QTextEdit {
                background-color: #2E2E2E;
                border: 1px solid #555555;
                border-radius: 5px;
                color: #FFFFFF;
                padding: 10px;
                font-family: "Consolas", monospace;
                font-size: 10pt;
            }
            QLabel {
                color: #FFFFFF;
                font-weight: normal;
                font-size: 10pt;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF8C00;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #1E90FF;
            }
            QScrollArea {
                border: none;
            }
            QGroupBox:hover {
                border: 2px solid #FF8C00;
            }
        """
        self.window().setStyleSheet(dark_stylesheet)

    def apply_light_theme(self):
        light_palette = QPalette()
        light_palette.setColor(QPalette.Window, Qt.white)
        light_palette.setColor(QPalette.WindowText, Qt.black)
        light_palette.setColor(QPalette.Base, QColor(240,240,240))
        light_palette.setColor(QPalette.AlternateBase, QColor(225,225,225))
        light_palette.setColor(QPalette.ToolTipBase, Qt.white)
        light_palette.setColor(QPalette.ToolTipText, Qt.black)
        light_palette.setColor(QPalette.Text, Qt.black)
        light_palette.setColor(QPalette.Button, QColor(240,240,240))
        light_palette.setColor(QPalette.ButtonText, Qt.black)
        light_palette.setColor(QPalette.BrightText, Qt.red)
        light_palette.setColor(QPalette.Highlight, QColor(0,120,215))
        light_palette.setColor(QPalette.HighlightedText, Qt.white)
        self.window().setPalette(light_palette)

        light_stylesheet = """
            QWidget {
                background-color: #F0F0F0;
                color: #000000;
                font-family: "Roboto", "Segoe UI", "Helvetica", sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background-color: #F0F0F0;
            }
            QTabBar::tab {
                background: #E0E0E0;
                color: #000000;
                padding: 10px;
                border: 1px solid #CCCCCC;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6A0DAD, stop:1 #FF8C00);
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #CCCCCC;
                border-radius: 10px;
                margin-top: 15px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #6A0DAD;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 11pt;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #63B8FF;
            }
            QPushButton:pressed {
                background-color: #104E8B;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 5px;
                color: #000000;
                height: 25px;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                color: #000000;
                padding: 10px;
                font-family: "Consolas", monospace;
                font-size: 10pt;
            }
            QLabel {
                color: #000000;
                font-weight: normal;
                font-size: 10pt;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #E0E0E0;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF8C00;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #6A0DAD;
            }
            QScrollArea {
                border: none;
            }
            QGroupBox:hover {
                border: 2px solid #FF8C00;
            }
        """
        self.window().setStyleSheet(light_stylesheet)

    ########################################################################
    # -------------- Tabs Creation --------------
    ########################################################################

    def create_main_system_tab(self):
        self.main_system_tab = QWidget()
        layout = QVBoxLayout(self.main_system_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        group = QGroupBox("Main System Parameters")
        form = QFormLayout(group)

        self.mu_box = QDoubleSpinBox()
        self.mu_box.setRange(-1e6,1e6)
        self.mu_box.setDecimals(6)

        self.landa_boxes = []
        for i in range(5):
            box = QDoubleSpinBox()
            box.setRange(-1e6,1e6)
            box.setDecimals(6)
            self.landa_boxes.append(box)

        self.nu_boxes = []
        for i in range(5):
            box = QDoubleSpinBox()
            box.setRange(-1e6,1e6)
            box.setDecimals(6)
            self.nu_boxes.append(box)

        self.a_low_box = QDoubleSpinBox()
        self.a_low_box.setRange(0,1e10)
        self.a_low_box.setDecimals(6)

        self.a_up_box = QDoubleSpinBox()
        self.a_up_box.setRange(0,1e10)
        self.a_up_box.setDecimals(6)

        self.f_1_box = QDoubleSpinBox()
        self.f_1_box.setRange(0,1e10)
        self.f_1_box.setDecimals(6)

        self.f_2_box = QDoubleSpinBox()
        self.f_2_box.setRange(0,1e10)
        self.f_2_box.setDecimals(6)

        self.omega_dc_box = QDoubleSpinBox()
        self.omega_dc_box.setRange(0,1e10)
        self.omega_dc_box.setDecimals(6)

        self.zeta_dc_box = QDoubleSpinBox()
        self.zeta_dc_box.setRange(0,1e10)
        self.zeta_dc_box.setDecimals(6)

        form.addRow("μ (MU):", self.mu_box)
        for i in range(5):
            form.addRow(f"Λ_{i+1}:", self.landa_boxes[i])
        for i in range(5):
            form.addRow(f"Ν_{i+1}:", self.nu_boxes[i])
        form.addRow("A_LOW:", self.a_low_box)
        form.addRow("A_UPP:", self.a_up_box)
        form.addRow("F_1:", self.f_1_box)
        form.addRow("F_2:", self.f_2_box)
        form.addRow("Ω_DC:", self.omega_dc_box)
        form.addRow("ζ_DC:", self.zeta_dc_box)

        sc_layout.addWidget(group)
        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_dva_parameters_tab(self):
        self.dva_tab = QWidget()
        layout = QVBoxLayout(self.dva_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        beta_group = QGroupBox("β (beta) Parameters")
        beta_form = QFormLayout(beta_group)
        self.beta_boxes = []
        for i in range(15):
            b = QDoubleSpinBox()
            b.setRange(-1e6,1e6)
            b.setDecimals(6)
            self.beta_boxes.append(b)
            beta_form.addRow(f"β_{i+1}:", b)
        sc_layout.addWidget(beta_group)

        lambda_group = QGroupBox("λ (lambda) Parameters")
        lambda_form = QFormLayout(lambda_group)
        self.lambda_boxes = []
        for i in range(15):
            l = QDoubleSpinBox()
            l.setRange(-1e6,1e6)
            l.setDecimals(6)
            self.lambda_boxes.append(l)
            lambda_form.addRow(f"λ_{i+1}:", l)
        sc_layout.addWidget(lambda_group)

        mu_group = QGroupBox("μ (mu) Parameters")
        mu_form = QFormLayout(mu_group)
        self.mu_dva_boxes = []
        for i in range(3):
            m = QDoubleSpinBox()
            m.setRange(-1e6,1e6)
            m.setDecimals(6)
            self.mu_dva_boxes.append(m)
            mu_form.addRow(f"μ_{i+1}:", m)
        sc_layout.addWidget(mu_group)

        nu_group = QGroupBox("ν (nu) Parameters")
        nu_form = QFormLayout(nu_group)
        self.nu_dva_boxes = []
        for i in range(15):
            n = QDoubleSpinBox()
            n.setRange(-1e6,1e6)
            n.setDecimals(6)
            self.nu_dva_boxes.append(n)
            nu_form.addRow(f"ν_{i+1}:", n)
        sc_layout.addWidget(nu_group)

        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_target_weights_tab(self):
        self.tw_tab = QWidget()
        layout = QVBoxLayout(self.tw_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        self.mass_target_spins = {}
        self.mass_weight_spins = {}

        for mass_num in range(1,6):
            mass_group = QGroupBox(f"Mass {mass_num} Targets & Weights")
            mg_layout = QVBoxLayout(mass_group)

            peak_group = QGroupBox("Peak Values & Weights")
            peak_form = QFormLayout(peak_group)
            for peak_num in range(1,5):
                pv = QDoubleSpinBox()
                pv.setRange(0,1e6)
                pv.setDecimals(6)
                wv = QDoubleSpinBox()
                wv.setRange(0,1e3)
                wv.setDecimals(6)
                peak_form.addRow(f"Peak Value {peak_num}:", pv)
                peak_form.addRow(f"Weight Peak Value {peak_num}:", wv)
                self.mass_target_spins[f"peak_value_{peak_num}_m{mass_num}"] = pv
                self.mass_weight_spins[f"peak_value_{peak_num}_m{mass_num}"] = wv
            mg_layout.addWidget(peak_group)

            bw_group = QGroupBox("Bandwidth Targets & Weights")
            bw_form = QFormLayout(bw_group)
            for i in range(1,5):
                for j in range(i+1,5):
                    bw = QDoubleSpinBox()
                    bw.setRange(0,1e6)
                    bw.setDecimals(6)
                    wbw = QDoubleSpinBox()
                    wbw.setRange(0,1e3)
                    wbw.setDecimals(6)
                    bw_form.addRow(f"Bandwidth {i}-{j}:", bw)
                    bw_form.addRow(f"Weight Bandwidth {i}-{j}:", wbw)
                    self.mass_target_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = bw
                    self.mass_weight_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = wbw
            mg_layout.addWidget(bw_group)

            slope_group = QGroupBox("Slope Targets & Weights")
            slope_form = QFormLayout(slope_group)
            for i in range(1,5):
                for j in range(i+1,5):
                    s = QDoubleSpinBox()
                    s.setRange(-1e6,1e6)
                    s.setDecimals(6)
                    ws = QDoubleSpinBox()
                    ws.setRange(0,1e3)
                    ws.setDecimals(6)
                    slope_form.addRow(f"Slope {i}-{j}:", s)
                    slope_form.addRow(f"Weight Slope {i}-{j}:", ws)
                    self.mass_target_spins[f"slope_{i}_{j}_m{mass_num}"] = s
                    self.mass_weight_spins[f"slope_{i}_{j}_m{mass_num}"] = ws
            mg_layout.addWidget(slope_group)

            auc_group = QGroupBox("Area Under Curve & Weight")
            auc_form = QFormLayout(auc_group)
            auc = QDoubleSpinBox()
            auc.setRange(0,1e6)
            auc.setDecimals(6)
            wauc = QDoubleSpinBox()
            wauc.setRange(0,1e3)
            wauc.setDecimals(6)
            auc_form.addRow("Area Under Curve:", auc)
            auc_form.addRow("Weight Area Under Curve:", wauc)
            self.mass_target_spins[f"area_under_curve_m{mass_num}"] = auc
            self.mass_weight_spins[f"area_under_curve_m{mass_num}"] = wauc
            mg_layout.addWidget(auc_group)

            mg_layout.addStretch()
            sc_layout.addWidget(mass_group)

        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_frequency_tab(self):
        self.freq_tab = QWidget()
        layout = QVBoxLayout(self.freq_tab)

        freq_group = QGroupBox("Frequency Range & Plot Options")
        freq_form = QFormLayout(freq_group)

        self.omega_start_box = QDoubleSpinBox()
        self.omega_start_box.setRange(0,1e6)
        self.omega_start_box.setDecimals(6)

        self.omega_end_box = QDoubleSpinBox()
        self.omega_end_box.setRange(0,1e6)
        self.omega_end_box.setDecimals(6)

        self.omega_points_box = QSpinBox()
        self.omega_points_box.setRange(1,1024)

        freq_form.addRow("Ω Start:", self.omega_start_box)
        freq_form.addRow("Ω End:", self.omega_end_box)
        freq_form.addRow("Ω Points:", self.omega_points_box)

        self.plot_figure_chk = QCheckBox("Plot Figure")
        self.show_peaks_chk = QCheckBox("Show Peaks")
        self.show_slopes_chk = QCheckBox("Show Slopes")

        freq_form.addRow(self.plot_figure_chk)
        freq_form.addRow(self.show_peaks_chk)
        freq_form.addRow(self.show_slopes_chk)

        layout.addWidget(freq_group)

    def create_sobol_analysis_tab(self):
        self.sobol_tab = QWidget()
        layout = QVBoxLayout(self.sobol_tab)

        splitter = QSplitter(Qt.Horizontal)

        sobol_group = QGroupBox("Sobol Analysis Settings")
        sobol_form = QFormLayout(sobol_group)

        self.num_samples_line = QLineEdit()
        self.num_samples_line.setPlaceholderText("e.g. 32,64,128")
        sobol_form.addRow("Num Samples List:", self.num_samples_line)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(1,64)
        sobol_form.addRow("Number of Jobs (n_jobs):", self.n_jobs_spin)

        sobol_group.setLayout(sobol_form)
        splitter.addWidget(sobol_group)

        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]

        dva_param_group = QGroupBox("DVA Parameter Settings")
        dva_param_layout = QVBoxLayout(dva_param_group)

        self.dva_param_table = QTableWidget()
        self.dva_param_table.setRowCount(len(dva_parameters))
        self.dva_param_table.setColumnCount(5)
        self.dva_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.dva_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dva_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.dva_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_fixed(state, r))
            self.dva_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6,1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.dva_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6,1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.dva_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6,1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.dva_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Set some defaults
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        dva_param_layout.addWidget(self.dva_param_table)
        dva_param_group.setLayout(dva_param_layout)

        splitter.addWidget(dva_param_group)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)
        layout.addStretch()
        self.sobol_tab.setLayout(layout)

    def create_ga_tab(self):
        self.ga_tab = QWidget()
        layout = QVBoxLayout(self.ga_tab)

        splitter = QSplitter(Qt.Horizontal)

        ga_group = QGroupBox("GA Settings")
        ga_form = QFormLayout(ga_group)
        self.ga_pop_size_box = QSpinBox()
        self.ga_pop_size_box.setRange(1,10000)
        self.ga_pop_size_box.setValue(800)

        self.ga_num_generations_box = QSpinBox()
        self.ga_num_generations_box.setRange(1,10000)
        self.ga_num_generations_box.setValue(100)

        self.ga_cxpb_box = QDoubleSpinBox()
        self.ga_cxpb_box.setRange(0,1)
        self.ga_cxpb_box.setValue(0.7)
        self.ga_cxpb_box.setDecimals(3)

        self.ga_mutpb_box = QDoubleSpinBox()
        self.ga_mutpb_box.setRange(0,1)
        self.ga_mutpb_box.setValue(0.2)
        self.ga_mutpb_box.setDecimals(3)

        self.ga_tol_box = QDoubleSpinBox()
        self.ga_tol_box.setRange(0,1e6)
        self.ga_tol_box.setValue(1e-3)
        self.ga_tol_box.setDecimals(6)

        self.ga_alpha_box = QDoubleSpinBox()
        self.ga_alpha_box.setRange(0.0,10.0)
        self.ga_alpha_box.setDecimals(4)
        self.ga_alpha_box.setSingleStep(0.01)
        self.ga_alpha_box.setValue(0.01)

        ga_form.addRow("Population Size:", self.ga_pop_size_box)
        ga_form.addRow("Number of Generations:", self.ga_num_generations_box)
        ga_form.addRow("Crossover Probability (cxpb):", self.ga_cxpb_box)
        ga_form.addRow("Mutation Probability (mutpb):", self.ga_mutpb_box)
        ga_form.addRow("Tolerance (tol):", self.ga_tol_box)
        ga_form.addRow("Sparsity Penalty (alpha):", self.ga_alpha_box)

        ga_param_group = QGroupBox("DVA Parameter Settings for GA")
        ga_param_layout = QVBoxLayout(ga_param_group)

        self.ga_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.ga_param_table.setRowCount(len(dva_parameters))
        self.ga_param_table.setColumnCount(5)
        self.ga_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.ga_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ga_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.ga_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_ga_fixed(state, r))
            self.ga_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6,1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.ga_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6,1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.ga_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6,1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.ga_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Default ranges
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        ga_param_layout.addWidget(self.ga_param_table)
        ga_param_group.setLayout(ga_param_layout)

        splitter.addWidget(ga_group)
        splitter.addWidget(ga_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.ga_results_text = QTextEdit()
        self.ga_results_text.setReadOnly(True)
        layout.addWidget(QLabel("GA Optimization Results:"))
        layout.addWidget(self.ga_results_text)

        self.run_ga_button_tab = QPushButton("Run GA")
        self.run_ga_button_tab.setFixedWidth(150)
        self.run_ga_button_tab.clicked.connect(self.run_ga)

        run_ga_layout = QHBoxLayout()
        run_ga_layout.addStretch()
        run_ga_layout.addWidget(self.run_ga_button_tab)
        layout.addLayout(run_ga_layout)

        layout.addStretch()
        self.ga_tab.setLayout(layout)

    ########################################################################
    # -------------- Toggles for "Fixed" in Tables --------------
    ########################################################################

    def toggle_ga_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.ga_param_table.cellWidget(row, 2)
        lower_bound_spin = self.ga_param_table.cellWidget(row, 3)
        upper_bound_spin = self.ga_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def toggle_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.dva_param_table.cellWidget(row, 2)
        lower_bound_spin = self.dva_param_table.cellWidget(row, 3)
        upper_bound_spin = self.dva_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    ########################################################################
    # -------------- Running FRF / SOBOL / GA --------------
    ########################################################################

    def run_frf(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.results_text.append("\n--- Running FRF Analysis ---\n")
        self.status_bar.showMessage("Running FRF Analysis...")

        main_params = (
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

        beta_vals = [b.value() for b in self.beta_boxes]
        lambda_vals = [l.value() for l in self.lambda_boxes]
        mu_vals = [m.value() for m in self.mu_dva_boxes]
        nu_vals = [n.value() for n in self.nu_dva_boxes]
        dva_params = tuple(beta_vals + lambda_vals + mu_vals + nu_vals)

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        plot_figure = self.plot_figure_chk.isChecked()
        show_peaks = self.show_peaks_chk.isChecked()
        show_slopes = self.show_slopes_chk.isChecked()

        # Instantiate your FRFWorker
        self.frf_worker = FRFWorker(
            main_params=main_params,
            dva_params=dva_params,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            target_values_dict=target_values,
            weights_dict=weights,
            plot_figure=plot_figure,
            show_peaks=show_peaks,
            show_slopes=show_slopes
        )
        self.frf_worker.finished.connect(self.display_frf_results)
        self.frf_worker.error.connect(self.handle_frf_error)
        self.frf_worker.start()

    def run_sobol(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()
        num_samples_list = self.get_num_samples_list()
        n_jobs = self.n_jobs_spin.value()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.sobol_results_text.append("\n--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        main_params = (
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
                    return
                dva_bounds[param_name] = (lower, upper)

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

        # Create and start SobolWorker
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
        self.sobol_worker.finished.connect(lambda res,warn: self.display_sobol_results(res,warn))
        self.sobol_worker.error.connect(self.handle_sobol_error)
        self.sobol_worker.start()

    def run_ga(self):
        pop_size = self.ga_pop_size_box.value()
        num_gens = self.ga_num_generations_box.value()
        cxpb = self.ga_cxpb_box.value()
        mutpb = self.ga_mutpb_box.value()
        tol = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()

        ga_dva_parameters = []
        row_count = self.ga_param_table.rowCount()
        for row in range(row_count):
            param_name = self.ga_param_table.item(row, 0).text()
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                ga_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                ga_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        # Create and start GAWorker
        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            ga_pop_size=pop_size,
            ga_num_generations=num_gens,
            ga_cxpb=cxpb,
            ga_mutpb=mutpb,
            ga_tol=tol,
            ga_parameter_data=ga_dva_parameters,
            alpha=alpha
        )
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.run_ga_button.setEnabled(False)
        self.ga_results_text.append("Running GA...")
        self.ga_worker.start()

    ########################################################################
    # -------------- Display & Error Handlers for FRF / Sobol / GA --------------
    ########################################################################

    def display_frf_results(self, results_with_dva, results_without_dva):
        """
        This slot is invoked when FRFWorker finishes successfully,
        giving you two sets of results:
        1) results_with_dva
        2) results_without_dva (for mass_1 and mass_2 with no DVAs).
        """
        self.frf_results = results_with_dva
        self.results_text.append("\n--- FRF Analysis Results (With DVA) ---")

        required_masses = [f'mass_{m}' for m in range(1,6)]

        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6e}"
            return str(val)

        # Print "with DVA" results
        for mass in required_masses:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in self.frf_results:
                for key, value in self.frf_results[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures:")
        if 'composite_measures' in self.frf_results:
            for mass, comp in self.frf_results['composite_measures'].items():
                self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response:")
        if 'singular_response' in self.frf_results:
            self.results_text.append(f"{format_float(self.frf_results['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        # Handle "without DVA" results
        self.results_text.append("\n--- FRF Analysis Results (Without DVAs for Mass 1 and Mass 2) ---")
        required_masses_without_dva = ['mass_1', 'mass_2']
        for mass in required_masses_without_dva:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in results_without_dva:
                for key, value in results_without_dva[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures (Without DVAs for Mass 1 and Mass 2):")
        if 'composite_measures' in results_without_dva:
            for mass, comp in results_without_dva['composite_measures'].items():
                if mass in ['mass_1', 'mass_2']:
                    self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response (Without DVAs for Mass 1 and Mass 2):")
        if 'singular_response' in results_without_dva:
            self.results_text.append(f"{format_float(results_without_dva['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        self.status_bar.showMessage("FRF Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        # Build the FRF plots for combo dropdown
        self.frf_combo.clear()
        self.frf_plots.clear()

        omega = np.linspace(self.omega_start_box.value(), self.omega_end_box.value(), self.omega_points_box.value())
        mass_labels = []
        for m_label in required_masses:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mass_labels.append(m_label)

        # Plot with DVAs, individually
        for m_label in mass_labels:
            fig = Figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax.plot(omega, mag, label=m_label)
                ax.set_xlabel('Frequency (rad/s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'FRF of {m_label} (With DVA)')
                ax.legend()
                ax.grid(True)
                self.frf_combo.addItem(f"{m_label} (With DVA)")
                self.frf_plots[f"{m_label} (With DVA)"] = fig
            else:
                QMessageBox.warning(self, "Plot Error", f"{m_label}: magnitude length != omega length.")

        # Combined plot with DVAs
        if mass_labels:
            fig_combined = Figure(figsize=(6,4))
            ax_combined = fig_combined.add_subplot(111)
            for m_label in mass_labels:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined.plot(omega, mag, label=m_label)
            ax_combined.set_xlabel('Frequency (rad/s)')
            ax_combined.set_ylabel('Amplitude')
            ax_combined.set_title('Combined FRF of All Masses (With DVAs)')
            ax_combined.grid(True)
            ax_combined.legend()
            self.frf_combo.addItem("All Masses Combined (With DVAs)")
            self.frf_plots["All Masses Combined (With DVAs)"] = fig_combined

        # Plot without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                fig = Figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax.plot(omega, mag, label=f"{m_label} (Without DVA)", color='green')
                    ax.set_xlabel('Frequency (rad/s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'FRF of {m_label} (Without DVA)')
                    ax.legend()
                    ax.grid(True)
                    self.frf_combo.addItem(f"{m_label} (Without DVA)")
                    self.frf_plots[f"{m_label} (Without DVA)"] = fig
                else:
                    QMessageBox.warning(self, "Plot Error", f"{m_label} (Without DVA): magnitude length mismatch.")

        # Combined plot with and without DVAs for Mass1 & Mass2
        fig_combined_with_without = Figure(figsize=(6,4))
        ax_combined_with_without = fig_combined_with_without.add_subplot(111)

        # With DVA lines
        for m_label in ['mass_1', 'mass_2']:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (With DVA)")

        # Without DVA lines
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')

        ax_combined_with_without.set_xlabel('Frequency (rad/s)')
        ax_combined_with_without.set_ylabel('Amplitude')
        ax_combined_with_without.set_title('FRF of Mass 1 & 2: With and Without DVAs')
        ax_combined_with_without.grid(True)
        ax_combined_with_without.legend()
        self.frf_combo.addItem("Mass 1 & 2: With and Without DVAs")
        self.frf_plots["Mass 1 & 2: With and Without DVAs"] = fig_combined_with_without

        # Plot all masses combined with and without DVAs for mass1 & mass2
        fig_all_combined = Figure(figsize=(6,4))
        ax_all_combined = fig_all_combined.add_subplot(111)

        # With DVAs (all masses)
        for m_label in mass_labels:
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax_all_combined.plot(omega, mag, label=f"{m_label} (With DVA)")

        # Without DVAs for mass1 & mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_all_combined.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')

        ax_all_combined.set_xlabel('Frequency (rad/s)')
        ax_all_combined.set_ylabel('Amplitude')
        ax_all_combined.set_title('Combined FRF (All Masses), \nMass1 & 2 with/without DVAs')
        ax_all_combined.grid(True)
        ax_all_combined.legend()
        self.frf_combo.addItem("All Masses Combined: With and Without DVAs for Mass 1 & 2")
        self.frf_plots["All Masses Combined: With and Without DVAs for Mass 1 & 2"] = fig_all_combined

        self.update_frf_plot()
        self.frf_canvas.draw()

    def handle_frf_error(self, err):
        self.results_text.append(f"Error running FRF: {err}")
        self.status_bar.showMessage("FRF Error encountered.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

    def display_sobol_results(self, all_results, warnings):
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

        # Build the various Sobol plots (see sobol_sensitivity.py for placeholders)
        self.generate_sobol_plots(all_results, param_names)
        self.update_sobol_plot()
        self.sobol_canvas.draw()

    def handle_sobol_error(self, err):
        self.sobol_results_text.append(f"Error running Sobol Analysis: {err}")
        self.status_bar.showMessage("Sobol Error encountered.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        self.run_ga_button.setEnabled(True)
        self.ga_results_text.append("\nGA Completed.\n")
        self.ga_results_text.append("Best individual parameters:")

        for name, val in zip(parameter_names, best_ind):
            self.ga_results_text.append(f"{name}: {val}")
        self.ga_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.ga_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.ga_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.ga_results_text.append(f"{section}: {data}")

    def handle_ga_error(self, err):
        self.run_ga_button.setEnabled(True)
        QMessageBox.warning(self, "GA Error", f"Error during GA: {err}")
        self.ga_results_text.append(f"\nError running GA: {err}")

    def handle_ga_update(self, msg):
        self.ga_results_text.append(msg)

    ########################################################################
    # -------------- Plotting Helpers --------------
    ########################################################################

    def update_frf_plot(self):
        key = self.frf_combo.currentText()
        if key in self.frf_plots:
            fig = self.frf_plots[key]
            self.frf_canvas.figure = fig
            self.frf_canvas.draw()
        else:
            self.frf_canvas.figure.clear()
            self.frf_canvas.draw()

    def update_sobol_plot(self):
        key = self.sobol_combo.currentText()
        if key in self.sobol_plots:
            fig = self.sobol_plots[key]
            self.sobol_canvas.figure = fig
            self.sobol_canvas.draw()
        else:
            self.sobol_canvas.figure.clear()
            self.sobol_canvas.draw()

    def save_plot(self, fig, plot_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {plot_type} Plot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"{plot_type} plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {e}")

    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path,'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self,"Success",f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self,"Error",f"Failed to save results: {e}")

    ########################################################################
    # -------------- Sobol Plots Generation --------------
    ########################################################################

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
    # -------------- Sobol Visualization Methods (placeholders) --------------
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
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_S1], rotation=90, fontsize=8)
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
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_ST], rotation=90, fontsize=8)
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
                ax.set_title(f"Convergence: {format_parameter_name(param)}", fontsize=10)
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
        ax.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)

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
        ax_s1.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)
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
        ax_st.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)
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
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001, format_parameter_name(param), fontsize=8)

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

    ########################################################################
    # -------------- Getting Inputs & Defaults --------------
    ########################################################################

    def get_target_values_weights(self):
        target_values_dict = {}
        weights_dict = {}

        for mass_num in range(1,6):
            t_dict = {}
            w_dict = {}
            for peak_num in range(1,5):
                pv_key = f"peak_value_{peak_num}_m{mass_num}"
                if pv_key in self.mass_target_spins:
                    t_dict[f"peak_value_{peak_num}"] = self.mass_target_spins[pv_key].value()
                    w_dict[f"peak_value_{peak_num}"] = self.mass_weight_spins[pv_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    bw_key = f"bandwidth_{i}_{j}_m{mass_num}"
                    if bw_key in self.mass_target_spins:
                        t_dict[f"bandwidth_{i}_{j}"] = self.mass_target_spins[bw_key].value()
                        w_dict[f"bandwidth_{i}_{j}"] = self.mass_weight_spins[bw_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    slope_key = f"slope_{i}_{j}_m{mass_num}"
                    if slope_key in self.mass_target_spins:
                        t_dict[f"slope_{i}_{j}"] = self.mass_target_spins[slope_key].value()
                        w_dict[f"slope_{i}_{j}"] = self.mass_weight_spins[slope_key].value()

            auc_key = f"area_under_curve_m{mass_num}"
            if auc_key in self.mass_target_spins:
                t_dict["area_under_curve"] = self.mass_target_spins[auc_key].value()
                w_dict["area_under_curve"] = self.mass_weight_spins[auc_key].value()

            target_values_dict[f"mass_{mass_num}"] = t_dict
            weights_dict[f"mass_{mass_num}"] = w_dict

        return target_values_dict, weights_dict

    def get_num_samples_list(self):
        text = self.num_samples_line.text().strip()
        if not text:
            return [32,64,128]
        try:
            parts = text.split(',')
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
            if not nums:
                raise ValueError("No valid integers found.")
            return nums
        except Exception:
            QMessageBox.warning(self, "Input Error", "Invalid Sample Sizes Input. Using default [32,64,128].")
            return [32,64,128]

    def set_default_values(self):
        self.mu_box.setValue(1.0)
        for i in range(5):
            if i < 2:
                self.landa_boxes[i].setValue(1.0)
            else:
                self.landa_boxes[i].setValue(0.5)
        for i in range(5):
            self.nu_boxes[i].setValue(0.75)

        self.a_low_box.setValue(0.05)
        self.a_up_box.setValue(0.05)
        self.f_1_box.setValue(100.0)
        self.f_2_box.setValue(100.0)
        self.omega_dc_box.setValue(5000.0)
        self.zeta_dc_box.setValue(0.01)

        self.omega_start_box.setValue(0.0)
        self.omega_end_box.setValue(10000.0)
        self.omega_points_box.setValue(1200)

        self.plot_figure_chk.setChecked(True)
        self.show_peaks_chk.setChecked(False)
        self.show_slopes_chk.setChecked(False)

        self.num_samples_line.setText("32,64,128")
        self.n_jobs_spin.setValue(5)

        # Default "Sobol" table
        for row in range(self.dva_param_table.rowCount()):
            param_name = self.dva_param_table.item(row, 0).text()
            if param_name.startswith("beta_") or param_name.startswith("lambda_") or param_name.startswith("nu_"):
                lower = 0.0001
                upper = 2.5
            elif param_name.startswith("mu_"):
                lower = 0.0001
                upper = 0.75
            else:
                lower = 0.0
                upper = 1.0
            lower_bound_spin = self.dva_param_table.cellWidget(row, 3)
            upper_bound_spin = self.dva_param_table.cellWidget(row, 4)
            lower_bound_spin.setValue(lower)
            upper_bound_spin.setValue(upper)
            fixed_checkbox = self.dva_param_table.cellWidget(row, 1)
            fixed_checkbox.setChecked(False)

        # Default "GA" table
        for row in range(self.ga_param_table.rowCount()):
            param_name = self.ga_param_table.item(row, 0).text()
            if param_name.startswith("beta_") or param_name.startswith("lambda_") or param_name.startswith("nu_"):
                lower = 0.0001
                upper = 2.5
            elif param_name.startswith("mu_"):
                lower = 0.0001
                upper = 0.75
            else:
                lower = 0.0
                upper = 1.0
            lower_bound_spin = self.ga_param_table.cellWidget(row, 3)
            upper_bound_spin = self.ga_param_table.cellWidget(row, 4)
            lower_bound_spin.setValue(lower)
            upper_bound_spin.setValue(upper)
            fixed_checkbox = self.ga_param_table.cellWidget(row, 1)
            fixed_checkbox.setChecked(False)


    ########################################################################
    # -------------- PSO GUI section --------------
    ########################################################################

    def create_pso_tab(self):
        self.pso_tab = QWidget()
        layout = QVBoxLayout(self.pso_tab)

        splitter = QSplitter(Qt.Horizontal)

        pso_group = QGroupBox("PSO Settings")
        pso_form = QFormLayout(pso_group)

        self.pso_swarm_size_box = QSpinBox()
        self.pso_swarm_size_box.setRange(1, 10000)
        self.pso_swarm_size_box.setValue(50)

        self.pso_num_iterations_box = QSpinBox()
        self.pso_num_iterations_box.setRange(1, 10000)
        self.pso_num_iterations_box.setValue(100)

        self.pso_inertia_box = QDoubleSpinBox()
        self.pso_inertia_box.setRange(0, 2)
        self.pso_inertia_box.setValue(0.7)
        self.pso_inertia_box.setDecimals(3)

        self.pso_cognitive_box = QDoubleSpinBox()
        self.pso_cognitive_box.setRange(0, 5)
        self.pso_cognitive_box.setValue(1.5)
        self.pso_cognitive_box.setDecimals(3)

        self.pso_social_box = QDoubleSpinBox()
        self.pso_social_box.setRange(0, 5)
        self.pso_social_box.setValue(1.5)
        self.pso_social_box.setDecimals(3)

        self.pso_tol_box = QDoubleSpinBox()
        self.pso_tol_box.setRange(0, 1e6)
        self.pso_tol_box.setValue(1e-3)
        self.pso_tol_box.setDecimals(6)

        self.pso_alpha_box = QDoubleSpinBox()
        self.pso_alpha_box.setRange(0.0, 10.0)
        self.pso_alpha_box.setDecimals(4)
        self.pso_alpha_box.setSingleStep(0.01)
        self.pso_alpha_box.setValue(0.01)

        pso_form.addRow("Swarm Size:", self.pso_swarm_size_box)
        pso_form.addRow("Number of Iterations:", self.pso_num_iterations_box)
        pso_form.addRow("Inertia Weight (w):", self.pso_inertia_box)
        pso_form.addRow("Cognitive Coefficient (c1):", self.pso_cognitive_box)
        pso_form.addRow("Social Coefficient (c2):", self.pso_social_box)
        pso_form.addRow("Tolerance (tol):", self.pso_tol_box)
        pso_form.addRow("Sparsity Penalty (alpha):", self.pso_alpha_box)

        pso_param_group = QGroupBox("DVA Parameter Settings for PSO")
        pso_param_layout = QVBoxLayout(pso_param_group)

        self.pso_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.pso_param_table.setRowCount(len(dva_parameters))
        self.pso_param_table.setColumnCount(5)
        self.pso_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.pso_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pso_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.pso_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_pso_fixed(state, r))
            self.pso_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.pso_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.pso_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.pso_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Default ranges
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        pso_param_layout.addWidget(self.pso_param_table)
        pso_param_group.setLayout(pso_param_layout)

        splitter.addWidget(pso_group)
        splitter.addWidget(pso_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.pso_results_text = QTextEdit()
        self.pso_results_text.setReadOnly(True)
        layout.addWidget(QLabel("PSO Optimization Results:"))
        layout.addWidget(self.pso_results_text)

        self.run_pso_button_tab = QPushButton("Run PSO")
        self.run_pso_button_tab.setFixedWidth(150)
        self.run_pso_button_tab.clicked.connect(self.run_pso)

        run_pso_layout = QHBoxLayout()
        run_pso_layout.addStretch()
        run_pso_layout.addWidget(self.run_pso_button_tab)
        layout.addLayout(run_pso_layout)

        layout.addStretch()
        self.pso_tab.setLayout(layout)

    def toggle_pso_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.pso_param_table.cellWidget(row, 2)
        lower_bound_spin = self.pso_param_table.cellWidget(row, 3)
        upper_bound_spin = self.pso_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def run_pso(self):
        # Retrieve PSO parameters from the GUI
        swarm_size = self.pso_swarm_size_box.value()
        num_iterations = self.pso_num_iterations_box.value()
        inertia = self.pso_inertia_box.value()
        c1 = self.pso_cognitive_box.value()
        c2 = self.pso_social_box.value()
        tol = self.pso_tol_box.value()
        alpha = self.pso_alpha_box.value()

        pso_dva_parameters = []
        row_count = self.pso_param_table.rowCount()
        for row in range(row_count):
            param_name = self.pso_param_table.item(row, 0).text()
            fixed_widget = self.pso_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.pso_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                pso_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.pso_param_table.cellWidget(row, 3)
                upper_bound_widget = self.pso_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                pso_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        # Create and start PSOWorker
        self.pso_worker = PSOWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            pso_swarm_size=swarm_size,
            pso_num_iterations=num_iterations,
            pso_w=inertia,
            pso_c1=c1,
            pso_c2=c2,
            pso_tol=tol,
            pso_parameter_data=pso_dva_parameters,
            alpha=alpha
        )
        self.pso_worker.finished.connect(self.handle_pso_finished)
        self.pso_worker.error.connect(self.handle_pso_error)
        self.pso_worker.update.connect(self.handle_pso_update)
        self.run_pso_button.setEnabled(False)
        self.pso_results_text.append("Running PSO...")
        self.pso_worker.start()

    def handle_pso_finished(self, results, best_particle, parameter_names, best_fitness):
        self.run_pso_button.setEnabled(True)
        self.pso_results_text.append("\nPSO Completed.\n")
        self.pso_results_text.append("Best particle parameters:")

        for name, val in zip(parameter_names, best_particle):
            self.pso_results_text.append(f"{name}: {val}")
        self.pso_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.pso_results_text.append(f"\nSingular response of best particle: {singular_response}")

        self.pso_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.pso_results_text.append(f"{section}: {data}")

    def handle_pso_error(self, err):
        self.run_pso_button.setEnabled(True)
        QMessageBox.warning(self, "PSO Error", f"Error during PSO: {err}")
        self.pso_results_text.append(f"\nError running PSO: {err}")

    def handle_pso_update(self, msg):
        self.pso_results_text.append(msg)


    ########################################################################
    # -------------- DE GUI section --------------
    ########################################################################

    def create_de_tab(self):
        self.de_tab = QWidget()
        layout = QVBoxLayout(self.de_tab)

        splitter = QSplitter(Qt.Horizontal)

        de_group = QGroupBox("DE Settings")
        de_form = QFormLayout(de_group)

        self.de_pop_size_box = QSpinBox()
        self.de_pop_size_box.setRange(1, 10000)
        self.de_pop_size_box.setValue(50)

        self.de_num_generations_box = QSpinBox()
        self.de_num_generations_box.setRange(1, 10000)
        self.de_num_generations_box.setValue(100)

        self.de_F_box = QDoubleSpinBox()
        self.de_F_box.setRange(0, 2)
        self.de_F_box.setValue(0.8)  # typical value between 0.5 and 0.9
        self.de_F_box.setDecimals(3)

        self.de_CR_box = QDoubleSpinBox()
        self.de_CR_box.setRange(0, 1)
        self.de_CR_box.setValue(0.9)  # typical value near 0.9
        self.de_CR_box.setDecimals(3)

        self.de_tol_box = QDoubleSpinBox()
        self.de_tol_box.setRange(0, 1e6)
        self.de_tol_box.setValue(1e-3)
        self.de_tol_box.setDecimals(6)

        self.de_alpha_box = QDoubleSpinBox()
        self.de_alpha_box.setRange(0.0, 10.0)
        self.de_alpha_box.setDecimals(4)
        self.de_alpha_box.setSingleStep(0.01)
        self.de_alpha_box.setValue(0.01)

        de_form.addRow("Population Size:", self.de_pop_size_box)
        de_form.addRow("Number of Generations:", self.de_num_generations_box)
        de_form.addRow("Mutation Factor (F):", self.de_F_box)
        de_form.addRow("Crossover Probability (CR):", self.de_CR_box)
        de_form.addRow("Tolerance (tol):", self.de_tol_box)
        de_form.addRow("Sparsity Penalty (alpha):", self.de_alpha_box)

        de_param_group = QGroupBox("DVA Parameter Settings for DE")
        de_param_layout = QVBoxLayout(de_param_group)

        self.de_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.de_param_table.setRowCount(len(dva_parameters))
        self.de_param_table.setColumnCount(5)
        self.de_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.de_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.de_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.de_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_de_fixed(state, r))
            self.de_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.de_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.de_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.de_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Set default ranges based on parameter name
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        de_param_layout.addWidget(self.de_param_table)
        de_param_group.setLayout(de_param_layout)

        splitter.addWidget(de_group)
        splitter.addWidget(de_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.de_results_text = QTextEdit()
        self.de_results_text.setReadOnly(True)
        layout.addWidget(QLabel("DE Optimization Results:"))
        layout.addWidget(self.de_results_text)

        self.run_de_button_tab = QPushButton("Run DE")
        self.run_de_button_tab.setFixedWidth(150)
        self.run_de_button_tab.clicked.connect(self.run_de)

        run_de_layout = QHBoxLayout()
        run_de_layout.addStretch()
        run_de_layout.addWidget(self.run_de_button_tab)
        layout.addLayout(run_de_layout)

        layout.addStretch()
        self.de_tab.setLayout(layout)


    def toggle_de_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.de_param_table.cellWidget(row, 2)
        lower_bound_spin = self.de_param_table.cellWidget(row, 3)
        upper_bound_spin = self.de_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)


    def run_de(self):
        # Retrieve DE parameters from the GUI
        pop_size = self.de_pop_size_box.value()
        num_gens = self.de_num_generations_box.value()
        F = self.de_F_box.value()
        CR = self.de_CR_box.value()
        tol = self.de_tol_box.value()
        alpha = self.de_alpha_box.value()

        de_dva_parameters = []
        row_count = self.de_param_table.rowCount()
        for row in range(row_count):
            param_name = self.de_param_table.item(row, 0).text()
            fixed_widget = self.de_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.de_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                de_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.de_param_table.cellWidget(row, 3)
                upper_bound_widget = self.de_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                de_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        # Create and start DEWorker
        self.de_worker = DEWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            de_pop_size=pop_size,
            de_num_generations=num_gens,
            de_F=F,
            de_CR=CR,
            de_tol=tol,
            de_parameter_data=de_dva_parameters,
            alpha=alpha
        )
        self.de_worker.finished.connect(self.handle_de_finished)
        self.de_worker.error.connect(self.handle_de_error)
        self.de_worker.update.connect(self.handle_de_update)
        self.run_de_button.setEnabled(False)
        self.de_results_text.append("Running DE...")
        self.de_worker.start()


    def handle_de_finished(self, results, best_individual, parameter_names, best_fitness):
        self.run_de_button.setEnabled(True)
        self.de_results_text.append("\nDE Completed.\n")
        self.de_results_text.append("Best individual parameters:")

        for name, val in zip(parameter_names, best_individual):
            self.de_results_text.append(f"{name}: {val}")
        self.de_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.de_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.de_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.de_results_text.append(f"{section}: {data}")


    def handle_de_error(self, err):
        self.run_de_button.setEnabled(True)
        QMessageBox.warning(self, "DE Error", f"Error during DE: {err}")
        self.de_results_text.append(f"\nError running DE: {err}")


    def handle_de_update(self, msg):
        self.de_results_text.append(msg)



    ########################################################################
    # -------------- SA GUI section --------------
    ########################################################################

    def create_sa_tab(self):
        self.sa_tab = QWidget()
        layout = QVBoxLayout(self.sa_tab)

        splitter = QSplitter(Qt.Horizontal)

        sa_group = QGroupBox("SA Settings")
        sa_form = QFormLayout(sa_group)

        self.sa_initial_temp_box = QDoubleSpinBox()
        self.sa_initial_temp_box.setRange(0, 1e6)
        self.sa_initial_temp_box.setValue(1000)  # default initial temperature
        self.sa_initial_temp_box.setDecimals(2)

        self.sa_cooling_rate_box = QDoubleSpinBox()
        self.sa_cooling_rate_box.setRange(0, 1)
        self.sa_cooling_rate_box.setValue(0.95)  # typical cooling rate
        self.sa_cooling_rate_box.setDecimals(3)

        self.sa_num_iterations_box = QSpinBox()
        self.sa_num_iterations_box.setRange(1, 10000)
        self.sa_num_iterations_box.setValue(1000)

        self.sa_tol_box = QDoubleSpinBox()
        self.sa_tol_box.setRange(0, 1e6)
        self.sa_tol_box.setValue(1e-3)
        self.sa_tol_box.setDecimals(6)

        self.sa_alpha_box = QDoubleSpinBox()
        self.sa_alpha_box.setRange(0.0, 10.0)
        self.sa_alpha_box.setDecimals(4)
        self.sa_alpha_box.setSingleStep(0.01)
        self.sa_alpha_box.setValue(0.01)

        sa_form.addRow("Initial Temperature:", self.sa_initial_temp_box)
        sa_form.addRow("Cooling Rate:", self.sa_cooling_rate_box)
        sa_form.addRow("Number of Iterations:", self.sa_num_iterations_box)
        sa_form.addRow("Tolerance (tol):", self.sa_tol_box)
        sa_form.addRow("Sparsity Penalty (alpha):", self.sa_alpha_box)

        sa_param_group = QGroupBox("DVA Parameter Settings for SA")
        sa_param_layout = QVBoxLayout(sa_param_group)

        self.sa_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.sa_param_table.setRowCount(len(dva_parameters))
        self.sa_param_table.setColumnCount(5)
        self.sa_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.sa_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sa_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.sa_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_sa_fixed(state, r))
            self.sa_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.sa_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.sa_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.sa_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Default ranges
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        sa_param_layout.addWidget(self.sa_param_table)
        sa_param_group.setLayout(sa_param_layout)

        splitter.addWidget(sa_group)
        splitter.addWidget(sa_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.sa_results_text = QTextEdit()
        self.sa_results_text.setReadOnly(True)
        layout.addWidget(QLabel("SA Optimization Results:"))
        layout.addWidget(self.sa_results_text)

        self.run_sa_button_tab = QPushButton("Run SA")
        self.run_sa_button_tab.setFixedWidth(150)
        self.run_sa_button_tab.clicked.connect(self.run_sa)

        run_sa_layout = QHBoxLayout()
        run_sa_layout.addStretch()
        run_sa_layout.addWidget(self.run_sa_button_tab)
        layout.addLayout(run_sa_layout)

        layout.addStretch()
        self.sa_tab.setLayout(layout)


    def toggle_sa_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.sa_param_table.cellWidget(row, 2)
        lower_bound_spin = self.sa_param_table.cellWidget(row, 3)
        upper_bound_spin = self.sa_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)


    def run_sa(self):
        # Retrieve SA parameters from the GUI
        initial_temp = self.sa_initial_temp_box.value()
        cooling_rate = self.sa_cooling_rate_box.value()
        num_iterations = self.sa_num_iterations_box.value()
        tol = self.sa_tol_box.value()
        alpha = self.sa_alpha_box.value()

        sa_dva_parameters = []
        row_count = self.sa_param_table.rowCount()
        for row in range(row_count):
            param_name = self.sa_param_table.item(row, 0).text()
            fixed_widget = self.sa_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.sa_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                sa_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.sa_param_table.cellWidget(row, 3)
                upper_bound_widget = self.sa_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                sa_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        # Create and start SAWorker
        self.sa_worker = SAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            sa_initial_temp=initial_temp,
            sa_cooling_rate=cooling_rate,
            sa_num_iterations=num_iterations,
            sa_tol=tol,
            sa_parameter_data=sa_dva_parameters,
            alpha=alpha
        )
        self.sa_worker.finished.connect(self.handle_sa_finished)
        self.sa_worker.error.connect(self.handle_sa_error)
        self.sa_worker.update.connect(self.handle_sa_update)
        self.run_sa_button.setEnabled(False)
        self.sa_results_text.append("Running SA...")
        self.sa_worker.start()


    def handle_sa_finished(self, results, best_candidate, parameter_names, best_fitness):
        self.run_sa_button.setEnabled(True)
        self.sa_results_text.append("\nSA Completed.\n")
        self.sa_results_text.append("Best candidate parameters:")

        for name, val in zip(parameter_names, best_candidate):
            self.sa_results_text.append(f"{name}: {val}")
        self.sa_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.sa_results_text.append(f"\nSingular response of best candidate: {singular_response}")

        self.sa_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.sa_results_text.append(f"{section}: {data}")


    def handle_sa_error(self, err):
        self.run_sa_button.setEnabled(True)
        QMessageBox.warning(self, "SA Error", f"Error during SA: {err}")
        self.sa_results_text.append(f"\nError running SA: {err}")


    def handle_sa_update(self, msg):
        self.sa_results_text.append(msg)


    ########################################################################
    # -------------- CMA-ES GUI section --------------
    ########################################################################

    def create_cmaes_tab(self):
        self.cmaes_tab = QWidget()
        layout = QVBoxLayout(self.cmaes_tab)

        splitter = QSplitter(Qt.Horizontal)

        cmaes_group = QGroupBox("CMA-ES Settings")
        cmaes_form = QFormLayout(cmaes_group)

        self.cmaes_sigma_box = QDoubleSpinBox()
        self.cmaes_sigma_box.setRange(0, 1e6)
        self.cmaes_sigma_box.setValue(100)  # default initial sigma
        self.cmaes_sigma_box.setDecimals(2)

        self.cmaes_max_iter_box = QSpinBox()
        self.cmaes_max_iter_box.setRange(1, 10000)
        self.cmaes_max_iter_box.setValue(500)

        self.cmaes_tol_box = QDoubleSpinBox()
        self.cmaes_tol_box.setRange(0, 1e6)
        self.cmaes_tol_box.setValue(1e-3)
        self.cmaes_tol_box.setDecimals(6)

        self.cmaes_alpha_box = QDoubleSpinBox()
        self.cmaes_alpha_box.setRange(0.0, 10.0)
        self.cmaes_alpha_box.setDecimals(4)
        self.cmaes_alpha_box.setSingleStep(0.01)
        self.cmaes_alpha_box.setValue(0.01)

        cmaes_form.addRow("Initial Sigma:", self.cmaes_sigma_box)
        cmaes_form.addRow("Max Iterations:", self.cmaes_max_iter_box)
        cmaes_form.addRow("Tolerance (tol):", self.cmaes_tol_box)
        cmaes_form.addRow("Sparsity Penalty (alpha):", self.cmaes_alpha_box)

        cmaes_param_group = QGroupBox("DVA Parameter Settings for CMA-ES")
        cmaes_param_layout = QVBoxLayout(cmaes_param_group)

        self.cmaes_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.cmaes_param_table.setRowCount(len(dva_parameters))
        self.cmaes_param_table.setColumnCount(5)
        self.cmaes_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.cmaes_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cmaes_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.cmaes_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_cmaes_fixed(state, r))
            self.cmaes_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.cmaes_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.cmaes_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.cmaes_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Set default ranges based on parameter name
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        cmaes_param_layout.addWidget(self.cmaes_param_table)
        cmaes_param_group.setLayout(cmaes_param_layout)

        splitter.addWidget(cmaes_group)
        splitter.addWidget(cmaes_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.cmaes_results_text = QTextEdit()
        self.cmaes_results_text.setReadOnly(True)
        layout.addWidget(QLabel("CMA-ES Optimization Results:"))
        layout.addWidget(self.cmaes_results_text)

        self.run_cmaes_button_tab = QPushButton("Run CMA-ES")
        self.run_cmaes_button_tab.setFixedWidth(150)
        self.run_cmaes_button_tab.clicked.connect(self.run_cmaes)

        run_cmaes_layout = QHBoxLayout()
        run_cmaes_layout.addStretch()
        run_cmaes_layout.addWidget(self.run_cmaes_button_tab)
        layout.addLayout(run_cmaes_layout)

        layout.addStretch()
        self.cmaes_tab.setLayout(layout)


    def toggle_cmaes_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.cmaes_param_table.cellWidget(row, 2)
        lower_bound_spin = self.cmaes_param_table.cellWidget(row, 3)
        upper_bound_spin = self.cmaes_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)


    def run_cmaes(self):
        # Retrieve CMA-ES parameters from the GUI
        sigma = self.cmaes_sigma_box.value()
        max_iter = self.cmaes_max_iter_box.value()
        tol = self.cmaes_tol_box.value()
        alpha = self.cmaes_alpha_box.value()

        cmaes_dva_parameters = []
        row_count = self.cmaes_param_table.rowCount()
        for row in range(row_count):
            param_name = self.cmaes_param_table.item(row, 0).text()
            fixed_widget = self.cmaes_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.cmaes_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                cmaes_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.cmaes_param_table.cellWidget(row, 3)
                upper_bound_widget = self.cmaes_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                cmaes_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        # Create and start CMAESWorker
        self.cmaes_worker = CMAESWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            cma_initial_sigma=sigma,
            cma_max_iter=max_iter,
            cma_tol=tol,
            cma_parameter_data=cmaes_dva_parameters,
            alpha=alpha
        )
        self.cmaes_worker.finished.connect(self.handle_cmaes_finished)
        self.cmaes_worker.error.connect(self.handle_cmaes_error)
        self.cmaes_worker.update.connect(self.handle_cmaes_update)
        self.run_cmaes_button.setEnabled(False)
        self.cmaes_results_text.append("Running CMA-ES...")
        self.cmaes_worker.start()


    def handle_cmaes_finished(self, results, best_candidate, parameter_names, best_fitness):
        self.run_cmaes_button.setEnabled(True)
        self.cmaes_results_text.append("\nCMA-ES Completed.\n")
        self.cmaes_results_text.append("Best candidate parameters:")
        for name, val in zip(parameter_names, best_candidate):
            self.cmaes_results_text.append(f"{name}: {val}")
        self.cmaes_results_text.append(f"\nBest fitness: {best_fitness:.6f}")
        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.cmaes_results_text.append(f"\nSingular response of best candidate: {singular_response}")
        self.cmaes_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.cmaes_results_text.append(f"{section}: {data}")


    def handle_cmaes_error(self, err):
        self.run_cmaes_button.setEnabled(True)
        QMessageBox.warning(self, "CMA-ES Error", f"Error during CMA-ES: {err}")
        self.cmaes_results_text.append(f"\nError running CMA-ES: {err}")


    def handle_cmaes_update(self, msg):
        self.cmaes_results_text.append(msg)


    ########################################################################
    # -------------- RL_1 GUI section --------------
    ########################################################################
    def create_rl_tab(self):
        # Create the main RL tab widget
        self.rl_tab = QWidget()
        rl_layout = QVBoxLayout(self.rl_tab)

        # Create a sub–tab widget to organize RL inputs into four groups
        self.rl_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: Hyperparameters --------------------
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        self.rl_num_episodes_box = QSpinBox()
        self.rl_num_episodes_box.setRange(1, 10000)
        self.rl_num_episodes_box.setValue(100)
        general_layout.addRow("Number of Episodes:", self.rl_num_episodes_box)
        
        self.rl_max_steps_box = QSpinBox()
        self.rl_max_steps_box.setRange(1, 10000)
        self.rl_max_steps_box.setValue(1000)
        general_layout.addRow("Max Steps per Episode:", self.rl_max_steps_box)
        
        self.rl_learning_rate_box = QDoubleSpinBox()
        self.rl_learning_rate_box.setRange(0, 1)
        self.rl_learning_rate_box.setValue(0.1)
        self.rl_learning_rate_box.setDecimals(4)
        general_layout.addRow("Learning Rate (alpha):", self.rl_learning_rate_box)
        
        self.rl_gamma_box = QDoubleSpinBox()
        self.rl_gamma_box.setRange(0, 1)
        self.rl_gamma_box.setValue(0.99)
        self.rl_gamma_box.setDecimals(4)
        general_layout.addRow("Discount Factor (gamma):", self.rl_gamma_box)
        
        self.rl_epsilon_box = QDoubleSpinBox()
        self.rl_epsilon_box.setRange(0, 1)
        self.rl_epsilon_box.setValue(1.0)
        self.rl_epsilon_box.setDecimals(4)
        general_layout.addRow("Initial Epsilon:", self.rl_epsilon_box)
        
        self.rl_epsilon_min_box = QDoubleSpinBox()
        self.rl_epsilon_min_box.setRange(0, 1)
        self.rl_epsilon_min_box.setValue(0.01)
        self.rl_epsilon_min_box.setDecimals(4)
        general_layout.addRow("Minimum Epsilon:", self.rl_epsilon_min_box)
        
        self.rl_epsilon_decay_box = QDoubleSpinBox()
        self.rl_epsilon_decay_box.setRange(0, 1)
        self.rl_epsilon_decay_box.setValue(0.99)
        self.rl_epsilon_decay_box.setDecimals(4)
        general_layout.addRow("Epsilon Decay:", self.rl_epsilon_decay_box)
        
        # Decay method combo
        self.rl_epsilon_decay_method_combo = QComboBox()
        self.rl_epsilon_decay_method_combo.addItems(
            ["exponential", "linear", "inverse", "step", "cosine"]
        )
        self.rl_epsilon_decay_method_combo.setCurrentText("exponential")
        self.rl_epsilon_decay_method_combo.currentIndexChanged.connect(self.update_decay_params_ui)
        general_layout.addRow("Decay Method:", self.rl_epsilon_decay_method_combo)
        
        self.rl_alpha_box = QDoubleSpinBox()
        self.rl_alpha_box.setRange(0.0, 10.0)
        self.rl_alpha_box.setDecimals(4)
        self.rl_alpha_box.setSingleStep(0.01)
        self.rl_alpha_box.setValue(0.01)
        general_layout.addRow("Sparsity Penalty (alpha):", self.rl_alpha_box)
        
        # --- Add Sobol parameters ---
        self.rl_sobol_sample_size_box = QSpinBox()
        self.rl_sobol_sample_size_box.setRange(2, 100000)
        self.rl_sobol_sample_size_box.setValue(32)
        general_layout.addRow("Sobol Sample Size:", self.rl_sobol_sample_size_box)
        
        self.rl_sobol_tolerance_box = QDoubleSpinBox()
        self.rl_sobol_tolerance_box.setRange(0, 1)
        self.rl_sobol_tolerance_box.setDecimals(4)
        self.rl_sobol_tolerance_box.setValue(0.05)
        general_layout.addRow("Sobol Tolerance:", self.rl_sobol_tolerance_box)
        
        # --- Add a small Run RL button in the Hyperparameters sub–tab ---
        self.hyper_run_rl_button = QPushButton("Run RL")
        self.hyper_run_rl_button.setFixedWidth(100)
        self.hyper_run_rl_button.clicked.connect(self.run_rl)
        general_layout.addRow("Run RL:", self.hyper_run_rl_button)
        
        general_tab.setLayout(general_layout)
        self.rl_sub_tabs.addTab(general_tab, "Hyperparameters")
        
        # -------------------- Sub-tab 2: Epsilon Decay Parameters --------------------
        self.decay_params_group = QGroupBox("Epsilon Decay Parameters")
        decay_layout = QVBoxLayout(self.decay_params_group)
        self.rl_decay_params_stack = QStackedWidget()
        # Exponential page (no extra parameters)
        exp_page = QWidget()
        exp_layout = QFormLayout(exp_page)
        exp_layout.addRow(QLabel("No extra parameters required for exponential decay."))
        exp_page.setLayout(exp_layout)
        self.rl_decay_params_stack.addWidget(exp_page)
        # Linear page
        linear_page = QWidget()
        linear_layout = QFormLayout(linear_page)
        self.rl_linear_decay_step_box = QDoubleSpinBox()
        self.rl_linear_decay_step_box.setRange(0, 1)
        self.rl_linear_decay_step_box.setDecimals(4)
        self.rl_linear_decay_step_box.setValue(0.01)
        linear_layout.addRow("Linear Decay Step:", self.rl_linear_decay_step_box)
        linear_page.setLayout(linear_layout)
        self.rl_decay_params_stack.addWidget(linear_page)
        # Inverse page
        inverse_page = QWidget()
        inverse_layout = QFormLayout(inverse_page)
        self.rl_inverse_decay_coeff_box = QDoubleSpinBox()
        self.rl_inverse_decay_coeff_box.setRange(0.1, 10)
        self.rl_inverse_decay_coeff_box.setDecimals(4)
        self.rl_inverse_decay_coeff_box.setValue(1.0)
        inverse_layout.addRow("Inverse Decay Coefficient:", self.rl_inverse_decay_coeff_box)
        inverse_page.setLayout(inverse_layout)
        self.rl_decay_params_stack.addWidget(inverse_page)
        # Step page
        step_page = QWidget()
        step_layout = QFormLayout(step_page)
        self.rl_step_interval_box = QSpinBox()
        self.rl_step_interval_box.setRange(1, 1000)
        self.rl_step_interval_box.setValue(10)
        self.rl_step_decay_amount_box = QDoubleSpinBox()
        self.rl_step_decay_amount_box.setRange(0, 1)
        self.rl_step_decay_amount_box.setDecimals(4)
        self.rl_step_decay_amount_box.setValue(self.rl_epsilon_decay_box.value())
        step_layout.addRow("Step Interval:", self.rl_step_interval_box)
        step_layout.addRow("Step Decay Amount:", self.rl_step_decay_amount_box)
        step_page.setLayout(step_layout)
        self.rl_decay_params_stack.addWidget(step_page)
        # Cosine page
        cosine_page = QWidget()
        cosine_layout = QFormLayout(cosine_page)
        self.rl_cosine_decay_amplitude_box = QDoubleSpinBox()
        self.rl_cosine_decay_amplitude_box.setRange(0.1, 5)
        self.rl_cosine_decay_amplitude_box.setDecimals(4)
        self.rl_cosine_decay_amplitude_box.setValue(1.0)
        cosine_layout.addRow("Cosine Decay Amplitude:", self.rl_cosine_decay_amplitude_box)
        cosine_page.setLayout(cosine_layout)
        self.rl_decay_params_stack.addWidget(cosine_page)
        
        decay_layout.addWidget(self.rl_decay_params_stack)
        self.decay_params_group.setLayout(decay_layout)
        decay_tab = QWidget()
        decay_tab_layout = QVBoxLayout(decay_tab)
        decay_tab_layout.addWidget(self.decay_params_group)
        decay_tab.setLayout(decay_tab_layout)
        self.rl_sub_tabs.addTab(decay_tab, "Epsilon Decay")
        
        # -------------------- Sub-tab 3: Reward System Settings --------------------
        self.reward_params_group = QGroupBox("Reward System Settings")
        reward_layout = QVBoxLayout(self.reward_params_group)
        # Reward system selection combo
        self.rl_reward_system_combo = QComboBox()
        self.rl_reward_system_combo.addItems([
            "Basic Absolute Error",          # System 1
            "Absolute Error + Simple Cost",  # System 2
            "Absolute Error + Sparsity",     # System 3
            "Original Reward (Abs+alpha*sum)",  # System 4 (original)
            "Absolute Error + Extended Cost" # System 5
        ])
        self.rl_reward_system_combo.setCurrentIndex(0)
        self.rl_reward_system_combo.currentIndexChanged.connect(self.update_reward_params_ui)
        reward_layout.addWidget(QLabel("Select Reward System:"))
        reward_layout.addWidget(self.rl_reward_system_combo)
        # Stacked widget for extra reward parameters
        self.rl_reward_params_stack = QStackedWidget()
        # Page 1: Basic Absolute Error - no extra parameters.
        reward_page1 = QWidget()
        r1_layout = QFormLayout(reward_page1)
        r1_layout.addRow(QLabel("No extra parameters required for Basic Absolute Error."))
        reward_page1.setLayout(r1_layout)
        self.rl_reward_params_stack.addWidget(reward_page1)
        # Page 2: Absolute Error + Simple Cost - cost table and coefficient.
        reward_page2 = QWidget()
        r2_layout = QFormLayout(reward_page2)
        self.rl_cost_table = QTableWidget()
        dva_params = ([f"beta_{i}" for i in range(1, 16)] +
                    [f"lambda_{i}" for i in range(1, 16)] +
                    [f"mu_{i}" for i in range(1, 4)] +
                    [f"nu_{i}" for i in range(1, 16)])
        self.rl_cost_table.setRowCount(len(dva_params))
        self.rl_cost_table.setColumnCount(2)
        self.rl_cost_table.setHorizontalHeaderLabels(["Parameter", "Cost"])
        for row, param in enumerate(dva_params):
            item = QTableWidgetItem(param)
            item.setFlags(Qt.ItemIsEnabled)
            self.rl_cost_table.setItem(row, 0, item)
            cost_spin = QDoubleSpinBox()
            cost_spin.setRange(0, 1000)
            cost_spin.setDecimals(4)
            cost_spin.setValue(1.0)
            self.rl_cost_table.setCellWidget(row, 1, cost_spin)
        r2_layout.addRow(self.rl_cost_table)
        self.rl_simple_cost_coeff_box = QDoubleSpinBox()
        self.rl_simple_cost_coeff_box.setRange(0, 100)
        self.rl_simple_cost_coeff_box.setDecimals(4)
        self.rl_simple_cost_coeff_box.setValue(1.0)
        r2_layout.addRow("Simple Cost Coefficient:", self.rl_simple_cost_coeff_box)
        reward_page2.setLayout(r2_layout)
        self.rl_reward_params_stack.addWidget(reward_page2)
        # Page 3: Absolute Error + Sparsity - one extra parameter.
        reward_page3 = QWidget()
        r3_layout = QFormLayout(reward_page3)
        self.rl_sparsity_penalty_box = QDoubleSpinBox()
        self.rl_sparsity_penalty_box.setRange(0, 10)
        self.rl_sparsity_penalty_box.setDecimals(4)
        self.rl_sparsity_penalty_box.setValue(0.01)
        r3_layout.addRow("Sparsity Penalty Factor:", self.rl_sparsity_penalty_box)
        reward_page3.setLayout(r3_layout)
        self.rl_reward_params_stack.addWidget(reward_page3)
        # Page 4: Original Reward (Abs Error + alpha_sparsity*sum(abs(params))) - one extra parameter.
        reward_page4 = QWidget()
        r4_layout = QFormLayout(reward_page4)
        self.rl_original_alpha_box = QDoubleSpinBox()
        self.rl_original_alpha_box.setRange(0, 10)
        self.rl_original_alpha_box.setDecimals(4)
        self.rl_original_alpha_box.setValue(0.01)
        r4_layout.addRow("Alpha (sparsity factor):", self.rl_original_alpha_box)
        reward_page4.setLayout(r4_layout)
        self.rl_reward_params_stack.addWidget(reward_page4)
        # Page 5: Absolute Error + Extended Cost - extended cost table and time penalty weight.
        reward_page5 = QWidget()
        r5_layout = QFormLayout(reward_page5)
        self.rl_extended_cost_table = QTableWidget()
        self.rl_extended_cost_table.setRowCount(len(dva_params))
        self.rl_extended_cost_table.setColumnCount(2)
        self.rl_extended_cost_table.setHorizontalHeaderLabels(["Parameter", "Extended Cost"])
        for row, param in enumerate(dva_params):
            item = QTableWidgetItem(param)
            item.setFlags(Qt.ItemIsEnabled)
            self.rl_extended_cost_table.setItem(row, 0, item)
            ext_cost_spin = QDoubleSpinBox()
            ext_cost_spin.setRange(0, 1000)
            ext_cost_spin.setDecimals(4)
            ext_cost_spin.setValue(1.0)
            self.rl_extended_cost_table.setCellWidget(row, 1, ext_cost_spin)
        r5_layout.addRow(self.rl_extended_cost_table)
        self.rl_time_penalty_weight_box = QDoubleSpinBox()
        self.rl_time_penalty_weight_box.setRange(0, 100)
        self.rl_time_penalty_weight_box.setDecimals(4)
        self.rl_time_penalty_weight_box.setValue(0.0)
        r5_layout.addRow("Time Penalty Weight:", self.rl_time_penalty_weight_box)
        reward_page5.setLayout(r5_layout)
        self.rl_reward_params_stack.addWidget(reward_page5)
        
        reward_layout.addWidget(self.rl_reward_params_stack)
        self.reward_params_group.setLayout(reward_layout)
        
        reward_tab = QWidget()
        reward_tab_layout = QVBoxLayout(reward_tab)
        reward_tab_layout.addWidget(self.reward_params_group)
        reward_tab.setLayout(reward_tab_layout)
        self.rl_sub_tabs.addTab(reward_tab, "Reward Settings")
        
        # -------------------- Sub-tab 4: DVA Parameter Settings --------------------
        self.rl_param_group = QGroupBox("DVA Parameter Settings for RL")
        rl_param_layout = QVBoxLayout(self.rl_param_group)
        self.rl_param_table = QTableWidget()
        dva_parameters = ([f"beta_{i}" for i in range(1, 16)] +
                        [f"lambda_{i}" for i in range(1, 16)] +
                        [f"mu_{i}" for i in range(1, 4)] +
                        [f"nu_{i}" for i in range(1, 16)])
        self.rl_param_table.setRowCount(len(dva_parameters))
        self.rl_param_table.setColumnCount(5)
        self.rl_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.rl_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rl_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.rl_param_table.setItem(row, 0, param_item)
            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_rl_fixed(state, r))
            self.rl_param_table.setCellWidget(row, 1, fixed_checkbox)
            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.rl_param_table.setCellWidget(row, 2, fixed_value_spin)
            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.rl_param_table.setCellWidget(row, 3, lower_bound_spin)
            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.rl_param_table.setCellWidget(row, 4, upper_bound_spin)
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0000)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0000)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)
        rl_param_layout.addWidget(self.rl_param_table)
        self.rl_param_group.setLayout(rl_param_layout)
        self.rl_sub_tabs.addTab(self.rl_param_group, "DVA Parameters")
        
        # Add the RL sub-tabs widget to the main RL tab layout
        rl_layout.addWidget(self.rl_sub_tabs)
        self.rl_tab.setLayout(rl_layout)
    

    def update_decay_params_ui(self):
        method = self.rl_epsilon_decay_method_combo.currentText()
        mapping = {
            "exponential": 0,
            "linear": 1,
            "inverse": 2,
            "step": 3,
            "cosine": 4
        }
        index = mapping.get(method, 0)
        self.rl_decay_params_stack.setCurrentIndex(index)


    def update_reward_params_ui(self):
        index = self.rl_reward_system_combo.currentIndex()
        self.rl_reward_params_stack.setCurrentIndex(index)


    def toggle_rl_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.rl_param_table.cellWidget(row, 2)
        lower_bound_spin = self.rl_param_table.cellWidget(row, 3)
        upper_bound_spin = self.rl_param_table.cellWidget(row, 4)
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)


    def run_rl(self):
        # Retrieve general RL parameters
        num_episodes = self.rl_num_episodes_box.value()
        max_steps = self.rl_max_steps_box.value()
        learning_rate = self.rl_learning_rate_box.value()
        gamma = self.rl_gamma_box.value()
        epsilon = self.rl_epsilon_box.value()
        epsilon_min = self.rl_epsilon_min_box.value()
        epsilon_decay = self.rl_epsilon_decay_box.value()
        epsilon_decay_method = self.rl_epsilon_decay_method_combo.currentText()
        alpha = self.rl_alpha_box.value()

        extra_decay_params = {}
        if epsilon_decay_method == "linear":
            extra_decay_params["rl_linear_decay_step"] = self.rl_linear_decay_step_box.value()
        elif epsilon_decay_method == "inverse":
            extra_decay_params["rl_inverse_decay_coefficient"] = self.rl_inverse_decay_coeff_box.value()
        elif epsilon_decay_method == "step":
            extra_decay_params["rl_step_interval"] = self.rl_step_interval_box.value()
            extra_decay_params["rl_step_decay_amount"] = self.rl_step_decay_amount_box.value()
        elif epsilon_decay_method == "cosine":
            extra_decay_params["rl_cosine_decay_amplitude"] = self.rl_cosine_decay_amplitude_box.value()

        # Retrieve Reward System selection and extra parameters
        reward_system_index = self.rl_reward_system_combo.currentIndex() + 1  # 1-based indexing
        extra_reward_params = {}
        if reward_system_index == 2:
            cost_values = []
            rows = self.rl_cost_table.rowCount()
            for row in range(rows):
                spin = self.rl_cost_table.cellWidget(row, 1)
                cost_values.append(spin.value())
            extra_reward_params["cost_values"] = cost_values
            extra_reward_params["simple_cost_coeff"] = self.rl_simple_cost_coeff_box.value()
        elif reward_system_index == 3:
            extra_reward_params["alpha_sparsity_simplified"] = self.rl_sparsity_penalty_box.value()
        elif reward_system_index == 4:
            extra_reward_params["alpha_sparsity"] = self.rl_original_alpha_box.value()
        elif reward_system_index == 5:
            extended_cost_values = []
            rows = self.rl_extended_cost_table.rowCount()
            for row in range(rows):
                spin = self.rl_extended_cost_table.cellWidget(row, 1)
                extended_cost_values.append(spin.value())
            extra_reward_params["extended_cost_values"] = extended_cost_values
            extra_reward_params["time_penalty_weight"] = self.rl_time_penalty_weight_box.value()

        # Retrieve Sobol settings for RL
        sobol_sample_size = self.rl_sobol_sample_size_box.value()
        sobol_tolerance = self.rl_sobol_tolerance_box.value()
        sobol_settings = {"sample_size": sobol_sample_size, "tolerance": sobol_tolerance}

        # Retrieve DVA parameters from the table
        rl_dva_parameters = []
        row_count = self.rl_param_table.rowCount()
        for row in range(row_count):
            param_name = self.rl_param_table.item(row, 0).text()
            fixed_widget = self.rl_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.rl_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                rl_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.rl_param_table.cellWidget(row, 3)
                upper_bound_widget = self.rl_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                rl_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
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

        target_values, weights = self.get_target_values_weights()
        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        self.rl_worker = RLWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            rl_num_episodes=num_episodes,
            rl_max_steps=max_steps,
            rl_alpha=learning_rate,
            rl_gamma=gamma,
            rl_epsilon=epsilon,
            rl_epsilon_min=epsilon_min,
            rl_epsilon_decay_type=epsilon_decay_method,
            rl_epsilon_decay=epsilon_decay,
            rl_parameter_data=rl_dva_parameters,
            alpha_sparsity=alpha,
            sobol_settings=sobol_settings,
            rl_reward_system=reward_system_index,
            **extra_reward_params,
            **extra_decay_params
        )
        self.rl_worker.finished.connect(self.handle_rl_finished)
        self.rl_worker.error.connect(self.handle_rl_error)
        self.rl_worker.update.connect(self.handle_rl_update)
        # Connect the run RL button from the hyperparameters tab to disable it during execution.
        self.run_rl_button = self.hyper_run_rl_button  
        self.run_rl_button.setEnabled(False)
        # Also, ensure there is an RL results text area in the RL tab:
        if not hasattr(self, "rl_results_text"):
            # Create it if it doesn't exist and add it to the RL tab layout
            self.rl_results_text = QTextEdit()
            self.rl_results_text.setReadOnly(True)
            # Insert it at the bottom of the RL tab
            self.rl_sub_tabs.addTab(self.rl_results_text, "RL Results")
        self.rl_results_text.append("Running RL...")
        self.rl_worker.start()


    def handle_rl_finished(self, results, best_solution, parameter_names, best_reward):
        self.run_rl_button.setEnabled(True)
        self.rl_results_text.append("\nRL Optimization Completed.\n")
        self.rl_results_text.append("Best parameter configuration:")
        for name, val in zip(parameter_names, best_solution):
            self.rl_results_text.append(f"{name}: {val}")
        self.rl_results_text.append(f"\nBest reward: {best_reward:.6f}")
        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.rl_results_text.append(f"\nSingular response of best configuration: {singular_response}")
        self.rl_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.rl_results_text.append(f"{section}: {data}")


    def handle_rl_error(self, err):
        self.run_rl_button.setEnabled(True)
        QMessageBox.warning(self, "RL Error", f"Error during RL: {err}")
        self.rl_results_text.append(f"\nError running RL: {err}")


    def handle_rl_update(self, msg):
        self.rl_results_text.append(msg)