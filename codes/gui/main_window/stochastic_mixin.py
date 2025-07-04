from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame, QPushButton,
    QTextEdit, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import ModernQTabWidget


class StochasticDesignMixin:
    def create_stochastic_design_page(self):
        """Create the stochastic design page with all existing tabs"""
        stochastic_page = QWidget()
        page_layout = QVBoxLayout(stochastic_page)
        page_layout.setContentsMargins(20, 20, 20, 20)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        title = QLabel("Stochastic Design")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header_layout.addWidget(title)

        description = QLabel("Design and optimize stochastic vibration systems with advanced algorithms")
        description.setFont(QFont("Segoe UI", 11))
        header_layout.addWidget(description)

        page_layout.addWidget(header)

        content_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.create_main_system_tab()
        self.create_dva_parameters_tab()
        self.create_target_weights_tab()
        self.create_frequency_tab()
        self.create_omega_sensitivity_tab()
        self.create_sobol_analysis_tab()
        self.create_ga_tab()
        self.create_pso_tab()
        self.create_de_tab()
        self.create_sa_tab()
        self.create_cmaes_tab()

        self.design_tabs = ModernQTabWidget()

        self.input_tabs = ModernQTabWidget()
        self.input_tabs.addTab(self.main_system_tab, "Main System")
        self.input_tabs.addTab(self.dva_tab, "DVA Parameters")
        self.input_tabs.addTab(self.tw_tab, "Targets & Weights")
        self.input_tabs.addTab(self.freq_tab, "Frequency & Plot")
        self.input_tabs.addTab(self.omega_sensitivity_tab, "Ω Sensitivity")

        self.sensitivity_tabs = ModernQTabWidget()
        self.sensitivity_tabs.addTab(self.sobol_tab, "Sobol Analysis")

        self.optimization_tabs = ModernQTabWidget()
        self.optimization_tabs.addTab(self.ga_tab, "GA Optimization")
        self.optimization_tabs.addTab(self.pso_tab, "PSO Optimization")
        self.optimization_tabs.addTab(self.de_tab, "DE Optimization")
        self.optimization_tabs.addTab(self.sa_tab, "SA Optimization")
        self.optimization_tabs.addTab(self.cmaes_tab, "CMA-ES Optimization")

        self.design_tabs.addTab(self.input_tabs, "Input")
        self.design_tabs.addTab(self.sensitivity_tabs, "Sensitivity Analysis")
        self.design_tabs.addTab(self.optimization_tabs, "Optimization")

        left_layout.addWidget(self.design_tabs)

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
        self.run_frf_button.setVisible(False)

        self.run_sobol_button = QPushButton("Run Sobol")
        self.run_sobol_button.setObjectName("primary-button")
        self.run_sobol_button.setMinimumHeight(40)
        self.run_sobol_button.clicked.connect(self.run_sobol)
        self.run_sobol_button.setVisible(False)

        self.run_ga_button = QPushButton("Run GA")
        self.run_ga_button.setObjectName("primary-button")
        self.run_ga_button.setMinimumHeight(40)
        self.run_ga_button.clicked.connect(self.run_ga)
        self.run_ga_button.setVisible(False)

        self.run_pso_button = QPushButton("Run PSO")
        self.run_pso_button.setObjectName("primary-button")
        self.run_pso_button.setMinimumHeight(40)
        self.run_pso_button.clicked.connect(self.run_pso)
        self.run_pso_button.setVisible(False)

        self.run_de_button = QPushButton("Run DE")
        self.run_de_button.setObjectName("primary-button")
        self.run_de_button.setMinimumHeight(40)
        self.run_de_button.clicked.connect(self.run_de)
        self.run_de_button.setVisible(False)

        self.run_sa_button = QPushButton("Run SA")
        self.run_sa_button.setObjectName("primary-button")
        self.run_sa_button.setMinimumHeight(40)
        self.run_sa_button.clicked.connect(self.run_sa)
        self.run_sa_button.setVisible(False)

        self.run_cmaes_button = QPushButton("Run CMA-ES")
        self.run_cmaes_button.setObjectName("primary-button")
        self.run_cmaes_button.setMinimumHeight(40)
        self.run_cmaes_button.clicked.connect(self.run_cmaes)
        self.run_cmaes_button.setVisible(False)

        run_buttons_layout.addWidget(self.run_frf_button)
        run_buttons_layout.addWidget(self.run_sobol_button)
        run_buttons_layout.addWidget(self.run_ga_button)
        run_buttons_layout.addWidget(self.run_pso_button)
        run_buttons_layout.addWidget(self.run_de_button)
        run_buttons_layout.addWidget(self.run_sa_button)
        run_buttons_layout.addWidget(self.run_cmaes_button)

        run_card_layout.addLayout(run_buttons_layout)
        run_card.setVisible(False)
        left_layout.addWidget(run_card)

        content_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        results_tabs = ModernQTabWidget()

        results_panel = QWidget()
        results_panel_layout = QVBoxLayout(results_panel)

        results_title = QLabel("Results")
        results_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        results_panel_layout.addWidget(results_title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        results_panel_layout.addWidget(self.results_text)

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

        results_tabs.addTab(results_panel, "Text Results")
        results_tabs.addTab(frf_panel, "FRF Visualization")
        results_tabs.addTab(comp_panel, "Comparative FRF")
        results_tabs.addTab(sobol_panel, "Sobol Visualization")

        right_layout.addWidget(results_tabs)

        content_splitter.addWidget(right_panel)

        content_splitter.setSizes([800, 800])

        page_layout.addWidget(content_splitter)

        self.content_stack.addWidget(stochastic_page)
