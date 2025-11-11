from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame, QPushButton,
    QTextEdit, QComboBox, QSizePolicy, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import ModernQTabWidget


class StochasticDesignMixin:
    def create_stochastic_design_page(self):
        """Create the stochastic design page with all existing tabs"""
        stochastic_page = QWidget()
        page_layout = QVBoxLayout(stochastic_page)
        page_layout.setContentsMargins(20, 20, 20, 20) # Add top margin for the page, and general padding
        page_layout.setSpacing(15) # Add space between banner and content

        # Create a short banner for the title
        self.stochastic_design_banner = QWidget() # Make banner an instance variable
        self.stochastic_design_banner.setFixedHeight(70)
        banner_layout = QHBoxLayout(self.stochastic_design_banner)
        banner_layout.setContentsMargins(20, 5, 20, 5) # Add vertical padding within the banner
        
        # Set banner background color
        banner_palette = self.stochastic_design_banner.palette()
        banner_palette.setColor(QPalette.Background, QColor("#3A004C")) # A new color, e.g., deep purple
        self.stochastic_design_banner.setAutoFillBackground(True)
        self.stochastic_design_banner.setPalette(banner_palette)

        self.stochastic_design_title_label = QLabel("Stochastic Design") # Make title an instance variable
        self.stochastic_design_title_label.setFont(QFont("Segoe UI", 28, QFont.Bold)) # Increase font size for better appearance
        self.stochastic_design_title_label.setStyleSheet("color: white;") # Ensure title text is visible on dark background
        banner_layout.addWidget(self.stochastic_design_title_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        
        page_layout.addWidget(self.stochastic_design_banner)
        
        content_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Initialize all tab widgets first
        self.design_tabs = ModernQTabWidget()
        self.input_tabs = ModernQTabWidget()
        self.sensitivity_tabs = ModernQTabWidget()
        self.optimization_tabs = ModernQTabWidget()
        self.moo_optimization_tabs = ModernQTabWidget() # New: Multi-Objective Optimizations tab widget

        # Create all tabs
        self.create_main_system_tab()
        self.create_dva_parameters_tab()
        self.create_target_weights_tab()
        self.create_frequency_tab()
        self.create_sobol_analysis_tab()
        self.create_omega_sensitivity_tab()
        self.create_ga_tab()
        self.create_pso_tab()
        self.create_sa_tab()
        self.create_cmaes_tab()
        if hasattr(self, 'create_rl_tab'):
            self.create_rl_tab()
        
        # New: Create NSGA-II and AdaVEA tabs using their mixin methods
        self.nsga2_tab = self.create_nsga2_tab()
        self.adavea_tab = self.create_adavea_tab()

        # Don't create DE tab here - it will be created by integrate_de_functionality method
        # if DEOptimizationMixin is available
        
        # Add tabs to input tabs
        self.input_tabs.addTab(self.main_system_tab, "Main System")
        self.input_tabs.addTab(self.dva_tab, "DVA Parameters")
        self.input_tabs.addTab(self.tw_tab, "Targets & Weights")
        self.input_tabs.addTab(self.freq_tab, "Frequency & Plot")
        self.input_tabs.addTab(self.omega_sensitivity_tab, "Ω Sensitivity")
        
        # Set the default input tab to Main System (index 0)
        self.input_tabs.setCurrentIndex(0)

        # Add tabs to sensitivity tabs
        self.sensitivity_tabs.addTab(self.sobol_tab, "Sobol Analysis")

        # Add tabs to optimization tabs
        self.optimization_tabs.addTab(self.ga_tab, "GA Optimization")
        self.optimization_tabs.addTab(self.pso_tab, "PSO Optimization")
        self.optimization_tabs.addTab(self.sa_tab, "SA Optimization")
        self.optimization_tabs.addTab(self.cmaes_tab, "CMA-ES Optimization")
        if hasattr(self, 'rl_tab'):
            self.optimization_tabs.addTab(self.rl_tab, "RL Optimization")
        
        # Don't add DE tab here - it will be handled by integrate_de_functionality method
        # Create a placeholder for now to avoid errors
        self.de_tab = QWidget()
        placeholder_layout = QVBoxLayout(self.de_tab)
        placeholder_label = QLabel("DE Optimization will be added after initialization")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        # Use a different tab text to identify this as a placeholder
        self.optimization_tabs.addTab(self.de_tab, "DE Optimization (Placeholder)")

        # New: Create Multi-Objective Optimizations tab and its sub-tabs
        self.moo_ga_tabs = QTabWidget() # Renamed from moo_ga_placeholder_tab
        self.moo_ga_tabs.addTab(self.nsga2_tab, "NSGA-II")
        self.moo_ga_tabs.addTab(self.adavea_tab, "AdaVEA")
        
        self.moo_optimization_tabs.addTab(self.moo_ga_tabs, "MOO-GA")

        # Add main tab groups to design tabs
        self.design_tabs.addTab(self.input_tabs, "Input")
        self.design_tabs.addTab(self.sensitivity_tabs, "Sensitivity Analysis")
        self.design_tabs.addTab(self.optimization_tabs, "Optimization")
        self.design_tabs.addTab(self.moo_optimization_tabs, "Multi-Objective Optimizations") # New tab
        
        # Set the default tab to Input (index 0)
        self.design_tabs.setCurrentIndex(0)

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

        # Restore DE button
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

        # New: Run MOO-GA button
        self.run_moo_ga_button = QPushButton("Run MOO-GA")
        self.run_moo_ga_button.setObjectName("primary-button")
        self.run_moo_ga_button.setMinimumHeight(40)
        self.run_moo_ga_button.clicked.connect(self.run_moo_ga)
        self.run_moo_ga_button.setVisible(False)

        run_buttons_layout.addWidget(self.run_frf_button)
        run_buttons_layout.addWidget(self.run_sobol_button)
        run_buttons_layout.addWidget(self.run_ga_button)
        run_buttons_layout.addWidget(self.run_pso_button)
        run_buttons_layout.addWidget(self.run_de_button)
        run_buttons_layout.addWidget(self.run_sa_button)
        run_buttons_layout.addWidget(self.run_cmaes_button)
        run_buttons_layout.addWidget(self.run_moo_ga_button) # Add new button

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

    def apply_optimized_dva_parameters(self):
        """Apply the best parameters from the last optimization run"""
        selected_optimizer = self.dva_optimizer_combo.currentText()
        
        # Get the best parameters based on selected optimizer
        best_params = None
        if "Genetic Algorithm" in selected_optimizer:
            if hasattr(self, 'current_ga_best_params'):
                best_params = self.current_ga_best_params
        elif "Particle Swarm" in selected_optimizer:
            if hasattr(self, 'current_pso_best_params'):
                best_params = self.current_pso_best_params
        elif "Differential Evolution" in selected_optimizer:
            if hasattr(self, 'current_de_best_params'):
                best_params = self.current_de_best_params
        elif "Simulated Annealing" in selected_optimizer:
            if hasattr(self, 'current_sa_best_params'):
                best_params = self.current_sa_best_params
        elif "CMA-ES" in selected_optimizer:
            if hasattr(self, 'current_cmaes_best_params'):
                best_params = self.current_cmaes_best_params
        elif "Reinforcement Learning" in selected_optimizer:
            if hasattr(self, 'current_rl_best_params'):
                best_params = self.current_rl_best_params
        
        if best_params is None:
            QMessageBox.warning(self, "No Parameters Available", 
                              f"No optimized parameters available from {selected_optimizer}.\n"
                              "Please run the optimization first.")
            return
        
        try:
            # Apply parameters to the spinboxes
            param_idx = 0
            
            # Apply beta parameters
            for i in range(15):
                if param_idx < len(best_params):
                    self.beta_boxes[i].setValue(best_params[param_idx])
                param_idx += 1
            
            # Apply lambda parameters
            for i in range(15):
                if param_idx < len(best_params):
                    self.lambda_boxes[i].setValue(best_params[param_idx])
                param_idx += 1
            
            # Apply mu parameters
            for i in range(3):
                if param_idx < len(best_params):
                    self.mu_dva_boxes[i].setValue(best_params[param_idx])
                param_idx += 1
            
            # Apply nu parameters
            for i in range(15):
                if param_idx < len(best_params):
                    self.nu_dva_boxes[i].setValue(best_params[param_idx])
                param_idx += 1
            
            QMessageBox.information(self, "Success", 
                                  f"Successfully applied optimized parameters from {selected_optimizer}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to apply parameters: {str(e)}\n"
                               "Please ensure the optimization was run successfully.")

    def create_de_tab(self):
        """Create a basic DE optimization tab."""
        # Check if de_tab is already created by DEOptimizationMixin
        if hasattr(self, 'de_tab') and self.de_tab is not None:
            return self.de_tab
            
        # Create a simple version of the DE tab if not already created
        self.de_tab = QWidget()
        layout = QVBoxLayout(self.de_tab)
        
        # Create a simple information label
        info_label = QLabel("DE (Differential Evolution) Optimization")
        info_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(info_label)
        
        description = QLabel("This tab provides options for Differential Evolution optimization.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create settings group
        settings_group = QGroupBox("DE Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Population size
        self.de_pop_size_box = QSpinBox()
        self.de_pop_size_box.setRange(10, 1000)
        self.de_pop_size_box.setValue(50)
        settings_layout.addRow("Population Size:", self.de_pop_size_box)
        
        # Number of generations
        self.de_generations_box = QSpinBox()
        self.de_generations_box.setRange(10, 10000)
        self.de_generations_box.setValue(100)
        settings_layout.addRow("Max Generations:", self.de_generations_box)
        
        # Mutation constant
        self.de_mutation_box = QDoubleSpinBox()
        self.de_mutation_box.setRange(0.1, 2.0)
        self.de_mutation_box.setValue(0.8)
        self.de_mutation_box.setSingleStep(0.1)
        settings_layout.addRow("Mutation Constant:", self.de_mutation_box)
        
        # Crossover probability
        self.de_crossover_box = QDoubleSpinBox()
        self.de_crossover_box.setRange(0.1, 1.0)
        self.de_crossover_box.setValue(0.7)
        self.de_crossover_box.setSingleStep(0.1)
        settings_layout.addRow("Crossover Probability:", self.de_crossover_box)
        
        layout.addWidget(settings_group)
        
        # Add placeholder for visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        placeholder = QLabel("DE optimization progress will be shown here")
        placeholder.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(placeholder)
        
        layout.addWidget(viz_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        return self.de_tab

    def run_de(self):
        """Run the differential evolution optimization"""
        # First check if we've been overridden by a real implementation
        if hasattr(self, '__class__') and hasattr(self.__class__, '__mro__'):
            # Check if there's a run_de method in any parent class other than StochasticDesignMixin
            for cls in self.__class__.__mro__:
                if cls != StochasticDesignMixin and hasattr(cls, 'run_de'):
                    # Found another implementation - delegate to it using super()
                    method_to_call = getattr(super(StochasticDesignMixin, self), 'run_de', None)
                    if method_to_call and callable(method_to_call):
                        return method_to_call()
        
        # If no other implementation found, show basic message
        # Try to show the DE tab
        try:
            # Show the DE tab when running DE optimization
            opt_tab_idx = self.design_tabs.indexOf(self.optimization_tabs)
            self.design_tabs.setCurrentIndex(opt_tab_idx)
            
            de_tab_idx = self.optimization_tabs.indexOf(self.de_tab)
            self.optimization_tabs.setCurrentIndex(de_tab_idx)
        except Exception:
            pass
            
        # Show the not implemented message
        QMessageBox.information(
            self, "Not Implemented", 
            "The DE optimization functionality is not fully implemented yet."
        )

    def run_moo_ga(self):
        """Run the Multi-Objective Optimization Genetic Algorithm (MOO-GA)"""
        try:
            # Switch to the Multi-Objective Optimizations tab
            moo_opt_tab_idx = self.design_tabs.indexOf(self.moo_optimization_tabs)
            self.design_tabs.setCurrentIndex(moo_opt_tab_idx)
            
            # Switch to the MOO-GA sub-tab (which is the first tab in moo_optimization_tabs)
            self.moo_optimization_tabs.setCurrentIndex(0)
        except Exception:
            pass
            
        QMessageBox.information(
            self, "Not Implemented", 
            "The Multi-Objective Optimization Genetic Algorithm (MOO-GA) functionality is not yet implemented."
        )
