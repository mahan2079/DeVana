from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from .widgets import ModernQTabWidget

class StochasticDesignMixin:
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
    
        def create_main_system_tab(self):
            """Create the main system parameters tab"""
            self.main_system_tab = QWidget()
            layout = QVBoxLayout(self.main_system_tab)
            
            # Create a scroll area for potentially large content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            
            # Create main container widget
            main_container = QWidget()
            main_layout = QVBoxLayout(main_container)
            
            # Main system parameters group
            params_group = QGroupBox("Main System Parameters")
            params_layout = QFormLayout(params_group)
            
            # MU parameter
            self.mu_box = QDoubleSpinBox()
            self.mu_box.setRange(-1e6, 1e6)
            self.mu_box.setDecimals(6)
            self.mu_box.setValue(1.0)
            params_layout.addRow("μ (MU):", self.mu_box)
    
            # LANDA parameters (Lambda)
            self.landa_boxes = []
            for i in range(5):
                box = QDoubleSpinBox()
                box.setRange(-1e6, 1e6)
                box.setDecimals(6)
                if i < 2:
                    box.setValue(1.0)
                else:
                    box.setValue(0.5)
                self.landa_boxes.append(box)
                params_layout.addRow(f"Λ_{i+1}:", box)
    
            # NU parameters
            self.nu_boxes = []
            for i in range(5):
                box = QDoubleSpinBox()
                box.setRange(-1e6, 1e6)
                box.setDecimals(6)
                box.setValue(0.75)
                self.nu_boxes.append(box)
                params_layout.addRow(f"Ν_{i+1}:", box)
    
            # A_LOW parameter
            self.a_low_box = QDoubleSpinBox()
            self.a_low_box.setRange(0, 1e10)
            self.a_low_box.setDecimals(6)
            self.a_low_box.setValue(0.05)
            params_layout.addRow("A_LOW:", self.a_low_box)
    
            # A_UPP parameter
            self.a_up_box = QDoubleSpinBox()
            self.a_up_box.setRange(0, 1e10)
            self.a_up_box.setDecimals(6)
            self.a_up_box.setValue(0.05)
            params_layout.addRow("A_UPP:", self.a_up_box)
    
            # F_1 parameter
            self.f_1_box = QDoubleSpinBox()
            self.f_1_box.setRange(0, 1e10)
            self.f_1_box.setDecimals(6)
            self.f_1_box.setValue(100.0)
            params_layout.addRow("F_1:", self.f_1_box)
    
            # F_2 parameter
            self.f_2_box = QDoubleSpinBox()
            self.f_2_box.setRange(0, 1e10)
            self.f_2_box.setDecimals(6)
            self.f_2_box.setValue(100.0)
            params_layout.addRow("F_2:", self.f_2_box)
    
            # OMEGA_DC parameter
            self.omega_dc_box = QDoubleSpinBox()
            self.omega_dc_box.setRange(0, 1e10)
            self.omega_dc_box.setDecimals(6)
            self.omega_dc_box.setValue(5000.0)
            params_layout.addRow("Ω_DC:", self.omega_dc_box)
    
            # ZETA_DC parameter
            self.zeta_dc_box = QDoubleSpinBox()
            self.zeta_dc_box.setRange(0, 1e10)
            self.zeta_dc_box.setDecimals(6)
            self.zeta_dc_box.setValue(0.01)
            params_layout.addRow("ζ_DC:", self.zeta_dc_box)
            
            main_layout.addWidget(params_group)
            main_layout.addStretch()
            
            # Set the container as the scroll area's widget
            scroll_area.setWidget(main_container)
            
            # Add scroll area to the tab's layout
            layout.addWidget(scroll_area)
    
        def create_dva_parameters_tab(self):
            """Create the DVA parameters tab"""
            self.dva_tab = QWidget()
            layout = QVBoxLayout(self.dva_tab)
            
            # Create a scroll area for potentially large content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            
            # Create main container widget
            main_container = QWidget()
            main_layout = QVBoxLayout(main_container)
            
            # Add button to apply optimized DVA parameters from GA
            apply_optimized_container = QWidget()
            apply_optimized_layout = QHBoxLayout(apply_optimized_container)
            apply_optimized_layout.setContentsMargins(0, 0, 0, 10)
            
            apply_optimized_button = QPushButton("Apply Optimized DVA Parameters")
            apply_optimized_button.setToolTip("Apply the best parameters from the last optimization run")
            apply_optimized_button.clicked.connect(self.apply_optimized_dva_parameters)
            apply_optimized_layout.addWidget(apply_optimized_button)
            
            apply_optimizer_combo = QComboBox()
            apply_optimizer_combo.addItems(["Genetic Algorithm (GA)", "Particle Swarm (PSO)", 
                                           "Differential Evolution (DE)", "Simulated Annealing (SA)", 
                                           "CMA-ES", "Reinforcement Learning (RL)"])
            apply_optimized_layout.addWidget(apply_optimizer_combo)
            self.dva_optimizer_combo = apply_optimizer_combo
            
            main_layout.addWidget(apply_optimized_container)
            
            # BETA parameters group
            beta_group = QGroupBox("β (beta) Parameters")
            beta_form = QFormLayout(beta_group)
            self.beta_boxes = []
            for i in range(15):
                b = QDoubleSpinBox()
                b.setRange(-1e6, 1e6)
                b.setDecimals(6)
                self.beta_boxes.append(b)
                beta_form.addRow(f"β_{i+1}:", b)
            main_layout.addWidget(beta_group)
    
            # LAMBDA parameters group
            lambda_group = QGroupBox("λ (lambda) Parameters")
            lambda_form = QFormLayout(lambda_group)
            self.lambda_boxes = []
            for i in range(15):
                l = QDoubleSpinBox()
                l.setRange(-1e6, 1e6)
                l.setDecimals(6)
                self.lambda_boxes.append(l)
                lambda_form.addRow(f"λ_{i+1}:", l)
            main_layout.addWidget(lambda_group)
    
            # MU parameters group
            mu_group = QGroupBox("μ (mu) Parameters")
            mu_form = QFormLayout(mu_group)
            self.mu_dva_boxes = []
            for i in range(3):
                m = QDoubleSpinBox()
                m.setRange(-1e6, 1e6)
                m.setDecimals(6)
                self.mu_dva_boxes.append(m)
                mu_form.addRow(f"μ_{i+1}:", m)
            main_layout.addWidget(mu_group)
    
            # NU parameters group
            nu_group = QGroupBox("ν (nu) Parameters")
            nu_form = QFormLayout(nu_group)
            self.nu_dva_boxes = []
            for i in range(15):
                n = QDoubleSpinBox()
                n.setRange(-1e6, 1e6)
                n.setDecimals(6)
                self.nu_dva_boxes.append(n)
                nu_form.addRow(f"ν_{i+1}:", n)
            main_layout.addWidget(nu_group)
            
            main_layout.addStretch()
            
            # Set the container as the scroll area's widget
            scroll_area.setWidget(main_container)
            
            # Add scroll area to the tab's layout
            layout.addWidget(scroll_area)
        
        def create_target_weights_tab(self):
            """Create the targets and weights tab"""
            self.tw_tab = QWidget()
            layout = QVBoxLayout(self.tw_tab)
            
            # Create a scroll area for potentially large content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            
            # Create main container widget
            main_container = QWidget()
            main_layout = QVBoxLayout(main_container)
            
            # Initialize dictionaries to store all target and weight spinboxes
            self.mass_target_spins = {}
            self.mass_weight_spins = {}
    
            # Create groups for each mass
            for mass_num in range(1, 6):
                mass_group = QGroupBox(f"Mass {mass_num} Targets & Weights")
                mg_layout = QVBoxLayout(mass_group)
    
                # Peak values group
                peak_group = QGroupBox("Peak Values & Weights")
                peak_form = QFormLayout(peak_group)
                for peak_num in range(1, 5):
                    pv = QDoubleSpinBox()
                    pv.setRange(0, 1e6)
                    pv.setDecimals(6)
                    wv = QDoubleSpinBox()
                    wv.setRange(0, 1e3)
                    wv.setDecimals(6)
                    
                    peak_form.addRow(f"Peak Value {peak_num}:", pv)
                    peak_form.addRow(f"Weight Peak Value {peak_num}:", wv)
                    
                    self.mass_target_spins[f"peak_value_{peak_num}_m{mass_num}"] = pv
                    self.mass_weight_spins[f"peak_value_{peak_num}_m{mass_num}"] = wv
                mg_layout.addWidget(peak_group)
                
                # Peak positions group (in a separate section)
                peak_pos_group = QGroupBox("Peak Positions & Weights")
                peak_pos_form = QFormLayout(peak_pos_group)
                for peak_num in range(1, 6):  # Note: 1-5 (not 1-4)
                    pp = QDoubleSpinBox()
                    pp.setRange(0, 1e6)
                    pp.setDecimals(6)
                    wpp = QDoubleSpinBox()
                    wpp.setRange(0, 1e3)
                    wpp.setDecimals(6)
                    
                    peak_pos_form.addRow(f"Peak Position {peak_num}:", pp)
                    peak_pos_form.addRow(f"Weight Peak Position {peak_num}:", wpp)
                    
                    self.mass_target_spins[f"peak_position_{peak_num}_m{mass_num}"] = pp
                    self.mass_weight_spins[f"peak_position_{peak_num}_m{mass_num}"] = wpp
                mg_layout.addWidget(peak_pos_group)
    
                # Bandwidth group
                bw_group = QGroupBox("Bandwidth Targets & Weights")
                bw_form = QFormLayout(bw_group)
                for i in range(1, 5):
                    for j in range(i+1, 5):
                        bw = QDoubleSpinBox()
                        bw.setRange(0, 1e6)
                        bw.setDecimals(6)
                        wbw = QDoubleSpinBox()
                        wbw.setRange(0, 1e3)
                        wbw.setDecimals(6)
                        bw_form.addRow(f"Bandwidth {i}-{j}:", bw)
                        bw_form.addRow(f"Weight Bandwidth {i}-{j}:", wbw)
                        self.mass_target_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = bw
                        self.mass_weight_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = wbw
                mg_layout.addWidget(bw_group)
    
                # Slope group
                slope_group = QGroupBox("Slope Targets & Weights")
                slope_form = QFormLayout(slope_group)
                for i in range(1, 5):
                    for j in range(i+1, 5):
                        s = QDoubleSpinBox()
                        s.setRange(-1e6, 1e6)
                        s.setDecimals(6)
                        ws = QDoubleSpinBox()
                        ws.setRange(0, 1e3)
                        ws.setDecimals(6)
                        slope_form.addRow(f"Slope {i}-{j}:", s)
                        slope_form.addRow(f"Weight Slope {i}-{j}:", ws)
                        self.mass_target_spins[f"slope_{i}_{j}_m{mass_num}"] = s
                        self.mass_weight_spins[f"slope_{i}_{j}_m{mass_num}"] = ws
                mg_layout.addWidget(slope_group)
    
                # Area under curve group
                auc_group = QGroupBox("Area Under Curve & Weight")
                auc_form = QFormLayout(auc_group)
                auc = QDoubleSpinBox()
                auc.setRange(0, 1e6)
                auc.setDecimals(6)
                wauc = QDoubleSpinBox()
                wauc.setRange(0, 1e3)
                wauc.setDecimals(6)
                auc_form.addRow("Area Under Curve:", auc)
                auc_form.addRow("Weight Area Under Curve:", wauc)
                self.mass_target_spins[f"area_under_curve_m{mass_num}"] = auc
                self.mass_weight_spins[f"area_under_curve_m{mass_num}"] = wauc
                mg_layout.addWidget(auc_group)
    
                mg_layout.addStretch()
                main_layout.addWidget(mass_group)
            
            main_layout.addStretch()
            
            # Set the container as the scroll area's widget
            scroll_area.setWidget(main_container)
            
            # Add scroll area to the tab's layout
            layout.addWidget(scroll_area)
        
        def create_omega_sensitivity_tab(self):
            """Create the Omega points sensitivity analysis tab"""
            self.omega_sensitivity_tab = QWidget()
            layout = QVBoxLayout(self.omega_sensitivity_tab)
    
            # Create tabs for parameters and visualization
            self.sensitivity_tabs = ModernQTabWidget()
            layout.addWidget(self.sensitivity_tabs)
            
            # --------- PARAMETERS TAB ---------
            params_tab = QWidget()
            params_layout = QVBoxLayout(params_tab)
            
            # Create a scroll area for parameters
            params_scroll_area = QScrollArea()
            params_scroll_area.setWidgetResizable(True)
            
            # Create main container widget for parameters
            params_container = QWidget()
            params_main_layout = QVBoxLayout(params_container)
            
            # Introduction group
            intro_group = QGroupBox("Omega Points Sensitivity Analysis")
            intro_layout = QVBoxLayout(intro_group)
            
            info_text = QLabel(
                "This tool analyzes how the number of frequency points affects slope calculations in "
                "the Frequency Response Function (FRF). It helps identify the minimum number of points "
                "needed for stable results by incrementally increasing the frequency resolution and "
                "observing the convergence of slope values."
            )
            info_text.setWordWrap(True)
            intro_layout.addWidget(info_text)
            
            # Parameters group
            params_group = QGroupBox("Analysis Parameters")
            params_group_layout = QFormLayout(params_group)
            
            # Initial points
            self.sensitivity_initial_points = QSpinBox()
            self.sensitivity_initial_points.setRange(50, 5000)
            self.sensitivity_initial_points.setValue(100)
            params_group_layout.addRow("Initial Ω Points:", self.sensitivity_initial_points)
            
            # Maximum points
            self.sensitivity_max_points = QSpinBox()
            self.sensitivity_max_points.setRange(500, 1000000000)  # Allow very large values (10^9)
            self.sensitivity_max_points.setValue(2000)
            params_group_layout.addRow("Maximum Ω Points:", self.sensitivity_max_points)
            
            # Step size
            self.sensitivity_step_size = QSpinBox()
            self.sensitivity_step_size.setRange(10, 100000)  # Allow larger step sizes
            self.sensitivity_step_size.setValue(1000)  # Increased default for large ranges
            params_group_layout.addRow("Step Size:", self.sensitivity_step_size)
            
            # Convergence threshold
            self.sensitivity_threshold = QDoubleSpinBox()
            self.sensitivity_threshold.setRange(1e-10, 0.1)
            self.sensitivity_threshold.setDecimals(10)
            self.sensitivity_threshold.setSingleStep(1e-10)
            self.sensitivity_threshold.setValue(0.01)
            params_group_layout.addRow("Convergence Threshold:", self.sensitivity_threshold)
            
            # Max iterations
            self.sensitivity_max_iterations = QSpinBox()
            self.sensitivity_max_iterations.setRange(5, 1000000)  # Allow extremely high iteration counts
            self.sensitivity_max_iterations.setValue(200)  # Set default to 200 to support larger ranges
            params_group_layout.addRow("Maximum Iterations:", self.sensitivity_max_iterations)
            
            # Mass of interest
            self.sensitivity_mass = QComboBox()
            for i in range(1, 6):
                self.sensitivity_mass.addItem(f"mass_{i}")
            params_group_layout.addRow("Mass of Interest:", self.sensitivity_mass)
            
            # Plot results checkbox
            self.sensitivity_plot_results = QCheckBox("Generate Convergence Plots")
            self.sensitivity_plot_results.setChecked(True)
            params_group_layout.addRow(self.sensitivity_plot_results)
            
            # Use optimal points checkbox
            self.sensitivity_use_optimal = QCheckBox("Use Optimal Points in FRF Analysis")
            self.sensitivity_use_optimal.setChecked(True)
            params_group_layout.addRow(self.sensitivity_use_optimal)
            
            # Results group
            self.sensitivity_results_group = QGroupBox("Analysis Results")
            self.sensitivity_results_layout = QVBoxLayout(self.sensitivity_results_group)
            
            self.sensitivity_results_text = QTextEdit()
            self.sensitivity_results_text.setReadOnly(True)
            self.sensitivity_results_layout.addWidget(self.sensitivity_results_text)
            
            # Run button container
            run_container = QWidget()
            run_layout = QHBoxLayout(run_container)
            run_layout.setContentsMargins(0, 20, 0, 0)  # Add some top margin
            
            # Add stretch to push button to center
            run_layout.addStretch()
            
            # Create and style the Run Analysis button
            self.run_sensitivity_btn = QPushButton("Run Sensitivity Analysis")
            self.run_sensitivity_btn.setObjectName("primary-button")
            self.run_sensitivity_btn.setMinimumWidth(200)
            self.run_sensitivity_btn.setMinimumHeight(40)
            self.run_sensitivity_btn.clicked.connect(self.run_omega_sensitivity)
            self.run_sensitivity_btn.setStyleSheet("""
                QPushButton#primary-button {
                    background-color: #4B67F0;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 20px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton#primary-button:hover {
                    background-color: #3B57E0;
                }
                QPushButton#primary-button:pressed {
                    background-color: #2B47D0;
                }
            """)
            run_layout.addWidget(self.run_sensitivity_btn)
            
            # Add stretch to push button to center
            run_layout.addStretch()
            
            # Add all groups to main layout
            params_main_layout.addWidget(intro_group)
            params_main_layout.addWidget(params_group)
            params_main_layout.addWidget(self.sensitivity_results_group)
            params_main_layout.addWidget(run_container)
            params_main_layout.addStretch()
            
            # Set the container as the scroll area's widget
            params_scroll_area.setWidget(params_container)
            params_layout.addWidget(params_scroll_area)
            
            # --------- VISUALIZATION TABS ---------
            # Create visualization tabs widget
            self.vis_tabs = ModernQTabWidget()
            
            # Common control panel for both visualization tabs
            vis_control_panel = QWidget()
            vis_control_layout = QHBoxLayout(vis_control_panel)
            
            # Save plot button
            self.sensitivity_save_plot_btn = QPushButton("Save Current Plot")
            self.sensitivity_save_plot_btn.setEnabled(False)
            self.sensitivity_save_plot_btn.clicked.connect(self.save_sensitivity_plot)
            vis_control_layout.addWidget(self.sensitivity_save_plot_btn)
            
            # Refresh plot button
            self.sensitivity_refresh_plot_btn = QPushButton("Refresh Plots")
            self.sensitivity_refresh_plot_btn.setEnabled(False)
            self.sensitivity_refresh_plot_btn.clicked.connect(self.refresh_sensitivity_plot)
            vis_control_layout.addWidget(self.sensitivity_refresh_plot_btn)
            
            # --------- CONVERGENCE PLOT TAB ---------
            convergence_tab = QWidget()
            convergence_layout = QVBoxLayout(convergence_tab)
            
            # Add control panel to layout
            convergence_layout.addWidget(vis_control_panel)
            
            # Create figure canvas for convergence plot
            self.convergence_fig = Figure(figsize=(10, 6))
            self.convergence_canvas = FigureCanvas(self.convergence_fig)
            self.convergence_canvas.setMinimumHeight(450)
            self.convergence_toolbar = NavigationToolbar(self.convergence_canvas, convergence_tab)
            
            # Add canvas and toolbar to layout
            convergence_layout.addWidget(self.convergence_canvas)
            convergence_layout.addWidget(self.convergence_toolbar)
            
            # --------- RELATIVE CHANGE PLOT TAB ---------
            rel_change_tab = QWidget()
            rel_change_layout = QVBoxLayout(rel_change_tab)
            
            # Create figure canvas for relative change plot
            self.rel_change_fig = Figure(figsize=(10, 6))
            self.rel_change_canvas = FigureCanvas(self.rel_change_fig)
            self.rel_change_canvas.setMinimumHeight(450)
            self.rel_change_toolbar = NavigationToolbar(self.rel_change_canvas, rel_change_tab)
            
            # Add canvas and toolbar to layout
            rel_change_layout.addWidget(self.rel_change_canvas)
            rel_change_layout.addWidget(self.rel_change_toolbar)
            
            # No data message (added to both tabs)
            self.convergence_no_data_label = QLabel("Run the sensitivity analysis to generate visualization")
            self.convergence_no_data_label.setAlignment(Qt.AlignCenter)
            self.convergence_no_data_label.setStyleSheet("color: #888; font-style: italic; font-size: 14px;")
            convergence_layout.addWidget(self.convergence_no_data_label)
            
            self.rel_change_no_data_label = QLabel("Run the sensitivity analysis to generate visualization")
            self.rel_change_no_data_label.setAlignment(Qt.AlignCenter)
            self.rel_change_no_data_label.setStyleSheet("color: #888; font-style: italic; font-size: 14px;")
            rel_change_layout.addWidget(self.rel_change_no_data_label)
            
            # Add tabs to the visualization tabs widget
            self.vis_tabs.addTab(convergence_tab, "Slope Convergence")
            self.vis_tabs.addTab(rel_change_tab, "Relative Change")
            
            # Create main visualization container tab
            vis_tab = QWidget()
            vis_layout = QVBoxLayout(vis_tab)
            vis_layout.addWidget(self.vis_tabs)
            
            # Add tabs to the main tab widget
            self.sensitivity_tabs.addTab(params_tab, "Parameters & Results")
            self.sensitivity_tabs.addTab(vis_tab, "Visualization")
            
        def create_comparative_visualization_options(self, parent_layout):
            """Create options for comparative visualization of multiple FRF inputs"""
            comp_group = QGroupBox("Comparative Visualization")
            comp_layout = QVBoxLayout(comp_group)
            
            # Introduction text
            intro_label = QLabel("This section allows you to create custom comparative plots by selecting multiple FRF results and customizing legends and title.")
            intro_label.setWordWrap(True)
            comp_layout.addWidget(intro_label)
            
            # Available plots section
            available_plots_group = QGroupBox("Available Plots")
            available_plots_layout = QVBoxLayout(available_plots_group)
            
            self.available_plots_list = QListWidget()
            self.available_plots_list.setSelectionMode(QAbstractItemView.MultiSelection)
            self.available_plots_list.itemSelectionChanged.connect(self._update_legend_table_from_selection)
            available_plots_layout.addWidget(self.available_plots_list)
            
            # Create button layout for management
            plots_btn_layout = QHBoxLayout()
            
            # Add clear button for plot history
            clear_plots_button = QPushButton("Clear All FRF Plots")
            clear_plots_button.clicked.connect(self.clear_all_frf_plots)
            plots_btn_layout.addWidget(clear_plots_button)
            
            # Add export/import buttons
            export_button = QPushButton("Export FRF Data")
            export_button.clicked.connect(self.export_frf_data)
            plots_btn_layout.addWidget(export_button)
            
            import_button = QPushButton("Import FRF Data")
            import_button.clicked.connect(self.import_frf_data)
            plots_btn_layout.addWidget(import_button)
            
            available_plots_layout.addLayout(plots_btn_layout)
            comp_layout.addWidget(available_plots_group)
            
            # Legend customization
            legend_group = QGroupBox("Legend & Style Customization")
            legend_layout = QVBoxLayout(legend_group)
            
            # Map of original plot names to custom properties
            self.legend_map = {}
            self.legend_table = QTableWidget()
            self.legend_table.setColumnCount(5)
            self.legend_table.setHorizontalHeaderLabels([
                "Original Name", 
                "Custom Legend", 
                "Line Style", 
                "Marker", 
                "Color"
            ])
            self.legend_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            legend_layout.addWidget(self.legend_table)
            
            comp_layout.addWidget(legend_group)
            
            # Plot title customization
            title_group = QGroupBox("Plot Title")
            title_layout = QFormLayout(title_group)
            
            self.plot_title_edit = QLineEdit()
            self.plot_title_edit.setPlaceholderText("Enter custom plot title here")
            title_layout.addRow("Custom Title:", self.plot_title_edit)
            
            # Font size for title
            self.title_font_size = QSpinBox()
            self.title_font_size.setRange(8, 24)
            self.title_font_size.setValue(14)
            title_layout.addRow("Title Font Size:", self.title_font_size)
            
            comp_layout.addWidget(title_group)
            
            # Plot customization options
            plot_options_group = QGroupBox("Plot Options")
            plot_options_layout = QFormLayout(plot_options_group)
            
            # Figure size
            fig_size_container = QWidget()
            fig_size_layout = QHBoxLayout(fig_size_container)
            fig_size_layout.setContentsMargins(0, 0, 0, 0)
            
            self.fig_width_spin = QSpinBox()
            self.fig_width_spin.setRange(4, 20)
            self.fig_width_spin.setValue(10)
            fig_size_layout.addWidget(QLabel("Width:"))
            fig_size_layout.addWidget(self.fig_width_spin)
            
            self.fig_height_spin = QSpinBox()
            self.fig_height_spin.setRange(3, 15)
            self.fig_height_spin.setValue(6)
            fig_size_layout.addWidget(QLabel("Height:"))
            fig_size_layout.addWidget(self.fig_height_spin)
            
            plot_options_layout.addRow("Figure Size:", fig_size_container)
            
            # Add normalization options
            norm_container = QWidget()
            norm_layout = QHBoxLayout(norm_container)
            norm_layout.setContentsMargins(0, 0, 0, 0)
            
            # X axis normalization
            self.x_norm_check = QCheckBox("X-Axis")
            norm_layout.addWidget(self.x_norm_check)
            
            self.x_norm_value = QDoubleSpinBox()
            self.x_norm_value.setRange(0.001, 1000000)
            self.x_norm_value.setValue(1.0)
            self.x_norm_value.setDecimals(3)
            self.x_norm_value.setSingleStep(0.1)
            self.x_norm_value.setEnabled(False)
            norm_layout.addWidget(self.x_norm_value)
            
            self.x_norm_check.toggled.connect(self.x_norm_value.setEnabled)
            
            norm_layout.addSpacing(20)
            
            # Y axis normalization
            self.y_norm_check = QCheckBox("Y-Axis")
            norm_layout.addWidget(self.y_norm_check)
            
            self.y_norm_value = QDoubleSpinBox()
            self.y_norm_value.setRange(0.001, 1000000)
            self.y_norm_value.setValue(1.0)
            self.y_norm_value.setDecimals(3)
            self.y_norm_value.setSingleStep(0.1)
            self.y_norm_value.setEnabled(False)
            norm_layout.addWidget(self.y_norm_value)
            
            self.y_norm_check.toggled.connect(self.y_norm_value.setEnabled)
            
            plot_options_layout.addRow("Normalize by:", norm_container)
            
            # Grid options
            self.show_grid_check = QCheckBox()
            self.show_grid_check.setChecked(True)
            plot_options_layout.addRow("Show Grid:", self.show_grid_check)
            
            # Legend position
            self.legend_position_combo = QComboBox()
            for pos in ["best", "upper right", "upper left", "lower left", "lower right", 
                       "right", "center left", "center right", "lower center", "upper center", "center"]:
                self.legend_position_combo.addItem(pos)
            plot_options_layout.addRow("Legend Position:", self.legend_position_combo)
            
            comp_layout.addWidget(plot_options_group)
            
            # Visualization actions
            actions_container = QWidget()
            actions_layout = QHBoxLayout(actions_container)
            
            self.create_comp_plot_btn = QPushButton("Create Comparative Plot")
            self.create_comp_plot_btn.setObjectName("primary-button")
            self.create_comp_plot_btn.clicked.connect(self.create_comparative_plot)
            actions_layout.addWidget(self.create_comp_plot_btn)
            
            self.save_comp_plot_btn = QPushButton("Save Plot")
            self.save_comp_plot_btn.setObjectName("secondary-button")
            self.save_comp_plot_btn.clicked.connect(lambda: self.save_plot(self.comp_fig, "Comparative FRF"))
            actions_layout.addWidget(self.save_comp_plot_btn)
            
            comp_layout.addWidget(actions_container)
            
            parent_layout.addWidget(comp_group)
            
        def _update_legend_table_from_selection(self):
            """Update the legend table based on the selected plots in the list widget"""
            # Clear current table contents
            self.legend_table.setRowCount(0)
            
            # Get selected items
            selected_items = self.available_plots_list.selectedItems()
            
            if not selected_items:
                return
                
            # Create a row for each selected plot
            self.legend_table.setRowCount(len(selected_items))
            
            for row, item in enumerate(selected_items):
                plot_name = item.text()
                
                # Original name column (non-editable)
                name_item = QTableWidgetItem(plot_name)
                name_item.setFlags(Qt.ItemIsEnabled)  # Make it non-editable
                self.legend_table.setItem(row, 0, name_item)
                
                # Custom legend name column
                if plot_name in self.legend_map and 'custom_name' in self.legend_map[plot_name]:
                    legend_name = self.legend_map[plot_name]['custom_name']
                else:
                    legend_name = plot_name
                    
                legend_item = QTableWidgetItem(legend_name)
                self.legend_table.setItem(row, 1, legend_item)
                
                # Line style combo box
                line_style_combo = QComboBox()
                line_styles = ['-', '--', '-.', ':']
                for style in line_styles:
                    line_style_combo.addItem(style)
                    
                # Set previously selected style if available
                if plot_name in self.legend_map and 'line_style' in self.legend_map[plot_name]:
                    line_style_value = self.legend_map[plot_name]['line_style']
                    # Convert empty string to "None" for the combobox
                    if line_style_value == "":
                        line_style_value = "None"
                    try:
                        index = line_styles.index(line_style_value)
                        if index >= 0:
                            line_style_combo.setCurrentIndex(index)
                    except ValueError:
                        # If line style value not in the list, use default
                        line_style_combo.setCurrentIndex(0)  # First style
                        
                self.legend_table.setCellWidget(row, 2, line_style_combo)
                
                # Marker combo box
                marker_combo = QComboBox()
                markers = ['None', '.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
                for marker in markers:
                    marker_combo.addItem(marker)
                    
                # Set previously selected marker if available
                if plot_name in self.legend_map and 'marker' in self.legend_map[plot_name]:
                    marker_value = self.legend_map[plot_name]['marker']
                    # Convert empty string to "None" for the combobox
                    if marker_value == "":
                        marker_value = "None"
                    try:
                        index = markers.index(marker_value)
                        if index >= 0:
                            marker_combo.setCurrentIndex(index)
                    except ValueError:
                        # If marker value not in the list, use default
                        marker_combo.setCurrentIndex(0)  # "None"
                        
                self.legend_table.setCellWidget(row, 3, marker_combo)
                
                # Color button
                color_button = QPushButton()
                color_button.setAutoFillBackground(True)
                
                # Set previously selected color if available
                if plot_name in self.legend_map and 'color' in self.legend_map[plot_name]:
                    color = self.legend_map[plot_name]['color']
                    color_button.setStyleSheet(f"background-color: {color};")
                else:
                    # Generate random color if no previous color exists
                    import random
                    r, g, b = [random.randint(0, 255) for _ in range(3)]
                    color = f"rgb({r},{g},{b})"
                    color_button.setStyleSheet(f"background-color: {color};")
                
                # Connect button to color picker
                color_button.clicked.connect(lambda checked, row=row: self._choose_color(row))
                
                self.legend_table.setCellWidget(row, 4, color_button)
                
                # Store initial values in the legend map
                if plot_name not in self.legend_map:
                    self.legend_map[plot_name] = {
                        'custom_name': legend_name,
                        'line_style': line_styles[line_style_combo.currentIndex()],
                        'marker': markers[marker_combo.currentIndex()],
                        'color': color
                    }
        
        def _choose_color(self, row):
            """Open a color dialog when a color button is clicked"""
            from PyQt5.QtWidgets import QColorDialog
            
            color_button = self.legend_table.cellWidget(row, 4)
            color_dialog = QColorDialog(self)
            
            if color_dialog.exec_():
                color = color_dialog.selectedColor()
                if color.isValid():
                    color_name = color.name()
                    color_button.setStyleSheet(f"background-color: {color_name};")
                    
                    # Update the legend map with the new color
                    plot_name = self.legend_table.item(row, 0).text()
                    if plot_name in self.legend_map:
                        self.legend_map[plot_name]['color'] = color_name
                    else:
                        self.legend_map[plot_name] = {'color': color_name}
        
        def clear_all_frf_plots(self):
            """Clear all FRF plots from the list and reset the legend map"""
            # Clear the list widget
            self.available_plots_list.clear()
            
            # Clear the legend table
            self.legend_table.setRowCount(0)
            
            # Reset the legend map
            self.legend_map = {}
            
            # Clear any existing plot
            try:
                if hasattr(self, 'comp_fig') and self.comp_fig:
                    import matplotlib.pyplot as plt
                    plt.close(self.comp_fig)
                    self.comp_fig = None
            except Exception as e:
                print(f"Error clearing figures: {str(e)}")
        
        def export_frf_data(self):
            """Export FRF data to a file"""
            from PyQt5.QtWidgets import QFileDialog
            import json
            import os
            
            # Check if there are any plots to export
            if self.available_plots_list.count() == 0:
                QMessageBox.warning(self, "Export Error", "No FRF data available to export.")
                return
            
            # Get export filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export FRF Data", os.path.expanduser("~"), "JSON Files (*.json)"
            )
            
            if not filename:
                return  # User cancelled
                
            # Gather data to export
            export_data = {
                'plots': {},
                'legend_map': self.legend_map
            }
            
            # Add individual plot data
            for i in range(self.available_plots_list.count()):
                plot_name = self.available_plots_list.item(i).text()
                if hasattr(self, f"frf_data_{plot_name}"):
                    plot_data = getattr(self, f"frf_data_{plot_name}")
                    export_data['plots'][plot_name] = plot_data
            
            # Write to file
            try:
                with open(filename, 'w') as f:
                    json.dump(export_data, f)
                QMessageBox.information(self, "Export Successful", f"FRF data has been exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
        
        def import_frf_data(self):
            """Import FRF data from a file"""
            from PyQt5.QtWidgets import QFileDialog
            import json
            import os
            
            # Get import filename
            filename, _ = QFileDialog.getOpenFileName(
                self, "Import FRF Data", os.path.expanduser("~"), "JSON Files (*.json)"
            )
            
            if not filename:
                return  # User cancelled
                
            # Read the file
            try:
                with open(filename, 'r') as f:
                    import_data = json.load(f)
                    
                # Validate imported data
                if 'plots' not in import_data or 'legend_map' not in import_data:
                    raise ValueError("Invalid file format: missing required data")
                    
                # Update legend map
                self.legend_map.update(import_data['legend_map'])
                
                # Import plot data
                for plot_name, plot_data in import_data['plots'].items():
                    # Store the data
                    setattr(self, f"frf_data_{plot_name}", plot_data)
                    
                    # Add to list if not already there
                    found = False
                    for i in range(self.available_plots_list.count()):
                        if self.available_plots_list.item(i).text() == plot_name:
                            found = True
                            break
                            
                    if not found:
                        self.available_plots_list.addItem(plot_name)
                        
                QMessageBox.information(self, "Import Successful", 
                                       f"Imported {len(import_data['plots'])} FRF datasets from {filename}")
                                       
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Error importing data: {str(e)}")
        
        def create_sobol_analysis_tab(self):
            self.sobol_tab = QWidget()
            layout = QVBoxLayout(self.sobol_tab)
    
            # Create sub-tabs widget
            self.sobol_sub_tabs = QTabWidget()
    
            # -------------------- Sub-tab 1: Sobol Analysis Settings --------------------
            sobol_settings_tab = QWidget()
            sobol_settings_layout = QFormLayout(sobol_settings_tab)
    
            self.num_samples_line = QLineEdit()
            self.num_samples_line.setPlaceholderText("e.g. 32,64,128")
            sobol_settings_layout.addRow("Num Samples List:", self.num_samples_line)
    
            self.n_jobs_spin = QSpinBox()
            self.n_jobs_spin.setRange(1, 64)
            self.n_jobs_spin.setValue(5)
            sobol_settings_layout.addRow("Number of Jobs (n_jobs):", self.n_jobs_spin)
    
            # Add a small Run Sobol button in the settings sub-tab
            self.hyper_run_sobol_button = QPushButton("Run Sobol")
            self.hyper_run_sobol_button.setFixedWidth(100)
            self.hyper_run_sobol_button.clicked.connect(self._run_sobol_implementation)
            sobol_settings_layout.addRow("Run Sobol:", self.hyper_run_sobol_button)
    
            # -------------------- Sub-tab 2: DVA Parameters --------------------
            dva_param_tab = QWidget()
            dva_param_layout = QVBoxLayout(dva_param_tab)
    
            self.dva_param_table = QTableWidget()
            dva_parameters = [
                *[f"beta_{i}" for i in range(1,16)],
                *[f"lambda_{i}" for i in range(1,16)],
                *[f"mu_{i}" for i in range(1,4)],
                *[f"nu_{i}" for i in range(1,16)]
            ]
            self.dva_param_table.setRowCount(len(dva_parameters))
            self.dva_param_table.setColumnCount(5)
            self.dva_param_table.setHorizontalHeaderLabels(
                ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
            )
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
                fixed_value_spin.setRange(-1e6, 1e6)
                fixed_value_spin.setDecimals(6)
                fixed_value_spin.setEnabled(False)
                self.dva_param_table.setCellWidget(row, 2, fixed_value_spin)
    
                lower_bound_spin = QDoubleSpinBox()
                lower_bound_spin.setRange(-1e6, 1e6)
                lower_bound_spin.setDecimals(6)
                lower_bound_spin.setEnabled(True)
                self.dva_param_table.setCellWidget(row, 3, lower_bound_spin)
    
                upper_bound_spin = QDoubleSpinBox()
                upper_bound_spin.setRange(-1e6, 1e6)
                upper_bound_spin.setDecimals(6)
                upper_bound_spin.setEnabled(True)
                self.dva_param_table.setCellWidget(row, 4, upper_bound_spin)
    
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
    
            dva_param_layout.addWidget(self.dva_param_table)
    
            # -------------------- Sub-tab 3: Sobol Results --------------------
            sobol_results_tab = QWidget()
            sobol_results_layout = QVBoxLayout(sobol_results_tab)
            
            # Results text area
            self.sobol_results_text = QTextEdit()
            self.sobol_results_text.setReadOnly(True)
            self.sobol_results_text.setStyleSheet("font-family: monospace;")
            sobol_results_layout.addWidget(self.sobol_results_text)
            
            # Placeholder for Sobol plots
            plot_container = QWidget()
            plot_layout = QVBoxLayout(plot_container)
            
            # Plotting controls section
            plot_controls = QWidget()
            plot_controls_layout = QHBoxLayout(plot_controls)
            
            self.sobol_combo = QComboBox()
            self.sobol_combo.currentIndexChanged.connect(self.update_sobol_plot)
            plot_controls_layout.addWidget(QLabel("Plot Type:"))
            plot_controls_layout.addWidget(self.sobol_combo)
            
            save_button = QPushButton("Save Results")
            save_button.clicked.connect(self.save_sobol_results)
            plot_controls_layout.addWidget(save_button)
            
            plot_layout.addWidget(plot_controls)
            
            # Figure canvas
            self.sobol_figure = Figure(figsize=(4, 4))
            self.sobol_canvas = FigureCanvas(self.sobol_figure)
            self.sobol_canvas.setMinimumHeight(300)
            toolbar = NavigationToolbar(self.sobol_canvas, self)
            
            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(self.sobol_canvas)
            
            sobol_results_layout.addWidget(plot_container)
            
            # Initialize plots dictionary
            self.sobol_plots = {}
    
            # Add sub-tabs to the Sobol tab widget
            self.sobol_sub_tabs.addTab(sobol_settings_tab, "Analysis Settings")
            self.sobol_sub_tabs.addTab(dva_param_tab, "DVA Parameters")
            self.sobol_sub_tabs.addTab(sobol_results_tab, "Results")
    
            # Add the Sobol sub-tabs widget to the main Sobol tab layout
            layout.addWidget(self.sobol_sub_tabs)
            self.sobol_tab.setLayout(layout)
    
        def create_ga_tab(self):
            """Create the genetic algorithm optimization tab"""
            self.ga_tab = QWidget()
            layout = QVBoxLayout(self.ga_tab)
    
            # Create sub-tabs widget
            self.ga_sub_tabs = QTabWidget()
    
            # -------------------- Sub-tab 1: GA Hyperparameters --------------------
            ga_hyper_tab = QWidget()
            ga_hyper_layout = QFormLayout(ga_hyper_tab)
    
            self.ga_pop_size_box = QSpinBox()
            self.ga_pop_size_box.setRange(1, 10000)
            self.ga_pop_size_box.setValue(800)
    
            self.ga_num_generations_box = QSpinBox()
            self.ga_num_generations_box.setRange(1, 10000)
            self.ga_num_generations_box.setValue(100)
    
            self.ga_cxpb_box = QDoubleSpinBox()
            self.ga_cxpb_box.setRange(0, 1)
            self.ga_cxpb_box.setValue(0.7)
            self.ga_cxpb_box.setDecimals(3)
    
            self.ga_mutpb_box = QDoubleSpinBox()
            self.ga_mutpb_box.setRange(0, 1)
            self.ga_mutpb_box.setValue(0.2)
            self.ga_mutpb_box.setDecimals(3)
    
            self.ga_tol_box = QDoubleSpinBox()
            self.ga_tol_box.setRange(0, 1e6)
            self.ga_tol_box.setValue(1e-3)
            self.ga_tol_box.setDecimals(6)
    
            self.ga_alpha_box = QDoubleSpinBox()
            self.ga_alpha_box.setRange(0.0, 10.0)
            self.ga_alpha_box.setDecimals(4)
            self.ga_alpha_box.setSingleStep(0.01)
            self.ga_alpha_box.setValue(0.01)
            
            # Add benchmarking runs box
            self.ga_benchmark_runs_box = QSpinBox()
            self.ga_benchmark_runs_box.setRange(1, 1000)
            self.ga_benchmark_runs_box.setValue(1)
            self.ga_benchmark_runs_box.setToolTip("Number of times to run the GA for benchmarking (1 = single run)")
            
            # Add adaptive rates checkbox
            self.adaptive_rates_checkbox = QCheckBox("Use Adaptive Rates")
            self.adaptive_rates_checkbox.setChecked(False)
            self.adaptive_rates_checkbox.setToolTip("Automatically adjust crossover and mutation rates during optimization")
            self.adaptive_rates_checkbox.stateChanged.connect(self.toggle_adaptive_rates_options)
            
            # Create a widget to hold adaptive rate options
            self.adaptive_rates_options = QWidget()
            adaptive_options_layout = QFormLayout(self.adaptive_rates_options)
            adaptive_options_layout.setContentsMargins(20, 0, 0, 0)  # Add left margin for indentation
            
            # Stagnation limit spinner
            self.stagnation_limit_box = QSpinBox()
            self.stagnation_limit_box.setRange(1, 50)
            self.stagnation_limit_box.setValue(5)
            self.stagnation_limit_box.setToolTip("Number of generations without improvement before adapting rates")
            adaptive_options_layout.addRow("Stagnation Limit:", self.stagnation_limit_box)
            
            # Create a widget for crossover bounds
            crossover_bounds_widget = QWidget()
            crossover_bounds_layout = QHBoxLayout(crossover_bounds_widget)
            crossover_bounds_layout.setContentsMargins(0, 0, 0, 0)
            
            self.cxpb_min_box = QDoubleSpinBox()
            self.cxpb_min_box.setRange(0.01, 0.5)
            self.cxpb_min_box.setValue(0.1)
            self.cxpb_min_box.setDecimals(2)
            self.cxpb_min_box.setSingleStep(0.05)
            self.cxpb_min_box.setToolTip("Minimum crossover probability")
            
            self.cxpb_max_box = QDoubleSpinBox()
            self.cxpb_max_box.setRange(0.5, 1.0)
            self.cxpb_max_box.setValue(0.9)
            self.cxpb_max_box.setDecimals(2)
            self.cxpb_max_box.setSingleStep(0.05)
            self.cxpb_max_box.setToolTip("Maximum crossover probability")
            
            crossover_bounds_layout.addWidget(QLabel("Min:"))
            crossover_bounds_layout.addWidget(self.cxpb_min_box)
            crossover_bounds_layout.addWidget(QLabel("Max:"))
            crossover_bounds_layout.addWidget(self.cxpb_max_box)
            
            adaptive_options_layout.addRow("Crossover Bounds:", crossover_bounds_widget)
            
            # Create a widget for mutation bounds
            mutation_bounds_widget = QWidget()
            mutation_bounds_layout = QHBoxLayout(mutation_bounds_widget)
            mutation_bounds_layout.setContentsMargins(0, 0, 0, 0)
            
            self.mutpb_min_box = QDoubleSpinBox()
            self.mutpb_min_box.setRange(0.01, 0.2)
            self.mutpb_min_box.setValue(0.05)
            self.mutpb_min_box.setDecimals(2)
            self.mutpb_min_box.setSingleStep(0.01)
            self.mutpb_min_box.setToolTip("Minimum mutation probability")
            
            self.mutpb_max_box = QDoubleSpinBox()
            self.mutpb_max_box.setRange(0.2, 0.8)
            self.mutpb_max_box.setValue(0.5)
            self.mutpb_max_box.setDecimals(2)
            self.mutpb_max_box.setSingleStep(0.05)
            self.mutpb_max_box.setToolTip("Maximum mutation probability")
            
            mutation_bounds_layout.addWidget(QLabel("Min:"))
            mutation_bounds_layout.addWidget(self.mutpb_min_box)
            mutation_bounds_layout.addWidget(QLabel("Max:"))
            mutation_bounds_layout.addWidget(self.mutpb_max_box)
            
            adaptive_options_layout.addRow("Mutation Bounds:", mutation_bounds_widget)
            
            # Initially hide adaptive options
            self.adaptive_rates_options.setVisible(False)
    
            ga_hyper_layout.addRow("Population Size:", self.ga_pop_size_box)
            ga_hyper_layout.addRow("Number of Generations:", self.ga_num_generations_box)
            ga_hyper_layout.addRow("Crossover Probability (cxpb):", self.ga_cxpb_box)
            ga_hyper_layout.addRow("Mutation Probability (mutpb):", self.ga_mutpb_box)
            ga_hyper_layout.addRow("Tolerance (tol):", self.ga_tol_box)
            ga_hyper_layout.addRow("Sparsity Penalty (alpha):", self.ga_alpha_box)
            ga_hyper_layout.addRow("Benchmark Runs:", self.ga_benchmark_runs_box)
            ga_hyper_layout.addRow("", self.adaptive_rates_checkbox)
            ga_hyper_layout.addRow("", self.adaptive_rates_options)
    
            # Add a small Run GA button in the hyperparameters sub-tab
            self.hyper_run_ga_button = QPushButton("Run GA")
            self.hyper_run_ga_button.setFixedWidth(100)
            self.hyper_run_ga_button.clicked.connect(self.run_ga)
            ga_hyper_layout.addRow("Run GA:", self.hyper_run_ga_button)
    
            # -------------------- Sub-tab 2: DVA Parameters --------------------
            ga_param_tab = QWidget()
            ga_param_layout = QVBoxLayout(ga_param_tab)
    
            self.ga_param_table = QTableWidget()
            dva_parameters = [
                *[f"beta_{i}" for i in range(1,16)],
                *[f"lambda_{i}" for i in range(1,16)],
                *[f"mu_{i}" for i in range(1,4)],
                *[f"nu_{i}" for i in range(1,16)]
            ]
            self.ga_param_table.setRowCount(len(dva_parameters))
            self.ga_param_table.setColumnCount(5)
            self.ga_param_table.setHorizontalHeaderLabels(
                ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
            )
            self.ga_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.ga_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    
            for row, param in enumerate(dva_parameters):
                param_item = QTableWidgetItem(param)
                param_item.setFlags(Qt.ItemIsEnabled)
                self.ga_param_table.setItem(row, 0, param_item)
    
                fixed_checkbox = QCheckBox()
                fixed_checkbox.setChecked(True)  # Set fixed to true by default
                fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_ga_fixed(state, r))
                self.ga_param_table.setCellWidget(row, 1, fixed_checkbox)
    
                fixed_value_spin = QDoubleSpinBox()
                fixed_value_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
                fixed_value_spin.setDecimals(6)
                fixed_value_spin.setValue(0.0)  # Set fixed value to 0
                fixed_value_spin.setEnabled(True)  # Enable because fixed is checked
                self.ga_param_table.setCellWidget(row, 2, fixed_value_spin)
    
                lower_bound_spin = QDoubleSpinBox()
                lower_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
                lower_bound_spin.setDecimals(6)
                lower_bound_spin.setValue(0.0)  # Set to 0
                lower_bound_spin.setEnabled(False)  # Disable because fixed is checked
                self.ga_param_table.setCellWidget(row, 3, lower_bound_spin)
    
                upper_bound_spin = QDoubleSpinBox()
                upper_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
                upper_bound_spin.setDecimals(6)
                upper_bound_spin.setValue(1.0)  # Set to 1
                upper_bound_spin.setEnabled(False)  # Disable because fixed is checked
                self.ga_param_table.setCellWidget(row, 4, upper_bound_spin)
    
            ga_param_layout.addWidget(self.ga_param_table)
    
            # -------------------- Sub-tab 3: Results --------------------
            ga_results_tab = QWidget()
            ga_results_layout = QVBoxLayout(ga_results_tab)
    
            # Create a header area for label and export button
            header_container = QWidget()
            header_layout = QHBoxLayout(header_container)
            header_layout.setContentsMargins(0, 0, 0, 0) # No margins for this internal layout
    
            results_label = QLabel("GA Optimization Results:")
            header_layout.addWidget(results_label)
            header_layout.addStretch() # Add spacer to push the export button to the right
    
            self.export_ga_results_button = QPushButton("Export GA Results")
            self.export_ga_results_button.setObjectName("secondary-button") # Use existing styling if desired
            self.export_ga_results_button.setToolTip("Export the GA optimization results to a JSON file")
            self.export_ga_results_button.setEnabled(False)  # Initially disabled
            # self.export_ga_results_button.clicked.connect(self.export_ga_results_to_file) # Will connect this later
            header_layout.addWidget(self.export_ga_results_button)
            
            ga_results_layout.addWidget(header_container) # Add the header with label and button
            
            self.ga_results_text = QTextEdit()
            self.ga_results_text.setReadOnly(True)
            ga_results_layout.addWidget(self.ga_results_text)
    
            # -------------------- Sub-tab 4: Benchmarking --------------------
            ga_benchmark_tab = QWidget()
            ga_benchmark_layout = QVBoxLayout(ga_benchmark_tab)
    
            # Create buttons for import/export
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(0, 0, 0, 10)  # Add some bottom margin
    
            self.import_benchmark_button = QPushButton("Import Benchmark Data")
            self.import_benchmark_button.setToolTip("Import previously saved GA benchmark data")
            self.import_benchmark_button.clicked.connect(self.import_ga_benchmark_data)
            button_layout.addWidget(self.import_benchmark_button)
    
            self.export_benchmark_button = QPushButton("Export Benchmark Data")
            self.export_benchmark_button.setToolTip("Export current GA benchmark data to a file")
            self.export_benchmark_button.setEnabled(False)  # Initially disabled until data is available
            self.export_benchmark_button.clicked.connect(self.export_ga_benchmark_data)
            button_layout.addWidget(self.export_benchmark_button)
    
            button_layout.addStretch()  # Add stretch to push buttons to the left
            ga_benchmark_layout.addWidget(button_container)
    
            # Create tabs for different benchmark visualizations
            self.benchmark_viz_tabs = QTabWidget()
            
            # Create tabs for different visualizations
            violin_tab = QWidget()
            violin_layout = QVBoxLayout(violin_tab)
            self.violin_plot_widget = QWidget()
            violin_layout.addWidget(self.violin_plot_widget)
            
            dist_tab = QWidget()
            dist_layout = QVBoxLayout(dist_tab)
            self.dist_plot_widget = QWidget()
            dist_layout.addWidget(self.dist_plot_widget)
            
            scatter_tab = QWidget()
            scatter_layout = QVBoxLayout(scatter_tab)
            self.scatter_plot_widget = QWidget()
            scatter_layout.addWidget(self.scatter_plot_widget)
            
            heatmap_tab = QWidget()
            heatmap_layout = QVBoxLayout(heatmap_tab)
            self.heatmap_plot_widget = QWidget()
            heatmap_layout.addWidget(self.heatmap_plot_widget)
            
            # Add Q-Q plot tab
            qq_tab = QWidget()
            qq_layout = QVBoxLayout(qq_tab)
            self.qq_plot_widget = QWidget()
            qq_layout.addWidget(self.qq_plot_widget)
            
            # Summary statistics tabs (create subtabs for better organization)
            stats_tab = QWidget()
            stats_layout = QVBoxLayout(stats_tab)
            
            # Create a tabbed widget for the statistics section
            stats_subtabs = QTabWidget()
            
            # ---- Subtab 1: Summary Statistics ----
            summary_tab = QWidget()
            summary_layout = QVBoxLayout(summary_tab)
            
            # Add summary statistics table
            self.benchmark_stats_table = QTableWidget()
            self.benchmark_stats_table.setColumnCount(5)
            self.benchmark_stats_table.setHorizontalHeaderLabels(["Metric", "Min", "Max", "Mean", "Std"])
            self.benchmark_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            summary_layout.addWidget(QLabel("Statistical Summary of All Runs:"))
            summary_layout.addWidget(self.benchmark_stats_table)
            
            # ---- Subtab 2: All Runs Table ----
            runs_tab = QWidget()
            runs_layout = QVBoxLayout(runs_tab)
            
            # Create a table for all runs
            self.benchmark_runs_table = QTableWidget()
            self.benchmark_runs_table.setColumnCount(4)
            self.benchmark_runs_table.setHorizontalHeaderLabels(["Run #", "Fitness", "Rank", "Details"])
            self.benchmark_runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.benchmark_runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.benchmark_runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.benchmark_runs_table.itemClicked.connect(self.show_run_details)
            
            runs_layout.addWidget(QLabel("All Benchmark Runs:"))
            runs_layout.addWidget(self.benchmark_runs_table)
            
            # Create run details text area
            details_tab = QWidget()
            details_layout = QVBoxLayout(details_tab)
            details_group = QGroupBox("Run Details")
            group_layout = QVBoxLayout(details_group)
            self.run_details_text = QTextEdit()
            self.run_details_text.setReadOnly(True)
            group_layout.addWidget(self.run_details_text)
            details_layout.addWidget(details_group)
            
            # Create GA Operations tab as a subtab
            ga_ops_tab = QWidget()
            ga_ops_layout = QVBoxLayout(ga_ops_tab)
            self.ga_ops_plot_widget = QWidget()
            ga_ops_layout.addWidget(self.ga_ops_plot_widget)
            
            # Add the subtabs to the stats tabbed widget
            stats_subtabs.addTab(summary_tab, "Summary Statistics")
            stats_subtabs.addTab(runs_tab, "All Runs")
            stats_subtabs.addTab(details_tab, "Run Details")
            stats_subtabs.addTab(ga_ops_tab, "GA Operations")
            
            # Add the stats tabbed widget to the stats tab
            stats_layout.addWidget(stats_subtabs)
            
            # Add all visualization tabs to the benchmark visualization tabs
            self.benchmark_viz_tabs.addTab(violin_tab, "Violin Plot")
            self.benchmark_viz_tabs.addTab(dist_tab, "Distribution")
            self.benchmark_viz_tabs.addTab(scatter_tab, "Scatter Plot")
            self.benchmark_viz_tabs.addTab(heatmap_tab, "Parameter Correlations")
            self.benchmark_viz_tabs.addTab(qq_tab, "Q-Q Plot")
            self.benchmark_viz_tabs.addTab(stats_tab, "Statistics")
            
            # GA Operations Performance Tab - already added as a subtab of Statistics
            
            # Add the benchmark visualization tabs to the benchmark tab
            ga_benchmark_layout.addWidget(self.benchmark_viz_tabs)
            
            # Add all sub-tabs to the GA tab widget
            # Initialize empty benchmark data storage
            self.ga_benchmark_data = []
    
            # Add all sub-tabs to the GA tab widget
            self.ga_sub_tabs.addTab(ga_hyper_tab, "GA Settings")
            self.ga_sub_tabs.addTab(ga_param_tab, "DVA Parameters")
            self.ga_sub_tabs.addTab(ga_results_tab, "Results")
            self.ga_sub_tabs.addTab(ga_benchmark_tab, "GA Benchmarking")
    
            # Add the GA sub-tabs widget to the main GA tab layout
            layout.addWidget(self.ga_sub_tabs)
            self.ga_tab.setLayout(layout)
            
        def toggle_fixed(self, state, row, table=None):
            """Toggle the fixed state of a DVA parameter row"""
            if table is None:
                table = self.dva_param_table
                
            fixed = (state == Qt.Checked)
            fixed_value_spin = table.cellWidget(row, 2)
            lower_bound_spin = table.cellWidget(row, 3)
            upper_bound_spin = table.cellWidget(row, 4)
    
            fixed_value_spin.setEnabled(fixed)
            lower_bound_spin.setEnabled(not fixed)
            upper_bound_spin.setEnabled(not fixed)
    
        def toggle_ga_fixed(self, state, row, table=None):
            """Toggle the fixed state of a GA parameter row"""
            if table is None:
                table = self.ga_param_table
                
            fixed = (state == Qt.Checked)
            fixed_value_spin = table.cellWidget(row, 2)
            lower_bound_spin = table.cellWidget(row, 3)
            upper_bound_spin = table.cellWidget(row, 4)
            
            # Enable/disable appropriate spinboxes
            fixed_value_spin.setEnabled(fixed)
            lower_bound_spin.setEnabled(not fixed)
            upper_bound_spin.setEnabled(not fixed)
            
            # If switching to fixed mode, copy current lower bound value to fixed value
            if fixed:
                fixed_value_spin.setValue(lower_bound_spin.value())
            # If switching to range mode, ensure lower bound is not greater than upper bound
            else:
                if lower_bound_spin.value() > upper_bound_spin.value():
                    upper_bound_spin.setValue(lower_bound_spin.value())
    
        def toggle_adaptive_rates_options(self, state):
            """Show or hide adaptive rates options based on checkbox state"""
            self.adaptive_rates_options.setVisible(state == Qt.Checked)
            
            # Enable/disable the fixed rate inputs based on adaptive rates setting
            self.ga_cxpb_box.setEnabled(state != Qt.Checked)
            self.ga_mutpb_box.setEnabled(state != Qt.Checked)
            
            # Update tooltips to indicate that rates will be adaptive
            if state == Qt.Checked:
                self.ga_cxpb_box.setToolTip("Starting crossover probability (will adapt during optimization)")
                self.ga_mutpb_box.setToolTip("Starting mutation probability (will adapt during optimization)")
            else:
                self.ga_cxpb_box.setToolTip("Crossover probability")
                self.ga_mutpb_box.setToolTip("Mutation probability")

