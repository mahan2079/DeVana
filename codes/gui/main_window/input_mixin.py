from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import ModernQTabWidget

class InputTabsMixin:
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
        
        # Create form layout for parameters
        params_form = QFormLayout()
        
        # MU parameter
        self.mu_box = QDoubleSpinBox()
        self.mu_box.setRange(-1e6, 1e6)
        self.mu_box.setDecimals(6)
        self.mu_box.setValue(1.0)
        params_form.addRow("μ (MU):", self.mu_box)

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
            params_form.addRow(f"Λ_{i+1}:", box)

        # NU parameters
        self.nu_boxes = []
        for i in range(5):
            box = QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setDecimals(6)
            box.setValue(0.75)
            self.nu_boxes.append(box)
            params_form.addRow(f"Ν_{i+1}:", box)

        # A_LOW parameter
        self.a_low_box = QDoubleSpinBox()
        self.a_low_box.setRange(0, 1e10)
        self.a_low_box.setDecimals(6)
        self.a_low_box.setValue(0.05)
        params_form.addRow("A_LOW:", self.a_low_box)

        # A_UPP parameter
        self.a_up_box = QDoubleSpinBox()
        self.a_up_box.setRange(0, 1e10)
        self.a_up_box.setDecimals(6)
        self.a_up_box.setValue(0.05)
        params_form.addRow("A_UPP:", self.a_up_box)

        # F_1 parameter
        self.f_1_box = QDoubleSpinBox()
        self.f_1_box.setRange(0, 1e10)
        self.f_1_box.setDecimals(6)
        self.f_1_box.setValue(100.0)
        params_form.addRow("F_1:", self.f_1_box)

        # F_2 parameter
        self.f_2_box = QDoubleSpinBox()
        self.f_2_box.setRange(0, 1e10)
        self.f_2_box.setDecimals(6)
        self.f_2_box.setValue(100.0)
        params_form.addRow("F_2:", self.f_2_box)

        # OMEGA_DC parameter
        self.omega_dc_box = QDoubleSpinBox()
        self.omega_dc_box.setRange(0, 1e10)
        self.omega_dc_box.setDecimals(6)
        self.omega_dc_box.setValue(5000.0)
        params_form.addRow("Ω_DC:", self.omega_dc_box)

        # ZETA_DC parameter
        self.zeta_dc_box = QDoubleSpinBox()
        self.zeta_dc_box.setRange(0, 1e10)
        self.zeta_dc_box.setDecimals(6)
        self.zeta_dc_box.setValue(0.01)
        params_form.addRow("ζ_DC:", self.zeta_dc_box)
        
        # Add form layout to main layout
        main_layout.addLayout(params_form)
        main_layout.addStretch()
        
        # Set the container as the scroll area's widget
        scroll_area.setWidget(main_container)
        
        # Add scroll area to the tab's layout
        layout.addWidget(scroll_area)
        
        return self.main_system_tab

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
                                       "CMA-ES", "Reinforcement Learning (RL)", "NSGA-II"])
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
        
        # --------- COMBINED VIEW TAB ---------
        combined_tab = QWidget()
        combined_layout = QVBoxLayout(combined_tab)

        self.combined_fig = Figure(figsize=(10, 6))
        self.combined_canvas = FigureCanvas(self.combined_fig)
        self.combined_canvas.setMinimumHeight(450)
        self.combined_toolbar = NavigationToolbar(self.combined_canvas, combined_tab)

        combined_layout.addWidget(self.combined_canvas)
        combined_layout.addWidget(self.combined_toolbar)

        # No data label for combined view
        self.combined_no_data_label = QLabel("Run the sensitivity analysis to generate visualization")
        self.combined_no_data_label.setAlignment(Qt.AlignCenter)
        self.combined_no_data_label.setStyleSheet("color: #888; font-style: italic; font-size: 14px;")
        combined_layout.addWidget(self.combined_no_data_label)

        # Add tabs to the visualization tabs widget
        self.vis_tabs.addTab(convergence_tab, "Slope Convergence")
        self.vis_tabs.addTab(rel_change_tab, "Relative Change")
        self.vis_tabs.addTab(combined_tab, "Combined View")

        # Create main visualization container tab
        vis_tab = QWidget()
        vis_layout = QVBoxLayout(vis_tab)
        vis_layout.addWidget(self.vis_tabs)

        # Add parameter and visualization tabs to the main widget
        self.sensitivity_tabs.addTab(params_tab, "Parameters")
        self.sensitivity_tabs.addTab(vis_tab, "Visualization")
        self.sensitivity_tabs.setCurrentIndex(0)

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
        
    def get_dva_params(self):
        """Get the DVA parameters in a dictionary format"""
        dva_params = {
            "beta": [box.value() for box in self.beta_boxes],
            "lambda": [box.value() for box in self.lambda_boxes],
            "mu": [box.value() for box in self.mu_dva_boxes],
            "nu": [box.value() for box in self.nu_dva_boxes]
        }
        return dva_params
        
    def get_target_values_weights(self):
        """
        Get the target values and weights for all masses.
        
        Returns:
            tuple: (target_values_dict, weights_dict) containing targets and weights for all masses
        """
        target_values_dict = {}
        weights_dict = {}

        for mass_num in range(1, 6):
            t_dict = {}
            w_dict = {}
            # Handle peak values (1-4)
            for peak_num in range(1, 5):
                pv_key = f"peak_value_{peak_num}_m{mass_num}"
                if pv_key in self.mass_target_spins:
                    t_dict[f"peak_value_{peak_num}"] = self.mass_target_spins[pv_key].value()
                    w_dict[f"peak_value_{peak_num}"] = self.mass_weight_spins[pv_key].value()
            
            # Handle peak positions (1-5)
            for peak_num in range(1, 6):
                pp_key = f"peak_position_{peak_num}_m{mass_num}"
                if pp_key in self.mass_target_spins:
                    t_dict[f"peak_position_{peak_num}"] = self.mass_target_spins[pp_key].value()
                    w_dict[f"peak_position_{peak_num}"] = self.mass_weight_spins[pp_key].value()

            for i in range(1, 5):
                for j in range(i+1, 5):
                    bw_key = f"bandwidth_{i}_{j}_m{mass_num}"
                    if bw_key in self.mass_target_spins:
                        t_dict[f"bandwidth_{i}_{j}"] = self.mass_target_spins[bw_key].value()
                        w_dict[f"bandwidth_{i}_{j}"] = self.mass_weight_spins[bw_key].value()

            for i in range(1, 5):
                for j in range(i+1, 5):
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

    def create_frequency_tab(self):
        """Create the frequency and plot tab"""
        self.freq_tab = QWidget()
        layout = QVBoxLayout(self.freq_tab)

        # Create a scroll area for potentially large content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create main container widget
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        
        # Frequency range group
        freq_group = QGroupBox("Frequency Range & Plot Options")
        freq_layout = QFormLayout(freq_group)

        # OMEGA start
        self.omega_start_box = QDoubleSpinBox()
        self.omega_start_box.setRange(0, 1e6)
        self.omega_start_box.setDecimals(6)
        self.omega_start_box.setValue(0.0)
        freq_layout.addRow("Ω Start:", self.omega_start_box)

        # OMEGA end
        self.omega_end_box = QDoubleSpinBox()
        self.omega_end_box.setRange(0, 1e6)
        self.omega_end_box.setDecimals(6)
        self.omega_end_box.setValue(10000.0)
        freq_layout.addRow("Ω End:", self.omega_end_box)

        # OMEGA points
        self.omega_points_box = QSpinBox()
        self.omega_points_box.setRange(1, 1000000000)  # Increased to 10^9
        self.omega_points_box.setValue(1200)
        freq_layout.addRow("Ω Points:", self.omega_points_box)

        # Plot options
        self.plot_figure_chk = QCheckBox("Plot Figure")
        self.plot_figure_chk.setChecked(True)
        freq_layout.addRow(self.plot_figure_chk)
        
        self.show_peaks_chk = QCheckBox("Show Peaks")
        self.show_peaks_chk.setChecked(False)
        freq_layout.addRow(self.show_peaks_chk)
        
        self.show_slopes_chk = QCheckBox("Show Slopes")
        self.show_slopes_chk.setChecked(False)
        freq_layout.addRow(self.show_slopes_chk)
        
        main_layout.addWidget(freq_group)
        
        # Add interpolation options section
        interp_group = QGroupBox("Curve Interpolation Options")
        interp_layout = QFormLayout(interp_group)
        
        # Interpolation method combo box
        self.interp_method_combo = QComboBox()
        from modules.FRF import INTERPOLATION_METHODS
        self.interp_method_combo.addItems(['none'] + INTERPOLATION_METHODS)
        self.interp_method_combo.setCurrentText('cubic')  # Default to cubic
        self.interp_method_combo.setToolTip(
            "none: No interpolation (raw data)\n"
            "linear: Straight line segments\n"
            "cubic: Smooth cubic spline (default)\n"
            "quadratic: Quadratic interpolation\n"
            "nearest: Nearest neighbor interpolation\n"
            "akima: Reduced oscillation spline\n"
            "pchip: Piecewise cubic Hermite\n"
            "smoothing_spline: Smoothing spline\n"
            "bspline: B-spline interpolation\n"
            "savgol: Savitzky-Golay filter (smoothing)\n"
            "moving_average: Moving average smoothing\n"
            "gaussian: Gaussian filter smoothing\n"
            "bessel: Bessel filter (good for frequency data)\n"
            "barycentric: Barycentric interpolation\n"
            "rbf: Radial basis function"
        )
        interp_layout.addRow("Interpolation Method:", self.interp_method_combo)
        
        # Points to use for interpolation
        self.interp_points_box = QSpinBox()
        self.interp_points_box.setRange(100, 10000)
        self.interp_points_box.setValue(1000)
        self.interp_points_box.setSingleStep(100)
        self.interp_points_box.setToolTip("Number of points to use in the interpolated curve")
        interp_layout.addRow("Interpolation Points:", self.interp_points_box)
        
        # Add info label about interpolation
        info_label = QLabel(
            "Interpolation smooths the frequency response curve for better visualization. "
            "Different methods provide various levels of smoothing and can affect how peaks "
            "and transitions appear. 'none' shows the raw calculation points."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        interp_layout.addRow(info_label)
        
        # Add interpolation group to main layout
        main_layout.addWidget(interp_group)

        # Add Run FRF button
        run_frf_container = QWidget()
        run_frf_layout = QHBoxLayout(run_frf_container)
        run_frf_layout.setContentsMargins(0, 20, 0, 0)  # Add some top margin
        
        # Add stretch to push button to center
        run_frf_layout.addStretch()
        
        # Create and style the Run FRF button
        self.run_frf_button = QPushButton("Run FRF Analysis")
        self.run_frf_button.setObjectName("primary-button")
        self.run_frf_button.setMinimumWidth(200)  # Make button wider
        self.run_frf_button.setMinimumHeight(40)  # Make button taller
        self.run_frf_button.clicked.connect(self.run_frf)
        self.run_frf_button.setStyleSheet("""
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
        run_frf_layout.addWidget(self.run_frf_button)
        
        # Add stretch to push button to center
        run_frf_layout.addStretch()
        
        main_layout.addWidget(run_frf_container)
        
        # Add comparative visualization options
        self.create_comparative_visualization_options(main_layout)
        
        main_layout.addStretch()
        
        # Set the container as the scroll area's widget
        scroll_area.setWidget(main_container)
        
        # Add scroll area to the tab's layout
        layout.addWidget(scroll_area)

    def refresh_sensitivity_plot(self):
        """Refresh the sensitivity analysis plots with current data"""
        try:
            # Check if we have sensitivity results
            if not hasattr(self, 'sensitivity_results') or self.sensitivity_results is None:
                QMessageBox.warning(self, "No Data", "No sensitivity analysis results available. Please run the analysis first.")
                return
                
            results = self.sensitivity_results
            
            # Extract data from results
            omega_points = results.get("omega_points", [])
            max_slopes = results.get("max_slopes", [])
            relative_changes = results.get("relative_changes", [])
            peak_position_changes = results.get("peak_position_changes", [])
            bandwidth_changes = results.get("bandwidth_changes", [])
            optimal_points = results.get("optimal_points", 0)
            
            if not omega_points or not max_slopes:
                QMessageBox.warning(self, "No Data", "No valid data found in sensitivity analysis results.")
                return
            
            # Get current tab index
            current_tab = self.vis_tabs.currentIndex()
            
            # Clear existing plots
            if current_tab == 0:  # Convergence plot
                self.convergence_fig.clear()
                ax = self.convergence_fig.add_subplot(111)
                
                # Plot slope convergence
                ax.plot(omega_points, max_slopes, 'o-', linewidth=2, markersize=6, 
                       color='#1f77b4', label='Maximum Slope')
                
                # Highlight the optimal point
                if optimal_points in omega_points:
                    opt_idx = omega_points.index(optimal_points)
                    ax.plot(optimal_points, max_slopes[opt_idx], 'ro', markersize=10, 
                           label=f'Optimal Point: {optimal_points}')
                
                ax.set_title('Slope Convergence Analysis', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Omega Points', fontsize=12)
                ax.set_ylabel('Maximum Slope', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format the plot nicely
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                self.convergence_canvas.draw()
                
            elif current_tab == 1:  # Relative change plot
                self.rel_change_fig.clear()
                ax = self.rel_change_fig.add_subplot(111)

                # Plot relative changes (skip first point as it has no relative change)
                def plot_change_series(changes, label, color, marker='o'):
                    if len(changes) > 0:
                        valid_idx = [i for i, val in enumerate(changes) if not np.isnan(val)]
                        pts = [omega_points[i] for i in valid_idx]
                        vals = [changes[i] for i in valid_idx]
                        if pts:
                            ax.semilogy(pts, vals, marker+'-', linewidth=2, markersize=6,
                                        color=color, label=label)

                plot_change_series(relative_changes, 'Max Relative Change', '#ff7f0e', 'o')
                plot_change_series(peak_position_changes, 'Peak Position Δ', '#2ca02c', '^')
                plot_change_series(bandwidth_changes, 'Bandwidth Δ', '#9467bd', 's')

                # Add convergence threshold line
                convergence_threshold = self.sensitivity_threshold.value()
                ax.axhline(y=convergence_threshold, color='red', linestyle='--',
                          linewidth=2, label=f'Convergence Threshold: {convergence_threshold}')

                # Highlight convergence point if it exists
                convergence_point = results.get("convergence_point")
                if convergence_point and convergence_point in omega_points:
                    conv_idx = omega_points.index(convergence_point)
                    if conv_idx < len(relative_changes) and not np.isnan(relative_changes[conv_idx]):
                        ax.plot(convergence_point, relative_changes[conv_idx], 'go',
                               markersize=10, label=f'Convergence Point: {convergence_point}')

                ax.set_title('Maximum Relative Change Across Metrics', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Omega Points', fontsize=12)
                ax.set_ylabel('Max Relative Change (log scale)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format the plot nicely
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                self.rel_change_canvas.draw()

            else:  # Combined view (current_tab == 2)
                self.combined_fig.clear()
                ax1 = self.combined_fig.add_subplot(111)
                ax2 = ax1.twinx()

                ax1.plot(omega_points, max_slopes, 'o-', color='#1f77b4', label='Maximum Slope')

                def plot_change_series_ax2(changes, label, color, marker='s'):
                    if len(changes) > 0:
                        valid_idx = [i for i, val in enumerate(changes) if not np.isnan(val)]
                        pts = [omega_points[i] for i in valid_idx]
                        vals = [changes[i] for i in valid_idx]
                        if pts:
                            ax2.semilogy(pts, vals, marker+'-', color=color, label=label)

                plot_change_series_ax2(relative_changes, 'Max Relative Change', '#ff7f0e', 's')
                plot_change_series_ax2(peak_position_changes, 'Peak Position Δ', '#2ca02c', '^')
                plot_change_series_ax2(bandwidth_changes, 'Bandwidth Δ', '#9467bd', 'd')
                convergence_threshold = self.sensitivity_threshold.value()
                ax2.axhline(y=convergence_threshold, color='red', linestyle='--', linewidth=2,
                            label=f'Threshold {convergence_threshold}')

                ax1.set_xlabel('Number of Omega Points', fontsize=12)
                ax1.set_ylabel('Maximum Slope', fontsize=12, color='#1f77b4')
                ax2.set_ylabel('Max Relative Change (log scale)', fontsize=12, color='#ff7f0e')
                ax1.grid(True, alpha=0.3)

                ax1.tick_params(axis='y', labelcolor='#1f77b4')
                ax2.tick_params(axis='y', labelcolor='#ff7f0e')

                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2)

                self.combined_canvas.draw()
            
            # Enable save button
            self.sensitivity_save_plot_btn.setEnabled(True)
            
            # Hide no data labels
            self.convergence_no_data_label.setVisible(False)
            self.rel_change_no_data_label.setVisible(False)
            self.combined_no_data_label.setVisible(False)
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to refresh sensitivity plot: {str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)


