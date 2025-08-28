from PyQt5.QtWidgets import QAction, QActionGroup, QToolBar, QPushButton, QWidget, QSizePolicy, QMessageBox
from PyQt5.QtCore import QSize
from app_info import APP_NAME, __version__

class MenuMixin:
    """Mixin providing menubar and toolbar creation"""

    def create_menubar(self):
        """Create the application menubar with modern styling"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # New Project
        new_action = QAction("&New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(lambda: self.status_bar.showMessage("New Project - Feature coming soon"))
        file_menu.addAction(new_action)

        # Open Project
        open_action = QAction("&Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(lambda: self.status_bar.showMessage("Open Project - Feature coming soon"))
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Save Project
        save_action = QAction("&Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(lambda: self.status_bar.showMessage("Save Project - Feature coming soon"))
        file_menu.addAction(save_action)

        # Save Project As
        save_as_action = QAction("Save Project &As", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(lambda: self.status_bar.showMessage("Save Project As - Feature coming soon"))
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Import
        import_action = QAction("&Import Parameters", self)
        import_action.triggered.connect(self.import_parameters)
        file_menu.addAction(import_action)

        # Export
        export_action = QAction("&Export Parameters", self)
        export_action.triggered.connect(self.export_parameters)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        # Default values
        default_action = QAction("Reset to &Default Values", self)
        default_action.triggered.connect(self.set_default_values)
        edit_menu.addAction(default_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Toggle Sidebar
        sidebar_action = QAction("Toggle &Sidebar", self)
        sidebar_action.setShortcut("Ctrl+B")
        sidebar_action.triggered.connect(lambda: self.status_bar.showMessage("Toggle Sidebar - Feature coming soon"))
        view_menu.addAction(sidebar_action)

        view_menu.addSeparator()

        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")

        # Theme action group to make selections exclusive
        theme_group = QActionGroup(self)

        # Dark theme action
        dark_action = QAction("&Dark Theme", self)
        dark_action.setCheckable(True)
        if self.current_theme == 'Dark':
            dark_action.setChecked(True)
        dark_action.triggered.connect(lambda: self.switch_theme('Dark'))
        theme_group.addAction(dark_action)
        theme_menu.addAction(dark_action)

        # Light theme action
        light_action = QAction("&Light Theme", self)
        light_action.setCheckable(True)
        if self.current_theme == 'Light':
            light_action.setChecked(True)
        light_action.triggered.connect(lambda: self.switch_theme('Light'))
        theme_group.addAction(light_action)
        theme_menu.addAction(light_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Run FRF
        run_frf_action = QAction("Run &FRF Analysis", self)
        run_frf_action.triggered.connect(self.run_frf)
        tools_menu.addAction(run_frf_action)

        # Run Sobol
        run_sobol_action = QAction("Run &Sobol Analysis", self)
        run_sobol_action.setToolTip("Run Sobol Analysis")
        run_sobol_action.setVisible(False)  # Hide button
        tools_menu.addAction(run_sobol_action)

        tools_menu.addSeparator()

        # Optimization submenu
        optimization_menu = tools_menu.addMenu("&Optimization")

        # GA Optimization
        ga_action = QAction("&Genetic Algorithm", self)
        ga_action.triggered.connect(self.run_ga)
        optimization_menu.addAction(ga_action)

        # PSO Optimization
        pso_action = QAction("&Particle Swarm", self)
        pso_action.triggered.connect(self.run_pso)
        optimization_menu.addAction(pso_action)

        # DE Optimization
        de_action = QAction("&Differential Evolution", self)
        de_action.triggered.connect(self.run_de)
        optimization_menu.addAction(de_action)

        # SA Optimization
        sa_action = QAction("&Simulated Annealing", self)
        sa_action.triggered.connect(self.run_sa)
        optimization_menu.addAction(sa_action)

        # CMAES Optimization
        cmaes_action = QAction("&CMA-ES", self)
        cmaes_action.triggered.connect(self.run_cmaes)
        optimization_menu.addAction(cmaes_action)

        # Playground menu
        playground_menu = menubar.addMenu("&Playground")

        clone_action = QAction("&New Playground Window...", self)
        clone_action.setShortcut("Ctrl+Shift+N")
        clone_action.triggered.connect(self.open_playground_clone)
        playground_menu.addAction(clone_action)

        rename_action = QAction("&Rename This Window...", self)
        rename_action.setShortcut("F2")
        rename_action.triggered.connect(self.rename_instance_window)
        playground_menu.addAction(rename_action)

        tile_action = QAction("&Tile Windows", self)
        tile_action.setShortcut("Ctrl+Shift+T")
        tile_action.triggered.connect(self.tile_playground_windows)
        playground_menu.addAction(tile_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # Documentation
        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(lambda: self.status_bar.showMessage("Documentation - Feature coming soon"))
        help_menu.addAction(docs_action)

        # About
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\n\n"
            "A modern application for designing and optimizing vibration systems.\n\n"
            "© 2023 DeVana Team\n"
            "All rights reserved."
        )

    def create_toolbar(self):
        """Create the application toolbar with modern styling"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Add spacer at the beginning
        spacer = QWidget()
        spacer.setFixedWidth(10)
        toolbar.addWidget(spacer)

        # New Project button
        new_button = QPushButton("New Project")
        new_button.setObjectName("toolbar-button")
        new_button.setToolTip("Create a new project")
        new_button.clicked.connect(lambda: self.status_bar.showMessage("New Project - Feature coming soon"))
        toolbar.addWidget(new_button)

        # Open Project button
        open_button = QPushButton("Open Project")
        open_button.setObjectName("toolbar-button")
        open_button.setToolTip("Open an existing project")
        open_button.clicked.connect(lambda: self.status_bar.showMessage("Open Project - Feature coming soon"))
        toolbar.addWidget(open_button)

        # Save Project button
        save_button = QPushButton("Save Project")
        save_button.setObjectName("toolbar-button")
        save_button.setToolTip("Save the current project")
        save_button.clicked.connect(lambda: self.status_bar.showMessage("Save Project - Feature coming soon"))
        toolbar.addWidget(save_button)

        # Add separator
        toolbar.addSeparator()

        # Run FRF button
        run_frf_button = QPushButton("Run FRF")
        run_frf_button.setObjectName("primary-button")
        run_frf_button.setToolTip("Run FRF Analysis")
        run_frf_button.clicked.connect(self.run_frf)
        run_frf_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_frf_button)

        # Run Sobol button
        run_sobol_button = QPushButton("Run Sobol")
        run_sobol_button.setObjectName("primary-button")
        run_sobol_button.setToolTip("Run Sobol Analysis")
        run_sobol_button.clicked.connect(self._run_sobol_implementation)
        run_sobol_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_sobol_button)

        # Run PSO button
        run_pso_button = QPushButton("Run PSO")
        run_pso_button.setObjectName("primary-button")
        run_pso_button.setToolTip("Run Particle Swarm Optimization")
        run_pso_button.clicked.connect(self.run_pso)
        run_pso_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_pso_button)

        # Run DE button
        run_de_button = QPushButton("Run DE")
        run_de_button.setObjectName("primary-button")
        run_de_button.setToolTip("Run Differential Evolution")
        run_de_button.clicked.connect(self.run_de)
        run_de_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_de_button)

        # Run SA button
        run_sa_button = QPushButton("Run SA")
        run_sa_button.setObjectName("primary-button")
        run_sa_button.setToolTip("Run Simulated Annealing")
        run_sa_button.clicked.connect(self.run_sa)
        run_sa_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_sa_button)

        # Run CMA-ES button
        run_cmaes_button = QPushButton("Run CMA-ES")
        run_cmaes_button.setObjectName("primary-button")
        run_cmaes_button.setToolTip("Run CMA-ES Optimization")
        run_cmaes_button.clicked.connect(self.run_cmaes)
        run_cmaes_button.setVisible(False)  # Hide button
        toolbar.addWidget(run_cmaes_button)

        # Add separator
        toolbar.addSeparator()

        # Theme toggle
        theme_button = QPushButton("Toggle Theme")
        theme_button.setObjectName("toolbar-button")
        theme_button.setToolTip(f"Switch to {'Light' if self.current_theme == 'Dark' else 'Dark'} Theme")
        theme_button.clicked.connect(self.toggle_theme)
        toolbar.addWidget(theme_button)

        # Add expanding spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Playground quick actions
        new_pg_button = QPushButton("New Window")
        new_pg_button.setObjectName("toolbar-button")
        new_pg_button.setToolTip("Open a new Playground window cloned from this one")
        new_pg_button.clicked.connect(self.open_playground_clone)
        toolbar.addWidget(new_pg_button)

        tile_button = QPushButton("Tile Windows")
        tile_button.setObjectName("toolbar-button")
        tile_button.setToolTip("Tile all Playground windows on screen")
        tile_button.clicked.connect(self.tile_playground_windows)
        toolbar.addWidget(tile_button)

    def switch_theme(self, theme):
        """Switch the application theme"""
        self.current_theme = theme
        if theme == 'Dark':
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

        # Update theme toggle button tooltip
        for action in self.findChildren(QAction):
            if action.text() == "Toggle &Theme":
                action.setToolTip(f"Switch to {'Light' if theme == 'Dark' else 'Dark'} Theme")

    def import_parameters(self):
        """Import parameters from a JSON file"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Parameters",
                "",
                "JSON Files (*.json);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            # Read JSON file
            with open(file_path, 'r') as f:
                params = json.load(f)
                
            # Update main system parameters
            if 'main_system' in params:
                main_params = params['main_system']
                if 'mu' in main_params:
                    self.mu_box.setValue(main_params['mu'])
                if 'landa' in main_params:
                    for i, val in enumerate(main_params['landa']):
                        if i < len(self.landa_boxes):
                            self.landa_boxes[i].setValue(val)
                if 'nu' in main_params:
                    for i, val in enumerate(main_params['nu']):
                        if i < len(self.nu_boxes):
                            self.nu_boxes[i].setValue(val)
                # Additional main system fields
                if 'a_low' in main_params and hasattr(self, 'a_low_box'):
                    self.a_low_box.setValue(main_params['a_low'])
                if 'a_up' in main_params and hasattr(self, 'a_up_box'):
                    self.a_up_box.setValue(main_params['a_up'])
                if 'f_1' in main_params and hasattr(self, 'f_1_box'):
                    self.f_1_box.setValue(main_params['f_1'])
                if 'f_2' in main_params and hasattr(self, 'f_2_box'):
                    self.f_2_box.setValue(main_params['f_2'])
                if 'omega_dc' in main_params and hasattr(self, 'omega_dc_box'):
                    self.omega_dc_box.setValue(main_params['omega_dc'])
                if 'zeta_dc' in main_params and hasattr(self, 'zeta_dc_box'):
                    self.zeta_dc_box.setValue(main_params['zeta_dc'])
                            
            # Update DVA parameters
            if 'dva' in params:
                dva_params = params['dva']
                if 'beta' in dva_params:
                    for i, val in enumerate(dva_params['beta']):
                        if i < len(self.beta_boxes):
                            self.beta_boxes[i].setValue(val)
                if 'lambda' in dva_params:
                    for i, val in enumerate(dva_params['lambda']):
                        if i < len(self.lambda_boxes):
                            self.lambda_boxes[i].setValue(val)
                if 'mu' in dva_params:
                    for i, val in enumerate(dva_params['mu']):
                        if i < len(self.mu_dva_boxes):
                            self.mu_dva_boxes[i].setValue(val)
                if 'nu' in dva_params:
                    for i, val in enumerate(dva_params['nu']):
                        if i < len(self.nu_dva_boxes):
                            self.nu_dva_boxes[i].setValue(val)
                            
            # Update frequency parameters
            if 'frequency' in params:
                freq_params = params['frequency']
                if 'omega_start' in freq_params:
                    self.omega_start_box.setValue(freq_params['omega_start'])
                if 'omega_end' in freq_params:
                    self.omega_end_box.setValue(freq_params['omega_end'])
                if 'omega_points' in freq_params:
                    self.omega_points_box.setValue(freq_params['omega_points'])
                # Plot and interpolation options (if present)
                if 'plot_figure' in freq_params and hasattr(self, 'plot_figure_chk'):
                    self.plot_figure_chk.setChecked(bool(freq_params['plot_figure']))
                if 'show_peaks' in freq_params and hasattr(self, 'show_peaks_chk'):
                    self.show_peaks_chk.setChecked(bool(freq_params['show_peaks']))
                if 'show_slopes' in freq_params and hasattr(self, 'show_slopes_chk'):
                    self.show_slopes_chk.setChecked(bool(freq_params['show_slopes']))
                if 'interpolation_method' in freq_params and hasattr(self, 'interp_method_combo'):
                    try:
                        self.interp_method_combo.setCurrentText(str(freq_params['interpolation_method']))
                    except Exception:
                        pass
                if 'interpolation_points' in freq_params and hasattr(self, 'interp_points_box'):
                    self.interp_points_box.setValue(int(freq_params['interpolation_points']))

            # Update target values and weights
            try:
                if 'targets' in params and 'weights' in params and \
                   hasattr(self, 'mass_target_spins') and hasattr(self, 'mass_weight_spins'):
                    targets = params['targets']
                    weights = params['weights']
                    # targets/weights schema: {"mass_1": {key: value}, ...}
                    for mass_idx in range(1, 6):
                        mass_key = f"mass_{mass_idx}"
                        t_mass = targets.get(mass_key, {}) if isinstance(targets, dict) else {}
                        w_mass = weights.get(mass_key, {}) if isinstance(weights, dict) else {}
                        # Set all known keys present in UI maps
                        for key, spin in self.mass_target_spins.items():
                            if key.endswith(f"_m{mass_idx}"):
                                simple_key = key.replace(f"_m{mass_idx}", "")
                                if simple_key in t_mass:
                                    try:
                                        spin.setValue(float(t_mass[simple_key]))
                                    except Exception:
                                        pass
                        for key, spin in self.mass_weight_spins.items():
                            if key.endswith(f"_m{mass_idx}"):
                                simple_key = key.replace(f"_m{mass_idx}", "")
                                if simple_key in w_mass:
                                    try:
                                        spin.setValue(float(w_mass[simple_key]))
                                    except Exception:
                                        pass
            except Exception:
                # Be permissive with targets/weights import
                pass

            # Update omega sensitivity settings
            if 'omega_sensitivity' in params:
                sens = params['omega_sensitivity']
                if hasattr(self, 'sensitivity_initial_points') and 'initial_points' in sens:
                    self.sensitivity_initial_points.setValue(int(sens['initial_points']))
                if hasattr(self, 'sensitivity_max_points') and 'max_points' in sens:
                    self.sensitivity_max_points.setValue(int(sens['max_points']))
                if hasattr(self, 'sensitivity_step_size') and 'step_size' in sens:
                    self.sensitivity_step_size.setValue(int(sens['step_size']))
                if hasattr(self, 'sensitivity_threshold') and 'threshold' in sens:
                    self.sensitivity_threshold.setValue(float(sens['threshold']))
                if hasattr(self, 'sensitivity_max_iterations') and 'max_iterations' in sens:
                    self.sensitivity_max_iterations.setValue(int(sens['max_iterations']))
                if hasattr(self, 'sensitivity_mass') and 'mass' in sens:
                    try:
                        self.sensitivity_mass.setCurrentText(str(sens['mass']))
                    except Exception:
                        pass
                if hasattr(self, 'sensitivity_plot_results') and 'plot_results' in sens:
                    self.sensitivity_plot_results.setChecked(bool(sens['plot_results']))
                if hasattr(self, 'sensitivity_use_optimal') and 'use_optimal' in sens:
                    self.sensitivity_use_optimal.setChecked(bool(sens['use_optimal']))

            # Update GA settings
            if 'ga_settings' in params:
                ga = params['ga_settings']
                try:
                    # Basic hyperparameters
                    pop = ga.get('population', {}) if isinstance(ga.get('population', {}), dict) else {}
                    if hasattr(self, 'ga_pop_size_box') and 'size' in pop:
                        self.ga_pop_size_box.setValue(int(pop['size']))
                    if hasattr(self, 'ga_pop_min_box') and 'min' in pop:
                        self.ga_pop_min_box.setValue(int(pop['min']))
                    if hasattr(self, 'ga_pop_max_box') and 'max' in pop:
                        self.ga_pop_max_box.setValue(int(pop['max']))
                    if hasattr(self, 'ga_num_generations_box') and 'generations' in ga:
                        self.ga_num_generations_box.setValue(int(ga['generations']))
                    if hasattr(self, 'ga_cxpb_box') and 'crossover_probability' in ga:
                        self.ga_cxpb_box.setValue(float(ga['crossover_probability']))
                    if hasattr(self, 'ga_mutpb_box') and 'mutation_probability' in ga:
                        self.ga_mutpb_box.setValue(float(ga['mutation_probability']))
                    if hasattr(self, 'ga_tol_box') and 'tolerance' in ga:
                        self.ga_tol_box.setValue(float(ga['tolerance']))
                    if hasattr(self, 'ga_alpha_box') and 'alpha_sparsity' in ga:
                        self.ga_alpha_box.setValue(float(ga['alpha_sparsity']))
                    if hasattr(self, 'ga_percentage_error_scale_box') and 'percentage_error_scale' in ga:
                        self.ga_percentage_error_scale_box.setValue(float(ga['percentage_error_scale']))
                    if hasattr(self, 'ga_benchmark_runs_box') and 'benchmark_runs' in ga:
                        self.ga_benchmark_runs_box.setValue(int(ga['benchmark_runs']))

                    # Controller selection
                    controller = str(ga.get('controller', 'fixed')).lower()
                    if hasattr(self, 'controller_none_radio') and controller == 'fixed':
                        self.controller_none_radio.setChecked(True)
                    if hasattr(self, 'controller_adaptive_radio') and controller == 'adaptive':
                        self.controller_adaptive_radio.setChecked(True)
                    if hasattr(self, 'controller_ml_radio') and controller == 'ml':
                        self.controller_ml_radio.setChecked(True)
                    if hasattr(self, 'controller_rl_radio') and controller == 'rl':
                        self.controller_rl_radio.setChecked(True)

                    # Adaptive rates
                    adaptive = ga.get('adaptive_rates', {}) if isinstance(ga.get('adaptive_rates', {}), dict) else {}
                    if hasattr(self, 'adaptive_rates_checkbox') and 'enabled' in adaptive:
                        self.adaptive_rates_checkbox.setChecked(bool(adaptive['enabled']))
                    if hasattr(self, 'stagnation_limit_box') and 'stagnation_limit' in adaptive:
                        self.stagnation_limit_box.setValue(int(adaptive['stagnation_limit']))
                    if hasattr(self, 'cxpb_min_box') and 'cxpb_min' in adaptive:
                        self.cxpb_min_box.setValue(float(adaptive['cxpb_min']))
                    if hasattr(self, 'cxpb_max_box') and 'cxpb_max' in adaptive:
                        self.cxpb_max_box.setValue(float(adaptive['cxpb_max']))
                    if hasattr(self, 'mutpb_min_box') and 'mutpb_min' in adaptive:
                        self.mutpb_min_box.setValue(float(adaptive['mutpb_min']))
                    if hasattr(self, 'mutpb_max_box') and 'mutpb_max' in adaptive:
                        self.mutpb_max_box.setValue(float(adaptive['mutpb_max']))

                    # ML controller options
                    ml = ga.get('ml_controller', {}) if isinstance(ga.get('ml_controller', {}), dict) else {}
                    if hasattr(self, 'ml_controller_checkbox') and 'enabled' in ml:
                        self.ml_controller_checkbox.setChecked(bool(ml['enabled']))
                    if hasattr(self, 'ml_pop_adapt_checkbox') and 'adapt_population' in ml:
                        self.ml_pop_adapt_checkbox.setChecked(bool(ml['adapt_population']))
                    if hasattr(self, 'ml_ucb_c_box') and 'ucb_c' in ml:
                        self.ml_ucb_c_box.setValue(float(ml['ucb_c']))
                    if hasattr(self, 'ml_diversity_weight_box') and 'diversity_weight' in ml:
                        self.ml_diversity_weight_box.setValue(float(ml['diversity_weight']))
                    if hasattr(self, 'ml_diversity_target_box') and 'diversity_target' in ml:
                        self.ml_diversity_target_box.setValue(float(ml['diversity_target']))
                    if hasattr(self, 'ml_historical_weight_box') and 'historical_weight' in ml:
                        self.ml_historical_weight_box.setValue(float(ml['historical_weight']))
                    if hasattr(self, 'ml_current_weight_box') and 'current_weight' in ml:
                        self.ml_current_weight_box.setValue(float(ml['current_weight']))

                    # RL controller options
                    rl = ga.get('rl_controller', {}) if isinstance(ga.get('rl_controller', {}), dict) else {}
                    if hasattr(self, 'rl_alpha_box') and 'alpha' in rl:
                        self.rl_alpha_box.setValue(float(rl['alpha']))
                    if hasattr(self, 'rl_gamma_box') and 'gamma' in rl:
                        self.rl_gamma_box.setValue(float(rl['gamma']))
                    if hasattr(self, 'rl_epsilon_box') and 'epsilon' in rl:
                        self.rl_epsilon_box.setValue(float(rl['epsilon']))
                    if hasattr(self, 'rl_decay_box') and 'epsilon_decay' in rl:
                        self.rl_decay_box.setValue(float(rl['epsilon_decay']))

                    # Surrogate options
                    surr = ga.get('surrogate', {}) if isinstance(ga.get('surrogate', {}), dict) else {}
                    if hasattr(self, 'surrogate_checkbox') and 'enabled' in surr:
                        self.surrogate_checkbox.setChecked(bool(surr['enabled']))
                    if hasattr(self, 'surr_pool_factor_box') and 'pool_factor' in surr:
                        self.surr_pool_factor_box.setValue(float(surr['pool_factor']))
                    if hasattr(self, 'surr_k_box') and 'k' in surr:
                        self.surr_k_box.setValue(int(surr['k']))
                    if hasattr(self, 'surr_explore_frac_box') and 'explore_fraction' in surr:
                        self.surr_explore_frac_box.setValue(float(surr['explore_fraction']))

                    # Seeding and neural options
                    seed = ga.get('seeding', {}) if isinstance(ga.get('seeding', {}), dict) else {}
                    if hasattr(self, 'seeding_method_combo') and 'method' in seed:
                        try:
                            self.seeding_method_combo.setCurrentText(str(seed['method']))
                        except Exception:
                            pass
                    neural = seed.get('neural', {}) if isinstance(seed.get('neural', {}), dict) else {}
                    if hasattr(self, 'neural_options_group') and 'enabled' in neural:
                        try:
                            self.neural_options_group.setChecked(bool(neural['enabled']))
                        except Exception:
                            pass
                    if hasattr(self, 'neural_acq_combo') and 'acquisition' in neural:
                        try:
                            self.neural_acq_combo.setCurrentText(str(neural['acquisition']))
                        except Exception:
                            pass
                    if hasattr(self, 'neural_beta_min') and 'beta_min' in neural:
                        self.neural_beta_min.setValue(float(neural['beta_min']))
                    if hasattr(self, 'neural_beta_max') and 'beta_max' in neural:
                        self.neural_beta_max.setValue(float(neural['beta_max']))
                    if hasattr(self, 'neural_eps') and 'epsilon' in neural:
                        self.neural_eps.setValue(float(neural['epsilon']))
                    if hasattr(self, 'neural_pool_mult') and 'pool_mult' in neural:
                        self.neural_pool_mult.setValue(float(neural['pool_mult']))
                    if hasattr(self, 'neural_ensemble') and 'ensemble' in neural:
                        self.neural_ensemble.setValue(int(neural['ensemble']))
                    if hasattr(self, 'neural_layers') and 'layers' in neural:
                        self.neural_layers.setValue(int(neural['layers']))
                    if hasattr(self, 'neural_hidden') and 'hidden' in neural:
                        self.neural_hidden.setValue(int(neural['hidden']))
                    if hasattr(self, 'neural_dropout') and 'dropout' in neural:
                        self.neural_dropout.setValue(float(neural['dropout']))
                    if hasattr(self, 'neural_wd') and 'weight_decay' in neural:
                        self.neural_wd.setValue(float(neural['weight_decay']))
                    if hasattr(self, 'neural_epochs') and 'epochs' in neural:
                        self.neural_epochs.setValue(int(neural['epochs']))
                    if hasattr(self, 'neural_time_cap') and 'time_cap_ms' in neural:
                        self.neural_time_cap.setValue(int(neural['time_cap_ms']))
                    if hasattr(self, 'neural_grad_refine_chk') and 'grad_refine' in neural:
                        self.neural_grad_refine_chk.setChecked(bool(neural['grad_refine']))
                    if hasattr(self, 'neural_grad_steps') and 'grad_steps' in neural:
                        self.neural_grad_steps.setValue(int(neural['grad_steps']))
                    if hasattr(self, 'neural_device_combo') and 'device' in neural:
                        try:
                            self.neural_device_combo.setCurrentText(str(neural['device']))
                        except Exception:
                            pass
                except Exception:
                    # Be permissive when some GA fields are missing
                    pass

            # Update GA parameter bounds/fixed
            if 'ga_parameters' in params and hasattr(self, 'ga_param_table'):
                try:
                    gp = params['ga_parameters']
                    bounds = gp.get('bounds', {}) if isinstance(gp.get('bounds', {}), dict) else {}
                    # Build name->row map
                    name_to_row = {}
                    for r in range(self.ga_param_table.rowCount()):
                        nm_item = self.ga_param_table.item(r, 0)
                        if nm_item:
                            name_to_row[nm_item.text()] = r
                    for pname, pinfo in bounds.items():
                        if pname not in name_to_row or not isinstance(pinfo, dict):
                            continue
                        row = name_to_row[pname]
                        fixed_chk = self.ga_param_table.cellWidget(row, 1)
                        if fixed_chk:
                            if bool(pinfo.get('fixed', False)):
                                fixed_chk.setChecked(True)
                                # Set fixed value if available
                                if 'fixed_value' in pinfo:
                                    fv_w = self.ga_param_table.cellWidget(row, 2)
                                    if fv_w:
                                        try:
                                            fv_w.setValue(float(pinfo['fixed_value']))
                                        except Exception:
                                            pass
                            else:
                                fixed_chk.setChecked(False)
                                lo_w = self.ga_param_table.cellWidget(row, 3)
                                hi_w = self.ga_param_table.cellWidget(row, 4)
                                if lo_w and 'lower' in pinfo:
                                    try:
                                        lo_w.setValue(float(pinfo['lower']))
                                    except Exception:
                                        pass
                                if hi_w and 'upper' in pinfo:
                                    try:
                                        hi_w.setValue(float(pinfo['upper']))
                                    except Exception:
                                        pass
                except Exception:
                    pass
                    
            self.status_bar.showMessage("Parameters imported successfully", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import parameters: {str(e)}"
            )
            
    def export_parameters(self):
        """Export parameters to a JSON file"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Parameters",
                "",
                "JSON Files (*.json);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            # Collect parameters
            # Targets and weights
            try:
                target_values_dict, weights_dict = self.get_target_values_weights()
            except Exception:
                target_values_dict, weights_dict = {}, {}

            # GA parameter bounds/fixed from table
            ga_param_bounds = {}
            try:
                if hasattr(self, 'ga_param_table') and self.ga_param_table is not None:
                    for row in range(self.ga_param_table.rowCount()):
                        name_item = self.ga_param_table.item(row, 0)
                        if not name_item:
                            continue
                        pname = name_item.text()
                        fixed_widget = self.ga_param_table.cellWidget(row, 1)
                        is_fixed = bool(fixed_widget.isChecked()) if fixed_widget else False
                        if is_fixed:
                            fv_w = self.ga_param_table.cellWidget(row, 2)
                            fv = fv_w.value() if fv_w else 0.0
                            ga_param_bounds[pname] = {
                                'fixed': True,
                                'fixed_value': float(fv)
                            }
                        else:
                            lo_w = self.ga_param_table.cellWidget(row, 3)
                            hi_w = self.ga_param_table.cellWidget(row, 4)
                            lb = lo_w.value() if lo_w else 0.0
                            ub = hi_w.value() if hi_w else 0.0
                            ga_param_bounds[pname] = {
                                'fixed': False,
                                'lower': float(lb),
                                'upper': float(ub)
                            }
            except Exception:
                ga_param_bounds = {}

            # GA hyperparameters and options
            ga_settings = {}
            try:
                ga_settings = {
                    'population': {
                        'size': int(self.ga_pop_size_box.value()),
                        'min': int(self.ga_pop_min_box.value()),
                        'max': int(self.ga_pop_max_box.value()),
                    },
                    'generations': int(self.ga_num_generations_box.value()),
                    'crossover_probability': float(self.ga_cxpb_box.value()),
                    'mutation_probability': float(self.ga_mutpb_box.value()),
                    'tolerance': float(self.ga_tol_box.value()),
                    'alpha_sparsity': float(self.ga_alpha_box.value()),
                    'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()),
                    'benchmark_runs': int(self.ga_benchmark_runs_box.value()),
                    'controller': (
                        'adaptive' if self.controller_adaptive_radio.isChecked() else
                        ('ml' if self.controller_ml_radio.isChecked() else
                         ('rl' if self.controller_rl_radio.isChecked() else 'fixed'))
                    ),
                    'adaptive_rates': {
                        'enabled': bool(self.adaptive_rates_checkbox.isChecked()),
                        'stagnation_limit': int(self.stagnation_limit_box.value()),
                        'cxpb_min': float(self.cxpb_min_box.value()),
                        'cxpb_max': float(self.cxpb_max_box.value()),
                        'mutpb_min': float(self.mutpb_min_box.value()),
                        'mutpb_max': float(self.mutpb_max_box.value()),
                    },
                    'ml_controller': {
                        'enabled': bool(self.ml_controller_checkbox.isChecked()),
                        'adapt_population': bool(self.ml_pop_adapt_checkbox.isChecked()),
                        'ucb_c': float(self.ml_ucb_c_box.value()),
                        'diversity_weight': float(self.ml_diversity_weight_box.value()),
                        'diversity_target': float(self.ml_diversity_target_box.value()),
                        'historical_weight': float(self.ml_historical_weight_box.value()),
                        'current_weight': float(self.ml_current_weight_box.value()),
                    },
                    'rl_controller': {
                        'alpha': float(self.rl_alpha_box.value()),
                        'gamma': float(self.rl_gamma_box.value()),
                        'epsilon': float(self.rl_epsilon_box.value()),
                        'epsilon_decay': float(self.rl_decay_box.value()),
                    },
                    'surrogate': {
                        'enabled': bool(self.surrogate_checkbox.isChecked()),
                        'pool_factor': float(self.surr_pool_factor_box.value()),
                        'k': int(self.surr_k_box.value()),
                        'explore_fraction': float(self.surr_explore_frac_box.value()),
                    },
                    'seeding': {
                        'method': str(self.seeding_method_combo.currentText()),
                        'neural': {
                            'enabled': bool(self.seeding_method_combo.currentText().lower().startswith('neural')),
                            'acquisition': str(self.neural_acq_combo.currentText()),
                            'beta_min': float(self.neural_beta_min.value()),
                            'beta_max': float(self.neural_beta_max.value()),
                            'epsilon': float(self.neural_eps.value()),
                            'pool_mult': float(self.neural_pool_mult.value()),
                            'ensemble': int(self.neural_ensemble.value()),
                            'layers': int(self.neural_layers.value()),
                            'hidden': int(self.neural_hidden.value()),
                            'dropout': float(self.neural_dropout.value()),
                            'weight_decay': float(self.neural_wd.value()),
                            'epochs': int(self.neural_epochs.value()),
                            'time_cap_ms': int(self.neural_time_cap.value()),
                            'grad_refine': bool(self.neural_grad_refine_chk.isChecked()),
                            'grad_steps': int(self.neural_grad_steps.value()),
                            'device': str(self.neural_device_combo.currentText()),
                        }
                    }
                }
            except Exception:
                ga_settings = {}

            params = {}

            # Reuse helper to collect current parameters
            try:
                params = self._collect_parameters_to_dict()
            except Exception:
                params = {
                    'main_system': {
                        'mu': self.mu_box.value(),
                        'landa': [box.value() for box in self.landa_boxes],
                        'nu': [box.value() for box in self.nu_boxes],
                        'a_low': self.a_low_box.value() if hasattr(self, 'a_low_box') else None,
                        'a_up': self.a_up_box.value() if hasattr(self, 'a_up_box') else None,
                        'f_1': self.f_1_box.value() if hasattr(self, 'f_1_box') else None,
                        'f_2': self.f_2_box.value() if hasattr(self, 'f_2_box') else None,
                        'omega_dc': self.omega_dc_box.value() if hasattr(self, 'omega_dc_box') else None,
                        'zeta_dc': self.zeta_dc_box.value() if hasattr(self, 'zeta_dc_box') else None,
                    },
                    'dva': {
                        'beta': [box.value() for box in self.beta_boxes],
                        'lambda': [box.value() for box in self.lambda_boxes],
                        'mu': [box.value() for box in self.mu_dva_boxes],
                        'nu': [box.value() for box in self.nu_dva_boxes]
                    },
                    'targets': target_values_dict,
                    'weights': weights_dict,
                    'frequency': {
                        'omega_start': self.omega_start_box.value(),
                        'omega_end': self.omega_end_box.value(),
                        'omega_points': self.omega_points_box.value(),
                        'plot_figure': bool(self.plot_figure_chk.isChecked()) if hasattr(self, 'plot_figure_chk') else None,
                        'show_peaks': bool(self.show_peaks_chk.isChecked()) if hasattr(self, 'show_peaks_chk') else None,
                        'show_slopes': bool(self.show_slopes_chk.isChecked()) if hasattr(self, 'show_slopes_chk') else None,
                        'interpolation_method': str(self.interp_method_combo.currentText()) if hasattr(self, 'interp_method_combo') else None,
                        'interpolation_points': int(self.interp_points_box.value()) if hasattr(self, 'interp_points_box') else None,
                    },
                    'omega_sensitivity': {
                        'initial_points': int(self.sensitivity_initial_points.value()) if hasattr(self, 'sensitivity_initial_points') else None,
                        'max_points': int(self.sensitivity_max_points.value()) if hasattr(self, 'sensitivity_max_points') else None,
                        'step_size': int(self.sensitivity_step_size.value()) if hasattr(self, 'sensitivity_step_size') else None,
                        'threshold': float(self.sensitivity_threshold.value()) if hasattr(self, 'sensitivity_threshold') else None,
                        'max_iterations': int(self.sensitivity_max_iterations.value()) if hasattr(self, 'sensitivity_max_iterations') else None,
                        'mass': str(self.sensitivity_mass.currentText()) if hasattr(self, 'sensitivity_mass') else None,
                        'plot_results': bool(self.sensitivity_plot_results.isChecked()) if hasattr(self, 'sensitivity_plot_results') else None,
                        'use_optimal': bool(self.sensitivity_use_optimal.isChecked()) if hasattr(self, 'sensitivity_use_optimal') else None,
                    },
                    'ga_settings': ga_settings,
                    'ga_parameters': {
                        'bounds': ga_param_bounds
                    }
                }
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
                
            self.status_bar.showMessage("Parameters exported successfully", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export parameters: {str(e)}"
            )

    # ---------------------------
    # Helpers for Playground cloning
    # ---------------------------
    def _collect_parameters_to_dict(self):
        """Collect the current UI state into a dict for cloning/export."""
        try:
            # Targets and weights
            try:
                target_values_dict, weights_dict = self.get_target_values_weights()
            except Exception:
                target_values_dict, weights_dict = {}, {}

            # GA param bounds
            ga_param_bounds = {}
            try:
                if hasattr(self, 'ga_param_table') and self.ga_param_table is not None:
                    for row in range(self.ga_param_table.rowCount()):
                        name_item = self.ga_param_table.item(row, 0)
                        if not name_item:
                            continue
                        pname = name_item.text()
                        fixed_widget = self.ga_param_table.cellWidget(row, 1)
                        is_fixed = bool(fixed_widget.isChecked()) if fixed_widget else False
                        if is_fixed:
                            fv_w = self.ga_param_table.cellWidget(row, 2)
                            fv = fv_w.value() if fv_w else 0.0
                            ga_param_bounds[pname] = {
                                'fixed': True,
                                'fixed_value': float(fv)
                            }
                        else:
                            lo_w = self.ga_param_table.cellWidget(row, 3)
                            hi_w = self.ga_param_table.cellWidget(row, 4)
                            lb = lo_w.value() if lo_w else 0.0
                            ub = hi_w.value() if hi_w else 0.0
                            ga_param_bounds[pname] = {
                                'fixed': False,
                                'lower': float(lb),
                                'upper': float(ub)
                            }
            except Exception:
                ga_param_bounds = {}

            # GA hyperparameters and options
            ga_settings = {}
            try:
                ga_settings = {
                    'population': {
                        'size': int(self.ga_pop_size_box.value()),
                        'min': int(self.ga_pop_min_box.value()),
                        'max': int(self.ga_pop_max_box.value()),
                    },
                    'generations': int(self.ga_num_generations_box.value()),
                    'crossover_probability': float(self.ga_cxpb_box.value()),
                    'mutation_probability': float(self.ga_mutpb_box.value()),
                    'tolerance': float(self.ga_tol_box.value()),
                    'alpha_sparsity': float(self.ga_alpha_box.value()),
                    'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()),
                    'benchmark_runs': int(self.ga_benchmark_runs_box.value()),
                    'controller': (
                        'adaptive' if self.controller_adaptive_radio.isChecked() else
                        ('ml' if self.controller_ml_radio.isChecked() else
                         ('rl' if self.controller_rl_radio.isChecked() else 'fixed'))
                    ),
                    'adaptive_rates': {
                        'enabled': bool(self.adaptive_rates_checkbox.isChecked()),
                        'stagnation_limit': int(self.stagnation_limit_box.value()),
                        'cxpb_min': float(self.cxpb_min_box.value()),
                        'cxpb_max': float(self.cxpb_max_box.value()),
                        'mutpb_min': float(self.mutpb_min_box.value()),
                        'mutpb_max': float(self.mutpb_max_box.value()),
                    },
                    'ml_controller': {
                        'enabled': bool(self.ml_controller_checkbox.isChecked()),
                        'adapt_population': bool(self.ml_pop_adapt_checkbox.isChecked()),
                        'ucb_c': float(self.ml_ucb_c_box.value()),
                        'diversity_weight': float(self.ml_diversity_weight_box.value()),
                        'diversity_target': float(self.ml_diversity_target_box.value()),
                        'historical_weight': float(self.ml_historical_weight_box.value()),
                        'current_weight': float(self.ml_current_weight_box.value()),
                    },
                    'rl_controller': {
                        'alpha': float(self.rl_alpha_box.value()),
                        'gamma': float(self.rl_gamma_box.value()),
                        'epsilon': float(self.rl_epsilon_box.value()),
                        'epsilon_decay': float(self.rl_decay_box.value()),
                    },
                    'surrogate': {
                        'enabled': bool(self.surrogate_checkbox.isChecked()),
                        'pool_factor': float(self.surr_pool_factor_box.value()),
                        'k': int(self.surr_k_box.value()),
                        'explore_fraction': float(self.surr_explore_frac_box.value()),
                    },
                    'seeding': {
                        'method': str(self.seeding_method_combo.currentText()),
                        'neural': {
                            'enabled': bool(self.seeding_method_combo.currentText().lower().startswith('neural')),
                            'acquisition': str(self.neural_acq_combo.currentText()),
                            'beta_min': float(self.neural_beta_min.value()),
                            'beta_max': float(self.neural_beta_max.value()),
                            'epsilon': float(self.neural_eps.value()),
                            'pool_mult': float(self.neural_pool_mult.value()),
                            'ensemble': int(self.neural_ensemble.value()),
                            'layers': int(self.neural_layers.value()),
                            'hidden': int(self.neural_hidden.value()),
                            'dropout': float(self.neural_dropout.value()),
                            'weight_decay': float(self.neural_wd.value()),
                            'epochs': int(self.neural_epochs.value()),
                            'time_cap_ms': int(self.neural_time_cap.value()),
                            'grad_refine': bool(self.neural_grad_refine_chk.isChecked()),
                            'grad_steps': int(self.neural_grad_steps.value()),
                            'device': str(self.neural_device_combo.currentText()),
                        }
                    }
                }
            except Exception:
                ga_settings = {}

            params = {
                'main_system': {
                    'mu': self.mu_box.value(),
                    'landa': [box.value() for box in self.landa_boxes],
                    'nu': [box.value() for box in self.nu_boxes],
                    'a_low': self.a_low_box.value() if hasattr(self, 'a_low_box') else None,
                    'a_up': self.a_up_box.value() if hasattr(self, 'a_up_box') else None,
                    'f_1': self.f_1_box.value() if hasattr(self, 'f_1_box') else None,
                    'f_2': self.f_2_box.value() if hasattr(self, 'f_2_box') else None,
                    'omega_dc': self.omega_dc_box.value() if hasattr(self, 'omega_dc_box') else None,
                    'zeta_dc': self.zeta_dc_box.value() if hasattr(self, 'zeta_dc_box') else None,
                },
                'dva': {
                    'beta': [box.value() for box in self.beta_boxes],
                    'lambda': [box.value() for box in self.lambda_boxes],
                    'mu': [box.value() for box in self.mu_dva_boxes],
                    'nu': [box.value() for box in self.nu_dva_boxes]
                },
                'targets': target_values_dict,
                'weights': weights_dict,
                'frequency': {
                    'omega_start': self.omega_start_box.value(),
                    'omega_end': self.omega_end_box.value(),
                    'omega_points': self.omega_points_box.value(),
                    'plot_figure': bool(self.plot_figure_chk.isChecked()) if hasattr(self, 'plot_figure_chk') else None,
                    'show_peaks': bool(self.show_peaks_chk.isChecked()) if hasattr(self, 'show_peaks_chk') else None,
                    'show_slopes': bool(self.show_slopes_chk.isChecked()) if hasattr(self, 'show_slopes_chk') else None,
                    'interpolation_method': str(self.interp_method_combo.currentText()) if hasattr(self, 'interp_method_combo') else None,
                    'interpolation_points': int(self.interp_points_box.value()) if hasattr(self, 'interp_points_box') else None,
                },
                'omega_sensitivity': {
                    'initial_points': int(self.sensitivity_initial_points.value()) if hasattr(self, 'sensitivity_initial_points') else None,
                    'max_points': int(self.sensitivity_max_points.value()) if hasattr(self, 'sensitivity_max_points') else None,
                    'step_size': int(self.sensitivity_step_size.value()) if hasattr(self, 'sensitivity_step_size') else None,
                    'threshold': float(self.sensitivity_threshold.value()) if hasattr(self, 'sensitivity_threshold') else None,
                    'max_iterations': int(self.sensitivity_max_iterations.value()) if hasattr(self, 'sensitivity_max_iterations') else None,
                    'mass': str(self.sensitivity_mass.currentText()) if hasattr(self, 'sensitivity_mass') else None,
                    'plot_results': bool(self.sensitivity_plot_results.isChecked()) if hasattr(self, 'sensitivity_plot_results') else None,
                    'use_optimal': bool(self.sensitivity_use_optimal.isChecked()) if hasattr(self, 'sensitivity_use_optimal') else None,
                },
                'ga_settings': ga_settings,
                'ga_parameters': {
                    'bounds': ga_param_bounds
                }
            }

            return params
        except Exception:
            return {}

    def _apply_parameters_from_dict(self, params):
        """Apply a parameters dict to this window (mirror of import)."""
        try:
            if not isinstance(params, dict):
                return

            # Update main system parameters
            if 'main_system' in params:
                main_params = params['main_system']
                if 'mu' in main_params:
                    self.mu_box.setValue(main_params['mu'])
                if 'landa' in main_params:
                    for i, val in enumerate(main_params['landa']):
                        if i < len(self.landa_boxes):
                            self.landa_boxes[i].setValue(val)
                if 'nu' in main_params:
                    for i, val in enumerate(main_params['nu']):
                        if i < len(self.nu_boxes):
                            self.nu_boxes[i].setValue(val)
                if 'a_low' in main_params and hasattr(self, 'a_low_box'):
                    self.a_low_box.setValue(main_params['a_low'])
                if 'a_up' in main_params and hasattr(self, 'a_up_box'):
                    self.a_up_box.setValue(main_params['a_up'])
                if 'f_1' in main_params and hasattr(self, 'f_1_box'):
                    self.f_1_box.setValue(main_params['f_1'])
                if 'f_2' in main_params and hasattr(self, 'f_2_box'):
                    self.f_2_box.setValue(main_params['f_2'])
                if 'omega_dc' in main_params and hasattr(self, 'omega_dc_box'):
                    self.omega_dc_box.setValue(main_params['omega_dc'])
                if 'zeta_dc' in main_params and hasattr(self, 'zeta_dc_box'):
                    self.zeta_dc_box.setValue(main_params['zeta_dc'])

            # Update DVA parameters
            if 'dva' in params:
                dva_params = params['dva']
                if 'beta' in dva_params:
                    for i, val in enumerate(dva_params['beta']):
                        if i < len(self.beta_boxes):
                            self.beta_boxes[i].setValue(val)
                if 'lambda' in dva_params:
                    for i, val in enumerate(dva_params['lambda']):
                        if i < len(self.lambda_boxes):
                            self.lambda_boxes[i].setValue(val)
                if 'mu' in dva_params:
                    for i, val in enumerate(dva_params['mu']):
                        if i < len(self.mu_dva_boxes):
                            self.mu_dva_boxes[i].setValue(val)
                if 'nu' in dva_params:
                    for i, val in enumerate(dva_params['nu']):
                        if i < len(self.nu_dva_boxes):
                            self.nu_dva_boxes[i].setValue(val)

            # Update frequency parameters
            if 'frequency' in params:
                freq_params = params['frequency']
                if 'omega_start' in freq_params:
                    self.omega_start_box.setValue(freq_params['omega_start'])
                if 'omega_end' in freq_params:
                    self.omega_end_box.setValue(freq_params['omega_end'])
                if 'omega_points' in freq_params:
                    self.omega_points_box.setValue(freq_params['omega_points'])
                if 'plot_figure' in freq_params and hasattr(self, 'plot_figure_chk'):
                    self.plot_figure_chk.setChecked(bool(freq_params['plot_figure']))
                if 'show_peaks' in freq_params and hasattr(self, 'show_peaks_chk'):
                    self.show_peaks_chk.setChecked(bool(freq_params['show_peaks']))
                if 'show_slopes' in freq_params and hasattr(self, 'show_slopes_chk'):
                    self.show_slopes_chk.setChecked(bool(freq_params['show_slopes']))
                if 'interpolation_method' in freq_params and hasattr(self, 'interp_method_combo'):
                    try:
                        self.interp_method_combo.setCurrentText(str(freq_params['interpolation_method']))
                    except Exception:
                        pass
                if 'interpolation_points' in freq_params and hasattr(self, 'interp_points_box'):
                    self.interp_points_box.setValue(int(freq_params['interpolation_points']))

            # Update targets and weights where possible
            try:
                if 'targets' in params and 'weights' in params and \
                   hasattr(self, 'mass_target_spins') and hasattr(self, 'mass_weight_spins'):
                    targets = params['targets']
                    weights = params['weights']
                    for mass_idx in range(1, 6):
                        mass_key = f"mass_{mass_idx}"
                        t_mass = targets.get(mass_key, {}) if isinstance(targets, dict) else {}
                        w_mass = weights.get(mass_key, {}) if isinstance(weights, dict) else {}
                        for key, spin in self.mass_target_spins.items():
                            if key.endswith(f"_m{mass_idx}"):
                                simple_key = key.replace(f"_m{mass_idx}", "")
                                if simple_key in t_mass:
                                    try:
                                        spin.setValue(float(t_mass[simple_key]))
                                    except Exception:
                                        pass
                        for key, spin in self.mass_weight_spins.items():
                            if key.endswith(f"_m{mass_idx}"):
                                simple_key = key.replace(f"_m{mass_idx}", "")
                                if simple_key in w_mass:
                                    try:
                                        spin.setValue(float(w_mass[simple_key]))
                                    except Exception:
                                        pass
            except Exception:
                pass

            # Update omega sensitivity settings
            if 'omega_sensitivity' in params:
                sens = params['omega_sensitivity']
                if hasattr(self, 'sensitivity_initial_points') and 'initial_points' in sens:
                    self.sensitivity_initial_points.setValue(int(sens['initial_points']))
                if hasattr(self, 'sensitivity_max_points') and 'max_points' in sens:
                    self.sensitivity_max_points.setValue(int(sens['max_points']))
                if hasattr(self, 'sensitivity_step_size') and 'step_size' in sens:
                    self.sensitivity_step_size.setValue(int(sens['step_size']))
                if hasattr(self, 'sensitivity_threshold') and 'threshold' in sens:
                    self.sensitivity_threshold.setValue(float(sens['threshold']))
                if hasattr(self, 'sensitivity_max_iterations') and 'max_iterations' in sens:
                    self.sensitivity_max_iterations.setValue(int(sens['max_iterations']))
                if hasattr(self, 'sensitivity_mass') and 'mass' in sens:
                    try:
                        self.sensitivity_mass.setCurrentText(str(sens['mass']))
                    except Exception:
                        pass
                if hasattr(self, 'sensitivity_plot_results') and 'plot_results' in sens:
                    self.sensitivity_plot_results.setChecked(bool(sens['plot_results']))
                if hasattr(self, 'sensitivity_use_optimal') and 'use_optimal' in sens:
                    self.sensitivity_use_optimal.setChecked(bool(sens['use_optimal']))

            # Update GA settings fields
            if 'ga_settings' in params:
                ga = params['ga_settings']
                try:
                    pop = ga.get('population', {}) if isinstance(ga.get('population', {}), dict) else {}
                    if hasattr(self, 'ga_pop_size_box') and 'size' in pop:
                        self.ga_pop_size_box.setValue(int(pop['size']))
                    if hasattr(self, 'ga_pop_min_box') and 'min' in pop:
                        self.ga_pop_min_box.setValue(int(pop['min']))
                    if hasattr(self, 'ga_pop_max_box') and 'max' in pop:
                        self.ga_pop_max_box.setValue(int(pop['max']))
                    if hasattr(self, 'ga_num_generations_box') and 'generations' in ga:
                        self.ga_num_generations_box.setValue(int(ga['generations']))
                    if hasattr(self, 'ga_cxpb_box') and 'crossover_probability' in ga:
                        self.ga_cxpb_box.setValue(float(ga['crossover_probability']))
                    if hasattr(self, 'ga_mutpb_box') and 'mutation_probability' in ga:
                        self.ga_mutpb_box.setValue(float(ga['mutation_probability']))
                    if hasattr(self, 'ga_tol_box') and 'tolerance' in ga:
                        self.ga_tol_box.setValue(float(ga['tolerance']))
                    if hasattr(self, 'ga_alpha_box') and 'alpha_sparsity' in ga:
                        self.ga_alpha_box.setValue(float(ga['alpha_sparsity']))
                    if hasattr(self, 'ga_percentage_error_scale_box') and 'percentage_error_scale' in ga:
                        self.ga_percentage_error_scale_box.setValue(float(ga['percentage_error_scale']))
                    if hasattr(self, 'ga_benchmark_runs_box') and 'benchmark_runs' in ga:
                        self.ga_benchmark_runs_box.setValue(int(ga['benchmark_runs']))

                    controller = str(ga.get('controller', 'fixed')).lower()
                    if hasattr(self, 'controller_none_radio') and controller == 'fixed':
                        self.controller_none_radio.setChecked(True)
                    if hasattr(self, 'controller_adaptive_radio') and controller == 'adaptive':
                        self.controller_adaptive_radio.setChecked(True)
                    if hasattr(self, 'controller_ml_radio') and controller == 'ml':
                        self.controller_ml_radio.setChecked(True)
                    if hasattr(self, 'controller_rl_radio') and controller == 'rl':
                        self.controller_rl_radio.setChecked(True)

                    adaptive = ga.get('adaptive_rates', {}) if isinstance(ga.get('adaptive_rates', {}), dict) else {}
                    if hasattr(self, 'adaptive_rates_checkbox') and 'enabled' in adaptive:
                        self.adaptive_rates_checkbox.setChecked(bool(adaptive['enabled']))
                    if hasattr(self, 'stagnation_limit_box') and 'stagnation_limit' in adaptive:
                        self.stagnation_limit_box.setValue(int(adaptive['stagnation_limit']))
                    if hasattr(self, 'cxpb_min_box') and 'cxpb_min' in adaptive:
                        self.cxpb_min_box.setValue(float(adaptive['cxpb_min']))
                    if hasattr(self, 'cxpb_max_box') and 'cxpb_max' in adaptive:
                        self.cxpb_max_box.setValue(float(adaptive['cxpb_max']))
                    if hasattr(self, 'mutpb_min_box') and 'mutpb_min' in adaptive:
                        self.mutpb_min_box.setValue(float(adaptive['mutpb_min']))
                    if hasattr(self, 'mutpb_max_box') and 'mutpb_max' in adaptive:
                        self.mutpb_max_box.setValue(float(adaptive['mutpb_max']))

                    ml = ga.get('ml_controller', {}) if isinstance(ga.get('ml_controller', {}), dict) else {}
                    if hasattr(self, 'ml_controller_checkbox') and 'enabled' in ml:
                        self.ml_controller_checkbox.setChecked(bool(ml['enabled']))
                    if hasattr(self, 'ml_pop_adapt_checkbox') and 'adapt_population' in ml:
                        self.ml_pop_adapt_checkbox.setChecked(bool(ml['adapt_population']))
                    if hasattr(self, 'ml_ucb_c_box') and 'ucb_c' in ml:
                        self.ml_ucb_c_box.setValue(float(ml['ucb_c']))
                    if hasattr(self, 'ml_diversity_weight_box') and 'diversity_weight' in ml:
                        self.ml_diversity_weight_box.setValue(float(ml['diversity_weight']))
                    if hasattr(self, 'ml_diversity_target_box') and 'diversity_target' in ml:
                        self.ml_diversity_target_box.setValue(float(ml['diversity_target']))
                    if hasattr(self, 'ml_historical_weight_box') and 'historical_weight' in ml:
                        self.ml_historical_weight_box.setValue(float(ml['historical_weight']))
                    if hasattr(self, 'ml_current_weight_box') and 'current_weight' in ml:
                        self.ml_current_weight_box.setValue(float(ml['current_weight']))

                    rl = ga.get('rl_controller', {}) if isinstance(ga.get('rl_controller', {}), dict) else {}
                    if hasattr(self, 'rl_alpha_box') and 'alpha' in rl:
                        self.rl_alpha_box.setValue(float(rl['alpha']))
                    if hasattr(self, 'rl_gamma_box') and 'gamma' in rl:
                        self.rl_gamma_box.setValue(float(rl['gamma']))
                    if hasattr(self, 'rl_epsilon_box') and 'epsilon' in rl:
                        self.rl_epsilon_box.setValue(float(rl['epsilon']))
                    if hasattr(self, 'rl_decay_box') and 'epsilon_decay' in rl:
                        self.rl_decay_box.setValue(float(rl['epsilon_decay']))

                    surr = ga.get('surrogate', {}) if isinstance(ga.get('surrogate', {}), dict) else {}
                    if hasattr(self, 'surrogate_checkbox') and 'enabled' in surr:
                        self.surrogate_checkbox.setChecked(bool(surr['enabled']))
                    if hasattr(self, 'surr_pool_factor_box') and 'pool_factor' in surr:
                        self.surr_pool_factor_box.setValue(float(surr['pool_factor']))
                    if hasattr(self, 'surr_k_box') and 'k' in surr:
                        self.surr_k_box.setValue(int(surr['k']))
                    if hasattr(self, 'surr_explore_frac_box') and 'explore_fraction' in surr:
                        self.surr_explore_frac_box.setValue(float(surr['explore_fraction']))

                    seed = ga.get('seeding', {}) if isinstance(ga.get('seeding', {}), dict) else {}
                    if hasattr(self, 'seeding_method_combo') and 'method' in seed:
                        try:
                            self.seeding_method_combo.setCurrentText(str(seed['method']))
                        except Exception:
                            pass
                    neural = seed.get('neural', {}) if isinstance(seed.get('neural', {}), dict) else {}
                    if hasattr(self, 'neural_options_group') and 'enabled' in neural:
                        try:
                            self.neural_options_group.setChecked(bool(neural['enabled']))
                        except Exception:
                            pass
                    if hasattr(self, 'neural_acq_combo') and 'acquisition' in neural:
                        try:
                            self.neural_acq_combo.setCurrentText(str(neural['acquisition']))
                        except Exception:
                            pass
                    if hasattr(self, 'neural_beta_min') and 'beta_min' in neural:
                        self.neural_beta_min.setValue(float(neural['beta_min']))
                    if hasattr(self, 'neural_beta_max') and 'beta_max' in neural:
                        self.neural_beta_max.setValue(float(neural['beta_max']))
                    if hasattr(self, 'neural_eps') and 'epsilon' in neural:
                        self.neural_eps.setValue(float(neural['epsilon']))
                    if hasattr(self, 'neural_pool_mult') and 'pool_mult' in neural:
                        self.neural_pool_mult.setValue(float(neural['pool_mult']))
                    if hasattr(self, 'neural_ensemble') and 'ensemble' in neural:
                        self.neural_ensemble.setValue(int(neural['ensemble']))
                    if hasattr(self, 'neural_layers') and 'layers' in neural:
                        self.neural_layers.setValue(int(neural['layers']))
                    if hasattr(self, 'neural_hidden') and 'hidden' in neural:
                        self.neural_hidden.setValue(int(neural['hidden']))
                    if hasattr(self, 'neural_dropout') and 'dropout' in neural:
                        self.neural_dropout.setValue(float(neural['dropout']))
                    if hasattr(self, 'neural_wd') and 'weight_decay' in neural:
                        self.neural_wd.setValue(float(neural['weight_decay']))
                    if hasattr(self, 'neural_epochs') and 'epochs' in neural:
                        self.neural_epochs.setValue(int(neural['epochs']))
                    if hasattr(self, 'neural_time_cap') and 'time_cap_ms' in neural:
                        self.neural_time_cap.setValue(int(neural['time_cap_ms']))
                    if hasattr(self, 'neural_grad_refine_chk') and 'grad_refine' in neural:
                        self.neural_grad_refine_chk.setChecked(bool(neural['grad_refine']))
                    if hasattr(self, 'neural_grad_steps') and 'grad_steps' in neural:
                        self.neural_grad_steps.setValue(int(neural['grad_steps']))
                    if hasattr(self, 'neural_device_combo') and 'device' in neural:
                        try:
                            self.neural_device_combo.setCurrentText(str(neural['device']))
                        except Exception:
                            pass
                except Exception:
                    pass

            # Update GA parameter bounds/fixed table
            if 'ga_parameters' in params and hasattr(self, 'ga_param_table'):
                try:
                    gp = params['ga_parameters']
                    bounds = gp.get('bounds', {}) if isinstance(gp.get('bounds', {}), dict) else {}
                    name_to_row = {}
                    for r in range(self.ga_param_table.rowCount()):
                        nm_item = self.ga_param_table.item(r, 0)
                        if nm_item:
                            name_to_row[nm_item.text()] = r
                    for pname, pinfo in bounds.items():
                        if pname not in name_to_row or not isinstance(pinfo, dict):
                            continue
                        row = name_to_row[pname]
                        fixed_chk = self.ga_param_table.cellWidget(row, 1)
                        if fixed_chk:
                            if bool(pinfo.get('fixed', False)):
                                fixed_chk.setChecked(True)
                                if 'fixed_value' in pinfo:
                                    fv_w = self.ga_param_table.cellWidget(row, 2)
                                    if fv_w:
                                        try:
                                            fv_w.setValue(float(pinfo['fixed_value']))
                                        except Exception:
                                            pass
                            else:
                                fixed_chk.setChecked(False)
                                lo_w = self.ga_param_table.cellWidget(row, 3)
                                hi_w = self.ga_param_table.cellWidget(row, 4)
                                if lo_w and 'lower' in pinfo:
                                    try:
                                        lo_w.setValue(float(pinfo['lower']))
                                    except Exception:
                                        pass
                                if hi_w and 'upper' in pinfo:
                                    try:
                                        hi_w.setValue(float(pinfo['upper']))
                                    except Exception:
                                        pass
                except Exception:
                    pass
        except Exception:
            pass
