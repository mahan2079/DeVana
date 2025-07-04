from PyQt5.QtWidgets import QAction, QActionGroup, QToolBar, QPushButton, QWidget, QSizePolicy, QMessageBox
from PyQt5.QtCore import QSize

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
            "About DeVana",
            "DeVana v2.0\n\n"
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
