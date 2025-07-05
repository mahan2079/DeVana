from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class SAOptimizationMixin:
    def create_sa_tab(self):
        """Create the simulated annealing optimization tab"""
        self.sa_tab = QWidget()
        layout = QVBoxLayout(self.sa_tab)
        
        # Create sub-tabs widget
        self.sa_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: SA Hyperparameters --------------------
        sa_hyper_tab = QWidget()
        sa_hyper_layout = QFormLayout(sa_hyper_tab)

        self.sa_initial_temp_box = QDoubleSpinBox()
        self.sa_initial_temp_box.setRange(0, 1e6)
        self.sa_initial_temp_box.setValue(1000)
        self.sa_initial_temp_box.setDecimals(2)

        self.sa_cooling_rate_box = QDoubleSpinBox()
        self.sa_cooling_rate_box.setRange(0, 1)
        self.sa_cooling_rate_box.setValue(0.95)
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

        sa_hyper_layout.addRow("Initial Temperature:", self.sa_initial_temp_box)
        sa_hyper_layout.addRow("Cooling Rate:", self.sa_cooling_rate_box)
        sa_hyper_layout.addRow("Number of Iterations:", self.sa_num_iterations_box)
        sa_hyper_layout.addRow("Tolerance (tol):", self.sa_tol_box)
        sa_hyper_layout.addRow("Sparsity Penalty (alpha):", self.sa_alpha_box)

        # Add a small Run SA button in the hyperparameters sub-tab
        self.hyper_run_sa_button = QPushButton("Run SA")
        self.hyper_run_sa_button.setFixedWidth(100)
        self.hyper_run_sa_button.clicked.connect(self.run_sa)
        sa_hyper_layout.addRow("Run SA:", self.hyper_run_sa_button)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        sa_param_tab = QWidget()
        sa_param_layout = QVBoxLayout(sa_param_tab)

        self.sa_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.sa_param_table.setRowCount(len(dva_parameters))
        self.sa_param_table.setColumnCount(5)
        self.sa_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.sa_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sa_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set up table rows
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

        # -------------------- Sub-tab 3: Results --------------------
        sa_results_tab = QWidget()
        sa_results_layout = QVBoxLayout(sa_results_tab)
        
        self.sa_results_text = QTextEdit()
        self.sa_results_text.setReadOnly(True)
        sa_results_layout.addWidget(QLabel("SA Optimization Results:"))
        sa_results_layout.addWidget(self.sa_results_text)

        # Add all sub-tabs to the SA tab widget
        self.sa_sub_tabs.addTab(sa_hyper_tab, "SA Settings")
        self.sa_sub_tabs.addTab(sa_param_tab, "DVA Parameters")
        self.sa_sub_tabs.addTab(sa_results_tab, "Results")

        # Add the SA sub-tabs widget to the main SA tab layout
        layout.addWidget(self.sa_sub_tabs)
        self.sa_tab.setLayout(layout)
        
    def toggle_sa_fixed(self, state, row, table=None):
        """Toggle the fixed state of a SA parameter row"""
        if table is None:
            table = self.sa_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
    def run_sa(self):
        """Run the simulated annealing optimization"""
        # Implementation already exists at line 2591
        pass
        
    def run_cmaes(self):
        """Run the CMA-ES optimization"""
        # Implementation already exists at line 2840
        pass
        
