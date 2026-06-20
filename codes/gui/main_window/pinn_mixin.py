from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
    QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from workers.PINNWorker import PINNWorker, TORCH_AVAILABLE

class PINNIdentificationMixin:
    def create_pinn_discretisizer_page(self):
        pinn_page = QWidget()
        page_layout = QVBoxLayout(pinn_page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(15)

        # Banner
        banner = QWidget()
        banner.setFixedHeight(70)
        banner_layout = QHBoxLayout(banner)
        banner_palette = banner.palette()
        banner_palette.setColor(QPalette.Background, QColor("#1A237E")) # Indigo
        banner.setAutoFillBackground(True)
        banner.setPalette(banner_palette)
        
        title_label = QLabel("PINN Discretisizer")
        title_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        banner_layout.addWidget(title_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        page_layout.addWidget(banner)

        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: Settings & Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Data Loading Group
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)
        self.pinn_file_path_label = QLabel("No file selected")
        self.pinn_load_btn = QPushButton("Load Sensor Data (CSV)")
        self.pinn_load_btn.clicked.connect(self.load_pinn_data)
        data_layout.addRow(self.pinn_load_btn)
        data_layout.addRow("File:", self.pinn_file_path_label)
        
        self.pinn_p_box = QSpinBox()
        self.pinn_p_box.setRange(1, 20)
        self.pinn_p_box.setValue(3)
        self.pinn_p_box.valueChanged.connect(self.update_topology_table)
        data_layout.addRow("Number of Sensors (P):", self.pinn_p_box)
        left_layout.addWidget(data_group)

        # NN Architecture Group
        nn_group = QGroupBox("Neural Network Configuration")
        nn_layout = QFormLayout(nn_group)
        self.pinn_layers_box = QSpinBox()
        self.pinn_layers_box.setRange(1, 10)
        self.pinn_layers_box.setValue(5)
        nn_layout.addRow("Hidden Layers:", self.pinn_layers_box)
        
        self.pinn_neurons_box = QSpinBox()
        self.pinn_neurons_box.setRange(8, 256)
        self.pinn_neurons_box.setValue(64)
        nn_layout.addRow("Neurons per Layer:", self.pinn_neurons_box)
        left_layout.addWidget(nn_group)

        # Fourier Features Group
        fourier_group = QGroupBox("Fourier Feature Embeddings")
        fourier_layout = QFormLayout(fourier_group)
        self.pinn_use_fourier_cb = QCheckBox("Enable Fourier Features")
        fourier_layout.addRow(self.pinn_use_fourier_cb)
        self.pinn_omega_max_box = QDoubleSpinBox()
        self.pinn_omega_max_box.setRange(1.0, 10000.0)
        self.pinn_omega_max_box.setValue(100.0)
        fourier_layout.addRow("Max Frequency (Hz):", self.pinn_omega_max_box)
        self.pinn_n_freqs_box = QSpinBox()
        self.pinn_n_freqs_box.setRange(1, 100)
        self.pinn_n_freqs_box.setValue(10)
        fourier_layout.addRow("Number of Frequencies:", self.pinn_n_freqs_box)
        left_layout.addWidget(fourier_group)

        # Topology Group
        topology_group = QGroupBox("System Topology (M, C, K Constraints)")
        topology_layout = QVBoxLayout(topology_group)
        self.pinn_topology_table = QTableWidget()
        topology_layout.addWidget(self.pinn_topology_table)
        
        topo_btns_layout = QHBoxLayout()
        self.pinn_chain_topo_btn = QPushButton("Set Chain Topology")
        self.pinn_chain_topo_btn.clicked.connect(lambda: self.set_topology_pattern('chain'))
        topo_btns_layout.addWidget(self.pinn_chain_topo_btn)
        
        self.pinn_dense_topo_btn = QPushButton("Set Dense Topology")
        self.pinn_dense_topo_btn.clicked.connect(lambda: self.set_topology_pattern('dense'))
        topo_btns_layout.addWidget(self.pinn_dense_topo_btn)
        topology_layout.addLayout(topo_btns_layout)
        left_layout.addWidget(topology_group)

        # Training Settings Group
        train_group = QGroupBox("Training Parameters")
        train_layout = QFormLayout(train_group)
        self.pinn_adam_epochs = QSpinBox()
        self.pinn_adam_epochs.setRange(0, 50000)
        self.pinn_adam_epochs.setValue(5000)
        train_layout.addRow("Adam Epochs:", self.pinn_adam_epochs)
        
        self.pinn_lbfgs_epochs = QSpinBox()
        self.pinn_lbfgs_epochs.setRange(0, 5000)
        self.pinn_lbfgs_epochs.setValue(1000)
        train_layout.addRow("L-BFGS Max Iter:", self.pinn_lbfgs_epochs)
        
        self.pinn_warmup_epochs = QSpinBox()
        self.pinn_warmup_epochs.setRange(0, 50000)
        self.pinn_warmup_epochs.setValue(1000)
        train_layout.addRow("Physics Warm-up Epochs:", self.pinn_warmup_epochs)
        
        self.pinn_lr_box = QDoubleSpinBox()
        self.pinn_lr_box.setRange(1e-5, 0.1)
        self.pinn_lr_box.setDecimals(5)
        self.pinn_lr_box.setValue(1e-3)
        train_layout.addRow("Learning Rate:", self.pinn_lr_box)
        left_layout.addWidget(train_group)

        # Controls
        self.run_pinn_btn = QPushButton("Start Identification")
        self.run_pinn_btn.setMinimumHeight(50)
        self.run_pinn_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        self.run_pinn_btn.clicked.connect(self.run_pinn_identification)
        left_layout.addWidget(self.run_pinn_btn)
        
        self.stop_pinn_btn = QPushButton("Stop")
        self.stop_pinn_btn.setEnabled(False)
        self.stop_pinn_btn.clicked.connect(self.stop_pinn_identification)
        left_layout.addWidget(self.stop_pinn_btn)
        
        self.pinn_progress_bar = QProgressBar()
        left_layout.addWidget(self.pinn_progress_bar)
        
        # Add to scroll area to handle long content
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll_widget = QWidget()
        left_scroll_widget.setLayout(left_layout)
        left_scroll.setWidget(left_scroll_widget)
        content_splitter.addWidget(left_scroll)

        # Right Panel: Results & Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Plots
        self.pinn_loss_fig = Figure()
        self.pinn_loss_canvas = FigureCanvas(self.pinn_loss_fig)
        right_layout.addWidget(NavigationToolbar(self.pinn_loss_canvas, right_panel))
        right_layout.addWidget(self.pinn_loss_canvas)
        
        self.pinn_results_text = QTextEdit()
        self.pinn_results_text.setReadOnly(True)
        right_layout.addWidget(self.pinn_results_text)
        
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([500, 700])
        page_layout.addWidget(content_splitter)

        self.content_stack.addWidget(pinn_page)
        self.pinn_data = None
        self.update_topology_table()

    def update_topology_table(self):
        P = self.pinn_p_box.value()
        self.pinn_topology_table.setRowCount(P)
        self.pinn_topology_table.setColumnCount(P)
        headers = [f"Sensor {i+1}" for i in range(P)]
        self.pinn_topology_table.setHorizontalHeaderLabels(headers)
        self.pinn_topology_table.setVerticalHeaderLabels(headers)
        
        self.set_topology_pattern('chain')
        self.pinn_topology_table.resizeColumnsToContents()

    def set_topology_pattern(self, pattern):
        P = self.pinn_p_box.value()
        for i in range(P):
            for j in range(P):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                
                if pattern == 'chain':
                    # Tridiagonal + diagonal
                    if abs(i - j) <= 1:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
                else: # dense
                    item.setCheckState(Qt.Checked)
                    
                if i == j:
                    item.setToolTip(f"Node {i+1} ground connection")
                else:
                    item.setToolTip(f"Connection between Node {i+1} and Node {j+1}")
                self.pinn_topology_table.setItem(i, j, item)

    def load_pinn_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Sensor Data", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.pinn_data = pd.read_csv(file_path)
                self.pinn_file_path_label.setText(file_path.split('/')[-1])
                # Validate columns
                P = self.pinn_p_box.value()
                expected_cols = 1 + P * 3 # t + x, v, a for each sensor
                if len(self.pinn_data.columns) < expected_cols:
                    QMessageBox.warning(self, "Data Warning", f"Expected at least {expected_cols} columns, found {len(self.pinn_data.columns)}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def get_topology_mask(self):
        P = self.pinn_p_box.value()
        mask = np.zeros((P, P))
        for i in range(P):
            for j in range(P):
                if self.pinn_topology_table.item(i, j).checkState() == Qt.Checked:
                    mask[i, j] = 1.0
        return mask

    def run_pinn_identification(self):
        if not TORCH_AVAILABLE:
            QMessageBox.critical(self, "Dependency Error", "PyTorch (torch) is required for PINN identification but it is not installed in the current environment.")
            return

        if self.pinn_data is None:
            QMessageBox.warning(self, "No Data", "Please load sensor data first.")
            return
        
        try:
            P = self.pinn_p_box.value()
            t = self.pinn_data.iloc[:, 0].values
            x = self.pinn_data.iloc[:, 1:P+1].values
            v = self.pinn_data.iloc[:, P+1:2*P+1].values
            a = self.pinn_data.iloc[:, 2*P+1:3*P+1].values
            
            mask = self.get_topology_mask()
            
            self.pinn_worker = PINNWorker(
                t, x, v, a, P, 
                layers=self.pinn_layers_box.value(),
                neurons=self.pinn_neurons_box.value(),
                adam_epochs=self.pinn_adam_epochs.value(),
                lbfgs_epochs=self.pinn_lbfgs_epochs.value(),
                lr=self.pinn_lr_box.value(),
                use_fourier=self.pinn_use_fourier_cb.isChecked(),
                omega_max=self.pinn_omega_max_box.value(),
                n_freqs=self.pinn_n_freqs_box.value(),
                warmup_epochs=self.pinn_warmup_epochs.value(),
                topology_mask=mask
            )
            
            self.pinn_worker.progress.connect(self.update_pinn_progress)
            self.pinn_worker.finished.connect(self.pinn_finished)
            self.pinn_worker.error.connect(self.pinn_error)
            
            self.run_pinn_btn.setEnabled(False)
            self.stop_pinn_btn.setEnabled(True)
            self.pinn_progress_bar.setValue(0)
            self.pinn_results_text.clear()
            self.pinn_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start identification: {str(e)}")

    def stop_pinn_identification(self):
        if hasattr(self, 'pinn_worker'):
            self.pinn_worker.stop()
            self.stop_pinn_btn.setEnabled(False)
            self.pinn_results_text.append("\nOptimization stopped by user.")

    def update_pinn_progress(self, epoch, loss, params):
        total_epochs = self.pinn_adam_epochs.value() + self.pinn_lbfgs_epochs.value()
        progress = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
        self.pinn_progress_bar.setValue(progress)
        self.pinn_results_text.append(f"Epoch/Iter {epoch}: Loss = {loss:.6f}")

    def pinn_finished(self, results):
        self.run_pinn_btn.setEnabled(True)
        self.stop_pinn_btn.setEnabled(False)
        self.pinn_progress_bar.setValue(100)
        QMessageBox.information(self, "Finished", "PINN System Identification completed.")
        self.display_pinn_results(results)

    def pinn_error(self, message):
        self.run_pinn_btn.setEnabled(True)
        self.stop_pinn_btn.setEnabled(False)
        QMessageBox.critical(self, "PINN Error", message)

    def display_pinn_results(self, results):
        M = np.array(results["M"])
        C = np.array(results["C"])
        K = np.array(results["K"])
        
        self.pinn_results_text.append("\n==================================")
        self.pinn_results_text.append("FINAL IDENTIFIED MATRICES")
        self.pinn_results_text.append("==================================")
        
        self.pinn_results_text.append("\nIdentified Mass Matrix (M):\n" + np.array2string(M, precision=4, suppress_small=True))
        self.pinn_results_text.append("\nIdentified Damping Matrix (C):\n" + np.array2string(C, precision=4, suppress_small=True))
        self.pinn_results_text.append("\nIdentified Stiffness Matrix (K):\n" + np.array2string(K, precision=4, suppress_small=True))
        
        try:
            from scipy.linalg import eigh
            evals, _ = eigh(K, M)
            freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
            self.pinn_results_text.append("\nNatural Frequencies (Hz):\n" + np.array2string(freqs, precision=2))
        except Exception as e:
            self.pinn_results_text.append(f"\nCould not calculate natural frequencies: {str(e)}")
