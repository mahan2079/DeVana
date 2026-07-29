from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QPushButton
from PyQt5.QtCore import Qt

class SAOptimizationMixin:
    def create_sa_tab(self):
        self.sa_tab = QWidget()
        layout = QVBoxLayout(self.sa_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for Simulated Annealing (SA) is currently under construction.<br><br><b>Note:</b> The SA functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        self.contextual_run_sa_button = QPushButton("Run SA")
        self.contextual_run_sa_button.setObjectName("primary-button")
        self.contextual_run_sa_button.setMinimumHeight(40)
        self.contextual_run_sa_button.clicked.connect(self.run_sa)
        layout.addWidget(self.contextual_run_sa_button)

    def run_sa(self):
        QMessageBox.information(self.sa_tab, "Under Construction", "The GUI for SA is currently under construction.\n\nThe functions and backend in the library work perfectly.")
        
    def run_cmaes(self):
        QMessageBox.information(self.sa_tab, "Under Construction", "The GUI is currently under construction.")
