from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt

class SAOptimizationMixin:
    def create_sa_tab(self):
        self.sa_tab = QWidget()
        layout = QVBoxLayout(self.sa_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for Simulated Annealing (SA) is currently under construction.<br><br><b>Note:</b> The SA functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def run_sa(self):
        QMessageBox.information(self.sa_tab, "Under Construction", "The GUI for SA is currently under construction.\n\nThe functions and backend in the library work perfectly.")
        
    def run_cmaes(self):
        QMessageBox.information(self.sa_tab, "Under Construction", "The GUI is currently under construction.")
