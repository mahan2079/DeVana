from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt

class CMAESOptimizationMixin:
    def create_cmaes_tab(self):
        self.cmaes_tab = QWidget()
        layout = QVBoxLayout(self.cmaes_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for CMA-ES is currently under construction.<br><br><b>Note:</b> The CMA-ES functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def run_cmaes(self):
        QMessageBox.information(self.cmaes_tab, "Under Construction", "The GUI for CMA-ES is currently under construction.\n\nThe functions and backend in the library work perfectly.")
