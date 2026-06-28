from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt

class DEOptimizationMixin:
    def create_de_tab(self):
        self.de_tab = QWidget()
        layout = QVBoxLayout(self.de_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for Differential Evolution (DE) is currently under construction.<br><br><b>Note:</b> The DE functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def run_de(self):
        QMessageBox.information(self.de_tab, "Under Construction", "The GUI for DE is currently under construction.\n\nThe functions and backend in the library work perfectly.")
