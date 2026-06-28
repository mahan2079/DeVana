from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt

class AdaVEAOptimizationMixin:
    def create_adavea_tab(self):
        self.adavea_tab = QWidget()
        layout = QVBoxLayout(self.adavea_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for AdaVEA is currently under construction.<br><br><b>Note:</b> The AdaVEA functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def run_adavea(self):
        QMessageBox.information(self.adavea_tab, "Under Construction", "The GUI for AdaVEA is currently under construction.\n\nThe functions and backend in the library work perfectly.")