from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt

class PSOMixin:
    def create_pso_tab(self):
        self.pso_tab = QWidget()
        layout = QVBoxLayout(self.pso_tab)
        label = QLabel("<h2>🚧 Under Construction 🚧</h2><br><br>The GUI for Particle Swarm Optimization (PSO) is currently under construction.<br><br><b>Note:</b> The PSO functions and the backend in the library are fully operational.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def run_pso(self):
        QMessageBox.information(self.pso_tab, "Under Construction", "The GUI for PSO is currently under construction.\n\nThe functions and backend in the library work perfectly.")
        
    def run_next_pso_benchmark(self):
        pass
