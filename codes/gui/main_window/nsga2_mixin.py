from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class NSGA2OptimizationMixin:
    def create_nsga2_tab(self):
        """Create the NSGA-II optimization tab."""
        nsga2_tab = QWidget()
        layout = QVBoxLayout(nsga2_tab)
        
        info_label = QLabel("NSGA-II Optimization (Placeholder)")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        return nsga2_tab
