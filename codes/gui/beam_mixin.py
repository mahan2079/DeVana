from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class ContinuousBeamMixin:
    def create_continuous_beam_page(self):
        """Create the continuous beam optimization page (Under Construction)."""
        beam_page = QWidget()
        layout = QVBoxLayout(beam_page)
        
        # Top-aligned content
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        top_layout.setContentsMargins(0, 40, 0, 0)
        
        # Construction message
        error_label = QLabel("Continuous Beam Module")
        error_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("color: #1976D2;")
        top_layout.addWidget(error_label)
        
        status_label = QLabel("--- UNDER CONSTRUCTION ---")
        status_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("color: #FFA000; margin-top: 10px; margin-bottom: 20px;")
        top_layout.addWidget(status_label)
        
        description = QLabel("This module is currently being refactored for improved performance and accuracy.")
        description.setFont(QFont("Segoe UI", 14))
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("color: #666666;")
        top_layout.addWidget(description)
        
        layout.addWidget(top_widget)
        layout.addStretch()
        self.content_stack.addWidget(beam_page)
