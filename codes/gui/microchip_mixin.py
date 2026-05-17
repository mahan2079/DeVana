from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class MicrochipPageMixin:
    def create_microchip_controller_page(self):
        """Create the microchip controller page (Under Construction)."""
        microchip_page = QWidget()
        layout = QVBoxLayout(microchip_page)

        # Top-aligned content
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        top_layout.setContentsMargins(0, 40, 0, 0)

        # Title
        title = QLabel("Microchip Controller")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976D2;")
        top_layout.addWidget(title)

        # Status
        status_label = QLabel("--- UNDER CONSTRUCTION ---")
        status_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("color: #FFA000; margin-top: 10px; margin-bottom: 20px;")
        top_layout.addWidget(status_label)

        # Description
        description = QLabel("Advanced interfaces for microchip-based vibration control systems are coming soon.")
        description.setFont(QFont("Segoe UI", 14))
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("color: #666666;")
        top_layout.addWidget(description)

        layout.addWidget(top_widget)
        layout.addStretch()
        self.content_stack.addWidget(microchip_page)
