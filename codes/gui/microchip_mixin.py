from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import os

class MicrochipPageMixin:
    def create_microchip_controller_page(self):
        """Create the empty microchip controller page for future implementation"""
        microchip_page = QWidget()
        layout = QVBoxLayout(microchip_page)

        # Centered content
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setAlignment(Qt.AlignCenter)

        # Icon or image placeholder
        placeholder = QLabel()
        placeholder.setPixmap(QPixmap("placeholder_image.png" if os.path.exists("placeholder_image.png") else ""))
        placeholder.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(placeholder)

        # Title
        title = QLabel("Microchip Controller")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title)

        # Description
        description = QLabel("This feature will provide interfaces for microchip-based vibration control systems.")
        description.setFont(QFont("Segoe UI", 12))
        description.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(description)

        # Coming soon label
        coming_soon = QLabel("Coming Soon!")
        coming_soon.setFont(QFont("Segoe UI", 14, QFont.Bold))
        coming_soon.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(coming_soon)

        layout.addWidget(center_widget)
        self.content_stack.addWidget(microchip_page)
