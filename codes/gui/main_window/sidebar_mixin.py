from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from gui.widgets import SidebarButton


class SidebarMixin:
    def create_sidebar(self, BEAM_IMPORTS_SUCCESSFUL):
        """Create the sidebar with navigation buttons"""
        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebar")
        sidebar_container.setFixedWidth(250)

        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        logo_container = QWidget()
        logo_container.setObjectName("logo-container")
        logo_container.setMinimumHeight(100)

        logo_layout = QVBoxLayout(logo_container)
        title = QLabel("DeVana")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        logo_layout.addWidget(title)

        version = QLabel("V0.3.0")
        version.setAlignment(Qt.AlignCenter)
        version.setFont(QFont("Segoe UI", 10))
        logo_layout.addWidget(version)

        sidebar_layout.addWidget(logo_container)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        sidebar_layout.addWidget(line)

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(10)

        self.stochastic_btn = SidebarButton(None, "Stochastic Design")
        self.stochastic_btn.setObjectName("active-nav-btn")
        self.stochastic_btn.mousePressEvent = lambda event: self.change_page(0)
        nav_layout.addWidget(self.stochastic_btn)

        self.microchip_btn = SidebarButton(None, "Microchip Controller")
        self.microchip_btn.mousePressEvent = lambda event: self.change_page(1)
        nav_layout.addWidget(self.microchip_btn)

        self.beam_btn = SidebarButton(None, "Continuous Beam")
        self.beam_btn.mousePressEvent = lambda event: self.change_page(2)
        if not BEAM_IMPORTS_SUCCESSFUL:
            self.beam_btn.setEnabled(False)
            self.beam_btn.setToolTip("Continuous Beam module not available")
        nav_layout.addWidget(self.beam_btn)

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_container)

        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)

        self.theme_toggle = QPushButton("Toggle Theme")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        bottom_layout.addWidget(self.theme_toggle)

        sidebar_layout.addWidget(bottom_container)
        sidebar_layout.addSpacing(20)

        self.main_layout.addWidget(sidebar_container)

    def change_page(self, index):
        """Change the active page in the content stack"""
        self.content_stack.setCurrentIndex(index)

        for btn in [self.stochastic_btn, self.microchip_btn, self.beam_btn]:
            btn.setObjectName("")
            btn.setStyleSheet("")

        if index == 0:
            self.stochastic_btn.setObjectName("active-nav-btn")
        elif index == 1:
            self.microchip_btn.setObjectName("active-nav-btn")
        elif index == 2:
            self.beam_btn.setObjectName("active-nav-btn")

        self.apply_current_theme()

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.current_theme == 'Dark':
            self.current_theme = 'Light'
            self.apply_light_theme()
            self.theme_toggle.setText("Switch to Dark Theme")
        else:
            self.current_theme = 'Dark'
            self.apply_dark_theme()
            self.theme_toggle.setText("Switch to Light Theme")
        
        # Update the sidebar button styles
        self.apply_current_theme()
