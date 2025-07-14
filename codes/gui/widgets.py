from PyQt5.QtWidgets import QTabWidget, QWidget, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QFont, QCursor
from PyQt5.QtCore import Qt

class ModernQTabWidget(QTabWidget):
    """Custom TabWidget with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setTabPosition(QTabWidget.North)
        self.setMovable(True)

class SidebarButton(QWidget):
    """Elegant sidebar button with modern styling and animations"""
    def __init__(self, icon_path, text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(12)

        if icon_path:
            icon = QLabel()
            pixmap = QPixmap(icon_path)
            icon.setPixmap(pixmap.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(icon)

        label = QLabel(text)
        label.setFont(QFont("Arial", 11, QFont.Medium))
        layout.addWidget(label)
        layout.addStretch()

        self.setMinimumHeight(56)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Set initial style
        self.setStyleSheet("""
            SidebarButton {
                border-radius: 6px;
                padding: 6px 12px;
                margin: 2px 8px;
                transition: all 0.2s ease;
            }
        """)

    def enterEvent(self, event):
        self.setStyleSheet("""
            SidebarButton {
                background-color: rgba(0, 150, 136, 0.15);
                border: 1px solid rgba(0, 150, 136, 0.4);
                border-radius: 6px;
                padding: 6px 12px;
                margin: 2px 8px;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("""
            SidebarButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 6px 12px;
                margin: 2px 8px;
            }
        """)
        super().leaveEvent(event)
