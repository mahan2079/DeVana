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
        label.setFont(QFont("Segoe UI", 11, QFont.Medium))
        layout.addWidget(label)
        layout.addStretch()

        self.setMinimumHeight(56)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Set initial style
        self.setStyleSheet("""
            SidebarButton {
                border-radius: 8px;
                padding: 8px 15px;
                margin: 2px 8px;
                transition: all 0.2s ease;
            }
        """)

    def enterEvent(self, event):
        self.setStyleSheet("""
            SidebarButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 rgba(94, 129, 255, 0.15), stop: 1 rgba(138, 43, 226, 0.15));
                border: 1px solid rgba(94, 129, 255, 0.3);
                border-radius: 10px;
                padding: 8px 15px;
                margin: 2px 8px;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("""
            SidebarButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 8px 15px;
                margin: 2px 8px;
            }
        """)
        super().leaveEvent(event)
