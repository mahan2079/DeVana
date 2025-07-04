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
    """Custom styled sidebar button"""
    def __init__(self, icon_path, text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        if icon_path:
            icon = QLabel()
            pixmap = QPixmap(icon_path)
            icon.setPixmap(pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(icon)

        label = QLabel(text)
        label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(label)
        layout.addStretch()

        self.setMinimumHeight(50)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def enterEvent(self, event):
        self.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); border-radius: 5px;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("")
        super().leaveEvent(event)
