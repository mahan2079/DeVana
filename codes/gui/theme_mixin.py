from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ThemeMixin:
    def toggle_theme(self):
        if self.current_theme == 'Dark':
            self.current_theme = 'Light'
            self.apply_light_theme()
        else:
            self.current_theme = 'Dark'
            self.apply_dark_theme()

    def apply_current_theme(self):
        if self.current_theme == 'Dark':
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_dark_theme(self):
        """Apply a modern dark theme with accent colors"""
        dark_palette = QPalette()
        
        # Base colors
        dark_color = QColor(18, 18, 18)
        darker_color = QColor(12, 12, 12)
        medium_color = QColor(40, 40, 40)
        light_color = QColor(60, 60, 60)
        text_color = QColor(240, 240, 240)
        disabled_text_color = QColor(128, 128, 128)
        
        # Accent colors
        primary_color = QColor(75, 111, 247)      # Blue
        secondary_color = QColor(107, 64, 216)    # Purple
        success_color = QColor(46, 204, 113)      # Green
        warning_color = QColor(241, 196, 15)      # Yellow
        danger_color = QColor(231, 76, 60)        # Red
        info_color = QColor(52, 152, 219)         # Light Blue
        
        # Set up the palette
        dark_palette.setColor(QPalette.Window, dark_color)
        dark_palette.setColor(QPalette.WindowText, text_color)
        dark_palette.setColor(QPalette.Base, darker_color)
        dark_palette.setColor(QPalette.AlternateBase, medium_color)
        dark_palette.setColor(QPalette.ToolTipBase, primary_color)
        dark_palette.setColor(QPalette.ToolTipText, text_color)
        dark_palette.setColor(QPalette.Text, text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text_color)
        dark_palette.setColor(QPalette.Button, dark_color)
        dark_palette.setColor(QPalette.ButtonText, text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text_color)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, primary_color)
        dark_palette.setColor(QPalette.Highlight, primary_color)
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)
        
        self.setPalette(dark_palette)
        
        # Dark theme stylesheet
        dark_stylesheet = f"""
            QMainWindow {{
                background-color: #121212;
            }}
            QWidget {{
                color: #F0F0F0;
                background-color: #121212;
            }}
            #sidebar {{
                background-color: #1A1A1A;
                border-right: 1px solid #2D2D2D;
            }}
            #logo-container {{
                background-color: #171717;
                border-bottom: 1px solid #2D2D2D;
            }}
            #run-card {{
                background-color: #1E1E1E;
                border-radius: 8px;
                border: 1px solid #2D2D2D;
            }}
            QTabWidget::pane {{
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                background-color: #1A1A1A;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: #1D1D1D;
                color: #B0B0B0;
                border: 1px solid #2D2D2D;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: #1A1A1A;
                color: #FFFFFF;
                border-bottom: 2px solid {primary_color.name()};
            }}
            QTabBar::tab:hover {{
                background-color: #2D2D2D;
            }}
            QScrollArea, QScrollBar {{
                border: none;
                background-color: #1A1A1A;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #1A1A1A;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #1A1A1A;
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: #3D3D3D;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background-color: #4D4D4D;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background: none;
                border: none;
            }}
            QScrollBar::up-arrow, QScrollBar::down-arrow, QScrollBar::left-arrow, QScrollBar::right-arrow {{
                background: none;
                border: none;
                color: none;
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: none;
            }}
            QGroupBox {{
                border: 1px solid #2D2D2D;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #CCCCCC;
            }}
            QGroupBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QPushButton {{
                background-color: #333333;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 16px;
                color: #FFFFFF;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #444444;
                border: 1px solid #555555;
            }}
            QPushButton:pressed {{
                background-color: #222222;
            }}
            QPushButton:disabled {{
                background-color: #2A2A2A;
                color: #666666;
                border: 1px solid #393939;
            }}
            #primary-button {{
                background-color: {primary_color.name()};
                border: none;
                color: white;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            #primary-button:hover {{
                background-color: {QColor(primary_color.red() + 20, primary_color.green() + 20, primary_color.blue() + 20).name()};
            }}
            #primary-button:pressed {{
                background-color: {QColor(primary_color.red() - 20, primary_color.green() - 20, primary_color.blue() - 20).name()};
            }}
            #secondary-button {{
                background-color: transparent;
                border: 1px solid {primary_color.name()};
                color: {primary_color.name()};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            #secondary-button:hover {{
                background-color: rgba(75, 111, 247, 0.1);
            }}
            #secondary-button:pressed {{
                background-color: rgba(75, 111, 247, 0.2);
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 5px 8px;
                background-color: #1E1E1E;
                color: #FFFFFF;
                selection-background-color: {primary_color.name()};
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {primary_color.name()};
                background-color: #252525;
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
                border: 1px solid #2A2A2A;
                background-color: #1A1A1A;
                color: #666666;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #333333;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #333333;
                background-color: #1E1E1E;
                selection-background-color: {primary_color.name()};
            }}
            QToolBar {{
                background-color: #1A1A1A;
                border-bottom: 1px solid #2D2D2D;
                spacing: 2px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 3px;
                margin: 1px;
            }}
            QToolButton:hover {{
                background-color: rgba(255, 255, 255, 0.1);
            }}
            QToolButton:pressed {{
                background-color: rgba(255, 255, 255, 0.2);
            }}
            QStatusBar {{
                background-color: #1A1A1A;
                color: #B0B0B0;
                border-top: 1px solid #2D2D2D;
            }}
            QMenuBar {{
                background-color: #1A1A1A;
                color: #F0F0F0;
                border-bottom: 1px solid #2D2D2D;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 6px 10px;
            }}
            QMenuBar::item:selected {{
                background-color: #333333;
            }}
            QMenu {{
                background-color: #1E1E1E;
                border: 1px solid #333333;
            }}
            QMenu::item {{
                padding: 6px 20px 6px 20px;
            }}
            QMenu::item:selected {{
                background-color: {primary_color.name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: #333333;
                margin: 5px 0px 5px 0px;
            }}
            QCheckBox {{
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #1E1E1E;
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary_color.name()};
                border: 1px solid {primary_color.name()};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QTableView {{
                background-color: #1E1E1E;
                alternate-background-color: #252525;
                border: 1px solid #333333;
                border-radius: 4px;
                gridline-color: #333333;
                selection-background-color: {primary_color.name()};
                selection-color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: #252525;
                color: #CCCCCC;
                padding: 5px;
                border: 1px solid #333333;
            }}
            QSplitter::handle {{
                background-color: #2D2D2D;
            }}
            QSplitter::handle:hover {{
                background-color: {primary_color.name()};
            }}
            QDockWidget {{
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }}
            QDockWidget::title {{
                text-align: center;
                background-color: #1A1A1A;
                padding: 5px;
                border-bottom: 1px solid #333333;
            }}
            QProgressBar {{
                border: 1px solid #333333;
                border-radius: 3px;
                background-color: #1E1E1E;
                text-align: center;
                color: #FFFFFF;
            }}
            QProgressBar::chunk {{
                background-color: {primary_color.name()};
                width: 1px;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 5px;
                background-color: #333333;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {primary_color.name()};
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {QColor(primary_color.red() + 20, primary_color.green() + 20, primary_color.blue() + 20).name()};
            }}
            QToolTip {{
                border: 1px solid #333333;
                background-color: #1E1E1E;
                color: #FFFFFF;
                padding: 5px;
                opacity: 220;
            }}
            #active-nav-btn {{
                background-color: rgba(75, 111, 247, 0.2);
                border-left: 3px solid {primary_color.name()};
                border-radius: 5px;
            }}
        """
        self.setStyleSheet(dark_stylesheet)

    def apply_light_theme(self):
        """Apply a modern light theme with accent colors"""
        light_palette = QPalette()
        
        # Base colors
        light_color = QColor(250, 250, 250)
        lighter_color = QColor(255, 255, 255)
        medium_color = QColor(240, 240, 240)
        dark_color = QColor(225, 225, 225)
        text_color = QColor(40, 40, 40)
        disabled_text_color = QColor(150, 150, 150)
        
        # Accent colors
        primary_color = QColor(66, 133, 244)      # Blue
        secondary_color = QColor(103, 58, 183)    # Purple
        success_color = QColor(76, 175, 80)       # Green
        warning_color = QColor(255, 152, 0)       # Orange
        danger_color = QColor(244, 67, 54)        # Red
        info_color = QColor(33, 150, 243)         # Light Blue
        
        # Set up the palette
        light_palette.setColor(QPalette.Window, light_color)
        light_palette.setColor(QPalette.WindowText, text_color)
        light_palette.setColor(QPalette.Base, lighter_color)
        light_palette.setColor(QPalette.AlternateBase, medium_color)
        light_palette.setColor(QPalette.ToolTipBase, primary_color)
        light_palette.setColor(QPalette.ToolTipText, Qt.white)
        light_palette.setColor(QPalette.Text, text_color)
        light_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text_color)
        light_palette.setColor(QPalette.Button, light_color)
        light_palette.setColor(QPalette.ButtonText, text_color)
        light_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text_color)
        light_palette.setColor(QPalette.BrightText, Qt.red)
        light_palette.setColor(QPalette.Link, primary_color)
        light_palette.setColor(QPalette.Highlight, primary_color)
        light_palette.setColor(QPalette.HighlightedText, Qt.white)
        light_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(200, 200, 200))
        light_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)
        
        self.setPalette(light_palette)
        
        # Light theme stylesheet
        light_stylesheet = f"""
            QMainWindow {{
                background-color: #FAFAFA;
            }}
            QWidget {{
                color: #282828;
                background-color: #FAFAFA;
            }}
            #sidebar {{
                background-color: #F0F0F0;
                border-right: 1px solid #E0E0E0;
            }}
            #logo-container {{
                background-color: #FFFFFF;
                border-bottom: 1px solid #E0E0E0;
            }}
            #run-card {{
                background-color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.05);
            }}
            QTabWidget::pane {{
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                background-color: #FFFFFF;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: #F5F5F5;
                color: #707070;
                border: 1px solid #E0E0E0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: #FFFFFF;
                color: #282828;
                border-bottom: 2px solid {primary_color.name()};
            }}
            QTabBar::tab:hover {{
                background-color: #EEEEEE;
            }}
            QScrollArea, QScrollBar {{
                border: none;
                background-color: #FFFFFF;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #FAFAFA;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #FAFAFA;
                height: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: #D0D0D0;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background-color: #B0B0B0;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background: none;
                border: none;
            }}
            QScrollBar::up-arrow, QScrollBar::down-arrow, QScrollBar::left-arrow, QScrollBar::right-arrow {{
                background: none;
                border: none;
                color: none;
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: none;
            }}
            QGroupBox {{
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                background-color: #FFFFFF;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #505050;
            }}
            QGroupBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QPushButton {{
                background-color: #F0F0F0;
                border: 1px solid #D0D0D0;
                border-radius: 4px;
                padding: 6px 16px;
                color: #404040;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #E5E5E5;
                border: 1px solid #C0C0C0;
            }}
            QPushButton:pressed {{
                background-color: #DEDEDE;
            }}
            QPushButton:disabled {{
                background-color: #F5F5F5;
                color: #B0B0B0;
                border: 1px solid #E0E0E0;
            }}
            #primary-button {{
                background-color: {primary_color.name()};
                border: none;
                color: white;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            #primary-button:hover {{
                background-color: {QColor(primary_color.red() + 20, primary_color.green() + 20, primary_color.blue() + 20).name()};
            }}
            #primary-button:pressed {{
                background-color: {QColor(primary_color.red() - 20, primary_color.green() - 20, primary_color.blue() - 20).name()};
            }}
            #secondary-button {{
                background-color: transparent;
                border: 1px solid {primary_color.name()};
                color: {primary_color.name()};
                border-radius: 4px;
                padding: 8px 16px;
            }}
            #secondary-button:hover {{
                background-color: rgba(66, 133, 244, 0.1);
            }}
            #secondary-button:pressed {{
                background-color: rgba(66, 133, 244, 0.2);
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 5px 8px;
                background-color: #FFFFFF;
                color: #202020;
                selection-background-color: {primary_color.name()};
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {primary_color.name()};
                background-color: #FFFFFF;
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
                border: 1px solid #E5E5E5;
                background-color: #F5F5F5;
                color: #B0B0B0;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #E0E0E0;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #E0E0E0;
                background-color: #FFFFFF;
                selection-background-color: {primary_color.name()};
            }}
            QToolBar {{
                background-color: #F5F5F5;
                border-bottom: 1px solid #E0E0E0;
                spacing: 2px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 3px;
                margin: 1px;
            }}
            QToolButton:hover {{
                background-color: rgba(0, 0, 0, 0.05);
            }}
            QToolButton:pressed {{
                background-color: rgba(0, 0, 0, 0.1);
            }}
            QStatusBar {{
                background-color: #F5F5F5;
                color: #606060;
                border-top: 1px solid #E0E0E0;
            }}
            QMenuBar {{
                background-color: #F5F5F5;
                color: #404040;
                border-bottom: 1px solid #E0E0E0;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 6px 10px;
            }}
            QMenuBar::item:selected {{
                background-color: #E5E5E5;
            }}
            QMenu {{
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
            }}
            QMenu::item {{
                padding: 6px 20px 6px 20px;
            }}
            QMenu::item:selected {{
                background-color: {primary_color.name()};
                color: white;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: #E0E0E0;
                margin: 5px 0px 5px 0px;
            }}
            QCheckBox {{
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                background-color: #FFFFFF;
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary_color.name()};
                border: 1px solid {primary_color.name()};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QTableView {{
                background-color: #FFFFFF;
                alternate-background-color: #F9F9F9;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                gridline-color: #E0E0E0;
                selection-background-color: {primary_color.name()};
                selection-color: white;
            }}
            QHeaderView::section {{
                background-color: #F0F0F0;
                color: #404040;
                padding: 5px;
                border: 1px solid #E0E0E0;
            }}
            QSplitter::handle {{
                background-color: #E0E0E0;
            }}
            QSplitter::handle:hover {{
                background-color: {primary_color.name()};
            }}
            QDockWidget {{
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }}
            QDockWidget::title {{
                text-align: center;
                background-color: #F5F5F5;
                padding: 5px;
                border-bottom: 1px solid #E0E0E0;
            }}
            QProgressBar {{
                border: 1px solid #E0E0E0;
                border-radius: 3px;
                background-color: #FFFFFF;
                text-align: center;
                color: #404040;
            }}
            QProgressBar::chunk {{
                background-color: {primary_color.name()};
                width: 1px;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 5px;
                background-color: #E0E0E0;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {primary_color.name()};
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {QColor(primary_color.red() + 20, primary_color.green() + 20, primary_color.blue() + 20).name()};
            }}
            QToolTip {{
                border: 1px solid #E0E0E0;
                background-color: #FFFFFF;
                color: #404040;
                padding: 5px;
                opacity: 220;
            }}
            #active-nav-btn {{
                background-color: rgba(66, 133, 244, 0.1);
                border-left: 3px solid {primary_color.name()};
                border-radius: 5px;
            }}
        """
        self.setStyleSheet(light_stylesheet)
