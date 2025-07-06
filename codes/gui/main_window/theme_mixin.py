from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

class ThemeMixin:
    """Mixin handling light/dark theme support with enhanced elegant styling"""
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.current_theme == 'Dark':
            self.current_theme = 'Light'
            self.apply_light_theme()
        else:
            self.current_theme = 'Dark'
            self.apply_dark_theme()
            
        # Update beam interface theme if it exists
        if hasattr(self, 'update_beam_interface_theme'):
            self.update_beam_interface_theme(self.current_theme)
    
    def apply_current_theme(self):
        """Apply the current theme"""
        if self.current_theme == 'Dark':
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
            
        # Update beam interface theme if it exists
        if hasattr(self, 'update_beam_interface_theme'):
            self.update_beam_interface_theme(self.current_theme)

    def apply_dark_theme(self):
        """Apply an elegant dark theme with gradient backgrounds and vibrant accents"""
        dark_palette = QPalette()
        
        # Enhanced base colors with depth
        dark_color = QColor(15, 15, 20)          # Deep dark blue-black
        darker_color = QColor(8, 8, 12)          # Even deeper background
        medium_color = QColor(35, 35, 45)        # Medium dark with blue tint
        light_color = QColor(55, 55, 70)         # Lighter dark with blue tint
        text_color = QColor(245, 245, 250)       # Almost white with slight blue tint
        disabled_text_color = QColor(120, 120, 135)
        
        # Vibrant accent colors for elegance
        primary_color = QColor(94, 129, 255)     # Vibrant blue
        secondary_color = QColor(138, 43, 226)   # Blue violet
        tertiary_color = QColor(255, 64, 129)    # Pink accent
        success_color = QColor(76, 217, 100)     # Vibrant green
        warning_color = QColor(255, 193, 7)      # Amber
        danger_color = QColor(255, 87, 87)       # Coral red
        info_color = QColor(41, 182, 246)        # Cyan
        
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
        dark_palette.setColor(QPalette.BrightText, tertiary_color)
        dark_palette.setColor(QPalette.Link, primary_color)
        dark_palette.setColor(QPalette.Highlight, primary_color)
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 90))
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)
        
        self.setPalette(dark_palette)
        
        # Enhanced dark theme stylesheet with gradients and elegant styling
        dark_stylesheet = f"""
            QMainWindow {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #0F0F14, stop: 0.5 #12121A, stop: 1 #0A0A0F);
                color: {text_color.name()};
            }}
            QWidget {{
                color: {text_color.name()};
                background: transparent;
            }}
            #sidebar {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1A1A25, stop: 0.3 #1E1E2D, stop: 1 #16161F);
                border-right: 2px solid qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 0.5 {secondary_color.name()}, stop: 1 {primary_color.name()});
                border-radius: 0px 8px 8px 0px;
            }}
            #logo-container {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1F1F2A, stop: 1 #181823);
                border-bottom: 1px solid {primary_color.name()};
                border-radius: 8px 8px 0px 0px;
            }}
            #run-card {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1E1E28, stop: 0.5 #1A1A24, stop: 1 #161620);
                border-radius: 12px;
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            QTabWidget::pane {{
                border: 2px solid {primary_color.name()};
                border-radius: 8px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1A1A25, stop: 1 #161620);
                top: -2px;
                padding: 10px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1D1D28, stop: 1 #191923);
                color: #B0B0C0;
                border: 1px solid #2D2D3D;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 3px;
                font-weight: 500;
                min-width: 120px;
                min-height: 30px;
            }}
            QTabBar::tab:selected {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                color: #FFFFFF;
                border-bottom: 3px solid {tertiary_color.name()};
                font-weight: bold;
                min-width: 120px;
                min-height: 30px;
            }}
            QTabBar::tab:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2D2D3D, stop: 1 #252530);
                color: #FFFFFF;
                min-width: 120px;
                min-height: 30px;
            }}
            QScrollArea, QScrollBar {{
                border: none;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1A1A25, stop: 1 #161620);
            }}
            QScrollBar:vertical {{
                border: none;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1A1A25, stop: 1 #1E1E2A);
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1A1A25, stop: 1 #1E1E2A);
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {QColor(primary_color.red() + 30, primary_color.green() + 30, primary_color.blue() + 30).name()}, 
                    stop: 1 {QColor(secondary_color.red() + 30, secondary_color.green() + 30, secondary_color.blue() + 30).name()});
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
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2D2D3D, stop: 1 #3D3D4D);
                border-radius: 10px;
                margin-top: 20px;
                padding-top: 25px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 15px;
                font-weight: bold;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 rgba(30, 30, 40, 0.3), stop: 1 rgba(20, 20, 30, 0.3));
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 6px 12px;
                color: #FFFFFF;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QGroupBox:hover {{
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
            }}
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #333344, stop: 1 #2A2A3A);
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #444455, stop: 1 #555566);
                border-radius: 8px;
                padding: 10px 24px;
                color: #FFFFFF;
                font-weight: 600;
                font-size: 11px;
                min-width: 80px;
                min-height: 32px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #444455, stop: 1 #3A3A4A);
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #222233, stop: 1 #1A1A2A);
                transform: translateY(1px);
            }}
            QPushButton:disabled {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2A2A35, stop: 1 #252530);
                color: #666677;
                border: 1px solid #393944;
            }}
            #primary-button {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                border: none;
                color: white;
                border-radius: 10px;
                padding: 12px 28px;
                font-weight: bold;
                font-size: 12px;
                min-width: 100px;
                min-height: 36px;
            }}
            #primary-button:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {QColor(primary_color.red() + 20, primary_color.green() + 20, primary_color.blue() + 20).name()}, 
                    stop: 1 {QColor(secondary_color.red() + 20, secondary_color.green() + 20, secondary_color.blue() + 20).name()});
                transform: translateY(-2px);
            }}
            #primary-button:pressed {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {QColor(primary_color.red() - 20, primary_color.green() - 20, primary_color.blue() - 20).name()}, 
                    stop: 1 {QColor(secondary_color.red() - 20, secondary_color.green() - 20, secondary_color.blue() - 20).name()});
                transform: translateY(0px);
            }}
            #secondary-button {{
                background: transparent;
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                color: {primary_color.name()};
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: 600;
                font-size: 11px;
                min-width: 90px;
                min-height: 34px;
            }}
            #secondary-button:hover {{
                background-color: rgba(75, 111, 247, 0.1);
            }}
            #secondary-button:pressed {{
                background-color: rgba(75, 111, 247, 0.2);
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 2px solid #333344;
                border-radius: 8px;
                padding: 8px 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1E1E28, stop: 1 #1A1A24);
                color: #FFFFFF;
                selection-background-color: {primary_color.name()};
                font-size: 11px;
                min-height: 24px;
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #252530, stop: 1 #20202C);
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 2px solid {primary_color.name()};
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #252530, stop: 1 #20202C);
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
        """Apply an elegant light theme with sophisticated gradients and vibrant accents"""
        light_palette = QPalette()
        
        # Enhanced base colors with warmth
        light_color = QColor(252, 252, 255)       # Slightly blue-tinted white
        lighter_color = QColor(255, 255, 255)     # Pure white
        medium_color = QColor(245, 245, 250)      # Light blue-gray
        dark_color = QColor(220, 220, 230)        # Soft gray
        text_color = QColor(30, 30, 40)           # Dark blue-gray
        disabled_text_color = QColor(140, 140, 150)
        
        # Enhanced accent colors
        primary_color = QColor(79, 122, 255)      # Vibrant blue
        secondary_color = QColor(138, 43, 226)    # Blue violet
        tertiary_color = QColor(255, 64, 129)     # Pink accent
        success_color = QColor(76, 217, 100)      # Vibrant green
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
        
        # Enhanced light theme stylesheet with gradients
        light_stylesheet = f"""
            QMainWindow {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FCFCFF, stop: 0.5 #F8F8FC, stop: 1 #F5F5FA);
                color: {text_color.name()};
            }}
            QWidget {{
                color: {text_color.name()};
                background: transparent;
            }}
            #sidebar {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #F5F5FA, stop: 0.3 #F0F0F5, stop: 1 #F8F8FC);
                border-right: 2px solid qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 0.5 {secondary_color.name()}, stop: 1 {primary_color.name()});
                border-radius: 0px 8px 8px 0px;
            }}
            #logo-container {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FFFFFF, stop: 1 #F8F8FC);
                border-bottom: 1px solid {primary_color.name()};
                border-radius: 8px 8px 0px 0px;
            }}
            #run-card {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FFFFFF, stop: 0.5 #FAFAFA, stop: 1 #F5F5FA);
                border-radius: 12px;
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }}
            QTabWidget::pane {{
                border: 2px solid {primary_color.name()};
                border-radius: 8px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FFFFFF, stop: 1 #F5F5FA);
                top: -2px;
                padding: 10px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #F5F5FA, stop: 1 #F0F0F5);
                color: #707080;
                border: 1px solid #E0E0E5;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 3px;
                font-weight: 500;
                min-width: 120px;
                min-height: 30px;
            }}
            QTabBar::tab:selected {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                color: #FFFFFF;
                border-bottom: 3px solid {tertiary_color.name()};
                font-weight: bold;
                min-width: 120px;
                min-height: 30px;
            }}
            QTabBar::tab:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #EEEEEE, stop: 1 #E5E5EA);
                color: {text_color.name()};
                min-width: 120px;
                min-height: 30px;
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
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E0E0E0, stop: 1 #D0D0D0);
                border-radius: 10px;
                margin-top: 20px;
                padding-top: 25px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 15px;
                font-weight: bold;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FFFFFF, stop: 1 #FAFAFA);
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 6px 12px;
                color: #FFFFFF;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }}
            QGroupBox:hover {{
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 {primary_color.name()}, stop: 1 {secondary_color.name()});
            }}
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #F5F5F5, stop: 1 #EEEEEE);
                border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #D0D0D0, stop: 1 #C0C0C0);
                border-radius: 8px;
                padding: 10px 24px;
                color: #404040;
                font-weight: 600;
                font-size: 11px;
                min-width: 80px;
                min-height: 32px;
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
