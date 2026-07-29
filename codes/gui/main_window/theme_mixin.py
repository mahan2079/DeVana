from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

class ThemeMixin:
    """Mixin handling light/dark theme support with refined industrial styling"""
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.current_theme == 'Dark':
            self.current_theme = 'Light'
            self.apply_light_theme()
        else:
            self.current_theme = 'Dark'
            self.apply_dark_theme()
            
        if hasattr(self, 'update_beam_interface_theme'):
            self.update_beam_interface_theme(self.current_theme)
        if hasattr(self, 'update_introduction_theme'):
            self.update_introduction_theme()
    
    def apply_current_theme(self):
        """Apply the current theme"""
        if self.current_theme == 'Dark':
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
            
        if hasattr(self, 'update_beam_interface_theme'):
            self.update_beam_interface_theme(self.current_theme)
        if hasattr(self, 'update_introduction_theme'):
            self.update_introduction_theme()

    def apply_dark_theme(self):
        """Apply a Refined Industrial dark theme (solid colors, strict geometry)"""
        dark_palette = QPalette()
        
        # Refined Industrial Base Colors
        dark_color = QColor(30, 30, 40)          # #1E1E28 - Base canvas
        darker_color = QColor(22, 22, 32)        # #161620 - Slightly darker for contrast
        medium_color = QColor(45, 45, 61)        # #2D2D3D - Borders and secondary elements
        light_color = QColor(61, 61, 77)         # #3D3D4D - Highlights
        text_color = QColor(235, 235, 235)       # Light text
        disabled_text_color = QColor(130, 130, 130)
        
        # Single Vibrant Accent (Teal)
        primary_color = QColor(0, 150, 136)      # #009688
        
        dark_palette.setColor(QPalette.Window, dark_color)
        dark_palette.setColor(QPalette.WindowText, text_color)
        dark_palette.setColor(QPalette.Base, darker_color)
        dark_palette.setColor(QPalette.AlternateBase, medium_color)
        dark_palette.setColor(QPalette.ToolTipBase, primary_color)
        dark_palette.setColor(QPalette.ToolTipText, text_color)
        dark_palette.setColor(QPalette.Text, text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text_color)
        dark_palette.setColor(QPalette.Button, darker_color)
        dark_palette.setColor(QPalette.ButtonText, text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text_color)
        dark_palette.setColor(QPalette.Link, primary_color)
        dark_palette.setColor(QPalette.Highlight, primary_color)
        dark_palette.setColor(QPalette.HighlightedText, Qt.white)
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 90))
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)
        
        self.setPalette(dark_palette)
        
        if hasattr(self, 'stochastic_design_banner') and self.stochastic_design_banner is not None:
            banner_palette = self.stochastic_design_banner.palette()
            banner_palette.setColor(QPalette.Background, darker_color) 
            self.stochastic_design_banner.setPalette(banner_palette)
            self.stochastic_design_banner.update()

        if hasattr(self, 'stochastic_design_title_label') and self.stochastic_design_title_label is not None:
            self.stochastic_design_title_label.setStyleSheet("color: white; font-size: 20pt; font-weight: bold;")
            self.stochastic_design_title_label.update()

        dark_stylesheet = f"""
            QMainWindow {{
                background-color: #1E1E28;
                color: {text_color.name()};
            }}
            QWidget {{
                color: {text_color.name()};
                background: transparent;
            }}
            #sidebar {{
                background-color: #161620;
                border-right: 1px solid #2D2D3D;
                border-radius: 0px;
            }}
            #logo-container {{
                background-color: #161620;
                border-bottom: 1px solid #2D2D3D;
                border-radius: 0px;
            }}
            #run-card {{
                background-color: #161620;
                border-radius: 6px;
                border: 1px solid #2D2D3D;
            }}
            QTabWidget::pane {{
                border: 1px solid #2D2D3D;
                border-radius: 6px;
                background-color: #1E1E28;
                top: -1px;
                padding: 10px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background-color: #161620;
                color: #B0B0C0;
                border: 1px solid #2D2D3D;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 20px;
                margin-right: 2px;
                font-weight: 500;
                min-height: 28px;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: #1E1E28;
                color: {primary_color.name()};
                border: 1px solid #2D2D3D;
                border-bottom: 2px solid {primary_color.name()};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #2D2D3D;
                color: #FFFFFF;
            }}
            QScrollArea, QScrollBar {{
                border: none;
                background-color: #1E1E28;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #161620;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #161620;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: #2D2D3D;
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background-color: {primary_color.name()};
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background: none;
                border: none;
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: none;
            }}
            QGroupBox {{
                border: 1px solid #2D2D3D;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 20px;
                padding-left: 15px;
                padding-right: 15px;
                padding-bottom: 15px;
                font-weight: bold;
                background-color: #161620;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 8px;
                color: {primary_color.name()};
                background-color: transparent;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton {{
                background-color: #2D2D3D;
                border: 1px solid #3D3D4D;
                border-radius: 6px;
                padding: 8px 16px;
                color: #FFFFFF;
                font-weight: 600;
                font-size: 12px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: #3D3D4D;
                border: 1px solid {primary_color.name()};
            }}
            QPushButton:pressed {{
                background-color: #161620;
            }}
            QPushButton:disabled {{
                background-color: #161620;
                color: #666677;
                border: 1px solid #2D2D3D;
            }}
            #primary-button {{
                background-color: {primary_color.name()};
                border: none;
                color: white;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
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
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 12px;
            }}
            #secondary-button:hover {{
                background-color: rgba(0, 150, 136, 0.1);
            }}
            #secondary-button:pressed {{
                background-color: rgba(0, 150, 136, 0.2);
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid #2D2D3D;
                border-radius: 6px;
                padding: 6px 10px;
                background-color: #161620;
                color: #FFFFFF;
                selection-background-color: {primary_color.name()};
                font-size: 12px;
                min-height: 24px;
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {primary_color.name()};
                background-color: #1E1E28;
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
                border: 1px solid #2D2D3D;
                background-color: #161620;
                color: #666666;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #2D2D3D;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #2D2D3D;
                background-color: #161620;
                selection-background-color: {primary_color.name()};
                border-radius: 6px;
            }}
            QToolBar {{
                background-color: #161620;
                border-bottom: 1px solid #2D2D3D;
                spacing: 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: #2D2D3D;
            }}
            QStatusBar {{
                background-color: #161620;
                color: #B0B0C0;
                border-top: 1px solid #2D2D3D;
            }}
            QMenuBar {{
                background-color: #161620;
                color: #EBEBEB;
                border-bottom: 1px solid #2D2D3D;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 6px 12px;
            }}
            QMenuBar::item:selected {{
                background-color: #2D2D3D;
                border-radius: 6px;
            }}
            QMenu {{
                background-color: #161620;
                border: 1px solid #2D2D3D;
                border-radius: 6px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 24px;
            }}
            QMenu::item:selected {{
                background-color: {primary_color.name()};
                border-radius: 6px;
            }}
            QCheckBox {{
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #2D2D3D;
                border-radius: 4px;
                background-color: #161620;
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary_color.name()};
                border: 1px solid {primary_color.name()};
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QTableView {{
                background-color: #1E1E28;
                alternate-background-color: #161620;
                border: 1px solid #2D2D3D;
                border-radius: 6px;
                gridline-color: #2D2D3D;
                selection-background-color: {primary_color.name()};
                selection-color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: #161620;
                color: #B0B0C0;
                padding: 6px;
                border: 1px solid #2D2D3D;
                font-weight: bold;
            }}
            QSplitter::handle {{
                background-color: #2D2D3D;
                margin: 2px;
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {primary_color.name()};
            }}
            QDockWidget::title {{
                text-align: center;
                background-color: #161620;
                padding: 6px;
                border-bottom: 1px solid #2D2D3D;
            }}
            QProgressBar {{
                border: 1px solid #2D2D3D;
                border-radius: 6px;
                background-color: #161620;
                text-align: center;
                color: #FFFFFF;
            }}
            QProgressBar::chunk {{
                background-color: {primary_color.name()};
                border-radius: 6px;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background-color: #2D2D3D;
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
                border: 1px solid #2D2D3D;
                background-color: #161620;
                color: #FFFFFF;
                padding: 6px;
                border-radius: 6px;
            }}
            #active-nav-btn {{
                background-color: rgba(0, 150, 136, 0.15);
                border-left: 3px solid {primary_color.name()};
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(dark_stylesheet)

    def apply_light_theme(self):
        """Apply a Refined Industrial light theme (solid colors, strict geometry)"""
        light_palette = QPalette()
        
        # Refined Industrial Base Colors
        light_color = QColor(245, 245, 250)      # #F5F5FA - Base canvas
        lighter_color = QColor(255, 255, 255)    # #FFFFFF - Slightly lighter for contrast
        medium_color = QColor(230, 230, 235)     # #E6E6EB - Borders and secondary elements
        dark_color = QColor(210, 210, 215)       # #D2D2D7 - Highlights
        text_color = QColor(45, 45, 45)          # Dark text
        disabled_text_color = QColor(150, 150, 150)
        
        # Single Vibrant Accent (Teal)
        primary_color = QColor(0, 150, 136)      # #009688
        
        light_palette.setColor(QPalette.Window, light_color)
        light_palette.setColor(QPalette.WindowText, text_color)
        light_palette.setColor(QPalette.Base, lighter_color)
        light_palette.setColor(QPalette.AlternateBase, medium_color)
        light_palette.setColor(QPalette.ToolTipBase, primary_color)
        light_palette.setColor(QPalette.ToolTipText, Qt.white)
        light_palette.setColor(QPalette.Text, text_color)
        light_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text_color)
        light_palette.setColor(QPalette.Button, lighter_color)
        light_palette.setColor(QPalette.ButtonText, text_color)
        light_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text_color)
        light_palette.setColor(QPalette.Link, primary_color)
        light_palette.setColor(QPalette.Highlight, primary_color)
        light_palette.setColor(QPalette.HighlightedText, Qt.white)
        light_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(200, 200, 200))
        light_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)
        
        self.setPalette(light_palette)
        
        if hasattr(self, 'stochastic_design_banner') and self.stochastic_design_banner is not None:
            banner_palette = self.stochastic_design_banner.palette()
            banner_palette.setColor(QPalette.Background, lighter_color)
            self.stochastic_design_banner.setPalette(banner_palette)
            self.stochastic_design_banner.update()

        if hasattr(self, 'stochastic_design_title_label') and self.stochastic_design_title_label is not None:
            self.stochastic_design_title_label.setStyleSheet("color: #2D2D2D; font-size: 20pt; font-weight: bold;")
            self.stochastic_design_title_label.update()

        light_stylesheet = f"""
            QMainWindow {{
                background-color: #F5F5FA;
                color: {text_color.name()};
            }}
            QWidget {{
                color: {text_color.name()};
                background: transparent;
            }}
            #sidebar {{
                background-color: #FFFFFF;
                border-right: 1px solid #E6E6EB;
                border-radius: 0px;
            }}
            #logo-container {{
                background-color: #FFFFFF;
                border-bottom: 1px solid #E6E6EB;
                border-radius: 0px;
            }}
            #run-card {{
                background-color: #FFFFFF;
                border-radius: 6px;
                border: 1px solid #E6E6EB;
            }}
            QTabWidget::pane {{
                border: 1px solid #E6E6EB;
                border-radius: 6px;
                background-color: #F5F5FA;
                top: -1px;
                padding: 10px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background-color: #FFFFFF;
                color: #707080;
                border: 1px solid #E6E6EB;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 20px;
                margin-right: 2px;
                font-weight: 500;
                min-height: 28px;
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: #F5F5FA;
                color: {primary_color.name()};
                border: 1px solid #E6E6EB;
                border-bottom: 2px solid {primary_color.name()};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #F0F0F5;
                color: #2D2D2D;
            }}
            QScrollArea, QScrollBar {{
                border: none;
                background-color: #F5F5FA;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #FFFFFF;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #FFFFFF;
                height: 10px;
                margin: 0px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
                background-color: #D2D2D7;
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
                background-color: {primary_color.name()};
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background: none;
                border: none;
            }}
            QScrollBar::add-page, QScrollBar::sub-page {{
                background: none;
            }}
            QGroupBox {{
                border: 1px solid #E6E6EB;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 20px;
                padding-left: 15px;
                padding-right: 15px;
                padding-bottom: 15px;
                font-weight: bold;
                background-color: #FFFFFF;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 8px;
                color: {primary_color.name()};
                background-color: transparent;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton {{
                background-color: #FFFFFF;
                border: 1px solid #D2D2D7;
                border-radius: 6px;
                padding: 8px 16px;
                color: #454545;
                font-weight: 600;
                font-size: 12px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: #F5F5FA;
                border: 1px solid {primary_color.name()};
            }}
            QPushButton:pressed {{
                background-color: #E6E6EB;
            }}
            QPushButton:disabled {{
                background-color: #F5F5FA;
                color: #B0B0B0;
                border: 1px solid #E6E6EB;
            }}
            #primary-button {{
                background-color: {primary_color.name()};
                border: none;
                color: white;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
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
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 12px;
            }}
            #secondary-button:hover {{
                background-color: rgba(0, 150, 136, 0.1);
            }}
            #secondary-button:pressed {{
                background-color: rgba(0, 150, 136, 0.2);
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 1px solid #E6E6EB;
                border-radius: 6px;
                padding: 6px 10px;
                background-color: #FFFFFF;
                color: #2D2D2D;
                selection-background-color: {primary_color.name()};
                font-size: 12px;
                min-height: 24px;
            }}
            QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border: 1px solid {primary_color.name()};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {primary_color.name()};
                background-color: #F5F5FA;
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
                border: 1px solid #E6E6EB;
                background-color: #F5F5FA;
                color: #A0A0A0;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #E6E6EB;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #E6E6EB;
                background-color: #FFFFFF;
                selection-background-color: {primary_color.name()};
                border-radius: 6px;
            }}
            QToolBar {{
                background-color: #FFFFFF;
                border-bottom: 1px solid #E6E6EB;
                spacing: 4px;
            }}
            QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: #F0F0F5;
            }}
            QStatusBar {{
                background-color: #FFFFFF;
                color: #707080;
                border-top: 1px solid #E6E6EB;
            }}
            QMenuBar {{
                background-color: #FFFFFF;
                color: #2D2D2D;
                border-bottom: 1px solid #E6E6EB;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 6px 12px;
            }}
            QMenuBar::item:selected {{
                background-color: #F0F0F5;
                border-radius: 6px;
            }}
            QMenu {{
                background-color: #FFFFFF;
                border: 1px solid #E6E6EB;
                border-radius: 6px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 24px;
            }}
            QMenu::item:selected {{
                background-color: {primary_color.name()};
                color: white;
                border-radius: 6px;
            }}
            QCheckBox {{
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #D2D2D7;
                border-radius: 4px;
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
                background-color: #F5F5FA;
                alternate-background-color: #FFFFFF;
                border: 1px solid #E6E6EB;
                border-radius: 6px;
                gridline-color: #E6E6EB;
                selection-background-color: {primary_color.name()};
                selection-color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: #FFFFFF;
                color: #707080;
                padding: 6px;
                border: 1px solid #E6E6EB;
                font-weight: bold;
            }}
            QSplitter::handle {{
                background-color: #E6E6EB;
                margin: 2px;
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {primary_color.name()};
            }}
            QDockWidget::title {{
                text-align: center;
                background-color: #FFFFFF;
                padding: 6px;
                border-bottom: 1px solid #E6E6EB;
            }}
            QProgressBar {{
                border: 1px solid #E6E6EB;
                border-radius: 6px;
                background-color: #FFFFFF;
                text-align: center;
                color: #2D2D2D;
            }}
            QProgressBar::chunk {{
                background-color: {primary_color.name()};
                border-radius: 6px;
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background-color: #E6E6EB;
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
                border: 1px solid #E6E6EB;
                background-color: #FFFFFF;
                color: #2D2D2D;
                padding: 6px;
                border-radius: 6px;
            }}
            #active-nav-btn {{
                background-color: rgba(0, 150, 136, 0.15);
                border-left: 3px solid {primary_color.name()};
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(light_stylesheet)
