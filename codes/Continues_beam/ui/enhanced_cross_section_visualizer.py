"""
Enhanced Cross-Section Visualizer for Composite Beams

This module provides a sophisticated visualization of composite beam cross-sections with:
- High-quality graphics and professional appearance
- Material property display with color coding
- Neutral axis and centroid visualization
- Layer thickness and position annotations
- Scientific accuracy with proper composite theory
- Interactive features and hover information
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, 
    QRadialGradient, QPainterPath, QPolygonF
)
import numpy as np
from ..beam.properties import calc_composite_properties


class EnhancedCrossSectionVisualizer(QWidget):
    """
    Enhanced widget for visualizing composite beam cross-sections with 
    professional graphics and scientific accuracy.
    """
    
    # Signal emitted when a layer is clicked
    layer_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None, theme='Dark'):
        super().__init__(parent)
        self.layers = []
        self.beam_width = 0.05
        self.temperature = 20.0
        self.show_neutral_axis = True
        self.show_dimensions = True
        self.show_material_props = True
        self.hovered_layer = -1
        self.theme = theme  # Store current theme
        
        self.setMinimumSize(400, 900)
        self.setMouseTracking(True)
        
        # Initialize layout
        self.init_ui()
        
    def set_theme(self, theme):
        """Update the theme and refresh the visualization."""
        self.theme = theme
        self.update_theme_styles()
        self.update()
        
    def update_theme_styles(self):
        """Update UI styles based on current theme."""
        if self.theme == 'Dark':
            title_color = "#64B5F6"
            panel_bg = "#2D2D30"
            panel_border = "#404040"
            text_color = "#F0F0F0"
        else:
            title_color = "#1976D2"
            panel_bg = "#F5F5F5"
            panel_border = "#E0E0E0"
            text_color = "#333333"
            
        # Update title style
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(f"color: {title_color}; padding: 5px;")
            
        # Update info panel style
        if hasattr(self, 'info_panel'):
            self.info_panel.setStyleSheet(f"""
                QFrame {{
                    background-color: {panel_bg};
                    border: 1px solid {panel_border};
                    border-radius: 4px;
                }}
            """)
            
        # Update label styles
        for label in [self.neutral_axis_label, self.total_thickness_label, self.area_label]:
            if hasattr(self, label.objectName()) if hasattr(label, 'objectName') else True:
                label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
        
    def init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        self.title_label = QLabel("Cross-Section View")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Main drawing area (this widget itself)
        layout.addStretch()
        
        # Information panel
        self.info_panel = self.create_info_panel()
        layout.addWidget(self.info_panel)
        
        # Apply initial theme
        self.update_theme_styles()
        
    def create_info_panel(self):
        """Create the information panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setMaximumHeight(80)
        
        layout = QHBoxLayout(panel)
        
        # Neutral axis info
        self.neutral_axis_label = QLabel("Neutral Axis: N/A")
        layout.addWidget(self.neutral_axis_label)
        
        # Total thickness info
        self.total_thickness_label = QLabel("Total Thickness: N/A")
        layout.addWidget(self.total_thickness_label)
        
        # Area info
        self.area_label = QLabel("Cross-Section Area: N/A")
        layout.addWidget(self.area_label)
        
        return panel
        
    def set_layers(self, layers, width, temperature=20.0):
        """Set the layers and parameters for visualization."""
        self.layers = layers
        self.beam_width = width
        self.temperature = temperature
        self.update_info_panel()
        self.update()
        
    def update_info_panel(self):
        """Update the information panel."""
        if not self.layers:
            self.neutral_axis_label.setText("Neutral Axis: N/A")
            self.total_thickness_label.setText("Total Thickness: N/A")
            self.area_label.setText("Cross-Section Area: N/A")
            return
            
        try:
            # Calculate total thickness
            total_thickness = sum(layer['thickness'] for layer in self.layers)
            
            # Calculate cross-sectional area
            area = self.beam_width * total_thickness
            
            # Calculate neutral axis position (simplified)
            # For a more accurate calculation, we'd need to use the transformed section method
            neutral_axis = self.calculate_neutral_axis()
            
            # Update labels
            self.neutral_axis_label.setText(f"Neutral Axis: {neutral_axis*1000:.2f} mm")
            self.total_thickness_label.setText(f"Total Thickness: {total_thickness*1000:.2f} mm")
            self.area_label.setText(f"Cross-Section Area: {area*1000000:.1f} mm²")
            
        except Exception as e:
            print(f"Error updating info panel: {e}")
            
    def calculate_neutral_axis(self):
        """Calculate the neutral axis position using composite beam theory."""
        if not self.layers:
            return 0.0
            
        # Calculate using the transformed section method
        sum_EAy = 0
        sum_EA = 0
        y_bottom = 0
        
        for layer in self.layers:
            # Get material properties at current temperature
            E = layer['E_func'](self.temperature)
            thickness = layer['thickness']
            area = self.beam_width * thickness
            
            # Centroid of this layer from bottom
            y_centroid = y_bottom + thickness / 2
            
            # Accumulate for neutral axis calculation
            sum_EAy += E * area * y_centroid
            sum_EA += E * area
            
            y_bottom += thickness
            
        # Neutral axis position from bottom
        if sum_EA > 0:
            return sum_EAy / sum_EA
        else:
            return sum(layer['thickness'] for layer in self.layers) / 2
            
    def paintEvent(self, event):
        """Handle paint events to draw the enhanced cross-section."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background with gradient
        self.draw_background(painter)
        
        if not self.layers:
            self.draw_placeholder(painter)
            return
            
        # Calculate drawing parameters
        total_height = sum(layer['thickness'] for layer in self.layers)
        margin = 50
        available_width = self.width() - 2 * margin
        available_height = self.height() - 150  # Leave space for info panel
        
        # Scale factors
        scale_x = available_width / self.beam_width
        scale_y = available_height / total_height
        scale = min(scale_x, scale_y) * 0.8  # Use 80% of available space
        
        # Calculate beam dimensions in pixels
        beam_width_px = self.beam_width * scale
        beam_height_px = total_height * scale
        
        # Center the beam
        center_x = self.width() / 2
        start_y = (self.height() - 150 - beam_height_px) / 2
        
        # Draw layers
        self.draw_layers(painter, center_x, start_y, beam_width_px, scale)
        
        # Draw neutral axis
        if self.show_neutral_axis:
            self.draw_neutral_axis(painter, center_x, start_y, beam_width_px, scale)
            
        # Draw dimensions
        if self.show_dimensions:
            self.draw_dimensions(painter, center_x, start_y, beam_width_px, beam_height_px)
            
        # Draw coordinate system
        self.draw_coordinate_system(painter, center_x, start_y, beam_width_px, beam_height_px)
        
    def draw_background(self, painter):
        """Draw professional background with theme-appropriate colors."""
        if self.theme == 'Dark':
            # Dark theme gradient background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(45, 45, 48))
            gradient.setColorAt(0.5, QColor(40, 40, 43))
            gradient.setColorAt(1, QColor(35, 35, 38))
            painter.fillRect(self.rect(), gradient)
            
            # Dark grid and border
            grid_color = QColor(60, 60, 65)
            border_color = QColor(80, 120, 160, 100)
        else:
            # Light theme gradient background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(250, 250, 252))
            gradient.setColorAt(0.5, QColor(245, 245, 248))
            gradient.setColorAt(1, QColor(240, 240, 245))
            painter.fillRect(self.rect(), gradient)
            
            # Light grid and border
            grid_color = QColor(220, 220, 225)
            border_color = QColor(180, 180, 190, 150)
        
        # Add subtle grid pattern for engineering feel
        painter.setPen(QPen(grid_color, 0.3))
        grid_spacing = 25
        
        # Draw vertical grid lines
        for x in range(grid_spacing, self.width(), grid_spacing):
            painter.drawLine(x, 0, x, self.height())
        
        # Draw horizontal grid lines  
        for y in range(grid_spacing, self.height(), grid_spacing):
            painter.drawLine(0, y, self.width(), y)
            
        # Add subtle border
        painter.setPen(QPen(border_color, 2))
        painter.drawRect(1, 1, self.width()-3, self.height()-3)
        
    def draw_placeholder(self, painter):
        """Draw placeholder text when no layers are defined."""
        if self.theme == 'Dark':
            text_color = QColor(160, 160, 165)
            outline_color = QColor(120, 120, 125)
        else:
            text_color = QColor(120, 120, 125)
            outline_color = QColor(180, 180, 185)
            
        painter.setPen(QPen(text_color, 2))
        painter.setFont(QFont("Segoe UI", 16, QFont.Light))
        
        # Draw icon-like graphics for better visual appeal
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Draw a simple beam outline
        painter.setPen(QPen(outline_color, 2, Qt.DashLine))
        painter.drawRect(int(center_x - 80), int(center_y - 30), 160, 60)
        
        # Draw placeholder text with better styling
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Segoe UI", 14, QFont.Light))
        painter.drawText(self.rect(), Qt.AlignCenter, "No layers defined\nAdd layers to see cross-section visualization")
        
    def draw_layers(self, painter, center_x, start_y, beam_width_px, scale):
        """Draw all layers with enhanced graphics."""
        # High-quality material colors
        material_colors = {
            'Steel': QColor(70, 130, 180),      # Steel Blue
            'Aluminum': QColor(192, 192, 192),   # Silver
            'Carbon Fiber': QColor(64, 64, 64), # Dark Gray
            'Titanium': QColor(135, 206, 235),  # Sky Blue
            'Foam': QColor(255, 218, 185),      # Peach
            'Custom': QColor(100, 149, 237),    # Cornflower Blue
        }
        
        y_pos = start_y
        
        for i, layer in enumerate(self.layers):
            thickness_px = layer['thickness'] * scale
            
            # Get material color
            material_type = layer.get('material_type', 'Custom')
            base_color = material_colors.get(material_type, QColor(100, 149, 237))
            
            # Create gradient for 3D effect
            gradient = QLinearGradient(center_x - beam_width_px/2, y_pos, 
                                     center_x + beam_width_px/2, y_pos)
            gradient.setColorAt(0, base_color.lighter(120))
            gradient.setColorAt(0.5, base_color)
            gradient.setColorAt(1, base_color.darker(120))
            
            # Highlight hovered layer
            if i == self.hovered_layer:
                gradient.setColorAt(0, base_color.lighter(150))
                gradient.setColorAt(0.5, base_color.lighter(130))
                gradient.setColorAt(1, base_color.lighter(110))
            
            # Draw layer rectangle
            rect = QRectF(center_x - beam_width_px/2, y_pos, beam_width_px, thickness_px)
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(base_color.darker(150), 1))
            painter.drawRect(rect)
            
            # Draw layer label with better contrast
            if thickness_px > 20:  # Only draw label if layer is thick enough
                # Use white text with shadow for better readability
                painter.setPen(QPen(Qt.white))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                label = f"{layer.get('name', f'Layer {i+1}')}\n{layer['thickness']*1000:.1f}mm"
                
                # Draw shadow first
                painter.setPen(QPen(Qt.black))
                shadow_rect = QRectF(rect.x() + 1, rect.y() + 1, rect.width(), rect.height())
                painter.drawText(shadow_rect, Qt.AlignCenter, label)
                
                # Draw main text
                painter.setPen(QPen(Qt.white))
                painter.drawText(rect, Qt.AlignCenter, label)
            
            # Draw material properties on the right
            if self.show_material_props:
                self.draw_material_properties(painter, layer, center_x + beam_width_px/2 + 10, 
                                            y_pos + thickness_px/2, i)
            
            y_pos += thickness_px
            
    def draw_material_properties(self, painter, layer, x, y, layer_index):
        """Draw material properties for a layer."""
        # Get material properties at current temperature
        E = layer['E_func'](self.temperature)
        rho = layer['rho_func'](self.temperature)
        
        # Create property text
        prop_text = f"E: {E/1e9:.1f} GPa\nρ: {rho:.0f} kg/m³"
        
        # Draw background box with theme-appropriate colors
        if self.theme == 'Dark':
            box_border = QColor(100, 100, 105)
            box_bg = QColor(55, 55, 60, 220)
            text_color = QColor(200, 200, 205)
        else:
            box_border = QColor(180, 180, 185)
            box_bg = QColor(245, 245, 250, 220)
            text_color = QColor(60, 60, 65)
            
        painter.setPen(QPen(box_border, 1))
        painter.setBrush(QBrush(box_bg))
        
        # Measure text size
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_width = metrics.width("ρ: 0000 kg/m³")  # Use longest line
        text_height = metrics.height() * 2 + 5
        
        prop_rect = QRectF(x, y - text_height/2, text_width + 10, text_height)
        painter.drawRect(prop_rect)
        
        # Draw text with theme-appropriate color
        painter.setPen(QPen(text_color))
        painter.drawText(prop_rect, Qt.AlignCenter, prop_text)
        
    def draw_neutral_axis(self, painter, center_x, start_y, beam_width_px, scale):
        """Draw the neutral axis."""
        neutral_axis_pos = self.calculate_neutral_axis()
        y_na = start_y + neutral_axis_pos * scale
        
        # Draw dashed line with glow effect (convert floats to ints)
        painter.setPen(QPen(QColor(255, 100, 100), 3, Qt.DashLine))
        painter.drawLine(int(center_x - beam_width_px/2 - 20), int(y_na), 
                        int(center_x + beam_width_px/2 + 20), int(y_na))
        
        # Draw label with background box (convert floats to ints)
        painter.setPen(QPen(QColor(255, 100, 100)))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        label_text = f"N.A. ({neutral_axis_pos*1000:.1f}mm)"
        
        # Draw background for label
        if self.theme == 'Dark':
            label_bg = QColor(60, 60, 65, 180)
        else:
            label_bg = QColor(245, 245, 250, 180)
            
        painter.setBrush(QBrush(label_bg))
        label_rect = QRectF(center_x - beam_width_px/2 - 90, y_na - 18, 80, 16)
        painter.drawRect(label_rect)
        
        painter.drawText(int(center_x - beam_width_px/2 - 80), int(y_na - 5), label_text)
        
    def draw_dimensions(self, painter, center_x, start_y, beam_width_px, beam_height_px):
        """Draw dimension lines and annotations."""
        if self.theme == 'Dark':
            dim_color = QColor(200, 200, 205)
        else:
            dim_color = QColor(80, 80, 85)
            
        painter.setPen(QPen(dim_color, 1))
        painter.setFont(QFont("Segoe UI", 9))
        
        # Width dimension
        y_dim = start_y + beam_height_px + 25
        left_x = center_x - beam_width_px/2
        right_x = center_x + beam_width_px/2
        
        # Dimension line (convert floats to ints)
        painter.drawLine(int(left_x), int(y_dim), int(right_x), int(y_dim))
        
        # End arrows
        arrow_size = 5
        painter.drawLine(int(left_x), int(y_dim - arrow_size), int(left_x), int(y_dim + arrow_size))
        painter.drawLine(int(right_x), int(y_dim - arrow_size), int(right_x), int(y_dim + arrow_size))
        
        # Dimension text
        painter.drawText(int(center_x - 20), int(y_dim + 15), f"{self.beam_width*1000:.1f} mm")
        
        # Height dimension
        x_dim = center_x - beam_width_px/2 - 40
        top_y = start_y
        bottom_y = start_y + beam_height_px
        
        # Dimension line (convert floats to ints)
        painter.drawLine(int(x_dim), int(top_y), int(x_dim), int(bottom_y))
        
        # End arrows
        painter.drawLine(int(x_dim - arrow_size), int(top_y), int(x_dim + arrow_size), int(top_y))
        painter.drawLine(int(x_dim - arrow_size), int(bottom_y), int(x_dim + arrow_size), int(bottom_y))
        
        # Dimension text (rotated)
        painter.save()
        painter.translate(int(x_dim - 15), int((top_y + bottom_y) / 2))
        painter.rotate(-90)
        total_height = sum(layer['thickness'] for layer in self.layers)
        painter.drawText(-20, 0, f"{total_height*1000:.1f} mm")
        painter.restore()
        
    def draw_coordinate_system(self, painter, center_x, start_y, beam_width_px, beam_height_px):
        """Draw coordinate system axes with professional styling."""
        # Origin point
        origin_x = center_x + beam_width_px/2 + 60
        origin_y = start_y + beam_height_px
        
        # Draw axes with bright colors for visibility
        painter.setPen(QPen(QColor(100, 180, 255), 2))
        
        # Y-axis (vertical) - convert floats to ints
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x), int(origin_y - 40))
        # Y-axis arrow
        painter.drawLine(int(origin_x), int(origin_y - 40), int(origin_x - 3), int(origin_y - 35))
        painter.drawLine(int(origin_x), int(origin_y - 40), int(origin_x + 3), int(origin_y - 35))
        
        # Z-axis (horizontal) - convert floats to ints
        painter.setPen(QPen(QColor(255, 180, 100), 2))
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x + 40), int(origin_y))
        # Z-axis arrow
        painter.drawLine(int(origin_x + 40), int(origin_y), int(origin_x + 35), int(origin_y - 3))
        painter.drawLine(int(origin_x + 40), int(origin_y), int(origin_x + 35), int(origin_y + 3))
        
        # Labels with background boxes
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        if self.theme == 'Dark':
            label_bg = QColor(60, 60, 65, 180)
        else:
            label_bg = QColor(245, 245, 250, 180)
        
        # Y-axis label
        painter.setPen(QPen(QColor(100, 180, 255)))
        painter.setBrush(QBrush(label_bg))
        y_label_rect = QRectF(origin_x - 12, origin_y - 55, 16, 16)
        painter.drawRect(y_label_rect)
        painter.drawText(int(origin_x - 8), int(origin_y - 45), "y")
        
        # Z-axis label
        painter.setPen(QPen(QColor(255, 180, 100)))
        painter.setBrush(QBrush(label_bg))
        z_label_rect = QRectF(origin_x + 40, origin_y - 8, 16, 16)
        painter.drawRect(z_label_rect)
        painter.drawText(int(origin_x + 45), int(origin_y + 5), "z")
        
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover effects."""
        if not self.layers:
            return
            
        # Calculate which layer the mouse is over
        total_height = sum(layer['thickness'] for layer in self.layers)
        margin = 50
        available_height = self.height() - 150
        scale = min((self.width() - 2*margin)/self.beam_width, available_height/total_height) * 0.8
        
        start_y = (self.height() - 150 - total_height * scale) / 2
        y_pos = start_y
        
        mouse_y = event.y()
        hovered_layer = -1
        
        for i, layer in enumerate(self.layers):
            thickness_px = layer['thickness'] * scale
            if y_pos <= mouse_y <= y_pos + thickness_px:
                hovered_layer = i
                break
            y_pos += thickness_px
            
        if hovered_layer != self.hovered_layer:
            self.hovered_layer = hovered_layer
            self.update()
            
    def mousePressEvent(self, event):
        """Handle mouse press events for layer selection."""
        if self.hovered_layer >= 0:
            self.layer_clicked.emit(self.hovered_layer)
            
    def leaveEvent(self, event):
        """Handle mouse leave events."""
        if self.hovered_layer >= 0:
            self.hovered_layer = -1
            self.update() 