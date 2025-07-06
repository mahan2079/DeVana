"""
Beam Side View Widget

This module provides a side view visualization of the composite beam showing:
- Beam length and overall proportions
- Layer structure visible from the side
- Support conditions and boundary conditions
- Coordinate system and dimensions
- Professional engineering drawing style
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, QPainterPath, QPolygonF
import numpy as np


class BeamSideViewWidget(QWidget):
    """
    Widget for displaying the side view of a composite beam with 
    professional engineering drawing style.
    """
    
    def __init__(self, parent=None, theme='Dark'):
        super().__init__(parent)
        self.beam_length = 1.0
        self.layers = []
        self.forces = []
        self.support_type = "cantilever"  # cantilever, simply_supported, fixed_fixed
        self.theme = theme  # Store current theme
        self.setMinimumSize(500, 200)
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
        else:
            title_color = "#1976D2"
            
        # Update title style
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(f"color: {title_color}; padding: 5px;")
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        self.title_label = QLabel("Beam Side View")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Apply initial theme
        self.update_theme_styles()
        
    def set_beam_geometry(self, length, layers):
        """Set the beam geometry and layers."""
        self.beam_length = length
        self.layers = layers
        self.update()
        
    def set_forces(self, forces):
        """Set the forces to be visualized."""
        self.forces = forces
        self.update()
        
    def set_support_type(self, support_type):
        """Set the support type for the beam."""
        self.support_type = support_type
        self.update()
        
    def paintEvent(self, event):
        """Paint the side view."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        self.draw_background(painter)
        
        if not self.layers or self.beam_length <= 0:
            self.draw_placeholder(painter)
            return
            
        margin = 80
        beam_area_width = self.width() - 2 * margin
        beam_area_height = self.height() - 120
        
        total_thickness = sum(layer['thickness'] for layer in self.layers)
        
        length_scale = beam_area_width / self.beam_length
        thickness_scale = beam_area_height / total_thickness if total_thickness > 0 else beam_area_height
        thickness_scale = min(thickness_scale, length_scale / 4)

        beam_length_px = self.beam_length * length_scale
        beam_thickness_px = total_thickness * thickness_scale
        
        start_x = margin
        center_y = (self.height() - 50) / 2
        start_y = center_y - beam_thickness_px / 2
        
        beam_rect = QRectF(start_x, start_y, beam_length_px, beam_thickness_px)
        
        self.draw_beam(painter, beam_rect)
        self.draw_forces(painter, beam_rect, length_scale)
        self.draw_supports(painter, beam_rect)
        self.draw_dimensions(painter, beam_rect)
        self.draw_coordinate_system(painter, start_x, center_y)
        
    def draw_background(self, painter):
        """Draw professional background with theme-appropriate colors."""
        if self.theme == 'Dark':
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(42, 42, 45))
            gradient.setColorAt(1, QColor(35, 35, 38))
            grid_color = QColor(55, 55, 60)
            border_color = QColor(70, 110, 150, 100)
        else:
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(248, 248, 250))
            gradient.setColorAt(1, QColor(238, 238, 242))
            grid_color = QColor(230, 230, 235)
            border_color = QColor(180, 180, 190, 150)
            
        painter.fillRect(self.rect(), gradient)
        
        painter.setPen(QPen(grid_color, 0.5))
        grid_size = 25
        for x in range(grid_size, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(grid_size, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
            
        painter.setPen(QPen(border_color, 2))
        painter.drawRect(1, 1, self.width()-3, self.height()-3)
        
    def draw_placeholder(self, painter):
        """Draw placeholder visualization."""
        if self.theme == 'Dark':
            text_color = QColor(160, 160, 165)
            outline_color = QColor(120, 120, 125)
        else:
            text_color = QColor(120, 120, 125)
            outline_color = QColor(180, 180, 185)

        painter.setPen(QPen(outline_color, 2, Qt.DashLine))
        center_x, center_y = self.width() / 2, self.height() / 2
        painter.drawRect(int(center_x - 100), int(center_y - 15), 200, 30)

        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(self.rect(), Qt.AlignCenter, "No beam geometry defined")

    def draw_beam(self, painter, beam_rect):
        """Draw the composite beam with layers."""
        if self.theme == 'Dark':
            grad_start = QColor(90, 120, 150)
            grad_end = QColor(50, 80, 110)
            pen_color = QColor(120, 150, 180)
            line_color = QColor(140, 170, 200)
        else:
            grad_start = QColor(210, 220, 230)
            grad_end = QColor(170, 180, 190)
            pen_color = QColor(100, 110, 120)
            line_color = QColor(130, 140, 150)

        gradient = QLinearGradient(beam_rect.topLeft(), beam_rect.bottomLeft())
        gradient.setColorAt(0, grad_start)
        gradient.setColorAt(1, grad_end)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(pen_color, 2))
        painter.drawRect(beam_rect)
        
        if len(self.layers) > 1:
            y_pos = beam_rect.top()
            total_thickness = sum(layer['thickness'] for layer in self.layers)
            
            painter.setPen(QPen(line_color, 1, Qt.DashLine))
            
            for layer in self.layers[:-1]:
                y_pos += (layer['thickness'] / total_thickness) * beam_rect.height()
                painter.drawLine(int(beam_rect.left()), int(y_pos), int(beam_rect.right()), int(y_pos))
        
        self.draw_hatching(painter, beam_rect)

    def draw_hatching(self, painter, rect):
        """Draw hatching pattern for engineering drawing style."""
        if self.theme == 'Dark':
            hatch_color = QColor(160, 190, 220)
        else:
            hatch_color = QColor(140, 160, 180)

        painter.setPen(QPen(hatch_color, 0.5))
        spacing = 10
        for i in range(int((rect.width() + rect.height()) / spacing)):
            x1 = rect.left() + i * spacing
            y1 = rect.top()
            x2 = rect.left()
            y2 = rect.top() + i * spacing
            
            p1 = QPointF(x1, y1)
            p2 = QPointF(x2, y2)
            
            path = QPainterPath()
            path.addRect(rect)
            
            line_path = QPainterPath()
            line_path.moveTo(p1)
            line_path.lineTo(p2)
            
            clipped_path = line_path.intersected(path)
            painter.drawPath(clipped_path)

    def draw_forces(self, painter, beam_rect, scale_x):
        """Draw forces on the beam with different colors and shapes for each type."""
        if not self.forces:
            return
        for force_region in self.forces:
            if force_region.spatial_type == 'point':
                self.draw_point_forces(painter, force_region, beam_rect, scale_x)
            elif force_region.spatial_type == 'distributed':
                self.draw_distributed_forces(painter, force_region, beam_rect, scale_x)
            # Add more types as needed

    def draw_point_forces(self, painter, region, beam_rect, scale_x):
        """Draw point forces as red arrows."""
        arrow_color = QColor(220, 40, 40)  # Red
        painter.setPen(QPen(arrow_color, 2))
        painter.setBrush(QBrush(arrow_color))
        for loc in region.locations:
            pos_x = beam_rect.left() + loc.get('position', 0) * scale_x
            arrow_start_y = beam_rect.top() - 20
            arrow_end_y = beam_rect.top()
            painter.drawLine(int(pos_x), int(arrow_start_y), int(pos_x), int(arrow_end_y))
            # Arrow head
            arrow_size = 8
            painter.drawPolygon(
                QPolygonF([
                    QPointF(pos_x, arrow_start_y),
                    QPointF(pos_x - arrow_size/2, arrow_start_y + arrow_size),
                    QPointF(pos_x + arrow_size/2, arrow_start_y + arrow_size)
                ])
            )
            # Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QPen(arrow_color))
            painter.drawText(int(pos_x) + 5, int(arrow_start_y) - 5, region.name)

    def draw_distributed_forces(self, painter, region, beam_rect, scale_x):
        """Draw distributed forces as blue bands/arrows."""
        band_color = QColor(40, 100, 220, 120)  # Blue, semi-transparent
        arrow_color = QColor(40, 100, 220)
        painter.setPen(QPen(arrow_color, 2))
        painter.setBrush(QBrush(band_color))
        for loc in region.locations:
            start_x = beam_rect.left() + loc.get('start', 0) * scale_x
            end_x = beam_rect.left() + loc.get('end', 0) * scale_x
            band_rect = QRectF(start_x, beam_rect.top() - 16, end_x - start_x, 8)
            painter.drawRect(band_rect)
            # Draw arrows along the band
            num_arrows = max(2, int((end_x - start_x) / 40))
            for i in range(num_arrows):
                pos_x = start_x + (i / (num_arrows - 1)) * (end_x - start_x)
                painter.drawLine(int(pos_x), int(beam_rect.top() - 20), int(pos_x), int(beam_rect.top()))
                painter.drawPolygon(
                    QPolygonF([
                        QPointF(pos_x, beam_rect.top() - 20),
                        QPointF(pos_x - 4, beam_rect.top() - 12),
                        QPointF(pos_x + 4, beam_rect.top() - 12)
                    ])
                )
            # Label
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QPen(arrow_color))
            painter.drawText(int(start_x) + 5, int(beam_rect.top()) - 25, region.name)

    def draw_supports(self, painter, beam_rect):
        """Draw beam supports based on type."""
        center_y = beam_rect.center().y()
        if self.support_type == 'cantilever':
            self.draw_fixed_support(painter, beam_rect.left(), center_y)
        elif self.support_type == 'simply_supported':
            self.draw_pin_support(painter, beam_rect.left(), center_y)
            self.draw_roller_support(painter, beam_rect.right(), center_y)
        elif self.support_type == 'fixed_fixed':
            self.draw_fixed_support(painter, beam_rect.left(), center_y)
            self.draw_fixed_support(painter, beam_rect.right(), center_y)
            
    def draw_fixed_support(self, painter, x, y):
        if self.theme == 'Dark':
            color = QColor(120, 150, 180)
        else:
            color = QColor(100, 110, 120)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        
        painter.drawLine(int(x), int(y - 20), int(x), int(y + 20))
        for i in range(5):
            dy = i * 8 - 16
            painter.drawLine(int(x), int(y + dy), int(x - 10), int(y + dy + 8))

    def draw_pin_support(self, painter, x, y):
        if self.theme == 'Dark':
            color = QColor(120, 150, 180)
        else:
            color = QColor(100, 110, 120)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        
        path = QPainterPath()
        path.moveTo(x, y)
        path.lineTo(x - 10, y + 15)
        path.lineTo(x + 10, y + 15)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(int(x - 15), int(y + 15), int(x + 15), int(y + 15))

    def draw_roller_support(self, painter, x, y):
        if self.theme == 'Dark':
            color = QColor(120, 150, 180)
        else:
            color = QColor(100, 110, 120)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        
        path = QPainterPath()
        path.moveTo(x, y)
        path.lineTo(x - 10, y + 15)
        path.lineTo(x + 10, y + 15)
        path.closeSubpath()
        painter.drawPath(path)
        
        painter.drawEllipse(int(x - 5), int(y + 15), 10, 5)
        painter.drawLine(int(x-15), int(y+20), int(x+15), int(y+20))

    def draw_dimensions(self, painter, beam_rect):
        if self.theme == 'Dark':
            color = QColor(200, 200, 205)
        else:
            color = QColor(80, 80, 85)
        painter.setPen(QPen(color, 1))
        painter.setFont(QFont("Segoe UI", 9))

        y_pos = beam_rect.bottom() + 30
        painter.drawLine(int(beam_rect.left()), int(y_pos - 10), int(beam_rect.left()), int(y_pos + 10))
        painter.drawLine(int(beam_rect.right()), int(y_pos - 10), int(beam_rect.right()), int(y_pos + 10))
        painter.drawLine(int(beam_rect.left()), int(y_pos), int(beam_rect.right()), int(y_pos))
        
        label = f"{self.beam_length:.2f} m"
        painter.drawText(beam_rect.translated(0, 35), Qt.AlignCenter, label)

    def draw_coordinate_system(self, painter, start_x, center_y):
        if self.theme == 'Dark':
            color = QColor(220, 220, 220, 150)
        else:
            color = QColor(50, 50, 50, 150)
            
        painter.setPen(QPen(color, 2))
        origin_x, origin_y = start_x - 40, center_y
        
        # X-axis
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x + 50), int(origin_y))
        painter.drawLine(int(origin_x + 50), int(origin_y), int(origin_x + 40), int(origin_y-5))
        painter.drawLine(int(origin_x + 50), int(origin_y), int(origin_x + 40), int(origin_y+5))
        
        # Y-axis
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x), int(origin_y - 40))
        painter.drawLine(int(origin_x), int(origin_y - 40), int(origin_x - 5), int(origin_y - 30))
        painter.drawLine(int(origin_x), int(origin_y - 40), int(origin_x + 5), int(origin_y - 30))

        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(int(origin_x + 45), int(origin_y + 20), "x")
        painter.drawText(int(origin_x - 20), int(origin_y - 35), "y") 