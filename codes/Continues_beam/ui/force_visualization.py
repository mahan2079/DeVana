"""
Force Visualization Widget

This module provides visualization of forces applied to the composite beam:
- Force arrows with proper scaling and direction
- Force magnitude and location labels
- Different force types (point, distributed, harmonic, etc.)
- Time-varying force visualization
- Scientific accuracy in force representation
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QLineF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath, QPolygonF, QLinearGradient
import numpy as np
import math


class ForceVisualizationWidget(QWidget):
    """
    Widget for visualizing forces applied to a composite beam with 
    scientific accuracy and beautiful graphics.
    """
    
    # Signal emitted when time changes (for time-varying forces)
    time_changed = pyqtSignal(float)
    
    def __init__(self, parent=None, theme='Dark'):
        super().__init__(parent)
        self.beam_length = 1.0
        self.layers = []
        self.forces = []
        self.current_time = 0.0
        self.max_force_magnitude = 1000.0
        self.animation_running = False
        self.theme = theme  # Store current theme
        
        self.setMinimumSize(600, 300)
        self.init_ui()
        self.setup_animation_timer()
        
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
        self.title_label = QLabel("Force Visualization")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Main drawing area
        layout.addStretch()
        
        # Apply initial theme
        self.update_theme_styles()
        
    def create_control_panel(self):
        """Create the control panel for force animation."""
        panel = QWidget()
        panel.setMaximumHeight(80)
        layout = QVBoxLayout(panel)
        
        # Time control
        time_layout = QHBoxLayout()
        
        time_layout.addWidget(QLabel("Time:"))
        
        self.time_label = QLabel("0.00 s")
        self.time_label.setMinimumWidth(60)
        self.time_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        time_layout.addWidget(self.time_label)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.on_time_changed)
        time_layout.addWidget(self.time_slider)
        
        # Animation controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        time_layout.addWidget(self.play_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_animation)
        time_layout.addWidget(self.reset_button)
        
        layout.addLayout(time_layout)
        
        # Force magnitude info
        info_layout = QHBoxLayout()
        
        self.max_force_label = QLabel("Max Force: N/A")
        self.max_force_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
        info_layout.addWidget(self.max_force_label)
        
        self.force_count_label = QLabel("Forces: 0")
        self.force_count_label.setStyleSheet("color: #388E3C; font-weight: bold;")
        info_layout.addWidget(self.force_count_label)
        
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        return panel
        
    def setup_animation_timer(self):
        """Setup the animation timer."""
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.advance_animation)
        
    def set_beam_and_forces(self, beam_length, layers, forces):
        """Set the beam geometry and forces."""
        self.beam_length = beam_length
        self.layers = layers
        self.forces = forces
        
        # Calculate maximum force magnitude for scaling
        self.calculate_max_force_magnitude()
        
        # Update info labels
        self.update_info_labels()
        
        self.update()
        
    def calculate_max_force_magnitude(self):
        """Calculate the maximum force magnitude for proper scaling."""
        max_mag = 0.0
        
        for region in self.forces:
            params = region.params
            amp_keys = ['amplitude', 'intensity']
            for key in amp_keys:
                if key in params:
                    max_mag = max(max_mag, abs(float(params[key])))
            
        self.max_force_magnitude = max(max_mag, 1.0)  # Minimum 1 N for scaling
        
    def update_info_labels(self):
        """Update the information labels."""
        self.max_force_label.setText(f"Max Force: {self.max_force_magnitude:.1f} N")
        self.force_count_label.setText(f"Forces: {len(self.forces)}")
        
    def paintEvent(self, event):
        """Paint the force visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw professional dark background
        self.draw_background(painter)
        
        if not self.layers:
            self.draw_placeholder(painter)
            return
            
        # Calculate drawing area
        margin = 80
        control_height = 80
        beam_area_width = self.width() - 2 * margin
        beam_area_height = self.height() - control_height - 60
        
        # Calculate beam dimensions
        total_thickness = sum(layer['thickness'] for layer in self.layers)
        length_scale = beam_area_width / self.beam_length if self.beam_length > 0 else 0
        thickness_scale = min(beam_area_height / total_thickness if total_thickness > 0 else 0, length_scale / 6 if length_scale > 0 else 0)
        
        # Beam position
        beam_length_px = self.beam_length * length_scale
        beam_thickness_px = total_thickness * thickness_scale
        start_x = margin
        center_y = control_height + beam_area_height / 2
        beam_y = center_y - beam_thickness_px / 2
        
        # Draw beam outline
        self.draw_beam_outline(painter, start_x, beam_y, beam_length_px, beam_thickness_px)
        
        # Draw forces
        self.draw_forces(painter, start_x, center_y, length_scale)
        
        # Draw coordinate system
        self.draw_coordinate_system(painter, start_x, center_y)
        
    def draw_background(self, painter):
        """Draw professional background with theme-appropriate colors."""
        if self.theme == 'Dark':
            # Dark theme gradient background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(40, 40, 43))
            gradient.setColorAt(1, QColor(32, 32, 35))
            painter.fillRect(self.rect(), gradient)
            
            grid_color = QColor(52, 52, 57)
            border_color = QColor(255, 140, 100, 80)
        else:
            # Light theme gradient background
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(250, 250, 252))
            gradient.setColorAt(1, QColor(240, 240, 245))
            painter.fillRect(self.rect(), gradient)
            
            grid_color = QColor(225, 225, 230)
            border_color = QColor(200, 150, 100, 120)
        
        # Draw subtle grid for engineering feel
        painter.setPen(QPen(grid_color, 0.5))
        grid_size = 20
        
        # Vertical lines
        for x in range(grid_size, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        
        # Horizontal lines
        for y in range(grid_size, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
            
        # Add subtle border
        painter.setPen(QPen(border_color, 2))
        painter.drawRect(1, 1, self.width()-3, self.height()-3)
        
    def draw_placeholder(self, painter):
        """Draw placeholder when no beam is defined."""
        painter.setPen(QPen(QColor(160, 160, 165), 2))
        painter.setFont(QFont("Segoe UI", 14, QFont.Light))
        
        # Draw simple beam and force outline for visual appeal
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Draw beam outline
        painter.setPen(QPen(QColor(120, 120, 125), 2, Qt.DashLine))
        painter.drawRect(int(center_x - 80), int(center_y - 10), 160, 20)
        
        # Draw force arrow
        painter.setPen(QPen(QColor(255, 120, 120), 2))
        painter.drawLine(int(center_x), int(center_y - 40), int(center_x), int(center_y - 15))
        # Arrow head
        painter.drawLine(int(center_x - 5), int(center_y - 20), int(center_x), int(center_y - 15))
        painter.drawLine(int(center_x + 5), int(center_y - 20), int(center_x), int(center_y - 15))
        
        # Draw placeholder text
        painter.setPen(QPen(QColor(160, 160, 165)))
        painter.setFont(QFont("Segoe UI", 12, QFont.Light))
        painter.drawText(self.rect(), Qt.AlignCenter, 
                        "No beam geometry defined\nSet beam parameters to see force visualization")
        
    def draw_beam_outline(self, painter, start_x, beam_y, length_px, thickness_px):
        """Draw a simple outline of the beam."""
        beam_rect = QRectF(start_x, beam_y, length_px, thickness_px)
        
        if self.theme == 'Dark':
            color = QColor(90, 120, 150)
        else:
            color = QColor(170, 180, 190)
            
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(beam_rect)
        
    def draw_forces(self, painter, start_x, center_y, length_scale):
        """Draw all forces on the beam."""
        for force in self.forces:
            if force.spatial_type == 'point':
                self.draw_point_force(painter, force, start_x, center_y, length_scale)
            elif force.spatial_type == 'distributed':
                self.draw_distributed_force(painter, force, start_x, center_y, length_scale)
            
    def get_current_force_magnitude(self, force_region):
        params = force_region.params
        force_type = force_region.force_type
        
        if force_type == 'harmonic':
            amp = float(params.get('amplitude', 0.0))
            freq = float(params.get('frequency', 1.0))
            phase = math.radians(float(params.get('phase', 0.0)))
            return amp * math.sin(2 * math.pi * freq * self.current_time + phase)
        elif force_type == 'step':
            if self.current_time >= float(params.get('start_time', 0.0)):
                return float(params.get('amplitude', 0.0))
        elif force_type == 'impulse':
            start = float(params.get('start_time', 0.0))
            duration = float(params.get('duration', 0.01))
            if start <= self.current_time <= start + duration:
                return float(params.get('amplitude', 0.0))
        return 0.0

    def draw_point_force(self, painter, force, start_x, center_y, length_scale):
        """Draw a point force."""
        if self.theme == 'Dark':
            arrow_color, text_color = QColor(255, 100, 100), QColor(255, 180, 180)
        else:
            arrow_color, text_color = QColor(200, 0, 0), QColor(180, 0, 0)

        for loc in force.locations:
            position = loc.get('position', 0.0)
            pos_x = start_x + position * length_scale
            
            magnitude = self.get_current_force_magnitude(force)
            
            arrow_length = (abs(magnitude) / self.max_force_magnitude) * 50 + 10
            
            y_start = center_y - (arrow_length if magnitude < 0 else 0)
            y_end = center_y + (arrow_length if magnitude > 0 else 0)
            
            self.draw_arrow(painter, pos_x, y_start, pos_x, y_end, arrow_color, 2)
            
            # Draw label with background
            painter.setPen(text_color)
            painter.setFont(QFont("Segoe UI", 8))
            label = f"{force.name}: {magnitude:.1f} N"
            
            if self.theme == 'Dark':
                bg_color = QColor(55, 55, 60, 200)
            else:
                bg_color = QColor(245, 245, 250, 200)
                
            metrics = painter.fontMetrics()
            text_width = metrics.width(label)
            text_height = metrics.height()
            
            label_rect = QRectF(pos_x - text_width/2 - 5, y_start - text_height - 5, text_width + 10, text_height + 4)
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(label_rect, 4, 4)
            
            painter.setPen(text_color)
            painter.drawText(label_rect, Qt.AlignCenter, label)
            
    def draw_distributed_force(self, painter, force_region, start_x, center_y, length_scale):
        """Draw a distributed force."""
        params = force_region.params
        
        if self.theme == 'Dark':
            fill_color, line_color, text_color = QColor(255, 100, 100, 40), QColor(255, 120, 120, 180), QColor(255, 180, 180)
        else:
            fill_color, line_color, text_color = QColor(200, 0, 0, 40), QColor(200, 0, 0, 180), QColor(180, 0, 0)

        for loc in force_region.locations:
            dist_start_x = start_x + loc.get('start', 0.0) * length_scale
            dist_end_x = start_x + loc.get('end', 0.0) * length_scale
            
            magnitude = self.get_current_force_magnitude(force_region)
            
            arrow_length = (abs(magnitude) / self.max_force_magnitude) * 50 + 10
            
            force_rect_y = center_y - arrow_length if magnitude < 0 else center_y
            force_rect_height = arrow_length
            
            force_rect = QRectF(dist_start_x, force_rect_y, dist_end_x - dist_start_x, force_rect_height)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(fill_color))
            painter.drawRect(force_rect)
            
            # Draw small arrows
            painter.setPen(QPen(line_color, 1))
            num_arrows = max(2, int((dist_end_x - dist_start_x) / 30))
            for i in range(num_arrows):
                pos_x = dist_start_x + (i / (num_arrows - 1)) * (dist_end_x - dist_start_x)
                y_start = center_y - (arrow_length if magnitude < 0 else 0)
                y_end = center_y + (arrow_length if magnitude > 0 else 0)
                self.draw_arrow(painter, pos_x, y_start, pos_x, y_end, line_color, 1)

            # Draw label
            painter.setPen(text_color)
            painter.setFont(QFont("Segoe UI", 8))
            label = f"{force_region.name}: {magnitude:.1f} N/m"
            painter.drawText(int(dist_start_x), int(force_rect_y - 5), label)
            
    def draw_harmonic_force(self, painter, force, start_x, center_y, beam_length_px, length_scale, force_index):
        """Draw a harmonic force (now handled by draw_point_force)."""
        pass # This is now handled by the generic force drawing methods
            
    def draw_harmonic_symbol(self, painter, x, y, color, frequency, time):
        """Draw a symbol representing a harmonic force."""
        painter.setPen(QPen(color, 1))
        
        # Draw sine wave
        points = []
        wave_width = 40
        wave_height = 15
        num_points = 20
        
        for i in range(num_points):
            t = i / (num_points - 1)
            wave_x = x + t * wave_width
            wave_y = y + wave_height * math.sin(2 * math.pi * frequency * time + 4 * math.pi * t)
            points.append((wave_x, wave_y))
            
        # Draw the wave
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1]))
            
    def draw_arrow(self, painter, x1, y1, x2, y2, color, width):
        """Draw an arrow from (x1,y1) to (x2,y2)."""
        painter.setPen(QPen(color, width))
        painter.setBrush(QBrush(color))
        
        # Draw line
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Calculate arrow head
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_length = 8
        arrow_angle = math.pi / 6  # 30 degrees
        
        # Arrow head points as QPointF
        p2 = QPointF(x2, y2)
        p3 = QPointF(x2 - arrow_length * math.cos(angle - arrow_angle), y2 - arrow_length * math.sin(angle - arrow_angle))
        p4 = QPointF(x2 - arrow_length * math.cos(angle + arrow_angle), y2 - arrow_length * math.sin(angle + arrow_angle))
        
        arrow_head = QPolygonF([p2, p3, p4])
        painter.drawPolygon(arrow_head)
        
    def draw_coordinate_system(self, painter, start_x, center_y):
        """Draw coordinate system with professional styling."""
        origin_x = start_x - 60
        origin_y = center_y + 50
        
        # X-axis (position)
        painter.setPen(QPen(QColor(255, 180, 100), 2))
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x + 30), int(origin_y))
        painter.drawLine(int(origin_x + 30), int(origin_y), int(origin_x + 25), int(origin_y - 3))
        painter.drawLine(int(origin_x + 30), int(origin_y), int(origin_x + 25), int(origin_y + 3))
        
        # Y-axis (force)
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(int(origin_x), int(origin_y), int(origin_x), int(origin_y - 30))
        painter.drawLine(int(origin_x), int(origin_y - 30), int(origin_x - 3), int(origin_y - 25))
        painter.drawLine(int(origin_x), int(origin_y - 30), int(origin_x + 3), int(origin_y - 25))
        
        # Labels with background boxes
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        # X-axis label
        painter.setPen(QPen(QColor(255, 180, 100)))
        painter.setBrush(QBrush(QColor(55, 55, 60, 180)))
        x_label_rect = QRectF(origin_x + 30, origin_y - 8, 16, 16)
        painter.drawRect(x_label_rect)
        painter.drawText(int(origin_x + 35), int(origin_y + 5), "x")
        
        # Y-axis label
        painter.setPen(QPen(QColor(255, 100, 100)))
        painter.setBrush(QBrush(QColor(55, 55, 60, 180)))
        f_label_rect = QRectF(origin_x - 12, origin_y - 45, 16, 16)
        painter.drawRect(f_label_rect)
        painter.drawText(int(origin_x - 8), int(origin_y - 35), "F")
        
    def on_time_changed(self, value):
        """Handle time slider changes."""
        # Convert slider value to time (0 to 5 seconds)
        self.current_time = value / 100.0 * 5.0
        self.time_label.setText(f"{self.current_time:.2f} s")
        self.time_changed.emit(self.current_time)
        self.update()
        
    def toggle_animation(self):
        """Toggle animation playback."""
        if self.animation_running:
            self.animation_timer.stop()
            self.play_button.setText("Play")
            self.animation_running = False
        else:
            self.animation_timer.start(50)  # 20 FPS
            self.play_button.setText("Pause")
            self.animation_running = True
            
    def advance_animation(self):
        """Advance the animation time by one step."""
        current_val = self.time_slider.value()
        if current_val < self.time_slider.maximum():
            self.time_slider.setValue(current_val + 1)
        else:
            self.animation_running = False
            self.play_button.setText("Play")
            self.animation_timer.stop()
            
    def reset_animation(self):
        """Reset the animation to the beginning."""
        self.animation_running = False
        self.play_button.setText("Play")
        self.animation_timer.stop()
        self.time_slider.setValue(0)
        
    def add_point_force(self, position, magnitude, direction=-1):
        """Legacy method - to be removed."""
        pass

    def add_harmonic_force(self, position, amplitude, frequency):
        """Legacy method - to be removed."""
        pass

    def clear_forces(self):
        """Legacy method - to be removed."""
        pass