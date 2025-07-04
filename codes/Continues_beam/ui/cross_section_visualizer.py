"""
Cross-section visualizer for composite beams.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen


class CrossSectionVisualizer(QWidget):
    """
    Widget for visualizing the cross-section of a multi-layered beam.
    
    This widget displays a visual representation of the beam cross-section
    with colored layers. The visualization is updated whenever the layers
    or beam width are changed.
    """
    
    def __init__(self, parent=None):
        """Initialize the visualizer."""
        super().__init__(parent)
        self.layers = []
        self.beam_width = 0.1  # Default width
        self.setMinimumHeight(150)
        
    def set_layers(self, layers, width):
        """
        Set the layers and width for visualization.
        
        Parameters:
        -----------
        layers : list of dict
            List of layer definitions (same as used in beam.properties)
        width : float
            Beam width
        """
        self.layers = layers
        self.beam_width = width
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        """Handle paint events to draw the cross-section."""
        if not self.layers:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # White background
        painter.fillRect(self.rect(), Qt.white)
        
        # Calculate total height of the beam
        total_height = sum(layer['height'] for layer in self.layers)
        
        # Setup scaling to fit in widget
        margin = 20
        w_available = self.width() - 2*margin
        h_available = self.height() - 2*margin
        
        # Scale factors to convert from meters to pixels
        scale_x = w_available / self.beam_width
        scale_y = h_available / total_height
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Center the beam in the widget
        x_center = self.width() / 2
        y_top = margin
        
        # Width in pixels
        width_px = self.beam_width * scale
        
        # Color palette for layers
        colors = [
            QColor(41, 128, 185),    # Blue
            QColor(39, 174, 96),     # Green
            QColor(192, 57, 43),     # Red
            QColor(142, 68, 173),    # Purple
            QColor(243, 156, 18),    # Orange
            QColor(22, 160, 133),    # Teal
            QColor(211, 84, 0),      # Dark Orange
            QColor(52, 73, 94),      # Dark Blue
            QColor(127, 140, 141)    # Gray
        ]
        
        # Draw each layer
        y_pos = y_top
        for i, layer in enumerate(self.layers):
            # Get layer properties
            height_px = layer['height'] * scale
            
            # Select color for this layer
            color_idx = i % len(colors)
            color = colors[color_idx]
            
            # Create rectangle for this layer
            rect = QRectF(
                x_center - width_px/2,
                y_pos,
                width_px,
                height_px
            )
            
            # Draw filled rectangle
            painter.setBrush(QBrush(color.lighter(130)))
            painter.setPen(QPen(color.darker(120), 1))
            painter.drawRect(rect)
            
            # Move y position for next layer
            y_pos += height_px
            
        # Draw axes and center line
        painter.setPen(QPen(Qt.darkGray, 1, Qt.DashLine))
        painter.drawLine(
            int(x_center),
            int(y_top),
            int(x_center),
            int(y_top + total_height * scale)
        )
        
        # Draw width dimension
        y_dim = y_top + total_height * scale + 15
        x_left = x_center - width_px/2
        x_right = x_center + width_px/2
        
        painter.setPen(QPen(Qt.black, 1))
        # Left arrow
        painter.drawLine(
            int(x_left),
            int(y_dim),
            int(x_left + 5),
            int(y_dim - 3)
        )
        painter.drawLine(
            int(x_left),
            int(y_dim),
            int(x_left + 5),
            int(y_dim + 3)
        )
        
        # Dimension line
        painter.drawLine(
            int(x_left),
            int(y_dim),
            int(x_right),
            int(y_dim)
        )
        
        # Right arrow
        painter.drawLine(
            int(x_right),
            int(y_dim),
            int(x_right - 5),
            int(y_dim - 3)
        )
        painter.drawLine(
            int(x_right),
            int(y_dim),
            int(x_right - 5),
            int(y_dim + 3)
        )
        
        # Width label
        painter.drawText(
            int(x_center - 15),
            int(y_dim + 15),
            f"{self.beam_width:.3f} m"
        ) 