"""
Matplotlib plotting canvas for the continuous beam application.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(QWidget):
    """
    A widget containing a matplotlib figure for plotting.
    
    This widget provides a convenient way to embed matplotlib plots
    in the Qt application.
    """
    
    def __init__(self, parent=None, figsize=(5, 4), dpi=100):
        """
        Initialize the plot canvas.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        figsize : tuple, optional
            Figure size in inches (width, height)
        dpi : int, optional
            Resolution in dots per inch
        """
        super().__init__(parent)
        
        # Create the figure and canvas
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        
        # Set layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        # Configure size policy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def clear(self):
        """Clear the figure."""
        self.figure.clear()
        self.canvas.draw()
        
    def draw(self):
        """Update the canvas."""
        self.canvas.draw() 