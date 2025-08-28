import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

sns.set(style="whitegrid")
plt.rc('text', usetex=True)

import random
from deap import base, creator, tools

class PlotWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        Flexible initializer to support both call styles across mixins:
        - PlotWindow(fig, title="Plot")
        - PlotWindow(parent, fig, title="Plot")
        """
        # Parse arguments
        parent = None
        title = "Plot"
        fig = None

        if not args:
            raise TypeError("PlotWindow requires at least a matplotlib Figure")

        # If first arg looks like a Figure, use (fig, [title]) signature
        if hasattr(args[0], "add_subplot"):
            fig = args[0]
            if len(args) >= 2 and isinstance(args[1], str):
                title = args[1]
        else:
            # Assume (parent, fig, [title])
            parent = args[0]
            if len(args) >= 2:
                fig = args[1]
            if len(args) >= 3 and isinstance(args[2], str):
                title = args[2]

        if fig is None:
            raise TypeError("PlotWindow requires a matplotlib Figure")

        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        try:
            # Remove any pre-existing "Open in New Window" to avoid recursion; this is a standalone window
            for act in list(self.toolbar.actions()):
                if act.text() == "Open in New Window":
                    self.toolbar.removeAction(act)
        except Exception:
            pass
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
