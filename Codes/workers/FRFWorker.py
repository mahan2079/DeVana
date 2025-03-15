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

import random
from deap import base, creator, tools



class FRFWorker(QThread):
    finished = pyqtSignal(dict, dict)  # Emitting two dicts: with DVA and without DVA
    error = pyqtSignal(str)
    
    def __init__(self, main_params, dva_params, omega_start, omega_end, omega_points,
                 target_values_dict, weights_dict, plot_figure, show_peaks, show_slopes):
        super().__init__()
        self.main_params = main_params
        self.dva_params = dva_params
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.plot_figure = plot_figure
        self.show_peaks = show_peaks
        self.show_slopes = show_slopes

    def run(self):
        try:
            # First FRF call: With DVAs
            results_with_dva = frf(
                main_system_parameters=self.main_params,
                dva_parameters=self.dva_params,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=self.target_values_dict['mass_1'],
                weights_mass1=self.weights_dict['mass_1'],
                target_values_mass2=self.target_values_dict['mass_2'],
                weights_mass2=self.weights_dict['mass_2'],
                target_values_mass3=self.target_values_dict['mass_3'],
                weights_mass3=self.weights_dict['mass_3'],
                target_values_mass4=self.target_values_dict['mass_4'],
                weights_mass4=self.weights_dict['mass_4'],
                target_values_mass5=self.target_values_dict['mass_5'],
                weights_mass5=self.weights_dict['mass_5'],
                plot_figure=False,    # Avoid plotting in worker
                show_peaks=self.show_peaks,
                show_slopes=self.show_slopes
            )
            
            # Second FRF call: Without DVAs for Mass 1 and Mass 2
            # Assuming Mass 1 and Mass 2 are main masses and their DVA parameters are not directly influencing them
            # Therefore, to remove DVAs, set all DVA parameters to zero
            dva_params_zero = list(self.dva_params)
            for i in range(len(dva_params_zero)):
                dva_params_zero[i] = 0.0
            results_without_dva = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(dva_params_zero),
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=self.target_values_dict['mass_1'],
                weights_mass1=self.weights_dict['mass_1'],
                target_values_mass2=self.target_values_dict['mass_2'],
                weights_mass2=self.weights_dict['mass_2'],
                target_values_mass3=self.target_values_dict['mass_3'],
                weights_mass3=self.weights_dict['mass_3'],
                target_values_mass4=self.target_values_dict['mass_4'],
                weights_mass4=self.weights_dict['mass_4'],
                target_values_mass5=self.target_values_dict['mass_5'],
                weights_mass5=self.weights_dict['mass_5'],
                plot_figure=False,    # Avoid plotting in worker
                show_peaks=self.show_peaks,
                show_slopes=self.show_slopes
            )
            
            self.finished.emit(results_with_dva, results_without_dva)
        except Exception as e:
            self.error.emit(str(e))
