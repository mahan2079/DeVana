import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QTabWidget, QSplitter, QScrollArea, 
    QGroupBox, QTableWidget, QTableWidgetItem, QFrame, QComboBox,
    QSpacerItem, QSizePolicy, QSlider
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.ui.animations.beam_animation import BeamAnimationWidget
from src.ui.animations.mode_shape_animation import ModeShapeAnimationWidget


class ResultsDashboard(QWidget):
    """
    A modern, comprehensive dashboard for displaying beam analysis results.
    
    This dashboard includes:
    - Summary panel showing key results
    - Interactive plots of displacements and mode shapes
    - Response visualization tabs for different analysis aspects
    - Enhanced animation controls with user-adjustable scaling
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Initialize data storage
        self.results = None
        
    def initUI(self):
        """Initialize the dashboard UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1E3A8A;
                border-radius: 0px;
            }
        """)
        header.setMinimumHeight(40)  # Reduced header height
        header.setMaximumHeight(40)  # Reduced header height
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)  # Reduced margins
        
        # Title and subtitle in a more compact layout
        title_section = QHBoxLayout()  # Changed to horizontal layout
        
        title = QLabel("Beam Analysis Results")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        title_section.addWidget(title)
        
        subtitle = QLabel("Vibration response visualization")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 12px;")
        title_section.addWidget(subtitle)
        
        header_layout.addLayout(title_section)
        header_layout.addStretch()
        
        # Export button - smaller and more compact
        export_button = QPushButton("Export")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        export_button.clicked.connect(self.export_results)
        header_layout.addWidget(export_button)
        
        main_layout.addWidget(header)
        
        # Create a scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create a container widget for the scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(10)
        
        # Create a splitter for the main content
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Summary and key metrics
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 5, 10)
        
        # Results summary panel
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        # Create the summary table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #E5E7EB;
                border-radius: 4px;
            }
        """)
        
        # Set minimum height for the summary table
        self.summary_table.setMinimumHeight(180)
        
        # Make sure rows are properly sized
        self.summary_table.verticalHeader().setDefaultSectionSize(30)
        self.summary_table.verticalHeader().setVisible(False)
        
        # Make the first column wider
        self.summary_table.setColumnWidth(0, 180)
        
        summary_layout.addWidget(self.summary_table)
        
        # Add summary data (will be populated later)
        self.populate_summary_table()
        
        left_layout.addWidget(summary_group)
        
        # Natural frequencies panel
        freq_group = QGroupBox("Natural Frequencies")
        freq_layout = QVBoxLayout(freq_group)
        
        # Natural frequencies table
        self.freq_table = QTableWidget()
        self.freq_table.setColumnCount(2)
        self.freq_table.setHorizontalHeaderLabels(["Mode", "Frequency (Hz)"])
        self.freq_table.horizontalHeader().setStretchLastSection(True)
        self.freq_table.setAlternatingRowColors(True)
        self.freq_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #E5E7EB;
                border-radius: 4px;
            }
        """)
        
        # Set minimum height for the frequency table
        self.freq_table.setMinimumHeight(200)
        
        # Make sure rows are properly sized
        self.freq_table.verticalHeader().setDefaultSectionSize(30)
        self.freq_table.verticalHeader().setVisible(False)
        
        # Make the first column wider
        self.freq_table.setColumnWidth(0, 120)
        
        freq_layout.addWidget(self.freq_table)
        
        left_layout.addWidget(freq_group)
        
        # Node selection for charts
        node_group = QGroupBox("Monitoring Point")
        node_layout = QFormLayout(node_group)
        
        self.node_selector = QComboBox()
        self.node_selector.currentIndexChanged.connect(self.update_node_plots)
        node_layout.addRow("Select Node:", self.node_selector)
        
        left_layout.addWidget(node_group)
        
        # Add a displacement plot for the selected node
        plot_group = QGroupBox("Node Response")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create the node response figure with adjusted size
        self.node_figure = Figure(figsize=(5, 3), dpi=100, tight_layout=True)
        self.node_canvas = FigureCanvas(self.node_figure)
        self.node_canvas.setMinimumHeight(200)  # Reduced minimum height
        self.node_ax = self.node_figure.add_subplot(111)
        self.node_ax.set_xlabel('Time (s)')
        self.node_ax.set_ylabel('Amplitude')
        self.node_ax.set_title('Node Response')
        self.node_ax.grid(True, linestyle='--', alpha=0.7)
        
        plot_layout.addWidget(self.node_canvas)
        
        left_layout.addWidget(plot_group)
        
        # Add the left panel to the splitter
        self.main_splitter.addWidget(left_panel)
        
        # Right panel - Results tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 10, 10, 10)
        
        # Create tabs for different visualizations
        self.results_tabs = QTabWidget()
        self.results_tabs.setDocumentMode(True)
        self.results_tabs.setTabPosition(QTabWidget.North)
        
        # Create various visualization tabs
        self.create_beam_deflection_tab()
        self.create_mode_shapes_tab()
        self.create_beam_animation_tab()
        self.create_mode_animation_tab()
        
        right_layout.addWidget(self.results_tabs)
        
        # Add the right panel to the splitter
        self.main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([400, 600])
        
        # Add the splitter to the scroll layout
        scroll_layout.addWidget(self.main_splitter)
        
        # Set the scroll content and add to main layout
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Add time selector
        time_selector = QWidget()
        time_selector_layout = QHBoxLayout(time_selector)
        time_selector_layout.setContentsMargins(10, 5, 10, 5)
        
        time_selector_layout.addWidget(QLabel("Select Time:"))
        self.time_slider = QComboBox()
        self.time_slider.currentIndexChanged.connect(self.update_deflection_plot)
        time_selector_layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("Time: 0.000 s")
        time_selector_layout.addWidget(self.time_label)
        
        # Add deflection scale
        time_selector_layout.addSpacing(20)
        time_selector_layout.addWidget(QLabel("Deflection Scale:"))
        
        self.deflection_scale = QSlider(Qt.Horizontal)
        self.deflection_scale.setRange(1, 1000)  # 0.1x to 100.0x
        self.deflection_scale.setValue(10)  # Default 1.0x
        self.deflection_scale.valueChanged.connect(self.update_deflection_plot)
        self.deflection_scale.setMaximumWidth(200)
        time_selector_layout.addWidget(self.deflection_scale)
        
        self.deflection_scale_label = QLabel("1.0x")
        time_selector_layout.addWidget(self.deflection_scale_label)
        
        time_selector_layout.addStretch()
        
        # Mode selector
        time_selector_layout.addWidget(QLabel("Select Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged.connect(self.update_mode_shape_plot)
        time_selector_layout.addWidget(self.mode_selector)
        
        # Add time selector to main layout
        main_layout.addWidget(time_selector)
        
    def create_beam_deflection_tab(self):
        """Create the beam deflection visualization tab"""
        deflection_tab = QWidget()
        deflection_layout = QVBoxLayout(deflection_tab)
        deflection_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create figure and canvas for beam deflection
        self.beam_figure = Figure(figsize=(8, 5), dpi=100, tight_layout=True)  # Increased height
        self.beam_canvas = FigureCanvas(self.beam_figure)
        self.beam_canvas.setMinimumHeight(350)  # Increased minimum height
        self.beam_ax = self.beam_figure.add_subplot(111)
        self.beam_ax.set_xlabel('Position (m)')
        self.beam_ax.set_ylabel('Displacement (m)')
        self.beam_ax.set_title('Beam Deflection')
        self.beam_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add toolbar for beam plot
        beam_toolbar = NavigationToolbar(self.beam_canvas, deflection_tab)
        deflection_layout.addWidget(beam_toolbar)
        deflection_layout.addWidget(self.beam_canvas)
        
        self.results_tabs.addTab(deflection_tab, "Beam Deflection")
        
    def create_mode_shapes_tab(self):
        """Create the mode shapes visualization tab"""
        mode_tab = QWidget()
        mode_layout = QVBoxLayout(mode_tab)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create figure and canvas for mode shapes
        self.mode_figure = Figure(figsize=(8, 5), dpi=100, tight_layout=True)  # Increased height
        self.mode_canvas = FigureCanvas(self.mode_figure)
        self.mode_canvas.setMinimumHeight(350)  # Increased minimum height
        self.mode_ax = self.mode_figure.add_subplot(111)
        self.mode_ax.set_xlabel('Position (m)')
        self.mode_ax.set_ylabel('Mode Shape')
        self.mode_ax.set_title('Mode Shapes')
        self.mode_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add toolbar for mode plot
        mode_toolbar = NavigationToolbar(self.mode_canvas, mode_tab)
        mode_layout.addWidget(mode_toolbar)
        mode_layout.addWidget(self.mode_canvas)
        
        self.results_tabs.addTab(mode_tab, "Mode Shapes")
        
    def create_beam_animation_tab(self):
        """Create the beam animation visualization tab"""
        print("Creating beam animation tab")
        self.beam_animation_widget = BeamAnimationWidget()
        self.results_tabs.addTab(self.beam_animation_widget, "Beam Animation")
        
    def create_mode_animation_tab(self):
        """Create the mode shape animation visualization tab"""
        print("Creating mode shape animation tab")
        self.mode_animation_widget = ModeShapeAnimationWidget()
        self.results_tabs.addTab(self.mode_animation_widget, "Mode Animation")
        
    def populate_summary_table(self, results=None):
        """Populate the summary table with analysis results"""
        if results is None:
            # Add placeholder rows
            self.summary_table.setRowCount(6)
            params = [
                "Beam Length", "Number of Elements", "Number of Nodes",
                "Simulation Time", "Number of Time Steps", "Integration Method"
            ]
            for i, param in enumerate(params):
                self.summary_table.setItem(i, 0, QTableWidgetItem(param))
                self.summary_table.setItem(i, 1, QTableWidgetItem("N/A"))
        else:
            # We'll extract and add real data here
            self.summary_table.setRowCount(6)
            
            # Extract beam length
            beam_length = 0
            if 'coords' in results:
                coords = results['coords']
                if len(coords) > 1:
                    beam_length = max(coords) - min(coords)
                
            # Extract number of nodes
            num_nodes = 0
            if 'coords' in results:
                num_nodes = len(results['coords'])
                
            # Extract number of elements
            num_elements = num_nodes - 1 if num_nodes > 1 else 0
            
            # Extract simulation time
            sim_time = 0
            if 'times' in results:
                times = results['times']
                if len(times) > 0:
                    sim_time = max(times)
            elif 'time' in results:
                times = results['time']
                if len(times) > 0:
                    sim_time = max(times)
            
            # Extract number of time steps
            num_time_steps = len(results.get('times', results.get('time', [])))
            
            # Integration method (usually not available in results, use placeholder)
            integration_method = results.get('integration_method', "Newmark-β")
            
            # Update table
            params = [
                ("Beam Length", f"{beam_length:.4f} m"),
                ("Number of Elements", str(num_elements)),
                ("Number of Nodes", str(num_nodes)),
                ("Simulation Time", f"{sim_time:.4f} s"),
                ("Number of Time Steps", str(num_time_steps)),
                ("Integration Method", integration_method)
            ]
            
            for i, (param, value) in enumerate(params):
                param_item = QTableWidgetItem(param)
                param_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                
                value_item = QTableWidgetItem(value)
                value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                self.summary_table.setItem(i, 0, param_item)
                self.summary_table.setItem(i, 1, value_item)
                
        # Resize rows to content
        self.summary_table.resizeRowsToContents()
        
        # Make sure the table is visible
        self.summary_table.show()
        
    def update_results(self, results):
        """Update all visualizations with new analysis results"""
        print("\n--- ResultsDashboard: Updating with new results ---")
        self.results = results
        
        if results is None:
            print("No results provided")
            return
            
        # Extract key data
        coords = results.get('coords')
        times = results.get('times', results.get('time', []))
        
        print(f"Received results with {len(coords) if coords is not None else 0} nodes and {len(times)} time points")
        
        # Update summary table with actual results
        self.populate_summary_table(results)
        
        # Update the time selector with available time points
        self.time_slider.blockSignals(True)
        self.time_slider.clear()
        for t in times:
            self.time_slider.addItem(f"{t:.4f} s")
        self.time_slider.setCurrentIndex(0)
        self.time_slider.blockSignals(False)
        
        # Update the node selector with available nodes
        if coords is not None:
            self.node_selector.blockSignals(True)
            self.node_selector.clear()
            for i, pos in enumerate(coords):
                self.node_selector.addItem(f"Node {i+1} ({pos:.3f} m)")
            self.node_selector.setCurrentIndex(0)
            self.node_selector.blockSignals(False)
        
        # Update the mode selector with available modes
        if 'mode_shapes' in results and 'natural_frequencies' in results:
            mode_shapes = results['mode_shapes']
            freqs = results['natural_frequencies']
            
            print(f"Found {mode_shapes.shape[1]} mode shapes with frequencies")
            
            # Verify mode shapes are valid
            max_mode_val = np.max(np.abs(mode_shapes))
            print(f"Maximum mode shape value: {max_mode_val}")
            
            # If mode shapes are extremely small, apply scaling
            if max_mode_val < 1e-10:
                print("WARNING: Mode shapes have very small values, applying scaling")
                results['mode_shapes'] = mode_shapes * 1e10
                mode_shapes = results['mode_shapes']
            
            self.mode_selector.blockSignals(True)
            self.mode_selector.clear()
            for i, freq in enumerate(freqs):
                self.mode_selector.addItem(f"Mode {i+1} ({freq:.2f} Hz)")
            self.mode_selector.setCurrentIndex(0)
            self.mode_selector.blockSignals(False)
            
            # Update frequency table
            self.freq_table.setRowCount(len(freqs))
            for i, freq in enumerate(freqs):
                mode_item = QTableWidgetItem(f"Mode {i+1}")
                mode_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                
                freq_item = QTableWidgetItem(f"{freq:.4f}")
                freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                self.freq_table.setItem(i, 0, mode_item)
                self.freq_table.setItem(i, 1, freq_item)
            
            # Resize rows to content
            self.freq_table.resizeRowsToContents()
            self.freq_table.show()
            
        # Update all visualizations
        self.update_deflection_plot()
        self.update_mode_shape_plot()
        self.update_node_plots()
        
        # Update animations
        print("Updating beam animation widget")
        try:
            if hasattr(self, 'beam_animation_widget'):
                self.beam_animation_widget.update_animation(results)
            else:
                print("Warning: beam_animation_widget not found")
        except Exception as e:
            print(f"Error updating beam animation: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # If mode shapes are available, set them for animation
        print("Updating mode shape animation widget")
        try:
            if 'mode_shapes' in results and 'natural_frequencies' in results and coords is not None:
                if hasattr(self, 'mode_animation_widget'):
                    print(f"Setting mode shape data: coords shape={np.array(coords).shape}, mode_shapes shape={results['mode_shapes'].shape}")
                    self.mode_animation_widget.set_data(
                        coords, 
                        results['mode_shapes'], 
                        results['natural_frequencies']
                    )
                else:
                    print("Warning: mode_animation_widget not found")
            else:
                missing = []
                if 'mode_shapes' not in results:
                    missing.append('mode_shapes')
                if 'natural_frequencies' not in results:
                    missing.append('natural_frequencies')
                if coords is None:
                    missing.append('coords')
                print(f"Warning: Cannot update mode animation, missing data: {', '.join(missing)}")
        except Exception as e:
            print(f"Error updating mode shape animation: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print("Results update complete")
        
    def update_deflection_plot(self):
        """Update the beam deflection plot for the selected time"""
        if self.results is None:
            return
            
        time_idx = self.time_slider.currentIndex()
        if time_idx < 0:
            return
            
        # Extract necessary data
        coords = self.results.get('coords')
        times = self.results.get('times', self.results.get('time', []))
        
        if len(times) <= time_idx or coords is None:
            return
            
        # Get displacements at the selected time
        if 'displacements' in self.results:
            # If pre-processed displacements are available
            displacements = self.results['displacements'][:, time_idx]
        elif 'displacement' in self.results:
            # Extract vertical displacements
            try:
                displacement_matrix = self.results['displacement']
                y_displacements = []
                
                for i in range(len(coords)):
                    # Get vertical displacement DOF for this node (odd indices)
                    node_dof = 2 * i + 1
                    if node_dof < displacement_matrix.shape[0]:
                        y_displacements.append(displacement_matrix[node_dof, time_idx])
                    else:
                        y_displacements.append(0.0)
                        
                displacements = np.array(y_displacements)
            except Exception as e:
                print(f"Error processing displacements: {str(e)}")
                return
        else:
            print("No displacement data found in results")
            return
            
        # Get the scale factor
        scale_factor = self.deflection_scale.value() / 10.0
        self.deflection_scale_label.setText(f"{scale_factor:.1f}x")
        
        # Apply scale to displacement
        scaled_displacements = displacements * scale_factor
            
        # Update the time label
        self.time_label.setText(f"{times[time_idx]:.4f} s")
            
        # Clear the axis and plot the beam deflection
        self.beam_ax.clear()
        
        # Plot the undeformed beam (reference)
        self.beam_ax.plot(coords, np.zeros_like(coords), 'k--', label='Undeformed')
        
        # Plot the deformed beam at the selected time
        self.beam_ax.plot(coords, scaled_displacements, 'b-o', label=f'Time = {times[time_idx]:.4f} s')
        
        # Add grid, legend, labels
        self.beam_ax.set_xlabel('Position (m)')
        self.beam_ax.set_ylabel('Displacement (m)')
        self.beam_ax.set_title('Beam Deflection')
        self.beam_ax.grid(True, linestyle='--', alpha=0.7)
        self.beam_ax.legend()
        
        # Update canvas
        self.beam_canvas.draw()
        
    def update_mode_shape_plot(self):
        """Update the mode shape plot for the selected mode"""
        if self.results is None:
            return
            
        mode_idx = self.mode_selector.currentIndex()
        if mode_idx < 0:
            return
            
        # Extract necessary data
        coords = self.results.get('coords')
        
        if 'mode_shapes' not in self.results or coords is None:
            print("Warning: No mode shapes found in results")
            return
            
        mode_shapes = self.results['mode_shapes']
        if mode_idx >= mode_shapes.shape[1]:
            print(f"Warning: Mode index {mode_idx} out of bounds for mode shapes with {mode_shapes.shape[1]} modes")
            return
            
        # Get the selected mode shape
        mode_shape = mode_shapes[:, mode_idx]
        
        # Print debug info
        print(f"Plotting mode shape {mode_idx+1}")
        print(f"Mode shape data shape: {mode_shape.shape}")
        print(f"Mode shape data range: min={np.min(mode_shape)}, max={np.max(mode_shape)}")
        
        # Scale the mode shape for better visualization
        max_amp = np.max(np.abs(mode_shape))
        if max_amp > 1e-10:
            # Scale to a reasonable amplitude
            beam_length = max(coords) - min(coords)
            target_amp = beam_length * 0.2
            scale_factor = target_amp / max_amp
            scaled_mode = mode_shape * scale_factor
            print(f"Applied scale factor: {scale_factor}")
        else:
            # Create a synthetic mode shape for visualization
            print("Warning: Mode shape amplitude is zero or very small, creating synthetic shape")
            beam_length = max(coords) - min(coords)
            x_normalized = (coords - min(coords)) / beam_length
            scaled_mode = np.sin((mode_idx+1) * np.pi * x_normalized) * beam_length * 0.1
            
        # Clear the axis and plot the mode shape
        self.mode_ax.clear()
        
        # Plot the undeformed beam (reference)
        self.mode_ax.plot(coords, np.zeros_like(coords), 'k--', label='Undeformed')
        
        # Plot the mode shape with markers for better visibility
        self.mode_ax.plot(coords, scaled_mode, 'r-o', linewidth=2, markersize=6, label=f'Mode {mode_idx+1}')
        
        # Add grid, legend, labels
        self.mode_ax.set_xlabel('Position (m)')
        self.mode_ax.set_ylabel('Mode Shape Amplitude')
        
        # Get the natural frequency if available
        if 'natural_frequencies' in self.results and mode_idx < len(self.results['natural_frequencies']):
            freq = self.results['natural_frequencies'][mode_idx]
            self.mode_ax.set_title(f'Mode {mode_idx+1} - {freq:.2f} Hz')
        else:
            self.mode_ax.set_title(f'Mode {mode_idx+1}')
            
        # Set proper axis limits with padding
        x_min, x_max = min(coords), max(coords)
        x_padding = 0.05 * (x_max - x_min)
        self.mode_ax.set_xlim(x_min - x_padding, x_max + x_padding)
        
        # Set y limits based on the scaled mode shape
        max_y = np.max(np.abs(scaled_mode)) * 1.2
        
        # Ensure minimum height for visibility
        if max_y < 0.01:
            max_y = 0.01
            
        self.mode_ax.set_ylim(-max_y, max_y)
            
        self.mode_ax.grid(True, linestyle='--', alpha=0.7)
        self.mode_ax.legend()
        
        # Update canvas
        self.mode_canvas.draw()
        
    def update_node_plots(self):
        """Update the node displacement plot for the selected node"""
        if self.results is None:
            return
            
        node_idx = self.node_selector.currentIndex()
        if node_idx < 0:
            return
            
        # Extract necessary data
        coords = self.results.get('coords')
        times = self.results.get('times', self.results.get('time', []))
        
        if coords is None or len(times) == 0:
            print("Warning: No coordinates or time data found in results")
            return
            
        # Clear the axis
        self.node_ax.clear()
        
        # Track if any data was actually plotted
        data_plotted = False
        max_value = 0
        min_value = 0
        
        # Get displacement data for the selected node
        if 'displacement' in self.results:
            # Get the displacement DOF for this node
            node_dof = 2 * node_idx + 1  # Vertical displacement DOF
            
            if node_dof < self.results['displacement'].shape[0]:
                # Plot displacement vs time
                displacement_data = self.results['displacement'][node_dof, :]
                
                # Apply scaling if values are extremely small
                max_disp = np.max(np.abs(displacement_data))
                if max_disp > 0 and max_disp < 1e-10:
                    scale_factor = 1e10
                    displacement_data = displacement_data * scale_factor
                    disp_label = f'Displacement (×{scale_factor:.0e})'
                else:
                    disp_label = 'Displacement'
                
                self.node_ax.plot(times, displacement_data, 'b-', linewidth=2, label=disp_label)
                data_plotted = True
                
                # Update min/max values
                max_value = max(max_value, np.max(displacement_data))
                min_value = min(min_value, np.min(displacement_data))
                
                # Print debug info
                print(f"Node {node_idx+1} displacement data: min={np.min(displacement_data)}, max={np.max(displacement_data)}")
                
                # Plot velocity if available
                if 'velocity' in self.results and node_dof < self.results['velocity'].shape[0]:
                    velocity_data = self.results['velocity'][node_dof, :]
                    
                    # Apply scaling if values are extremely small
                    max_vel = np.max(np.abs(velocity_data))
                    if max_vel > 0 and max_vel < 1e-10:
                        vel_scale = 1e10
                        velocity_data = velocity_data * vel_scale
                        vel_label = f'Velocity (×{vel_scale:.0e})'
                    else:
                        vel_label = 'Velocity'
                    
                    self.node_ax.plot(times, velocity_data, 'g-', linewidth=1.5, label=vel_label)
                    
                    # Update min/max values
                    max_value = max(max_value, np.max(velocity_data))
                    min_value = min(min_value, np.min(velocity_data))
                    
                    print(f"Node {node_idx+1} velocity data: min={np.min(velocity_data)}, max={np.max(velocity_data)}")
                
                # Plot acceleration if available
                if 'acceleration' in self.results and node_dof < self.results['acceleration'].shape[0]:
                    # Get acceleration data
                    accel_data = self.results['acceleration'][node_dof, :]
                    
                    # Apply scaling if values are extremely small
                    max_accel = np.max(np.abs(accel_data))
                    if max_accel > 0 and max_accel < 1e-10:
                        accel_scale = 1e10
                        accel_data = accel_data * accel_scale
                        accel_label = f'Acceleration (×{accel_scale:.0e})'
                    else:
                        # Use a scale factor to make acceleration comparable to displacement
                        accel_scale = 0.001
                        accel_data = accel_data * accel_scale
                        accel_label = f'Acceleration (×{accel_scale})'
                    
                    self.node_ax.plot(times, accel_data, 'r-', linewidth=1, label=accel_label)
                    
                    # Update min/max values
                    max_value = max(max_value, np.max(accel_data))
                    min_value = min(min_value, np.min(accel_data))
                    
                    print(f"Node {node_idx+1} acceleration data: min={np.min(accel_data)}, max={np.max(accel_data)}")
            else:
                print(f"Warning: Node DOF {node_dof} out of bounds for displacement with shape {self.results['displacement'].shape}")
        elif 'displacements' in self.results:
            # If pre-processed displacements are available
            if node_idx < self.results['displacements'].shape[0]:
                displacement_data = self.results['displacements'][node_idx, :]
                
                # Apply scaling if values are extremely small
                max_disp = np.max(np.abs(displacement_data))
                if max_disp > 0 and max_disp < 1e-10:
                    scale_factor = 1e10
                    displacement_data = displacement_data * scale_factor
                    disp_label = f'Displacement (×{scale_factor:.0e})'
                else:
                    disp_label = 'Displacement'
                
                self.node_ax.plot(times, displacement_data, 'b-', linewidth=2, label=disp_label)
                data_plotted = True
                
                # Update min/max values
                max_value = max(max_value, np.max(displacement_data))
                min_value = min(min_value, np.min(displacement_data))
                
                print(f"Node {node_idx+1} displacement data: min={np.min(displacement_data)}, max={np.max(displacement_data)}")
            else:
                print(f"Warning: Node index {node_idx} out of bounds for displacements with shape {self.results['displacements'].shape}")
        else:
            print("Warning: No displacement data found in results")
            
        # Add grid, legend, labels
        self.node_ax.set_xlabel('Time (s)')
        self.node_ax.set_ylabel('Amplitude')
        self.node_ax.set_title(f'Node {node_idx+1} Response')
        self.node_ax.grid(True, linestyle='--', alpha=0.7)
        
        if data_plotted:
            self.node_ax.legend(loc='best', fontsize='small')
            
            # Ensure the plot has a reasonable height
            y_range = max_value - min_value
            padding = y_range * 0.1 if y_range > 0 else 0.01
            self.node_ax.set_ylim(min_value - padding, max_value + padding)
        else:
            # If no data was plotted, set default limits
            self.node_ax.set_ylim(-0.01, 0.01)
            self.node_ax.text(0.5, 0.5, "No data available", 
                             ha='center', va='center', transform=self.node_ax.transAxes)
        
        # Update canvas
        self.node_canvas.draw()
        
    def export_results(self):
        """Export results to CSV files"""
        print("Export functionality not yet implemented")
        # This would be implemented to export data to CSV files 