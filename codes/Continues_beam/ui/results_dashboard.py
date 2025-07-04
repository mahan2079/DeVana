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

from .beam_animation import BeamAnimationWidget
from .mode_shape_animation import ModeShapeAnimationWidget


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
        header.setMinimumHeight(60)
        header.setMaximumHeight(60)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 10, 20, 10)
        
        # Title and subtitle
        title_section = QVBoxLayout()
        
        title = QLabel("Beam Analysis Results")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        title_section.addWidget(title)
        
        subtitle = QLabel("Comprehensive visualization of beam vibration response")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.7); font-size: 12px;")
        title_section.addWidget(subtitle)
        
        header_layout.addLayout(title_section)
        header_layout.addStretch()
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
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
        
        self.node_figure = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        self.node_canvas = FigureCanvas(self.node_figure)
        self.node_ax = self.node_figure.add_subplot(111)
        self.node_ax.set_xlabel('Time (s)')
        self.node_ax.set_ylabel('Displacement (m)')
        self.node_ax.set_title('Node Displacement vs Time')
        self.node_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add toolbar for node plot
        node_toolbar = NavigationToolbar(self.node_canvas, plot_group)
        plot_layout.addWidget(node_toolbar)
        plot_layout.addWidget(self.node_canvas)
        
        left_layout.addWidget(plot_group)
        
        # Add the left panel to the splitter
        self.main_splitter.addWidget(left_panel)
        
        # Right panel - Results tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 10, 10, 10)
        
        # Create tab widget for different visualizations
        self.results_tabs = QTabWidget()
        self.results_tabs.setDocumentMode(True)
        self.results_tabs.setTabPosition(QTabWidget.North)
        
        # Create tabs for different visualizations
        
        # 1. Beam Deflection tab
        self.create_beam_deflection_tab()
        
        # 2. Mode Shapes tab
        self.create_mode_shapes_tab()
        
        # 3. Beam Animation tab
        self.create_beam_animation_tab()
        
        # 4. Mode Shape Animation tab
        self.create_mode_animation_tab()
        
        right_layout.addWidget(self.results_tabs)
        
        # Add the right panel to the splitter
        self.main_splitter.addWidget(right_panel)
        
        # Set initial sizes for splitter (40% left, 60% right)
        self.main_splitter.setSizes([400, 600])
        
        # Add the splitter to the scroll layout
        scroll_layout.addWidget(self.main_splitter)
        
        # Set the scroll content
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
    def create_beam_deflection_tab(self):
        """Create the beam deflection visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        
        # Time slider section
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Time:"))
        
        self.time_slider = QComboBox()  # Using combo box instead of slider for precise time selection
        self.time_slider.currentIndexChanged.connect(self.update_deflection_plot)
        slider_layout.addWidget(self.time_slider, 1)  # Give it a stretch factor of 1
        
        self.time_label = QLabel("0.00 s")
        slider_layout.addWidget(self.time_label)
        
        layout.addLayout(slider_layout)
        
        # Beam deflection plot
        self.beam_figure = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self.beam_canvas = FigureCanvas(self.beam_figure)
        self.beam_ax = self.beam_figure.add_subplot(111)
        self.beam_ax.set_xlabel('Position (m)')
        self.beam_ax.set_ylabel('Displacement (m)')
        self.beam_ax.set_title('Beam Deflection')
        self.beam_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add toolbar for beam deflection plot
        beam_toolbar = NavigationToolbar(self.beam_canvas, tab)
        layout.addWidget(beam_toolbar)
        layout.addWidget(self.beam_canvas, 1)  # Give it a stretch factor of 1
        
        # Add scale control
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Displacement Scale:"))
        
        self.deflection_scale = QSlider(Qt.Horizontal)
        self.deflection_scale.setRange(1, 1000)  # 0.1x to 100.0x
        self.deflection_scale.setValue(10)  # Default to 1.0x
        self.deflection_scale.setTickPosition(QSlider.TicksBelow)
        self.deflection_scale.setTickInterval(100)
        self.deflection_scale.valueChanged.connect(self.update_deflection_plot)
        scale_layout.addWidget(self.deflection_scale)
        
        self.deflection_scale_label = QLabel("1.0x")
        scale_layout.addWidget(self.deflection_scale_label)
        
        layout.addLayout(scale_layout)
        
        self.results_tabs.addTab(tab, "Beam Deflection")
        
    def create_mode_shapes_tab(self):
        """Create the mode shapes visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        
        # Mode selection section
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        
        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged.connect(self.update_mode_shape_plot)
        mode_layout.addWidget(self.mode_selector, 1)
        
        self.freq_display = QLabel("0.00 Hz")
        mode_layout.addWidget(self.freq_display)
        
        layout.addLayout(mode_layout)
        
        # Mode shape plot
        self.mode_figure = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self.mode_canvas = FigureCanvas(self.mode_figure)
        self.mode_ax = self.mode_figure.add_subplot(111)
        self.mode_ax.set_xlabel('Position (m)')
        self.mode_ax.set_ylabel('Mode Shape Amplitude')
        self.mode_ax.set_title('Mode Shape')
        self.mode_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add toolbar for mode shape plot
        mode_toolbar = NavigationToolbar(self.mode_canvas, tab)
        layout.addWidget(mode_toolbar)
        layout.addWidget(self.mode_canvas, 1)
        
        # Add scale factor control
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale Factor:"))
        
        self.mode_scale_slider = QSlider(Qt.Horizontal)
        self.mode_scale_slider.setRange(1, 1000)  # 0.1x to 100.0x
        self.mode_scale_slider.setValue(10)  # Default to 1.0x
        self.mode_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.mode_scale_slider.setTickInterval(100)
        self.mode_scale_slider.valueChanged.connect(self.update_mode_shape_plot)
        scale_layout.addWidget(self.mode_scale_slider)
        
        self.mode_scale_label = QLabel("1.0x")
        scale_layout.addWidget(self.mode_scale_label)
        
        layout.addLayout(scale_layout)
        
        self.results_tabs.addTab(tab, "Mode Shapes")
        
    def create_beam_animation_tab(self):
        """Create the beam animation visualization tab"""
        self.beam_animation_widget = BeamAnimationWidget()
        self.results_tabs.addTab(self.beam_animation_widget, "Beam Animation")
        
    def create_mode_animation_tab(self):
        """Create the mode shape animation visualization tab"""
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
                self.summary_table.setItem(i, 0, QTableWidgetItem(param))
                self.summary_table.setItem(i, 1, QTableWidgetItem(value))
            
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
            
            self.mode_selector.blockSignals(True)
            self.mode_selector.clear()
            for i, freq in enumerate(freqs):
                self.mode_selector.addItem(f"Mode {i+1} ({freq:.2f} Hz)")
            self.mode_selector.setCurrentIndex(0)
            self.mode_selector.blockSignals(False)
            
            # Update frequency table
            self.freq_table.setRowCount(len(freqs))
            for i, freq in enumerate(freqs):
                self.freq_table.setItem(i, 0, QTableWidgetItem(f"Mode {i+1}"))
                self.freq_table.setItem(i, 1, QTableWidgetItem(f"{freq:.4f}"))
                
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
                    self.mode_animation_widget.set_data(
                        coords, 
                        results['mode_shapes'], 
                        results['natural_frequencies']
                    )
                else:
                    print("Warning: mode_animation_widget not found")
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
        self.beam_ax.grid(True, linestyle='--', alpha=0.7)
        self.beam_ax.legend()
        self.beam_ax.set_xlabel('Position (m)')
        self.beam_ax.set_ylabel('Displacement (m)')
        self.beam_ax.set_title(f'Beam Deflection at t = {times[time_idx]:.4f} s (Scale: {scale_factor:.1f}x)')
        
        # Set y-axis limits with some margin
        max_disp = max(np.max(np.abs(scaled_displacements)), 0.001)
        self.beam_ax.set_ylim(-max_disp * 1.2, max_disp * 1.2)
        
        # Update the canvas
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
        
        if 'mode_shapes' not in self.results or 'natural_frequencies' not in self.results or coords is None:
            return
            
        mode_shapes = self.results['mode_shapes']
        freqs = self.results['natural_frequencies']
        
        if mode_idx >= mode_shapes.shape[1] or mode_idx >= len(freqs):
            return
            
        # Get the scale factor
        scale_factor = self.mode_scale_slider.value() / 10.0
        self.mode_scale_label.setText(f"{scale_factor:.1f}x")
        
        # Get the selected mode shape and apply scaling
        mode_shape = mode_shapes[:, mode_idx] * scale_factor
        
        # Update the frequency display
        self.freq_display.setText(f"{freqs[mode_idx]:.4f} Hz")
        
        # Clear the axis and plot the mode shape
        self.mode_ax.clear()
        
        # Plot the reference line (zero displacement)
        self.mode_ax.plot(coords, np.zeros_like(coords), 'k--', label='Reference')
        
        # Plot the mode shape
        self.mode_ax.plot(coords, mode_shape, 'r-o', label=f'Mode {mode_idx+1}')
        
        # Add grid, legend, labels
        self.mode_ax.grid(True, linestyle='--', alpha=0.7)
        self.mode_ax.legend()
        self.mode_ax.set_xlabel('Position (m)')
        self.mode_ax.set_ylabel('Mode Shape Amplitude')
        self.mode_ax.set_title(f'Mode {mode_idx+1} - {freqs[mode_idx]:.4f} Hz (Scale: {scale_factor:.1f}x)')
        
        # Set y-axis limits with some margin
        max_amp = max(np.max(np.abs(mode_shape)), 0.001)
        self.mode_ax.set_ylim(-max_amp * 1.2, max_amp * 1.2)
        
        # Update the canvas
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
        
        if len(times) <= 0 or coords is None or node_idx >= len(coords):
            return
            
        # Get time-history of displacement for the selected node
        if 'displacements' in self.results:
            # If pre-processed displacements are available
            node_disp = self.results['displacements'][node_idx, :]
        elif 'displacement' in self.results:
            # Extract vertical displacements for this node over time
            try:
                displacement_matrix = self.results['displacement']
                node_dof = 2 * node_idx + 1  # Vertical displacement DOF
                
                if node_dof < displacement_matrix.shape[0]:
                    node_disp = displacement_matrix[node_dof, :]
                else:
                    return
            except Exception as e:
                print(f"Error processing node displacements: {str(e)}")
                return
        else:
            print("No displacement data found for node plot")
            return
            
        # Clear the axis and plot the node displacement time history
        self.node_ax.clear()
        
        # Plot the node displacement vs time
        self.node_ax.plot(times, node_disp, 'b-', linewidth=1.5)
        
        # Add grid, labels
        self.node_ax.grid(True, linestyle='--', alpha=0.7)
        self.node_ax.set_xlabel('Time (s)')
        self.node_ax.set_ylabel('Displacement (m)')
        node_position = coords[node_idx]
        self.node_ax.set_title(f'Node {node_idx+1} Displacement at x = {node_position:.3f} m')
        
        # Update the canvas
        self.node_canvas.draw()
        
    def export_results(self):
        """Export analysis results to file(s)"""
        # This will be implemented later
        print("Export results functionality will be implemented")
        
def main():
    """Simple test function to demonstrate the dashboard"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create dark fusion style
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    dashboard = ResultsDashboard()
    dashboard.setWindowTitle("Beam Analysis Results")
    dashboard.resize(1200, 800)
    dashboard.show()
    
    # Exit
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main() 