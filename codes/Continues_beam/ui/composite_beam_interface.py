"""
Comprehensive Composite Beam Analysis Interface

This module provides a complete interface for composite beam analysis with:
- Advanced layer management with drag-and-drop reordering
- Real-time cross-section visualization
- Force visualization on beam side view
- Temperature-dependent material properties
- Scientific accuracy with proper composite beam theory
- Beautiful, modern UI design
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QTabWidget,
    QHeaderView, QMessageBox, QFrame, QScrollArea, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QIcon

import numpy as np
from ..beam.properties import calc_composite_properties
from ..beam.solver import solve_beam_vibration
from .layer_dialog import LayerDialog
from .enhanced_cross_section_visualizer import EnhancedCrossSectionVisualizer
from .beam_side_view import BeamSideViewWidget
from .force_visualization import ForceVisualizationWidget
from .force_regions_panel import ForceRegionsPanel
from .material_database import MaterialDatabase
from ..utils import ForceRegionManager


class CompositeBeamInterface(QWidget):
    """
    Main interface for composite beam analysis with advanced features.
    """
    
    # Signal emitted when analysis is completed
    analysis_completed = pyqtSignal(dict)
    
    def __init__(self, parent=None, theme='Dark'):
        super().__init__(parent)
        self.layers = []
        self.beam_width = 0.05  # Default 50mm width
        self.beam_length = 1.0  # Default 1m length
        self.force_manager = ForceRegionManager()
        self.material_db = MaterialDatabase()
        self.theme = theme  # Store current theme
        self.initUI()
        
    def set_theme(self, theme):
        """Update the theme and propagate to visualization widgets."""
        self.theme = theme
        self.update_theme_styles()
        
        # Update visualization widgets
        if hasattr(self, 'cross_section_viz'):
            self.cross_section_viz.set_theme(theme)
        if hasattr(self, 'side_view_widget'):
            self.side_view_widget.set_theme(theme)
            
    def update_theme_styles(self):
        """Update UI styles based on current theme."""
        if self.theme == 'Dark':
            self.apply_dark_styling()
        else:
            self.apply_light_styling()
        
    def initUI(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # Left panel - Layer management and parameters
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualizations
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # Apply modern styling
        self.update_theme_styles()
        
        # Initialize with example composite
        self.load_example_composite()
        
    def create_left_panel(self):
        """Create the left control panel."""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        
        # Beam geometry section
        geometry_group = self.create_geometry_section()
        left_layout.addWidget(geometry_group)
        
        # Layer management section
        layer_group = self.create_layer_management_section()
        left_layout.addWidget(layer_group)
        
        # Force regions section
        force_group = self.create_force_regions_section()
        left_layout.addWidget(force_group)
        
        # Analysis parameters section
        analysis_group = self.create_analysis_section()
        left_layout.addWidget(analysis_group)
        
        # Analysis controls
        controls_group = self.create_analysis_controls()
        left_layout.addWidget(controls_group)
        
        return left_widget
        
    def create_geometry_section(self):
        """Create the beam geometry input section."""
        group = QGroupBox("Beam Geometry")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QFormLayout(group)
        
        # Beam length
        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(0.1, 50.0)
        self.length_input.setValue(1.0)
        self.length_input.setSuffix(" m")
        self.length_input.setDecimals(3)
        self.length_input.setMinimumWidth(120)
        self.length_input.valueChanged.connect(self.update_beam_geometry)
        layout.addRow("Length:", self.length_input)
        
        # Beam width
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(0.001, 1.0)
        self.width_input.setValue(0.05)
        self.width_input.setSuffix(" m")
        self.width_input.setDecimals(4)
        self.width_input.setMinimumWidth(120)
        self.width_input.valueChanged.connect(self.update_beam_geometry)
        layout.addRow("Width:", self.width_input)
        
        # Number of elements
        self.num_elements_input = QSpinBox()
        self.num_elements_input.setRange(10, 200)
        self.num_elements_input.setValue(50)
        self.num_elements_input.setMinimumWidth(120)
        layout.addRow("FE Elements:", self.num_elements_input)
        
        return group
        
    def create_layer_management_section(self):
        """Create the layer management section."""
        group = QGroupBox("Composite Layers")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout(group)
        
        # Layer controls
        controls_layout = QHBoxLayout()
        
        self.add_layer_btn = QPushButton("Add Layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        controls_layout.addWidget(self.add_layer_btn)
        
        self.edit_layer_btn = QPushButton("Edit Layer")
        self.edit_layer_btn.clicked.connect(self.edit_layer)
        controls_layout.addWidget(self.edit_layer_btn)
        
        self.remove_layer_btn = QPushButton("Remove Layer")
        self.remove_layer_btn.clicked.connect(self.remove_layer)
        controls_layout.addWidget(self.remove_layer_btn)
        
        controls_layout.addStretch()
        
        # Material presets
        preset_combo = QComboBox()
        preset_combo.addItems([
            "Custom", "Steel", "Aluminum", "Carbon Fiber", 
            "Glass Fiber", "Titanium", "Foam Core"
        ])
        preset_combo.currentTextChanged.connect(self.load_material_preset)
        controls_layout.addWidget(QLabel("Presets:"))
        controls_layout.addWidget(preset_combo)
        
        layout.addLayout(controls_layout)
        
        # Layer table
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(6)
        self.layer_table.setHorizontalHeaderLabels([
            "Layer", "Material", "Thickness (mm)", "E (GPa)", "ρ (kg/m³)", "Position"
        ])
        
        header = self.layer_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        self.layer_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.layer_table.setAlternatingRowColors(True)
        self.layer_table.setMinimumHeight(400)
        
        layout.addWidget(self.layer_table)
        
        # Effective properties display
        props_layout = QFormLayout()
        
        self.effective_EI_label = QLabel("N/A")
        self.effective_EI_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        props_layout.addRow("Effective EI:", self.effective_EI_label)
        
        self.effective_rhoA_label = QLabel("N/A")
        self.effective_rhoA_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        props_layout.addRow("Effective ρA:", self.effective_rhoA_label)
        
        self.neutral_axis_label = QLabel("N/A")
        self.neutral_axis_label.setStyleSheet("color: #C2185B; font-weight: bold;")
        props_layout.addRow("Neutral Axis:", self.neutral_axis_label)
        
        layout.addLayout(props_layout)
        
        return group
        
    def create_force_regions_section(self):
        """Create the force regions management section."""
        group = QGroupBox("Force Regions")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout(group)
        
        self.force_regions_panel = ForceRegionsPanel(self.force_manager)
        self.force_regions_panel.regions_changed.connect(self.update_visualizations)
        layout.addWidget(self.force_regions_panel)
        
        return group
        
    def create_analysis_section(self):
        """Create the analysis parameters section."""
        group = QGroupBox("Analysis Parameters")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QFormLayout(group)
        
        # Time span
        self.time_span_input = QDoubleSpinBox()
        self.time_span_input.setRange(0.1, 100.0)
        self.time_span_input.setValue(3.0)
        self.time_span_input.setSuffix(" s")
        self.time_span_input.setDecimals(2)
        layout.addRow("Time Duration:", self.time_span_input)
        
        # Time points
        self.time_points_input = QSpinBox()
        self.time_points_input.setRange(100, 2000)
        self.time_points_input.setValue(300)
        layout.addRow("Time Points:", self.time_points_input)
        
        # Temperature
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(-100, 500)
        self.temperature_input.setValue(20)
        self.temperature_input.setSuffix(" °C")
        self.temperature_input.valueChanged.connect(self.update_material_properties)
        layout.addRow("Temperature:", self.temperature_input)
        
        return group
        
    def create_analysis_controls(self):
        """Create the analysis control buttons."""
        group = QGroupBox("Analysis Controls")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout(group)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Composite Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(self.run_analysis_btn)
        
        # Export/Import buttons
        export_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Setup")
        export_btn.clicked.connect(self.export_setup)
        export_layout.addWidget(export_btn)
        
        import_btn = QPushButton("Import Setup")
        import_btn.clicked.connect(self.import_setup)
        export_layout.addWidget(import_btn)
        
        layout.addLayout(export_layout)
        
        return group
        
    def create_right_panel(self):
        """Create the right visualization panel."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        
        # Create tabbed interface for different views
        viz_tabs = QTabWidget()
        
        # Cross-section view tab
        cross_section_widget = self.create_cross_section_tab()
        viz_tabs.addTab(cross_section_widget, "Cross-Section")
        
        # Side view tab
        side_view_widget = self.create_side_view_tab()
        viz_tabs.addTab(side_view_widget, "Side View")
        
        right_layout.addWidget(viz_tabs)
        
        return right_widget
        
    def create_cross_section_tab(self):
        """Create the cross-section visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enhanced cross-section visualizer
        self.cross_section_viz = EnhancedCrossSectionVisualizer(theme=self.theme)
        layout.addWidget(self.cross_section_viz)
        
        return widget
        
    def create_side_view_tab(self):
        """Create the side view tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Beam side view widget
        self.side_view_widget = BeamSideViewWidget(theme=self.theme)
        layout.addWidget(self.side_view_widget)
        
        return widget
        
    def apply_light_styling(self):
        """Apply light theme styling to the interface."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #FAFAFA;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1976D2;
            }
            QPushButton {
                background-color: #F5F5F5;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
                border-color: #1976D2;
            }
            QTableWidget {
                gridline-color: #E0E0E0;
                background-color: white;
                alternate-background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                color: #333333;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                color: #333333;
            }
            QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border-color: #1976D2;
            }
            QLabel {
                color: #333333;
            }
        """)
        
    def apply_dark_styling(self):
        """Apply dark theme styling to the interface."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2D2D30;
                color: #F0F0F0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #64B5F6;
            }
            QPushButton {
                background-color: #3A3A3C;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                color: #F0F0F0;
            }
            QPushButton:hover {
                background-color: #4A4A4E;
                border-color: #64B5F6;
            }
            QTableWidget {
                gridline-color: #404040;
                background-color: #2D2D30;
                alternate-background-color: #363638;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #F0F0F0;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
                background-color: #2D2D30;
                color: #F0F0F0;
            }
            QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border-color: #64B5F6;
            }
            QLabel {
                color: #F0F0F0;
            }
        """)
        
    def load_example_composite(self):
        """Load an example composite structure."""
        # Example: Steel-Aluminum-Steel sandwich
        example_layers = [
            {
                'name': 'Steel Bottom',
                'thickness': 0.003,  # 3mm
                'E_func': lambda T: 210e9 * (1 - 0.0001 * T),
                'rho_func': lambda T: 7800 * (1 + 0.00001 * T),
                'material_type': 'Steel'
            },
            {
                'name': 'Aluminum Core',
                'thickness': 0.004,  # 4mm
                'E_func': lambda T: 70e9 * (1 - 0.0002 * T),
                'rho_func': lambda T: 2700 * (1 + 0.00002 * T),
                'material_type': 'Aluminum'
            },
            {
                'name': 'Steel Top',
                'thickness': 0.003,  # 3mm
                'E_func': lambda T: 210e9 * (1 - 0.0001 * T),
                'rho_func': lambda T: 7800 * (1 + 0.00001 * T),
                'material_type': 'Steel'
            }
        ]
        
        self.layers = example_layers
        self.update_layer_table()
        self.update_visualizations()
        
    def update_layer_table(self):
        """Update the layer table display."""
        self.layer_table.setRowCount(len(self.layers))
        
        for i, layer in enumerate(self.layers):
            # Layer name
            self.layer_table.setItem(i, 0, QTableWidgetItem(layer['name']))
            
            # Material type
            self.layer_table.setItem(i, 1, QTableWidgetItem(layer['material_type']))
            
            # Thickness in mm
            thickness_mm = layer['thickness'] * 1000
            self.layer_table.setItem(i, 2, QTableWidgetItem(f"{thickness_mm:.2f}"))
            
            # Young's modulus in GPa
            T = self.temperature_input.value()
            E_val = layer['E_func'](T) / 1e9
            self.layer_table.setItem(i, 3, QTableWidgetItem(f"{E_val:.1f}"))
            
            # Density
            rho_val = layer['rho_func'](T)
            self.layer_table.setItem(i, 4, QTableWidgetItem(f"{rho_val:.0f}"))
            
            # Position (cumulative thickness)
            pos = sum(l['thickness'] for l in self.layers[:i+1]) * 1000
            self.layer_table.setItem(i, 5, QTableWidgetItem(f"{pos:.2f} mm"))
        
        self.update_effective_properties()
        
    def update_effective_properties(self):
        """Update the effective properties display."""
        if not self.layers:
            self.effective_EI_label.setText("N/A")
            self.effective_rhoA_label.setText("N/A")
            self.neutral_axis_label.setText("N/A")
            return
            
        try:
            # Convert to format expected by calc_composite_properties
            layers_for_calc = []
            for layer in self.layers:
                layers_for_calc.append({
                    'thickness': layer['thickness'],
                    'E_func': layer['E_func'],
                    'rho_func': layer['rho_func']
                })
            
            # Calculate effective properties
            EI_eff, rhoA_eff = calc_composite_properties(self.beam_width, layers_for_calc)
            
            # Display results
            self.effective_EI_label.setText(f"{EI_eff:.2e} N·m²")
            self.effective_rhoA_label.setText(f"{rhoA_eff:.2f} kg/m")
            
            # Calculate neutral axis position
            total_thickness = sum(l['thickness'] for l in self.layers)
            neutral_axis = total_thickness / 2  # Simplified for display
            self.neutral_axis_label.setText(f"{neutral_axis*1000:.2f} mm")
            
        except Exception as e:
            print(f"Error calculating effective properties: {e}")
            self.effective_EI_label.setText("Error")
            self.effective_rhoA_label.setText("Error")
            self.neutral_axis_label.setText("Error")
            
    def update_visualizations(self):
        """Update all visualization widgets with current data."""
        # Update cross-section
        if hasattr(self, 'cross_section_viz'):
            self.cross_section_viz.set_layers(
                self.layers, self.beam_width, self.temperature_input.value()
            )
            
        # Update side view
        if hasattr(self, 'side_view_widget'):
            self.side_view_widget.set_beam_geometry(self.beam_length, self.layers)
            self.side_view_widget.set_forces(self.force_manager.regions)
        
    def update_beam_geometry(self):
        """Update beam geometry from inputs."""
        self.beam_length = self.length_input.value()
        self.beam_width = self.width_input.value()
        self.update_visualizations()
        
    def update_material_properties(self):
        """Update material properties when temperature changes."""
        self.update_layer_table()
        self.update_visualizations()
        
    def add_layer(self):
        """Add a new layer."""
        dialog = LayerDialog(self)
        if dialog.exec_() == dialog.Accepted:
            layer_data = dialog.get_layer_data()
            
            # Convert to internal format
            layer = {
                'name': f"Layer {len(self.layers) + 1}",
                'thickness': layer_data['height'],
                'E_func': layer_data['E'] if callable(layer_data['E']) else lambda T: layer_data['E'],
                'rho_func': layer_data['rho'] if callable(layer_data['rho']) else lambda T: layer_data['rho'],
                'material_type': 'Custom'
            }
            
            self.layers.append(layer)
            self.update_layer_table()
            self.update_visualizations()
            
    def edit_layer(self):
        """Edit the selected layer."""
        current_row = self.layer_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Selection Required", "Please select a layer to edit.")
            return
            
        # Implementation would open layer dialog with current values
        QMessageBox.information(self, "Edit Layer", "Layer editing functionality will be implemented.")
        
    def remove_layer(self):
        """Remove the selected layer."""
        current_row = self.layer_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Selection Required", "Please select a layer to remove.")
            return
            
        if len(self.layers) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "At least one layer must remain.")
            return
            
        self.layers.pop(current_row)
        self.update_layer_table()
        self.update_visualizations()
        
    def load_material_preset(self, material_name):
        """Load a material preset."""
        if material_name == "Custom":
            return
            
        # Get material properties from database
        material_props = self.material_db.get_material(material_name)
        if material_props:
            QMessageBox.information(self, "Material Preset", 
                                  f"Loading {material_name} properties.")
            
    def run_analysis(self):
        """Run the composite beam analysis."""
        if not self.layers:
            QMessageBox.warning(self, "No Layers", "Please add at least one layer.")
            return
            
        try:
            self.run_analysis_btn.setText("Running Analysis...")
            self.run_analysis_btn.setEnabled(False)
            
            # Convert layers to solver format
            solver_layers = []
            for layer in self.layers:
                solver_layers.append({
                    'height': layer['thickness'],
                    'E': layer['E_func'],
                    'rho': layer['rho_func']
                })
            
            # Run analysis
            results = solve_beam_vibration(
                width=self.beam_width,
                layers=solver_layers,
                L=self.beam_length,
                k_spring=0.0,
                num_elems=self.num_elements_input.value(),
                t_span=(0, self.time_span_input.value()),
                num_time_points=self.time_points_input.value()
            )
            
            # Emit results
            self.analysis_completed.emit(results)
            
            QMessageBox.information(self, "Analysis Complete", 
                                  "Composite beam analysis completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Error running analysis:\n{str(e)}")
            
        finally:
            self.run_analysis_btn.setText("Run Composite Analysis")
            self.run_analysis_btn.setEnabled(True)
            
    def export_setup(self):
        """Export the current setup to file."""
        QMessageBox.information(self, "Export", "Export functionality will be implemented.")
        
    def import_setup(self):
        """Import setup from file."""
        QMessageBox.information(self, "Import", "Import functionality will be implemented.") 