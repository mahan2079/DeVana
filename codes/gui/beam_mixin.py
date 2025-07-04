from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import pandas as pd

from Continues_beam.utils import ForceRegionManager

class ContinuousBeamMixin:
        def create_continuous_beam_page(self):
            """Create the continuous beam analysis page"""
            if not BEAM_IMPORTS_SUCCESSFUL:
                # Create placeholder page if imports failed
                beam_page = QWidget()
                layout = QVBoxLayout(beam_page)
                
                # Centered content
                center_widget = QWidget()
                center_layout = QVBoxLayout(center_widget)
                center_layout.setAlignment(Qt.AlignCenter)
                
                # Error message
                error_label = QLabel("Continuous Beam Module Not Available")
                error_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
                error_label.setAlignment(Qt.AlignCenter)
                center_layout.addWidget(error_label)
                
                description = QLabel("Please make sure the 'Continues beam' module is correctly installed.")
                description.setFont(QFont("Segoe UI", 12))
                description.setAlignment(Qt.AlignCenter)
                center_layout.addWidget(description)
                
                layout.addWidget(center_widget)
                self.content_stack.addWidget(beam_page)
                return
                
            # Create tab container
            beam_page = QWidget()
            page_layout = QVBoxLayout(beam_page)
            page_layout.setContentsMargins(10, 10, 10, 10)
            page_layout.setSpacing(10)
            
            # Header - more compact
            header = QWidget()
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(5, 5, 5, 5)
            
            title_container = QVBoxLayout()
            title = QLabel("Continuous Beam Analysis")
            title.setFont(QFont("Segoe UI", 16, QFont.Bold))
            title_container.addWidget(title)
            
            description = QLabel("Analyze and optimize vibration in continuous beams")
            description.setFont(QFont("Segoe UI", 10))
            title_container.addWidget(description)
            
            header_layout.addLayout(title_container)
            header_layout.addStretch()
            
            # Add header to page layout with minimal height
            header.setMaximumHeight(70)
            page_layout.addWidget(header)
            
            # Initialize the beam specific properties
            self.layers = []
            self.force_regions_manager = ForceRegionManager()
            
            # Add a default layer
            default_layer = {
                'height': 0.05,
                'E': 210e9,  # Steel
                'rho': 7800
            }
            self.layers.append(default_layer)
            
            # Pre-initialize canvases to avoid NoneType errors
            self.beam_anim = None
            self.beam_canvas = None
            self.node_anim = None
            self.node_canvas = None
            
            # Create tabs
            self.beam_tabs = ModernQTabWidget()
            
            # Add tabs
            self.init_beam_parameters_tab()
            self.init_layers_tab()
            self.init_loads_tab()
            self.init_beam_results_tab()
            
            # Add tabs to the page
            page_layout.addWidget(self.beam_tabs)
            
            # Create button area
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(0, 10, 0, 0)
            
            # Add run button
            self.run_beam_button = QPushButton("▶ Run Simulation")
            self.run_beam_button.clicked.connect(self.run_beam_simulation)
            self.run_beam_button.setMinimumHeight(40)
            self.run_beam_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #388E3C;
                }
            """)
            
            # Create spacer to push buttons to the right
            button_layout.addStretch()
            button_layout.addWidget(self.run_beam_button)
            
            # Add button container to main layout
            page_layout.addWidget(button_container)
            
            # Add the page to the stack
            self.content_stack.addWidget(beam_page)
            
        def init_beam_parameters_tab(self):
            """Initialize the beam parameters tab"""
            beam_tab = QWidget()
            beam_layout = QVBoxLayout(beam_tab)
            beam_layout.setContentsMargins(10, 10, 10, 10)
            beam_layout.setSpacing(10)
            
            # Material properties group
            material_group = QGroupBox("Material Properties")
            material_layout = QFormLayout(material_group)
            material_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            material_layout.setContentsMargins(10, 20, 10, 10)
            material_layout.setSpacing(10)
            
            # Young's modulus
            self.young_modulus = QDoubleSpinBox()
            self.young_modulus.setRange(1e9, 1000e9)
            self.young_modulus.setValue(210e9)
            self.young_modulus.setSuffix(" Pa")
            self.young_modulus.setDecimals(2)
            self.young_modulus.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            material_layout.addRow("Young's Modulus:", self.young_modulus)
            
            # Density
            self.density = QDoubleSpinBox()
            self.density.setRange(100, 20000)
            self.density.setValue(7800)
            self.density.setSuffix(" kg/m³")
            self.density.setDecimals(0)
            material_layout.addRow("Density:", self.density)
            
            # Beam geometry group
            geometry_group = QGroupBox("Beam Geometry")
            geometry_layout = QFormLayout(geometry_group)
            geometry_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            geometry_layout.setContentsMargins(10, 20, 10, 10)
            geometry_layout.setSpacing(10)
            
            # Length
            self.beam_length = QDoubleSpinBox()
            self.beam_length.setRange(0.1, 100)
            self.beam_length.setValue(10.0)
            self.beam_length.setSuffix(" m")
            self.beam_length.setDecimals(2)
            geometry_layout.addRow("Length:", self.beam_length)
            
            # Width
            self.beam_width = QDoubleSpinBox()
            self.beam_width.setRange(0.001, 10)
            self.beam_width.setValue(0.2)
            self.beam_width.setSuffix(" m")
            self.beam_width.setDecimals(3)
            geometry_layout.addRow("Width:", self.beam_width)
            
            # Height
            self.beam_height = QDoubleSpinBox()
            self.beam_height.setRange(0.001, 10)
            self.beam_height.setValue(0.3)
            self.beam_height.setSuffix(" m")
            self.beam_height.setDecimals(3)
            geometry_layout.addRow("Height:", self.beam_height)
            
            # Number of elements
            self.num_elements = QSpinBox()
            self.num_elements.setRange(2, 100)
            self.num_elements.setValue(10)
            geometry_layout.addRow("Number of Elements:", self.num_elements)
            
            # Spring stiffness
            self.k_spring = QDoubleSpinBox()
            self.k_spring.setRange(0, 1e9)
            self.k_spring.setValue(1e5)
            self.k_spring.setSuffix(" N/m")
            self.k_spring.setDecimals(0)
            self.k_spring.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            geometry_layout.addRow("Tip Spring Stiffness:", self.k_spring)
            
            # Add groups to layout
            beam_layout.addWidget(material_group)
            beam_layout.addWidget(geometry_group)
            beam_layout.addStretch()
            
            # Add to tabs
            self.beam_tabs.addTab(beam_tab, "Beam Parameters")
            
        def init_layers_tab(self):
            """Initialize the layers tab for composite beam"""
            layers_tab = QWidget()
            main_layout = QHBoxLayout(layers_tab)
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(15)
            
            # Left side - Cross-section visualization (takes 60% of width)
            viz_container = QWidget()
            viz_layout = QVBoxLayout(viz_container)
            viz_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add title and description
            viz_title = QLabel("Cross-Section Visualization")
            viz_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
            viz_layout.addWidget(viz_title)
            
            viz_desc = QLabel("Visual representation of the beam's cross-section with layers")
            viz_desc.setStyleSheet("color: #666; font-style: italic;")
            viz_layout.addWidget(viz_desc)
            
            # Cross-section visualization - make it larger
            self.cross_section_visualizer = CrossSectionVisualizer()
            self.cross_section_visualizer.setMinimumHeight(350)
            viz_layout.addWidget(self.cross_section_visualizer, 1)  # stretch factor of 1
            
            # Add dimension information
            dimension_label = QLabel("Total Height: 0.0 m")
            dimension_label.setAlignment(Qt.AlignCenter)
            self.dimension_label = dimension_label  # Store for later updates
            viz_layout.addWidget(dimension_label)
            
            main_layout.addWidget(viz_container, 60)  # 60% of width
            
            # Right side - Layers table and controls
            controls_container = QWidget()
            controls_layout = QVBoxLayout(controls_container)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add title
            controls_title = QLabel("Layer Properties")
            controls_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
            controls_layout.addWidget(controls_title)
            
            # Layers table
            self.layers_table = QTableWidget()
            self.layers_table.setColumnCount(3)
            self.layers_table.setHorizontalHeaderLabels(["Height (m)", "Young's Modulus (Pa)", "Density (kg/m³)"])
            self.layers_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.layers_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.layers_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.layers_table.setMinimumHeight(200)
            controls_layout.addWidget(self.layers_table)
            
            # Layer buttons
            btn_layout = QHBoxLayout()
            
            self.add_layer_btn = QPushButton("Add Layer")
            self.add_layer_btn.clicked.connect(self.add_new_layer)
            btn_layout.addWidget(self.add_layer_btn)
            
            self.edit_layer_btn = QPushButton("Edit Layer")
            self.edit_layer_btn.clicked.connect(self.edit_layer)
            self.edit_layer_btn.setEnabled(False)
            btn_layout.addWidget(self.edit_layer_btn)
            
            self.remove_layer_btn = QPushButton("Remove Layer")
            self.remove_layer_btn.clicked.connect(self.remove_layer)
            self.remove_layer_btn.setEnabled(False)
            btn_layout.addWidget(self.remove_layer_btn)
            
            # Connect selection change
            self.layers_table.itemSelectionChanged.connect(self.update_layer_buttons)
            
            # Add button layout
            controls_layout.addLayout(btn_layout)
            
            # Add help text
            help_text = QLabel(
                "Add layers to define a composite beam. The total height "
                "will be the sum of all layer heights. Layers are stacked "
                "from top to bottom as shown in the visualization."
            )
            help_text.setWordWrap(True)
            help_text.setStyleSheet("color: #666; font-style: italic;")
            controls_layout.addWidget(help_text)
            
            # Add a stretch to push everything up
            controls_layout.addStretch()
            
            main_layout.addWidget(controls_container, 40)  # 40% of width
            
            # Add existing layers to the table
            for layer in self.layers:
                self.add_layer_to_table(layer)
            
            # Update the cross-section
            self.update_cross_section()
            
            # Add to tabs
            self.beam_tabs.addTab(layers_tab, "Layers")
        
        def update_layer_buttons(self):
            """Enable or disable layer editing buttons based on selection"""
            selected = len(self.layers_table.selectedItems()) > 0
            self.edit_layer_btn.setEnabled(selected)
            self.remove_layer_btn.setEnabled(selected)
        
        def add_new_layer(self):
            """Open dialog to add a new layer"""
            dialog = LayerDialog(self)
            if dialog.exec_():
                layer_data = dialog.get_layer_data()
                
                # Convert string expressions to callables if needed
                for key in ['E', 'rho']:
                    if isinstance(layer_data.get(key), str):
                        from Continues_beam.utils import parse_expression
                        try:
                            layer_data[key] = parse_expression(layer_data[key])
                        except ValueError as e:
                            QMessageBox.warning(self, "Expression Error", str(e))
                            return
                
                # Add to layers list
                self.layers.append(layer_data)
                
                # Add to table
                self.add_layer_to_table(layer_data)
                
                # Update visualization
                self.update_cross_section()
        
        def edit_layer(self):
            """Edit the selected layer"""
            selected_row = self.layers_table.currentRow()
            if selected_row >= 0 and selected_row < len(self.layers):
                # Get current layer data
                layer_data = self.layers[selected_row]
                
                # Open dialog with current data
                dialog = LayerDialog(self, layer_data)
                if dialog.exec_():
                    new_layer_data = dialog.get_layer_data()
                    
                    # Convert string expressions to callables if needed
                    for key in ['E', 'rho']:
                        if isinstance(new_layer_data.get(key), str):
                            from Continues_beam.utils import parse_expression
                            try:
                                new_layer_data[key] = parse_expression(new_layer_data[key])
                            except ValueError as e:
                                QMessageBox.warning(self, "Expression Error", str(e))
                                return
                    
                    # Update layers list
                    self.layers[selected_row] = new_layer_data
                    
                    # Update table display
                    self.layers_table.setRowCount(0)
                    for layer in self.layers:
                        self.add_layer_to_table(layer)
                    
                    # Update visualization
                    self.update_cross_section()    
        def remove_layer(self):
            """Remove the selected layer"""
            selected_row = self.layers_table.currentRow()
            if selected_row >= 0 and selected_row < len(self.layers):
                # Confirm deletion
                reply = QMessageBox.question(
                    self, 
                    "Confirm Deletion",
                    f"Remove layer {selected_row + 1}?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Remove from layers list
                    del self.layers[selected_row]
                    
                    # Update table display
                    self.layers_table.setRowCount(0)
                    for layer in self.layers:
                        self.add_layer_to_table(layer)
                    
                    # Update visualization
                    self.update_cross_section()
                    
        def add_layer_to_table(self, layer):
            """Add a layer to the table display
            
            Args:
                layer (dict): Layer data dictionary
            """
            row = self.layers_table.rowCount()
            self.layers_table.insertRow(row)
            
            # Height column
            height_item = QTableWidgetItem(f"{layer['height']:.4f}")
            height_item.setTextAlignment(Qt.AlignCenter)
            self.layers_table.setItem(row, 0, height_item)
            
            # Young's modulus column
            if 'E_expr' in layer:
                E_text = layer['E_expr']
            else:
                E_text = f"{layer['E']:.2e}"
            E_item = QTableWidgetItem(E_text)
            E_item.setTextAlignment(Qt.AlignCenter)
            self.layers_table.setItem(row, 1, E_item)
            
            # Density column
            if 'rho_expr' in layer:
                rho_text = layer['rho_expr']
            else:
                rho_text = f"{layer['rho']:.2f}"
            rho_item = QTableWidgetItem(rho_text)
            rho_item.setTextAlignment(Qt.AlignCenter)
            self.layers_table.setItem(row, 2, rho_item)
            
        def init_loads_tab(self):
            """Initialize the loads & time tab for continuous beam analysis"""
            loads_tab = QWidget()
            layout = QVBoxLayout(loads_tab)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Create panel for managing force regions
            regions_group = QGroupBox("Force Regions")
            regions_layout = QVBoxLayout(regions_group)
            self.force_regions_panel = ForceRegionsPanel(self.force_regions_manager)
            regions_layout.addWidget(self.force_regions_panel)
            layout.addWidget(regions_group)
            
            # Time settings group
            time_group = QGroupBox("Time Settings")
            time_layout = QFormLayout(time_group)
            time_group.setLayout(time_layout)
            
            # Start time
            self.time_start_spin = QDoubleSpinBox()
            self.time_start_spin.setRange(0, 1000)
            self.time_start_spin.setValue(0.0)
            self.time_start_spin.setSuffix(" s")
            self.time_start_spin.setDecimals(2)
            time_layout.addRow("Start Time:", self.time_start_spin)
            
            # End time
            self.time_end_spin = QDoubleSpinBox()
            self.time_end_spin.setRange(0.01, 1000)
            self.time_end_spin.setValue(3.0)
            self.time_end_spin.setSuffix(" s")
            self.time_end_spin.setDecimals(2)
            time_layout.addRow("End Time:", self.time_end_spin)
            
            # Number of time points
            self.time_points_spin = QSpinBox()
            self.time_points_spin.setRange(10, 10000)
            self.time_points_spin.setValue(300)
            time_layout.addRow("Number of Time Points:", self.time_points_spin)
            
            layout.addWidget(time_group)
            
            # Stretch at the end to push everything to the top
            layout.addStretch()
            
            # Add to tabs
            self.beam_tabs.addTab(loads_tab, "Loads & Time")
            
        def update_cross_section(self):
            """Update the cross-section visualizer with current layers"""
            beam_width = self.beam_width.value()
            self.cross_section_visualizer.set_layers(self.layers, beam_width)
            
            # Calculate the total height of all layers
            total_height = sum(layer.get('height', 0.0) for layer in self.layers) if self.layers else 0.0
            
            # Update the dimension label with the total height
            self.dimension_label.setText(f"Total Height: {total_height:.4f} m")
            
            # Ensure the visualizer repaints itself
            self.cross_section_visualizer.update()
    
        def init_beam_results_tab(self):
            """Initialize the results tab for continuous beam analysis"""
            # Create the new comprehensive results dashboard
            try:
                # Try to import from the new location first
                try:
                    from src.ui.components.results_dashboard import ResultsDashboard
                except ImportError:
                    # Fall back to the old location
                    from Continues_beam.ui.results_dashboard import ResultsDashboard
                
                results_tab = QWidget()
                layout = QVBoxLayout(results_tab)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
                
                # Add saving capability header
                header_layout = QHBoxLayout()
                header_layout.setContentsMargins(10, 5, 10, 5)
                
                # Add save results button in the header
                self.save_results_btn = QPushButton("💾 Save Results")
                self.save_results_btn.clicked.connect(self.save_beam_results)
                self.save_results_btn.setEnabled(False)  # Disabled until results are available
                header_layout.addStretch()
                header_layout.addWidget(self.save_results_btn)
                
                layout.addLayout(header_layout)
                
                # Create the dashboard
                self.results_dashboard = ResultsDashboard()
                
                # Set size policy to allow the dashboard to expand
                self.results_dashboard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                # Add the dashboard to the layout with a stretch factor
                layout.addWidget(self.results_dashboard, 1)  # Give it a stretch factor of 1
                
                # For backward compatibility, maintain the frequency text widget
                self.freq_text = QTextEdit()
                self.freq_text.setVisible(False)  # Hide it as it's not needed in the new UI
                
                # Add to tabs
                self.beam_tabs.addTab(results_tab, "Results")
                
            except ImportError as e:
                # Fallback to old implementation if ResultsDashboard is not available
                print(f"Warning: Could not import ResultsDashboard: {e}")
                print("Using legacy results display instead.")
                
                results_tab = QWidget()
                layout = QVBoxLayout(results_tab)
                layout.setContentsMargins(5, 5, 5, 5)
                layout.setSpacing(5)
                
                # Results display area
                results_group = QGroupBox("Analysis Results")
                results_layout = QVBoxLayout(results_group)
                results_layout.setContentsMargins(5, 10, 5, 5)
                results_layout.setSpacing(5)
                
                # Add natural frequencies display in a compact form
                freq_header_layout = QHBoxLayout()
                freq_label = QLabel("Natural Frequencies:")
                freq_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
                freq_header_layout.addWidget(freq_label)
                freq_header_layout.addStretch()
                
                # Add save results button in the header
                self.save_results_btn = QPushButton("💾 Save Results")
                self.save_results_btn.clicked.connect(self.save_beam_results)
                self.save_results_btn.setEnabled(False)  # Disabled until results are available
                freq_header_layout.addWidget(self.save_results_btn)
                
                results_layout.addLayout(freq_header_layout)
                
                self.freq_text = QTextEdit()
                self.freq_text.setReadOnly(True)
                self.freq_text.setMaximumHeight(80)  # Reduce height to save space
                results_layout.addWidget(self.freq_text)
                
                # Create a splitter for the visualization area
                viz_splitter = QSplitter(Qt.Horizontal)
                viz_splitter.setChildrenCollapsible(False)
                
                # Left side - Static visualizations
                left_viz = QTabWidget()
                left_viz.setDocumentMode(True)
                
                # Beam deflection tab
                beam_tab = QWidget()
                beam_layout = QVBoxLayout(beam_tab)
                beam_layout.setContentsMargins(2, 2, 2, 2)
                self.beam_canvas = PlotCanvas(beam_tab)
                beam_layout.addWidget(self.beam_canvas)
                
                # Node displacement over time tab
                node_tab = QWidget()
                node_layout = QVBoxLayout(node_tab)
                node_layout.setContentsMargins(2, 2, 2, 2)
                
                # Node selection
                node_select_layout = QHBoxLayout()
                node_select_layout.addWidget(QLabel("Select Node:"))
                
                self.node_combo = QComboBox()
                node_select_layout.addWidget(self.node_combo)
                node_select_layout.addStretch()
                
                node_layout.addLayout(node_select_layout)
                
                self.node_canvas = PlotCanvas(node_tab)
                node_layout.addWidget(self.node_canvas)
                
                # Add tabs to left visualization
                left_viz.addTab(beam_tab, "Beam Deflection")
                left_viz.addTab(node_tab, "Node Displacement")
                
                # Right side - Animations
                right_viz = QTabWidget()
                right_viz.setDocumentMode(True)
                
                # Beam animation tab
                beam_anim_tab = QWidget()
                beam_anim_layout = QVBoxLayout(beam_anim_tab)
                beam_anim_layout.setContentsMargins(2, 2, 2, 2)
                self.beam_animation_adapter = BeamAnimationAdapter()
                beam_anim_layout.addWidget(self.beam_animation_adapter)
                
                # Mode shape animation tab
                mode_shape_tab = QWidget()
                mode_shape_layout = QVBoxLayout(mode_shape_tab)
                mode_shape_layout.setContentsMargins(2, 2, 2, 2)
                self.mode_shape_adapter = ModeShapeAdapter()
                mode_shape_layout.addWidget(self.mode_shape_adapter)
                
                # Add tabs to right visualization
                right_viz.addTab(beam_anim_tab, "Beam Animation")
                right_viz.addTab(mode_shape_tab, "Mode Shapes")
                
                # Add both sides to the splitter
                viz_splitter.addWidget(left_viz)
                viz_splitter.addWidget(right_viz)
                
                # Set initial sizes to make them equal
                viz_splitter.setSizes([500, 500])
                
                # Make the visualization area taller
                viz_splitter.setMinimumHeight(1000)
                
                # Add the splitter to the results layout with stretch factor
                results_layout.addWidget(viz_splitter, 1)  # Give it a stretch factor of 1
                
                # Add the results group to the main layout with stretch factor
                layout.addWidget(results_group, 1)  # Give it a stretch factor of 1
                
                # Add to tabs
                self.beam_tabs.addTab(results_tab, "Results")
            
        def run_beam_simulation(self):
            """Run the beam vibration simulation"""
            try:
                # Update status
                self.status_bar.showMessage("Running simulation...")
                
                # Get beam parameters
                beam_length = self.beam_length.value()
                beam_width = self.beam_width.value()
                spring_constant = self.k_spring.value()
                num_elements = self.num_elements.value()
                
                # Validate inputs - ensure we have at least one layer
                if not self.layers or all(layer.get('height', 0) <= 0 for layer in self.layers):
                    QMessageBox.warning(self, "Simulation Error", 
                                       "You must define at least one layer with a positive height.")
                    self.statusBar().showMessage("Simulation failed: No valid layers defined")
                    return
                
                # Prepare layers for solver
                layers = []
                for layer in self.layers:
                    height = layer.get('height', 0)
                    E = layer.get('E', 0)
                    rho = layer.get('rho', 0)
                    
                    layers.append({'height': height, 'E': E, 'rho': rho})
                
                # Set up force function
                force_gens = get_force_generators()
                
                # Check if we have any defined force regions
                if not self.force_regions_manager.regions:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("No Force Regions")
                    msg.setText("No force regions are defined.")
                    msg.setInformativeText("The simulation will run with a zero force function.\n"
                                           "Consider adding force regions in the Loads & Time tab for meaningful results.")
                    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                    response = msg.exec_()
                    
                    if response == QMessageBox.Cancel:
                        self.statusBar().showMessage("Simulation cancelled")
                        return
                    
                    # Create a lambda function that always returns 0.0 as the force profile
                    f_profile = lambda x, t: 0.0
                else:
                    # Create force profile from defined regions
                    f_profile = self.force_regions_manager.create_force_function(force_gens)
                
                # Run simulation
                results = solve_beam_vibration(
                    width=beam_width,
                    layers=layers,
                    L=beam_length,
                    k_spring=spring_constant,
                    num_elems=num_elements,
                    f_profile=f_profile
                )
                
                # Store results
                self.simulation_results = results
                
                # Update display
                self.update_results_display()
                
                # Update status
                self.status_bar.showMessage("Simulation completed successfully")
                
            except Exception as e:
                error_message = str(e)
                detailed_message = "An error occurred during the simulation."
                
                # Check for common errors
                if "no object chosen" in error_message.lower():
                    detailed_message = "A reference to an undefined object was detected. "
                    detailed_message += "This may happen if no force regions are defined or if there's an issue with the layers configuration."
                
                # Show error message
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Simulation Error")
                msg_box.setText(detailed_message)
                msg_box.setDetailedText(f"Error details: {error_message}")
                msg_box.exec_()
                
                # Update status bar
                self.status_bar.showMessage(f"Simulation failed: {error_message}")
        
        def update_results_display(self):
            """Update the results display with simulation results"""
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                return
            
            # Check if we're using the new results dashboard
            if hasattr(self, 'results_dashboard'):
                # Update the comprehensive dashboard with all results
                self.results_dashboard.update_results(self.simulation_results)
                
                # Ensure the freq_text is also populated for backward compatibility
                self.freq_text.clear()
                for i, freq in enumerate(self.simulation_results['natural_frequencies_hz']):
                    if i < 10:  # Show only first 10 modes
                        self.freq_text.append(f"Mode {i+1}: {freq:.2f} Hz")
            else:
                # Use the legacy display method
                # Display natural frequencies
                self.freq_text.clear()
                for i, freq in enumerate(self.simulation_results['natural_frequencies_hz']):
                    if i < 10:  # Show only first 10 modes
                        self.freq_text.append(f"Mode {i+1}: {freq:.2f} Hz")
                
                # Update beam deflection plot
                self.plot_beam_deflection()
                
                # Update node selection combo
                self.node_combo.clear()
                for i in range(len(self.simulation_results['coords'])):
                    x = self.simulation_results['coords'][i]
                    self.node_combo.addItem(f"Node {i+1} (x={x:.2f}m)")
                
                # Connect node selection to plot update
                self.node_combo.currentIndexChanged.connect(self.plot_node_displacement)
                
                # Initial node plot
                if self.node_combo.count() > 0:
                    self.node_combo.setCurrentIndex(0)
                    self.plot_node_displacement()
                    
                # Update animations
                if hasattr(self, 'beam_animation_adapter'):
                    self.beam_animation_adapter.update_animation(self.simulation_results)
                    
                if hasattr(self, 'mode_shape_adapter'):
                    self.mode_shape_adapter.update_results(self.simulation_results)
                
            # Enable save results button
            self.save_results_btn.setEnabled(True)
            
        def plot_beam_deflection(self):
            """Plot the beam deflection"""
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                return
            
            # Check if canvas exists
            if not hasattr(self, 'beam_canvas') or self.beam_canvas is None:
                return
            
            # Clear the canvas
            self.beam_canvas.clear()
            ax = self.beam_canvas.figure.add_subplot(111)
            
            # Get data
            x = self.simulation_results['coords']
            t = self.simulation_results['time']
            u = self.simulation_results['displacement']
            
            # Plot beam deflection at selected time points
            num_frames = min(10, len(t))
            step = len(t) // num_frames
            
            # Extract displacement at even nodes (translations)
            u_disp = u[::2, :]
            
            for i in range(0, len(t), step):
                if i >= len(t):
                    continue
                
                # Get displacement at this time
                deflection = u_disp[:, i]
                
                # Scale factor for visualization
                scale = 1.0
                if np.max(np.abs(deflection)) > 0:
                    scale = 0.2 * np.max(x) / np.max(np.abs(deflection))
                
                # Plot the deflected beam
                ax.plot(x, scale * deflection, label=f"t={t[i]:.2f}s")
            
            # Plot settings
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Deflection (scaled)')
            ax.set_title('Beam Deflection')
            ax.grid(True)
            ax.legend()
            
            # Update canvas
            self.beam_canvas.draw()
            
        def plot_node_displacement(self):
            """Plot the displacement of the selected node over time"""
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                return
            
            # Check if canvas exists
            if not hasattr(self, 'node_canvas') or self.node_canvas is None:
                return
            
            # Get selected node
            if self.node_combo.count() == 0:
                return
            
            node_idx = self.node_combo.currentIndex()
            
            # Clear the canvas
            self.node_canvas.clear()
            ax = self.node_canvas.figure.add_subplot(111)
            
            # Get data
            t = self.simulation_results['time']
            u = self.simulation_results['displacement']
            v = self.simulation_results['velocity']
            a = self.simulation_results['acceleration']
            
            # Get the displacement DOF for this node
            dof = node_idx * 2  # 2 DOFs per node (displacement and rotation)
            
            # Plot displacement, velocity, and acceleration
            ax.plot(t, u[dof, :], label='Displacement')
            ax.plot(t, v[dof, :], label='Velocity')
            ax.plot(t, a[dof, :] / 1000, label='Acceleration (÷1000)')
            
            # Plot settings
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Node {node_idx+1} Response')
            ax.grid(True)
            ax.legend()
            
            # Update canvas
            self.node_canvas.draw()
            
        def save_beam_results(self):
            """Save beam simulation results to CSV files"""
            if not hasattr(self, 'simulation_results') or not self.simulation_results:
                QMessageBox.warning(self, "No Results", "Please run a simulation before saving results.")
                return
                
            try:
                # Ask for directory to save results
                save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results")
                if not save_dir:
                    return
                    
                # Save natural frequencies
                freq_file = os.path.join(save_dir, "natural_frequencies.csv")
                with open(freq_file, 'w') as f:
                    f.write("Mode,Frequency (Hz)\n")
                    for i, freq in enumerate(self.simulation_results['natural_frequencies_hz']):
                        f.write(f"{i+1},{freq:.6f}\n")
                        
                # Save displacement results
                time = self.simulation_results['time']
                disp = self.simulation_results['displacement']
                coords = self.simulation_results['coords']
                
                # Create a DataFrame for displacement over time
                disp_data = pd.DataFrame()
                disp_data['Time (s)'] = time
                
                # Add columns for each node's displacement
                for i in range(len(coords)):
                    dof = i * 2  # 2 DOFs per node (displacement and rotation)
                    disp_data[f'Node {i+1} (x={coords[i]:.2f}m)'] = disp[dof, :]
                    
                # Save to CSV
                disp_file = os.path.join(save_dir, "displacement_results.csv")
                disp_data.to_csv(disp_file, index=False)
                
                # Create a DataFrame for the currently selected node's detailed response
                if self.node_combo.count() > 0:
                    node_idx = self.node_combo.currentIndex()
                    dof = node_idx * 2
                    
                    node_data = pd.DataFrame()
                    node_data['Time (s)'] = time
                    node_data['Displacement'] = self.simulation_results['displacement'][dof, :]
                    node_data['Velocity'] = self.simulation_results['velocity'][dof, :]
                    node_data['Acceleration'] = self.simulation_results['acceleration'][dof, :]
                    
                    # Save to CSV
                    node_file = os.path.join(save_dir, f"node_{node_idx+1}_response.csv")
                    node_data.to_csv(node_file, index=False)
                
                # Show success message
                QMessageBox.information(self, "Results Saved", 
                                       f"Results successfully saved to {save_dir}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Results", f"Error: {str(e)}")
    

