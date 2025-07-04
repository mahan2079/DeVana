"""
Dialog for creating and editing force regions.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QDialogButtonBox, QLineEdit, QComboBox, QTabWidget,
    QListWidget, QListWidgetItem, QPushButton, QWidget,
    QRadioButton, QButtonGroup, QStackedWidget, QMessageBox,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt

from .force_widgets import (
    createForceWidget, createPointSpatialWidget, createDistributedSpatialWidget
)


class ForceRegionDialog(QDialog):
    """
    Dialog for creating or editing a force region.
    
    A force region defines a type of force (harmonic, step, impulse, or custom)
    and where it is applied to the beam (point locations or distributed regions).
    """
    
    def __init__(self, parent=None, region=None):
        """
        Initialize the dialog.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        region : ForceRegion, optional
            Existing region for editing
        """
        super().__init__(parent)
        self.setWindowTitle("Force Region")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Set up the layout
        main_layout = QVBoxLayout(self)
        
        # Basic information layout
        form_layout = QFormLayout()
        
        # Region name
        self.name_edit = QLineEdit()
        if region:
            self.name_edit.setText(region.name)
        else:
            self.name_edit.setText("Force Region")
        form_layout.addRow("Name:", self.name_edit)
        
        # Force type combobox
        self.force_type_combo = QComboBox()
        self.force_type_combo.addItem("Harmonic", "harmonic")
        self.force_type_combo.addItem("Step", "step")
        self.force_type_combo.addItem("Impulse", "impulse")
        self.force_type_combo.addItem("Custom", "custom")
        
        if region:
            index = self.force_type_combo.findData(region.force_type)
            if index >= 0:
                self.force_type_combo.setCurrentIndex(index)
        
        self.force_type_combo.currentIndexChanged.connect(self.update_force_widget)
        form_layout.addRow("Force Type:", self.force_type_combo)
        
        # Spatial type radio buttons
        spatial_layout = QHBoxLayout()
        self.point_radio = QRadioButton("Point")
        self.distributed_radio = QRadioButton("Distributed")
        
        self.spatial_type_group = QButtonGroup()
        self.spatial_type_group.addButton(self.point_radio, 0)
        self.spatial_type_group.addButton(self.distributed_radio, 1)
        
        if region and region.spatial_type == 'distributed':
            self.distributed_radio.setChecked(True)
        else:
            self.point_radio.setChecked(True)
            
        self.spatial_type_group.buttonClicked.connect(self.update_spatial_widgets)
        
        spatial_layout.addWidget(self.point_radio)
        spatial_layout.addWidget(self.distributed_radio)
        form_layout.addRow("Spatial Type:", spatial_layout)
        
        main_layout.addLayout(form_layout)
        
        # Force parameters section
        force_group_label = QLabel("Force Parameters")
        force_group_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(force_group_label)
        
        # Create stacked widget for different force types
        self.force_stack = QStackedWidget()
        
        # Add widgets for different force types
        for i in range(self.force_type_combo.count()):
            force_type = self.force_type_combo.itemData(i)
            self.force_stack.addWidget(createForceWidget(force_type))
            
        main_layout.addWidget(self.force_stack)
        
        # Location parameters section
        location_group_label = QLabel("Location Parameters")
        location_group_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(location_group_label)
        
        # Create tabs for locations list and editor
        self.location_tabs = QTabWidget()
        
        # Locations list tab
        self.locations_list_widget = QWidget()
        locations_list_layout = QVBoxLayout(self.locations_list_widget)
        
        self.locations_list = QListWidget()
        locations_list_layout.addWidget(self.locations_list)
        
        # Buttons for managing locations
        locations_btn_layout = QHBoxLayout()
        
        self.add_location_btn = QPushButton("Add")
        self.add_location_btn.clicked.connect(self.add_location)
        
        self.edit_location_btn = QPushButton("Edit")
        self.edit_location_btn.clicked.connect(self.edit_location)
        
        self.remove_location_btn = QPushButton("Remove")
        self.remove_location_btn.clicked.connect(self.remove_location)
        
        # Add spacer to push buttons to the right
        locations_btn_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        locations_btn_layout.addWidget(self.add_location_btn)
        locations_btn_layout.addWidget(self.edit_location_btn)
        locations_btn_layout.addWidget(self.remove_location_btn)
        
        locations_list_layout.addLayout(locations_btn_layout)
        
        # Location editor tab
        self.location_editor_widget = QWidget()
        location_editor_layout = QVBoxLayout(self.location_editor_widget)
        
        # Create stacked widget for different spatial types
        self.spatial_stack = QStackedWidget()
        self.spatial_stack.addWidget(createPointSpatialWidget())
        self.spatial_stack.addWidget(createDistributedSpatialWidget())
        
        location_editor_layout.addWidget(self.spatial_stack)
        
        # Add button for adding location from editor
        add_from_editor_layout = QHBoxLayout()
        
        self.add_from_editor_btn = QPushButton("Add to Locations")
        self.add_from_editor_btn.clicked.connect(self.add_from_editor)
        
        add_from_editor_layout.addStretch()
        add_from_editor_layout.addWidget(self.add_from_editor_btn)
        
        location_editor_layout.addLayout(add_from_editor_layout)
        
        # Add tabs to tab widget
        self.location_tabs.addTab(self.locations_list_widget, "Locations")
        self.location_tabs.addTab(self.location_editor_widget, "Editor")
        
        main_layout.addWidget(self.location_tabs)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Initialize the locations list if editing an existing region
        self.locations = []
        if region:
            self.locations = region.locations.copy()
            
            # Set force parameters
            if region.params:
                self.set_force_params(region.force_type, region.params)
                
        # Refresh the locations list
        self.refresh_locations_list()
        
        # Update widgets based on initial values
        self.update_force_widget(self.force_type_combo.currentIndex())
        self.update_spatial_widgets()
        
    def update_force_widget(self, index):
        """Update the force widget based on the selected force type."""
        if index >= 0:
            self.force_stack.setCurrentIndex(index)
            
    def update_spatial_widgets(self):
        """Update spatial widgets based on the selected spatial type."""
        self.spatial_stack.setCurrentIndex(self.spatial_type_group.checkedId())
        
    def refresh_locations_list(self):
        """Refresh the list of locations."""
        self.locations_list.clear()
        
        for i, location in enumerate(self.locations):
            item = QListWidgetItem()
            
            # Create a descriptive label
            if self.point_radio.isChecked():
                position = location.get('position', 0)
                scale = location.get('scale', 1)
                item.setText(f"Point at x = {position:.2f} m (scale {scale:.2f})")
            else:  # distributed
                start = location.get('start', 0)
                end = location.get('end', 0)
                scale = location.get('scale', 1)
                item.setText(f"Region from x = {start:.2f} to {end:.2f} m (scale {scale:.2f})")
                
            self.locations_list.addItem(item)
            
    def add_location(self):
        """Add a new location using the editor."""
        # Switch to editor tab
        self.location_tabs.setCurrentIndex(1)
        
    def edit_location(self):
        """Edit the selected location."""
        selected_items = self.locations_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a location to edit.")
            return
            
        # Get the selected index
        selected_index = self.locations_list.row(selected_items[0])
        
        # Get the location
        location = self.locations[selected_index]
        
        # Set the editor values
        if self.point_radio.isChecked():
            position_spin = self.spatial_stack.widget(0).findChild(QWidget, "position")
            scale_spin = self.spatial_stack.widget(0).findChild(QWidget, "scale")
            
            if position_spin:
                position_spin.setValue(location.get('position', 0))
            if scale_spin:
                scale_spin.setValue(location.get('scale', 1))
        else:  # distributed
            start_spin = self.spatial_stack.widget(1).findChild(QWidget, "start")
            end_spin = self.spatial_stack.widget(1).findChild(QWidget, "end")
            scale_spin = self.spatial_stack.widget(1).findChild(QWidget, "scale")
            
            if start_spin:
                start_spin.setValue(location.get('start', 0))
            if end_spin:
                end_spin.setValue(location.get('end', 0))
            if scale_spin:
                scale_spin.setValue(location.get('scale', 1))
                
        # Switch to editor tab
        self.location_tabs.setCurrentIndex(1)
        
        # Remove the old location (will be replaced when the user adds from editor)
        self.locations.pop(selected_index)
        self.refresh_locations_list()
        
    def remove_location(self):
        """Remove the selected location."""
        selected_items = self.locations_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a location to remove.")
            return
            
        # Get the selected index
        selected_index = self.locations_list.row(selected_items[0])
        
        # Remove the location
        self.locations.pop(selected_index)
        
        # Refresh the list
        self.refresh_locations_list()
        
    def add_from_editor(self):
        """Add the current location from the editor to the locations list."""
        spatial_type = "point" if self.point_radio.isChecked() else "distributed"
        
        if spatial_type == "point":
            # Get values from point spatial widget
            position_spin = self.spatial_stack.widget(0).findChild(QWidget, "position")
            scale_spin = self.spatial_stack.widget(0).findChild(QWidget, "scale")
            
            if position_spin and scale_spin:
                location = {
                    'position': position_spin.value(),
                    'scale': scale_spin.value()
                }
                self.locations.append(location)
        else:  # distributed
            # Get values from distributed spatial widget
            start_spin = self.spatial_stack.widget(1).findChild(QWidget, "start")
            end_spin = self.spatial_stack.widget(1).findChild(QWidget, "end")
            scale_spin = self.spatial_stack.widget(1).findChild(QWidget, "scale")
            
            if start_spin and end_spin and scale_spin:
                location = {
                    'start': start_spin.value(),
                    'end': end_spin.value(),
                    'scale': scale_spin.value()
                }
                self.locations.append(location)
                
        # Refresh the list
        self.refresh_locations_list()
        
        # Switch back to locations tab
        self.location_tabs.setCurrentIndex(0)
        
    def get_force_params(self, force_type):
        """
        Get the force parameters from the current force widget.
        
        Parameters:
        -----------
        force_type : str
            Type of force ('harmonic', 'step', 'impulse', 'custom')
            
        Returns:
        --------
        dict : Parameters for the force function
        """
        widget = self.force_stack.currentWidget()
        params = {}
        
        if force_type == 'harmonic':
            amplitude_spin = widget.findChild(QWidget, "amplitude")
            frequency_spin = widget.findChild(QWidget, "frequency")
            phase_spin = widget.findChild(QWidget, "phase")
            
            if amplitude_spin and frequency_spin and phase_spin:
                params = {
                    'amplitude': amplitude_spin.value(),
                    'frequency': frequency_spin.value(),
                    'phase': phase_spin.value()
                }
                
        elif force_type == 'step':
            amplitude_spin = widget.findChild(QWidget, "amplitude")
            start_time_spin = widget.findChild(QWidget, "start_time")
            ramp_time_spin = widget.findChild(QWidget, "ramp_time")
            
            if amplitude_spin and start_time_spin and ramp_time_spin:
                params = {
                    'amplitude': amplitude_spin.value(),
                    'start_time': start_time_spin.value(),
                    'ramp_time': ramp_time_spin.value()
                }
                
        elif force_type == 'impulse':
            amplitude_spin = widget.findChild(QWidget, "amplitude")
            start_time_spin = widget.findChild(QWidget, "start_time")
            duration_spin = widget.findChild(QWidget, "duration")
            
            if amplitude_spin and start_time_spin and duration_spin:
                params = {
                    'amplitude': amplitude_spin.value(),
                    'start_time': start_time_spin.value(),
                    'duration': duration_spin.value()
                }
                
        elif force_type == 'custom':
            expression_edit = widget.findChild(QWidget, "expression")
            
            if expression_edit:
                params = {
                    'expression': expression_edit.text()
                }
                
        return params
        
    def set_force_params(self, force_type, params):
        """
        Set the force parameters in the force widget.
        
        Parameters:
        -----------
        force_type : str
            Type of force ('harmonic', 'step', 'impulse', 'custom')
        params : dict
            Parameters for the force function
        """
        # Find the correct index for the force type
        index = self.force_type_combo.findData(force_type)
        if index >= 0:
            self.force_type_combo.setCurrentIndex(index)
            
            # Get the widget
            widget = self.force_stack.widget(index)
            
            if force_type == 'harmonic':
                amplitude_spin = widget.findChild(QWidget, "amplitude")
                frequency_spin = widget.findChild(QWidget, "frequency")
                phase_spin = widget.findChild(QWidget, "phase")
                
                if amplitude_spin and 'amplitude' in params:
                    amplitude_spin.setValue(params['amplitude'])
                if frequency_spin and 'frequency' in params:
                    frequency_spin.setValue(params['frequency'])
                if phase_spin and 'phase' in params:
                    phase_spin.setValue(params['phase'])
                    
            elif force_type == 'step':
                amplitude_spin = widget.findChild(QWidget, "amplitude")
                start_time_spin = widget.findChild(QWidget, "start_time")
                ramp_time_spin = widget.findChild(QWidget, "ramp_time")
                
                if amplitude_spin and 'amplitude' in params:
                    amplitude_spin.setValue(params['amplitude'])
                if start_time_spin and 'start_time' in params:
                    start_time_spin.setValue(params['start_time'])
                if ramp_time_spin and 'ramp_time' in params:
                    ramp_time_spin.setValue(params['ramp_time'])
                    
            elif force_type == 'impulse':
                amplitude_spin = widget.findChild(QWidget, "amplitude")
                start_time_spin = widget.findChild(QWidget, "start_time")
                duration_spin = widget.findChild(QWidget, "duration")
                
                if amplitude_spin and 'amplitude' in params:
                    amplitude_spin.setValue(params['amplitude'])
                if start_time_spin and 'start_time' in params:
                    start_time_spin.setValue(params['start_time'])
                if duration_spin and 'duration' in params:
                    duration_spin.setValue(params['duration'])
                    
            elif force_type == 'custom':
                expression_edit = widget.findChild(QWidget, "expression")
                
                if expression_edit and 'expression' in params:
                    expression_edit.setText(params['expression'])
    
    def get_region_data(self):
        """
        Get the region data from the dialog inputs.
        
        Returns:
        --------
        dict : Region data with name, force_type, params, spatial_type, and locations
        """
        # Get the force type
        force_type = self.force_type_combo.currentData()
        
        # Get the spatial type
        spatial_type = "point" if self.point_radio.isChecked() else "distributed"
        
        # Get the force parameters
        params = self.get_force_params(force_type)
        
        # Create the region data
        region_data = {
            'name': self.name_edit.text(),
            'force_type': force_type,
            'params': params,
            'spatial_type': spatial_type,
            'locations': self.locations
        }
        
        return region_data 