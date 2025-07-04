"""
Layer dialog for editing beam layers.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QDialogButtonBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QGroupBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt


class LayerDialog(QDialog):
    """
    Dialog for adding or editing a beam layer.
    
    This dialog allows the user to specify the height, Young's modulus,
    and density of a beam layer. The material properties can be specified
    either as constants or as functions of temperature.
    """
    
    def __init__(self, parent=None, layer_data=None):
        """
        Initialize the dialog.
        
        Parameters:
        -----------
        parent : QWidget
            Parent widget
        layer_data : dict, optional
            Existing layer data for editing
        """
        super().__init__(parent)
        self.setWindowTitle("Layer Properties")
        self.setMinimumWidth(400)
        
        # Set up the layout
        main_layout = QVBoxLayout(self)
        
        # Geometry group
        geometry_group = QGroupBox("Geometry")
        geometry_layout = QFormLayout(geometry_group)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.001, 1.0)
        self.height_spin.setValue(0.05)  # Default 5 cm
        self.height_spin.setSuffix(" m")
        self.height_spin.setDecimals(3)
        geometry_layout.addRow("Height:", self.height_spin)
        
        main_layout.addWidget(geometry_group)
        
        # Young's modulus group
        e_group = QGroupBox("Young's Modulus")
        e_layout = QVBoxLayout(e_group)
        
        e_type_layout = QHBoxLayout()
        self.e_const_radio = QRadioButton("Constant")
        self.e_func_radio = QRadioButton("Function")
        self.e_type_group = QButtonGroup()
        self.e_type_group.addButton(self.e_const_radio)
        self.e_type_group.addButton(self.e_func_radio)
        self.e_const_radio.setChecked(True)
        
        e_type_layout.addWidget(self.e_const_radio)
        e_type_layout.addWidget(self.e_func_radio)
        e_layout.addLayout(e_type_layout)
        
        # Constant E input
        e_const_layout = QFormLayout()
        self.e_const_spin = QDoubleSpinBox()
        self.e_const_spin.setRange(1e6, 1e12)
        self.e_const_spin.setValue(210e9)  # Default steel
        self.e_const_spin.setSuffix(" Pa")
        self.e_const_spin.setDecimals(2)
        self.e_const_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        e_const_layout.addRow("Value:", self.e_const_spin)
        e_layout.addLayout(e_const_layout)
        
        # Function E input
        e_func_layout = QFormLayout()
        self.e_func_edit = QLineEdit()
        self.e_func_edit.setPlaceholderText("e.g., 210e9 * (1 - 0.0001 * T)")
        e_func_layout.addRow("Expression:", self.e_func_edit)
        e_layout.addLayout(e_func_layout)
        
        main_layout.addWidget(e_group)
        
        # Density group
        rho_group = QGroupBox("Density")
        rho_layout = QVBoxLayout(rho_group)
        
        rho_type_layout = QHBoxLayout()
        self.rho_const_radio = QRadioButton("Constant")
        self.rho_func_radio = QRadioButton("Function")
        self.rho_type_group = QButtonGroup()
        self.rho_type_group.addButton(self.rho_const_radio)
        self.rho_type_group.addButton(self.rho_func_radio)
        self.rho_const_radio.setChecked(True)
        
        rho_type_layout.addWidget(self.rho_const_radio)
        rho_type_layout.addWidget(self.rho_func_radio)
        rho_layout.addLayout(rho_type_layout)
        
        # Constant rho input
        rho_const_layout = QFormLayout()
        self.rho_const_spin = QDoubleSpinBox()
        self.rho_const_spin.setRange(100, 20000)
        self.rho_const_spin.setValue(7800)  # Default steel
        self.rho_const_spin.setSuffix(" kg/m³")
        self.rho_const_spin.setDecimals(0)
        rho_const_layout.addRow("Value:", self.rho_const_spin)
        rho_layout.addLayout(rho_const_layout)
        
        # Function rho input
        rho_func_layout = QFormLayout()
        self.rho_func_edit = QLineEdit()
        self.rho_func_edit.setPlaceholderText("e.g., 7800 * (1 - 0.00001 * T)")
        rho_func_layout.addRow("Expression:", self.rho_func_edit)
        rho_layout.addLayout(rho_func_layout)
        
        main_layout.addWidget(rho_group)
        
        # Connect radio buttons to enable/disable appropriate inputs
        self.e_const_radio.toggled.connect(self.update_e_inputs)
        self.rho_const_radio.toggled.connect(self.update_rho_inputs)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Set initial state of inputs
        self.update_e_inputs(True)
        self.update_rho_inputs(True)
        
        # Fill with existing data if editing
        if layer_data:
            self.set_layer_data(layer_data)
    
    def update_e_inputs(self, checked):
        """Update enabled state of Young's modulus inputs."""
        self.e_const_spin.setEnabled(self.e_const_radio.isChecked())
        self.e_func_edit.setEnabled(self.e_func_radio.isChecked())
    
    def update_rho_inputs(self, checked):
        """Update enabled state of density inputs."""
        self.rho_const_spin.setEnabled(self.rho_const_radio.isChecked())
        self.rho_func_edit.setEnabled(self.rho_func_radio.isChecked())
    
    def get_layer_data(self):
        """
        Get the layer data from the dialog inputs.
        
        Returns:
        --------
        dict : Layer data with height, E, and rho
        """
        # Layer height
        layer = {
            'height': self.height_spin.value()
        }
        
        # Young's modulus
        if self.e_const_radio.isChecked():
            # Constant E
            e_val = self.e_const_spin.value()
            layer['E'] = e_val
        else:
            # Function E (as string)
            e_expr = self.e_func_edit.text().strip()
            if e_expr:
                # Parse later in utils.parse_expression
                layer['E'] = e_expr
            else:
                # Default if empty
                layer['E'] = self.e_const_spin.value()
        
        # Density
        if self.rho_const_radio.isChecked():
            # Constant rho
            rho_val = self.rho_const_spin.value()
            layer['rho'] = rho_val
        else:
            # Function rho (as string)
            rho_expr = self.rho_func_edit.text().strip()
            if rho_expr:
                # Parse later in utils.parse_expression
                layer['rho'] = rho_expr
            else:
                # Default if empty
                layer['rho'] = self.rho_const_spin.value()
        
        return layer
    
    def set_layer_data(self, layer_data):
        """
        Set the dialog inputs from layer data.
        
        Parameters:
        -----------
        layer_data : dict
            Layer data with height, E, and rho
        """
        # Set height
        if 'height' in layer_data:
            self.height_spin.setValue(layer_data['height'])
        
        # Set Young's modulus
        if 'E' in layer_data:
            E = layer_data['E']
            if isinstance(E, (int, float)):
                # Constant E
                self.e_const_radio.setChecked(True)
                self.e_const_spin.setValue(E)
            elif isinstance(E, str):
                # Function E as string
                self.e_func_radio.setChecked(True)
                self.e_func_edit.setText(E)
            else:
                # Function E as callable
                self.e_func_radio.setChecked(True)
                self.e_func_edit.setText("Custom function")
        
        # Set density
        if 'rho' in layer_data:
            rho = layer_data['rho']
            if isinstance(rho, (int, float)):
                # Constant rho
                self.rho_const_radio.setChecked(True)
                self.rho_const_spin.setValue(rho)
            elif isinstance(rho, str):
                # Function rho as string
                self.rho_func_radio.setChecked(True)
                self.rho_func_edit.setText(rho)
            else:
                # Function rho as callable
                self.rho_func_radio.setChecked(True)
                self.rho_func_edit.setText("Custom function") 