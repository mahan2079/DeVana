"""
Force input widgets for different force types.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QDoubleSpinBox, QLineEdit, QGroupBox, QComboBox
)


def createForceWidget(force_type, parent=None):
    """
    Create a widget for a specific force type.
    
    Parameters:
    -----------
    force_type : str
        Type of force ('harmonic', 'step', 'impulse', 'custom')
    parent : QWidget, optional
        Parent widget
        
    Returns:
    --------
    QWidget : Force widget for the specified type
    """
    if force_type == 'harmonic':
        return createHarmonicForceWidget(parent)
    elif force_type == 'step':
        return createStepForceWidget(parent)
    elif force_type == 'impulse':
        return createImpulseForceWidget(parent)
    elif force_type == 'custom':
        return createCustomForceWidget(parent)
    else:
        raise ValueError(f"Unknown force type: {force_type}")


def createHarmonicForceWidget(parent=None):
    """Create a widget for harmonic force input."""
    widget = QWidget(parent)
    layout = QFormLayout(widget)
    
    # Amplitude
    amplitude_spin = QDoubleSpinBox()
    amplitude_spin.setRange(0, 1e6)
    amplitude_spin.setValue(1000.0)
    amplitude_spin.setSuffix(" N")
    amplitude_spin.setDecimals(1)
    amplitude_spin.setObjectName("amplitude")
    layout.addRow("Amplitude:", amplitude_spin)
    
    # Frequency
    frequency_spin = QDoubleSpinBox()
    frequency_spin.setRange(0.01, 1000)
    frequency_spin.setValue(10.0)
    frequency_spin.setSuffix(" Hz")
    frequency_spin.setDecimals(2)
    frequency_spin.setObjectName("frequency")
    layout.addRow("Frequency:", frequency_spin)
    
    # Phase
    phase_spin = QDoubleSpinBox()
    phase_spin.setRange(0, 360)
    phase_spin.setValue(0.0)
    phase_spin.setSuffix(" °")
    phase_spin.setDecimals(1)
    phase_spin.setObjectName("phase")
    layout.addRow("Phase:", phase_spin)
    
    return widget


def createStepForceWidget(parent=None):
    """Create a widget for step force input."""
    widget = QWidget(parent)
    layout = QFormLayout(widget)
    
    # Amplitude
    amplitude_spin = QDoubleSpinBox()
    amplitude_spin.setRange(0, 1e6)
    amplitude_spin.setValue(1000.0)
    amplitude_spin.setSuffix(" N")
    amplitude_spin.setDecimals(1)
    amplitude_spin.setObjectName("amplitude")
    layout.addRow("Amplitude:", amplitude_spin)
    
    # Start time
    start_time_spin = QDoubleSpinBox()
    start_time_spin.setRange(0, 100)
    start_time_spin.setValue(0.5)
    start_time_spin.setSuffix(" s")
    start_time_spin.setDecimals(2)
    start_time_spin.setObjectName("start_time")
    layout.addRow("Start Time:", start_time_spin)
    
    # Ramp time
    ramp_time_spin = QDoubleSpinBox()
    ramp_time_spin.setRange(0, 10)
    ramp_time_spin.setValue(0.1)
    ramp_time_spin.setSuffix(" s")
    ramp_time_spin.setDecimals(2)
    ramp_time_spin.setObjectName("ramp_time")
    layout.addRow("Ramp Time:", ramp_time_spin)
    
    return widget


def createImpulseForceWidget(parent=None):
    """Create a widget for impulse force input."""
    widget = QWidget(parent)
    layout = QFormLayout(widget)
    
    # Amplitude
    amplitude_spin = QDoubleSpinBox()
    amplitude_spin.setRange(0, 1e6)
    amplitude_spin.setValue(1000.0)
    amplitude_spin.setSuffix(" N")
    amplitude_spin.setDecimals(1)
    amplitude_spin.setObjectName("amplitude")
    layout.addRow("Amplitude:", amplitude_spin)
    
    # Start time
    start_time_spin = QDoubleSpinBox()
    start_time_spin.setRange(0, 100)
    start_time_spin.setValue(0.5)
    start_time_spin.setSuffix(" s")
    start_time_spin.setDecimals(2)
    start_time_spin.setObjectName("start_time")
    layout.addRow("Start Time:", start_time_spin)
    
    # Duration
    duration_spin = QDoubleSpinBox()
    duration_spin.setRange(0.001, 1)
    duration_spin.setValue(0.01)
    duration_spin.setSuffix(" s")
    duration_spin.setDecimals(3)
    duration_spin.setObjectName("duration")
    layout.addRow("Duration:", duration_spin)
    
    return widget


def createCustomForceWidget(parent=None):
    """Create a widget for custom force input."""
    widget = QWidget(parent)
    layout = QFormLayout(widget)
    
    # Expression
    expression_edit = QLineEdit()
    expression_edit.setPlaceholderText("e.g., 1000 * sin(2 * pi * 10 * t)")
    expression_edit.setObjectName("expression")
    layout.addRow("Expression:", expression_edit)
    
    # Help text
    help_label = QLabel(
        "Available functions: sin, cos, exp, log, sqrt, etc.\n"
        "Use 't' as the time variable."
    )
    help_label.setWordWrap(True)
    help_label.setStyleSheet("color: gray;")
    layout.addRow("", help_label)
    
    return widget


def createPointSpatialWidget(parent=None):
    """Create a widget for point force spatial configuration."""
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)
    
    # Position
    position_layout = QFormLayout()
    position_spin = QDoubleSpinBox()
    position_spin.setRange(0, 100)
    position_spin.setValue(5.0)
    position_spin.setSuffix(" m")
    position_spin.setDecimals(2)
    position_spin.setObjectName("position")
    position_layout.addRow("Position:", position_spin)
    
    # Scale
    scale_spin = QDoubleSpinBox()
    scale_spin.setRange(0, 10)
    scale_spin.setValue(1.0)
    scale_spin.setDecimals(2)
    scale_spin.setObjectName("scale")
    position_layout.addRow("Scale:", scale_spin)
    
    layout.addLayout(position_layout)
    
    return widget


def createDistributedSpatialWidget(parent=None):
    """Create a widget for distributed force spatial configuration."""
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)
    
    # Region bounds
    bounds_layout = QFormLayout()
    
    # Start position
    start_spin = QDoubleSpinBox()
    start_spin.setRange(0, 100)
    start_spin.setValue(0.0)
    start_spin.setSuffix(" m")
    start_spin.setDecimals(2)
    start_spin.setObjectName("start")
    bounds_layout.addRow("Start:", start_spin)
    
    # End position
    end_spin = QDoubleSpinBox()
    end_spin.setRange(0, 100)
    end_spin.setValue(10.0)
    end_spin.setSuffix(" m")
    end_spin.setDecimals(2)
    end_spin.setObjectName("end")
    bounds_layout.addRow("End:", end_spin)
    
    # Scale
    scale_spin = QDoubleSpinBox()
    scale_spin.setRange(0, 10)
    scale_spin.setValue(1.0)
    scale_spin.setDecimals(2)
    scale_spin.setObjectName("scale")
    bounds_layout.addRow("Scale:", scale_spin)
    
    layout.addLayout(bounds_layout)
    
    return widget 