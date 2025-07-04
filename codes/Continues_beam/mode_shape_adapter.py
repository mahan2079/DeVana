from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt
from Continues_beam.ui.mode_shape_animation import ModeShapeAnimationWidget

class ModeShapeAdapter(QWidget):
    """
    Adapter class to integrate the mode shape animation widget into DeVana
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create the mode shape animation widget
        self.animation_widget = ModeShapeAnimationWidget(self)
        layout.addWidget(self.animation_widget)
        
        # Add additional control buttons at the top for easier access
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(5)
        
        # Mode selection controls
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)
        
        mode_label = QLabel("Select Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(200)
        self.mode_combo.setMinimumHeight(30)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)
        
        # Frequency display
        self.freq_label = QLabel("Frequency: 0.00 Hz")
        self.freq_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        mode_layout.addWidget(self.freq_label)
        
        mode_layout.addStretch()
        control_layout.addLayout(mode_layout)
        
        # Animation control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play Mode Animation")
        self.play_button.clicked.connect(self.toggle_animation)
        self.play_button.setMinimumHeight(30)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.play_button)
        
        # Reset button
        self.reset_button = QPushButton("⟳ Reset Mode Animation")
        self.reset_button.clicked.connect(self.reset_animation)
        self.reset_button.setMinimumHeight(30)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a69b7;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.reset_button)
        
        control_layout.addLayout(button_layout)
        
        # Add control layout to main layout
        layout.insertLayout(0, control_layout)
        
        # Initially disable controls until animation data is loaded
        self.play_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        
    def update_results(self, results):
        """Update the animation with new results"""
        self.animation_widget.update_results(results)
        
        # Update mode selection combo box
        self.mode_combo.clear()
        if hasattr(self.animation_widget, 'mode_shapes') and self.animation_widget.mode_shapes is not None:
            for i in range(self.animation_widget.mode_shapes.shape[1]):
                freq_text = ""
                if hasattr(self.animation_widget, 'natural_frequencies') and self.animation_widget.natural_frequencies is not None:
                    if i < len(self.animation_widget.natural_frequencies):
                        freq_text = f" ({self.animation_widget.natural_frequencies[i]:.2f} Hz)"
                self.mode_combo.addItem(f"Mode {i+1}{freq_text}")
        
        # Update frequency label
        if hasattr(self.animation_widget, 'natural_frequencies') and self.animation_widget.natural_frequencies is not None:
            if len(self.animation_widget.natural_frequencies) > 0:
                self.freq_label.setText(f"Frequency: {self.animation_widget.natural_frequencies[0]:.2f} Hz")
        
        # Enable controls after animation data is loaded
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        
    def change_mode(self, index):
        """Change the current mode being animated"""
        if hasattr(self.animation_widget, 'change_mode'):
            self.animation_widget.change_mode(index)
            
            # Update frequency label
            if hasattr(self.animation_widget, 'natural_frequencies') and self.animation_widget.natural_frequencies is not None:
                if index < len(self.animation_widget.natural_frequencies):
                    self.freq_label.setText(f"Frequency: {self.animation_widget.natural_frequencies[index]:.2f} Hz")
        
    def toggle_animation(self):
        """Toggle animation play/pause"""
        if hasattr(self.animation_widget, 'toggle_animation'):
            self.animation_widget.toggle_animation()
            # Update button text based on animation state
            if hasattr(self.animation_widget, 'animation_running') and self.animation_widget.animation_running:
                self.play_button.setText("⏸ Pause Mode Animation")
            else:
                self.play_button.setText("▶ Play Mode Animation")
        
    def reset_animation(self):
        """Reset the animation"""
        if hasattr(self.animation_widget, 'reset_animation'):
            self.animation_widget.reset_animation()
            # Reset play button text
            self.play_button.setText("▶ Play Mode Animation") 