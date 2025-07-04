from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from Continues_beam.ui.beam_animation import BeamAnimationWidget

class BeamAnimationAdapter(QWidget):
    """
    Adapter class to integrate the beam animation widget into DeVana
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create the beam animation widget
        self.animation_widget = BeamAnimationWidget(self)
        layout.addWidget(self.animation_widget)
        
        # Add additional control panel at the top for easier access
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setSpacing(5)
        
        # Time display and speed controls
        time_layout = QHBoxLayout()
        time_layout.setSpacing(10)
        
        time_label = QLabel("Current Time:")
        time_label.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(time_label)
        
        self.current_time = QLabel("0.00 s")
        self.current_time.setStyleSheet("font-weight: bold; color: #2196F3;")
        time_layout.addWidget(self.current_time)
        
        # Speed control
        time_layout.addSpacing(20)
        speed_label = QLabel("Animation Speed:")
        speed_label.setStyleSheet("font-weight: bold;")
        time_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.setMinimumWidth(150)
        self.speed_slider.setMaximumWidth(200)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self.update_speed)
        time_layout.addWidget(self.speed_slider)
        
        self.speed_value = QLabel("100%")
        time_layout.addWidget(self.speed_value)
        
        time_layout.addStretch()
        control_layout.addLayout(time_layout)
        
        # Animation control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play Animation")
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
        self.reset_button = QPushButton("⟳ Reset Animation")
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
        self.speed_slider.setEnabled(False)
        
        # Set up timer to update the time display
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_time_display)
        self.update_timer.start(100)  # Update every 100ms
        
    def update_animation(self, results):
        """Update the animation with new results"""
        self.animation_widget.update_animation(results)
        # Enable controls after animation data is loaded
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.speed_slider.setEnabled(True)
        
    def update_time_display(self):
        """Update the time display from the animation widget"""
        if hasattr(self.animation_widget, 'times') and self.animation_widget.times is not None:
            if hasattr(self.animation_widget, 'current_time_idx'):
                idx = int(self.animation_widget.current_time_idx)
                if idx < len(self.animation_widget.times):
                    self.current_time.setText(f"{self.animation_widget.times[idx]:.2f} s")
        
    def toggle_animation(self):
        """Toggle animation play/pause"""
        if hasattr(self.animation_widget, 'toggle_animation'):
            self.animation_widget.toggle_animation()
            # Update button text based on animation state
            if hasattr(self.animation_widget, 'timer') and self.animation_widget.timer.isActive():
                self.play_button.setText("⏸ Pause Animation")
            else:
                self.play_button.setText("▶ Play Animation")
        
    def reset_animation(self):
        """Reset the animation"""
        if hasattr(self.animation_widget, 'reset_animation'):
            self.animation_widget.reset_animation()
            # Reset play button text
            self.play_button.setText("▶ Play Animation")
            
    def update_speed(self, value):
        """Update animation speed"""
        if hasattr(self.animation_widget, 'update_speed'):
            self.animation_widget.update_speed(value)
        self.speed_value.setText(f"{value}%") 