import sys
import os
import platform
import traceback
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QSplashScreen, QVBoxLayout, QWidget, QProgressBar, QMessageBox, QTableWidget, QHeaderView, QPushButton, QHBoxLayout, QTableWidgetItem, QGraphicsOpacityEffect
from PyQt5.QtGui import QColor, QFont, QPixmap, QPainter, QLinearGradient, QBrush, QPen
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QRect, QThread, QEasingCurve

from mainwindow import MainWindow

# Store the original methods
original_init = MainWindow.__init__
original_qthread_del = getattr(QThread, "__del__", None)

# Define a closeEvent method to handle thread cleanup
def closeEvent(self, event):
    """Handle cleanup when window is closed"""
    print("Application closing - cleaning up threads...")
    
    # Clean up PSO worker
    if hasattr(self, 'pso_worker') and self.pso_worker is not None:
        if self.pso_worker.isRunning():
            print("Terminating PSO worker thread...")
            try:
                # Use our custom terminate method if available
                if hasattr(self.pso_worker, 'terminate'):
                    self.pso_worker.terminate()
                
                # Wait for a short time to let the thread finish
                if not self.pso_worker.wait(1000):  # 1 second timeout
                    print("PSO worker did not finish in time, forcing termination...")
                    # Use QThread's terminate as a last resort
                    self.pso_worker.terminate()
                    self.pso_worker.wait()
            except Exception as e:
                print(f"Error terminating PSO worker: {str(e)}")
    
    # Clean up GA worker
    if hasattr(self, 'ga_worker') and self.ga_worker is not None:
        if self.ga_worker.isRunning():
            print("Terminating GA worker thread...")
            try:
                # Wait for a short time to let the thread finish
                if not self.ga_worker.wait(1000):  # 1 second timeout
                    print("GA worker did not finish in time, forcing termination...")
                    self.ga_worker.terminate()
                    self.ga_worker.wait()
            except Exception as e:
                print(f"Error terminating GA worker: {str(e)}")
    
    # Clean up RL worker
    if hasattr(self, 'rl_worker') and self.rl_worker is not None:
        if self.rl_worker.isRunning():
            print("Terminating RL worker thread...")
            try:
                # Use our custom terminate method if available
                if hasattr(self.rl_worker, 'terminate'):
                    self.rl_worker.terminate()
                
                # Wait for a short time to let the thread finish
                if not self.rl_worker.wait(1000):  # 1 second timeout
                    print("RL worker did not finish in time, forcing termination...")
                    # Use QThread's terminate as a last resort
                    self.rl_worker.terminate()
                    self.rl_worker.wait()
            except Exception as e:
                print(f"Error terminating RL worker: {str(e)}")
    
    # Allow the close event to proceed
    event.accept()

# Safe QThread __del__ method
def safe_qthread_del(self):
    """Safe QThread destructor that terminates the thread if it's still running"""
    try:
        if self.isRunning():
            print(f"Warning: QThread is being destroyed while still running. Forcing termination.")
            self.terminate()
            self.wait(500)  # Short wait to give it a chance to terminate
    except Exception as e:
        print(f"Error in safe QThread destructor: {str(e)}")
    
    # Call original __del__ if it exists
    if original_qthread_del:
        try:
            original_qthread_del(self)
        except Exception:
            pass

# Apply the monkey patches
MainWindow.closeEvent = closeEvent

# Only patch QThread.__del__ if it exists and safe_qthread_del can be added
try:
    QThread.__del__ = safe_qthread_del
    print("PSO Thread Fix: QThread.__del__ patched for safer thread termination")
except Exception as e:
    print(f"Warning: Could not patch QThread.__del__: {str(e)}")

# Print a confirmation message
print("PSO Thread Fix: Patch applied successfully") 

# Decoupler modules removed

# Global exception hook to catch unhandled exceptions
def exception_hook(exctype, value, tb):
    """
    Global function to catch unhandled exceptions.
    
    @param exctype: Exception type
    @param value: Exception value
    @param tb: Exception traceback
    """
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(error_msg, file=sys.stderr)
    
    # Show error dialog with details
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle("Application Error")
    msg_box.setText("An unexpected error occurred. The application will continue running.")
    msg_box.setDetailedText(error_msg)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()
    
    # Let the application continue
    sys.__excepthook__(exctype, value, tb)

# Set the exception hook
sys.excepthook = exception_hook

# DeVana v0.4.1

class WelcomePage(QWidget):
    def __init__(self):
        """Create an elegant welcome page with sophisticated animations"""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Welcome to DeVana")
        self.setFixedSize(800, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        # Animation state variables
        self.gradient_phase = 0.0
        self.pulse_phase = 0.0
        self.particle_positions = []
        self.init_particles()
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Create animated elements
        self.title_label = QLabel("DeVana")
        self.title_label.setFont(QFont("Segoe UI Light", 48, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            color: white;
            background: transparent;
            padding: 20px;
        """)
        
        self.version_label = QLabel("Version V0.4.1")
        self.version_label.setFont(QFont("Segoe UI", 18, QFont.Normal))
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.9);
            background: transparent;
            letter-spacing: 2px;
        """)
        
        self.description_label = QLabel("Advanced Vibration Optimization System")
        self.description_label.setFont(QFont("Segoe UI", 16, QFont.Normal))
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.8);
            background: transparent;
            padding: 10px;
        """)
        
        self.tagline_label = QLabel("Designed for Mechanical Engineers and Vibration Specialists")
        self.tagline_label.setFont(QFont("Segoe UI", 13, QFont.Normal))
        self.tagline_label.setAlignment(Qt.AlignCenter)
        self.tagline_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.7);
            background: transparent;
            padding: 5px;
        """)
        
        # Elegant progress indicator
        self.progress_container = QWidget()
        self.progress_container.setFixedHeight(60)
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(100, 10, 100, 10)
        
        self.loading_label = QLabel("Loading...")
        self.loading_label.setFont(QFont("Segoe UI", 12))
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 2px;
                background-color: rgba(255, 255, 255, 0.1);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4DB6AC, stop:0.5 #26A69A, stop:1 #009688);
                border-radius: 2px;
            }
        """)
        
        progress_layout.addWidget(self.loading_label)
        progress_layout.addWidget(self.progress_bar)
        
        # Add widgets to layout with proper spacing
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addSpacing(10)
        layout.addWidget(self.version_label)
        layout.addSpacing(40)
        layout.addWidget(self.description_label)
        layout.addSpacing(15)
        layout.addWidget(self.tagline_label)
        layout.addStretch(1)
        layout.addWidget(self.progress_container)
        layout.addStretch(1)
        
        # Set initial opacity for fade-in effect
        self.title_label.setGraphicsEffect(self.create_opacity_effect(0))
        self.version_label.setGraphicsEffect(self.create_opacity_effect(0))
        self.description_label.setGraphicsEffect(self.create_opacity_effect(0))
        self.tagline_label.setGraphicsEffect(self.create_opacity_effect(0))
        self.progress_container.setGraphicsEffect(self.create_opacity_effect(0))
        
        # Setup animation timers
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animations)
        
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        
        # Setup property animations
        self.setup_animations()
        
    def init_particles(self):
        """Initialize floating particles for background effect"""
        import random
        self.particle_positions = []
        for _ in range(15):
            x = random.randint(0, 800)
            y = random.randint(0, 600)
            size = random.randint(2, 6)
            speed = random.uniform(0.5, 2.0)
            opacity = random.uniform(0.1, 0.4)
            self.particle_positions.append([x, y, size, speed, opacity])
    
    def create_opacity_effect(self, opacity):
        """Create a graphics effect for opacity animation"""
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(opacity)
        return effect
        
    def setup_animations(self):
        """Setup smooth property animations for UI elements"""
        
        # Title animation
        self.title_animation = QPropertyAnimation(self.title_label.graphicsEffect(), b"opacity")
        self.title_animation.setDuration(1500)
        self.title_animation.setStartValue(0.0)
        self.title_animation.setEndValue(1.0)
        self.title_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Version animation
        self.version_animation = QPropertyAnimation(self.version_label.graphicsEffect(), b"opacity")
        self.version_animation.setDuration(1200)
        self.version_animation.setStartValue(0.0)
        self.version_animation.setEndValue(1.0)
        self.version_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Description animation
        self.description_animation = QPropertyAnimation(self.description_label.graphicsEffect(), b"opacity")
        self.description_animation.setDuration(1000)
        self.description_animation.setStartValue(0.0)
        self.description_animation.setEndValue(1.0)
        self.description_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Tagline animation
        self.tagline_animation = QPropertyAnimation(self.tagline_label.graphicsEffect(), b"opacity")
        self.tagline_animation.setDuration(800)
        self.tagline_animation.setStartValue(0.0)
        self.tagline_animation.setEndValue(1.0)
        self.tagline_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Progress animation
        self.progress_animation = QPropertyAnimation(self.progress_container.graphicsEffect(), b"opacity")
        self.progress_animation.setDuration(600)
        self.progress_animation.setStartValue(0.0)
        self.progress_animation.setEndValue(1.0)
        self.progress_animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def showEvent(self, event):
        """Start sophisticated animations when window is shown"""
        super().showEvent(event)
        
        # Start background animation
        self.animation_timer.start(16)  # ~60 FPS
        
        # Start staggered fade-in animations
        QTimer.singleShot(300, self.title_animation.start)
        QTimer.singleShot(800, self.version_animation.start)
        QTimer.singleShot(1300, self.description_animation.start)
        QTimer.singleShot(1800, self.tagline_animation.start)
        QTimer.singleShot(2300, self.progress_animation.start)
        
        # Start progress animation
        QTimer.singleShot(2500, lambda: self.progress_timer.start(50))
        
    def update_animations(self):
        """Update background animations"""
        try:
            # Update gradient phase for animated background
            self.gradient_phase += 0.01
            if self.gradient_phase >= 1.0:
                self.gradient_phase = 0.0
                
            # Update pulse phase for subtle pulsing effect
            self.pulse_phase += 0.03
            if self.pulse_phase >= 6.28:  # 2π
                self.pulse_phase = 0.0
                
            # Update particle positions
            for particle in self.particle_positions:
                particle[1] += particle[3]  # Move particle down
                if particle[1] > 600:  # Reset particle to top
                    particle[1] = -10
                    import random
                    particle[0] = random.randint(0, 800)
                    
            # Trigger repaint - only if the widget is visible and not already repainting
            if self.isVisible() and not hasattr(self, '_repainting'):
                self.update()
        except Exception as e:
            print(f"Error in update_animations: {e}")
            # Stop the animation timer to prevent cascading errors
            if hasattr(self, 'animation_timer'):
                self.animation_timer.stop()
        
    def update_progress(self):
        """Update progress bar with smooth animation"""
        if self.progress_value < 100:
            # Use easing function for smooth progress
            increment = max(1, (100 - self.progress_value) / 30)
            self.progress_value = min(100, self.progress_value + increment)
            self.progress_bar.setValue(int(self.progress_value))
            
            # Update loading text
            if self.progress_value < 30:
                self.loading_label.setText("Initializing modules...")
            elif self.progress_value < 60:
                self.loading_label.setText("Loading optimization algorithms...")
            elif self.progress_value < 90:
                self.loading_label.setText("Preparing user interface...")
            else:
                self.loading_label.setText("Almost ready...")
        else:
            self.progress_timer.stop()
            self.loading_label.setText("Welcome!")
            
    def paintEvent(self, event):
        """Custom paint event with sophisticated visual effects and elegant gradients"""
        # Prevent recursive repainting
        if hasattr(self, '_repainting'):
            return
        
        self._repainting = True
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
        
            # Enhanced animated gradient background
            import math
            gradient = QLinearGradient(0, 0, self.width(), self.height())
            
            # Create more sophisticated shifting colors
            base_hue = 170 + (self.gradient_phase * 40)  # Teal to blue
            
            # More elegant color palette
            color1 = QColor.fromHsv(int(base_hue) % 360, 85, 25)          # Deep rich base
            color2 = QColor.fromHsv(int(base_hue + 40) % 360, 70, 35)     # Mid tone
            color3 = QColor.fromHsv(int(base_hue + 80) % 360, 60, 20)     # Dark accent
            color4 = QColor.fromHsv(int(base_hue + 120) % 360, 50, 30)    # Complementary
            
            gradient.setColorAt(0.0, color1)
            gradient.setColorAt(0.3, color2)
            gradient.setColorAt(0.7, color3)
            gradient.setColorAt(1.0, color4)
            painter.fillRect(self.rect(), gradient)
            
            # Enhanced floating particles with better colors
            painter.setPen(Qt.NoPen)
            for i, particle in enumerate(self.particle_positions):
                x, y, size, speed, opacity = particle
                
                # Create more vibrant particle colors
                particle_hue = (base_hue + i * 15) % 360
                glow_color = QColor.fromHsv(int(particle_hue), 100, 80, int(opacity * 180))
                center_color = QColor.fromHsv(int(particle_hue), 60, 95, int(opacity * 220))
                
                # Draw glowing outer ring
                painter.setBrush(QBrush(glow_color))
                painter.drawEllipse(int(x), int(y), size, size)
                
                # Add bright center
                painter.setBrush(QBrush(center_color))
                painter.drawEllipse(int(x + size/4), int(y + size/4), size//2, size//2)
            
            # Enhanced geometric patterns
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
            painter.setBrush(Qt.NoBrush)
            
            # Multiple animated circle sets
            pulse_scale = 1.0 + 0.08 * math.sin(self.pulse_phase)
            center_x, center_y = self.width() // 2, self.height() // 2
            
            # Main circles
            for i in range(4):
                radius = (40 + i * 25) * pulse_scale
                painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                                  int(radius * 2), int(radius * 2))
            
            # Secondary offset circles
            painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
            offset_scale = 1.0 + 0.05 * math.sin(self.pulse_phase + 1.5)
            for i in range(3):
                radius = (30 + i * 35) * offset_scale
                painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                                  int(radius * 2), int(radius * 2))
            
            painter.end()
        except Exception as e:
            print(f"Error in paintEvent: {e}")
        finally:
            # Always clear the repainting flag
            if hasattr(self, '_repainting'):
                delattr(self, '_repainting')


class SplashScreen(QSplashScreen):
    def __init__(self):
        """Create a splash screen with gradient background and app info"""
        # Create a pixmap for the splash screen
        pixmap = QPixmap(QSize(600, 400))
        pixmap.fill(Qt.transparent)
        
        # Create gradient background
        self.painter = QPainter(pixmap)
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0.0, QColor(0, 120, 110))
        gradient.setColorAt(1.0, QColor(0, 60, 55))
        self.painter.setBrush(QBrush(gradient))
        self.painter.setPen(Qt.NoPen)
        self.painter.drawRect(0, 0, 600, 400)
        
        # Add app info
        self.painter.setPen(Qt.white)
        font = QFont("Segoe UI", 36, QFont.Bold)
        self.painter.setFont(font)
        self.painter.drawText(50, 150, "DeVana")
        
        font.setPointSize(14)
        self.painter.setFont(font)
        self.painter.drawText(50, 200, "Version V0.4.1")
        
        font.setPointSize(12)
        font.setBold(False)
        self.painter.setFont(font)
        self.painter.drawText(50, 240, "Advanced Vibration Optimization System")
        
        # Finish painting
        self.painter.end()
        
        # Initialize splash screen with the pixmap
        super().__init__(pixmap)
        
        # Set window properties
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        
    def mousePressEvent(self, event):
        """Prevent closing the splash screen when clicked"""
        pass


def main():
    """Main application entry point with robust error handling"""
    try:
        app = QApplication(sys.argv)
        
        # Install the exception hook for the Qt message handler
        sys._excepthook = sys.excepthook
        
        # Set application style and global font
        app.setStyle("Fusion")
        app.setFont(QFont("Arial", 10))
        
        # Create and show welcome page
        welcome = WelcomePage()
        welcome.show()
        
        # Create main window but don't show it yet
        window = None
        
        # Function to safely create and show main window
        def show_main_window():
            try:
                nonlocal window
                nonlocal welcome  # Ensure welcome is accessible within this function
                
                # First make sure the welcome window is properly closed
                if welcome is not None and welcome.isVisible():
                    welcome.close()
                
                # Create the main window with proper error handling
                try:
                    window = MainWindow()
                    window.show()
                    
                    # Ensure the window is brought to the front
                    window.raise_()
                    window.activateWindow()
                    
                    print("Main window successfully created and shown")
                except Exception as e:
                    error_msg = f"Failed to initialize MainWindow: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg, file=sys.stderr)
                    QMessageBox.critical(None, "Initialization Error", 
                                        f"Failed to initialize the main window.\n\nError: {str(e)}")
                    raise  # Re-raise to trigger the outer exception handler
                
                # Setup watchdog timer to check if window is responsive
                def check_window_responsive():
                    if window is not None and window.isVisible():
                        try:
                            # Try to access some property to check if window is responsive
                            _ = window.size()
                        except Exception as e:
                            print(f"Warning: Main window not responsive: {str(e)}")
                            # Don't let the application crash, just log the error
                
                # Start watchdog timer
                watchdog = QTimer()
                watchdog.timeout.connect(check_window_responsive)
                watchdog.start(5000)  # Check every 5 seconds
                
            except Exception as e:
                error_msg = f"Error creating main window: {str(e)}\n{traceback.format_exc()}"
                print(error_msg, file=sys.stderr)
                QMessageBox.critical(None, "Error Starting Application", 
                                    "Failed to create the main window. The application will restart.\n\nError: " + str(e))
                # Attempt to restart the welcome page
                try:
                    new_welcome = WelcomePage()  # Create a new welcome page instead of referencing existing one
                    new_welcome.show()
                    new_welcome.raise_()
                    new_welcome.activateWindow()
                    QTimer.singleShot(7000, show_main_window)
                except Exception as restart_err:
                    print(f"Failed to restart welcome page: {str(restart_err)}")
                    # If we can't even create the welcome page, exit gracefully
                    QMessageBox.critical(None, "Critical Error", 
                                        "Unable to restart application. Please manually restart the program.")
                    app.quit()
        
        # 7-second timer to show main window
        QTimer.singleShot(7000, show_main_window)
        
        # Set up autosave functionality
        def auto_save():
            try:
                # Only run autosave if window exists and is visible
                if window is not None and window.isVisible():
                    # Implement autosave logic here if needed
                    print("Auto-save triggered")
            except Exception as e:
                print(f"Auto-save failed: {str(e)}")
        
        # Start autosave timer
        autosave_timer = QTimer()
        autosave_timer.timeout.connect(auto_save)
        autosave_timer.start(300000)  # Every 5 minutes
        
        # Start application main loop with exception handling
        result = app.exec_()
        
        # Clean up on application exit
        try:
            # Clean up window if it exists
            if window is not None:
                window.close()
                window.deleteLater()
                
            # Make sure any welcome page is closed
            if welcome is not None and welcome.isVisible():
                welcome.close()
                welcome.deleteLater()
                
            # Clean up any other app resources
            app.processEvents()
        except Exception as e:
            print(f"Error during application cleanup: {str(e)}")
            
        return result
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        
        # Show error in message box
        if QApplication.instance():
            QMessageBox.critical(None, "Critical Error", 
                                "A critical error occurred. Please restart the application.\n\nError: " + str(e))
        
        return 1


if __name__ == "__main__":
    # Execute the main function with additional exception handling
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Unhandled exception in application startup: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
