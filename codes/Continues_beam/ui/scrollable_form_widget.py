"""
Scrollable form widget for the continuous beam application.
"""

from PyQt5.QtWidgets import QWidget, QScrollArea, QFormLayout


class ScrollableFormWidget(QScrollArea):
    """
    A scrollable area containing a form layout.
    
    This widget is useful for forms that might have many fields and need
    to fit into a limited space.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the scrollable form widget.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Make the scroll area take the size of its content
        self.setWidgetResizable(True)
        
        # Create a container widget and form layout
        self.container = QWidget()
        self.form_layout = QFormLayout(self.container)
        
        # Set the container as the scroll area's widget
        self.setWidget(self.container)
        
    def addRow(self, label, widget):
        """
        Add a row to the form layout.
        
        Parameters:
        -----------
        label : str or QWidget
            Label for the form row
        widget : QWidget
            Widget for the form row
        """
        self.form_layout.addRow(label, widget)
        
    def layout(self):
        """
        Get the form layout.
        
        Returns:
        --------
        QFormLayout : The form layout
        """
        return self.form_layout 