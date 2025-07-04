"""
Panel for managing force regions.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, 
    QListWidget, QListWidgetItem, QLabel, QSpacerItem, QSizePolicy,
    QMessageBox
)
from PyQt5.QtCore import Qt

from .force_region_dialog import ForceRegionDialog
from ..utils import ForceRegion


class ForceRegionsPanel(QWidget):
    """
    Panel for managing multiple force regions.
    
    This panel provides a list of force regions and buttons for adding,
    editing, and removing them. It interacts with a ForceRegionManager
    to keep track of the regions.
    """
    
    def __init__(self, region_manager, parent=None):
        """
        Initialize the panel.
        
        Parameters:
        -----------
        region_manager : ForceRegionManager
            Manager for force regions
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        self.region_manager = region_manager
        
        # Set up the layout
        main_layout = QVBoxLayout(self)
        
        # List of regions
        self.regions_list = QListWidget()
        self.regions_list.setSelectionMode(QListWidget.SingleSelection)
        main_layout.addWidget(self.regions_list)
        
        # Create a description label
        desc_label = QLabel("Define regions where forces act on the beam.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(desc_label)
        
        # Buttons for managing regions
        buttons_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Region")
        self.add_btn.clicked.connect(self.add_region)
        
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self.edit_region)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_region)
        
        # Add spacer to push buttons to the right
        buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        buttons_layout.addWidget(self.add_btn)
        buttons_layout.addWidget(self.edit_btn)
        buttons_layout.addWidget(self.remove_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Refresh the list
        self.refresh_regions_list()
        
    def refresh_regions_list(self):
        """Refresh the list of regions from the manager."""
        self.regions_list.clear()
        
        for i, region in enumerate(self.region_manager.regions):
            item = QListWidgetItem()
            
            # Create a descriptive label
            if region.spatial_type == 'point':
                if len(region.locations) == 1:
                    location_text = f"at x = {region.locations[0].get('position', 0):.2f} m"
                else:
                    location_text = f"at {len(region.locations)} points"
            else:  # distributed
                if len(region.locations) == 1:
                    loc = region.locations[0]
                    location_text = f"from x = {loc.get('start', 0):.2f} to {loc.get('end', 0):.2f} m"
                else:
                    location_text = f"in {len(region.locations)} regions"
            
            item.setText(f"{region.name} ({region.force_type} force {location_text})")
            self.regions_list.addItem(item)
            
    def add_region(self):
        """Add a new force region."""
        dialog = ForceRegionDialog(parent=self)
        if dialog.exec_():
            # Get the region data
            region_data = dialog.get_region_data()
            
            # Create a region and add it to the manager
            region = ForceRegion(
                name=region_data['name'],
                force_type=region_data['force_type'],
                params=region_data['params'],
                spatial_type=region_data['spatial_type'],
                locations=region_data['locations']
            )
            
            self.region_manager.add_region(region)
            
            # Refresh the list
            self.refresh_regions_list()
            
    def edit_region(self):
        """Edit the selected force region."""
        selected_items = self.regions_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a region to edit.")
            return
            
        # Get the selected index
        selected_index = self.regions_list.row(selected_items[0])
        
        # Get the region from the manager
        region = self.region_manager.regions[selected_index]
        
        # Create a dialog with the region data
        dialog = ForceRegionDialog(parent=self, region=region)
        if dialog.exec_():
            # Get the updated region data
            region_data = dialog.get_region_data()
            
            # Update the region
            region.name = region_data['name']
            region.force_type = region_data['force_type']
            region.params = region_data['params']
            region.spatial_type = region_data['spatial_type']
            region.locations = region_data['locations']
            
            # Refresh the list
            self.refresh_regions_list()
            
    def remove_region(self):
        """Remove the selected force region."""
        selected_items = self.regions_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a region to remove.")
            return
            
        # Get the selected index
        selected_index = self.regions_list.row(selected_items[0])
        
        # Remove the region from the manager
        self.region_manager.remove_region(selected_index)
        
        # Refresh the list
        self.refresh_regions_list() 