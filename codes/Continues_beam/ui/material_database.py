"""
Material Database for Composite Beam Analysis

This module provides a comprehensive database of engineering materials with:
- Temperature-dependent material properties
- Common composite materials
- Metal alloys and their properties
- Foam and core materials
- Scientifically accurate property relationships
"""

import numpy as np


class MaterialDatabase:
    """
    Database of engineering materials with temperature-dependent properties.
    All properties are functions of temperature in Celsius.
    """
    
    def __init__(self):
        self.materials = self._initialize_material_database()
        
    def _initialize_material_database(self):
        """Initialize the material database with common materials."""
        materials = {}
        
        # Steel (AISI 1020)
        materials['Steel'] = {
            'name': 'Steel (AISI 1020)',
            'category': 'Metals',
            'E_func': lambda T: 210e9 * (1 - 0.0001 * T),  # Young's modulus decreases with temperature
            'rho_func': lambda T: 7800 * (1 + 0.000012 * T),  # Density increases slightly with temperature
            'description': 'Common structural steel with good strength and ductility',
            'typical_thickness_range': (0.001, 0.1),  # 1mm to 100mm
            'color': '#4A90E2'  # Steel blue
        }
        
        # Aluminum (6061-T6)
        materials['Aluminum'] = {
            'name': 'Aluminum (6061-T6)',
            'category': 'Metals',
            'E_func': lambda T: 70e9 * (1 - 0.0002 * T),
            'rho_func': lambda T: 2700 * (1 + 0.000023 * T),
            'description': 'Lightweight aluminum alloy with good corrosion resistance',
            'typical_thickness_range': (0.0005, 0.05),  # 0.5mm to 50mm
            'color': '#C0C0C0'  # Silver
        }
        
        # Carbon Fiber (Unidirectional)
        materials['Carbon Fiber'] = {
            'name': 'Carbon Fiber (Unidirectional)',
            'category': 'Composites',
            'E_func': lambda T: 150e9 * (1 - 0.00005 * T),  # Less temperature sensitive
            'rho_func': lambda T: 1600 * (1 + 0.000001 * T),  # Very stable density
            'description': 'High-strength, lightweight composite material',
            'typical_thickness_range': (0.0001, 0.01),  # 0.1mm to 10mm
            'color': '#2F2F2F'  # Dark gray
        }
        
        # Glass Fiber (E-Glass)
        materials['Glass Fiber'] = {
            'name': 'Glass Fiber (E-Glass)',
            'category': 'Composites',
            'E_func': lambda T: 72e9 * (1 - 0.00008 * T),
            'rho_func': lambda T: 2600 * (1 + 0.000005 * T),
            'description': 'Cost-effective composite material with good electrical properties',
            'typical_thickness_range': (0.0002, 0.02),  # 0.2mm to 20mm
            'color': '#90EE90'  # Light green
        }
        
        # Titanium (Ti-6Al-4V)
        materials['Titanium'] = {
            'name': 'Titanium (Ti-6Al-4V)',
            'category': 'Metals',
            'E_func': lambda T: 114e9 * (1 - 0.00015 * T),
            'rho_func': lambda T: 4430 * (1 + 0.000009 * T),
            'description': 'High-strength, low-density aerospace material',
            'typical_thickness_range': (0.0005, 0.05),  # 0.5mm to 50mm
            'color': '#87CEEB'  # Sky blue
        }
        
        # Foam Core (PVC)
        materials['Foam Core'] = {
            'name': 'PVC Foam Core',
            'category': 'Cores',
            'E_func': lambda T: 100e6 * (1 - 0.001 * T),  # Much softer, more temperature sensitive
            'rho_func': lambda T: 100 * (1 + 0.0001 * T),
            'description': 'Lightweight core material for sandwich structures',
            'typical_thickness_range': (0.005, 0.1),  # 5mm to 100mm
            'color': '#FFE4B5'  # Moccasin
        }
        
        # Honeycomb Core (Aluminum)
        materials['Honeycomb Core'] = {
            'name': 'Aluminum Honeycomb',
            'category': 'Cores',
            'E_func': lambda T: 500e6 * (1 - 0.0003 * T),  # Higher stiffness than foam
            'rho_func': lambda T: 50 * (1 + 0.00002 * T),   # Very light
            'description': 'High-strength, ultra-lightweight honeycomb core',
            'typical_thickness_range': (0.01, 0.15),  # 10mm to 150mm
            'color': '#FFD700'  # Gold
        }
        
        # Stainless Steel (316L)
        materials['Stainless Steel'] = {
            'name': 'Stainless Steel (316L)',
            'category': 'Metals',
            'E_func': lambda T: 200e9 * (1 - 0.00012 * T),
            'rho_func': lambda T: 8000 * (1 + 0.000015 * T),
            'description': 'Corrosion-resistant steel for harsh environments',
            'typical_thickness_range': (0.001, 0.1),  # 1mm to 100mm
            'color': '#778899'  # Light slate gray
        }
        
        # Copper
        materials['Copper'] = {
            'name': 'Copper (C11000)',
            'category': 'Metals',
            'E_func': lambda T: 110e9 * (1 - 0.0002 * T),
            'rho_func': lambda T: 8960 * (1 + 0.000017 * T),
            'description': 'High conductivity metal for electrical applications',
            'typical_thickness_range': (0.0001, 0.05),  # 0.1mm to 50mm
            'color': '#B87333'  # Dark goldenrod
        }
        
        # Aramid Fiber (Kevlar)
        materials['Aramid Fiber'] = {
            'name': 'Aramid Fiber (Kevlar)',
            'category': 'Composites',
            'E_func': lambda T: 124e9 * (1 - 0.00003 * T),  # Very stable
            'rho_func': lambda T: 1440 * (1 + 0.000002 * T),
            'description': 'High-impact resistance fiber for ballistic applications',
            'typical_thickness_range': (0.0002, 0.015),  # 0.2mm to 15mm
            'color': '#FFFF00'  # Yellow
        }
        
        return materials
        
    def get_material(self, material_name):
        """Get material properties by name."""
        return self.materials.get(material_name, None)
        
    def get_material_names(self):
        """Get list of all available material names."""
        return list(self.materials.keys())
        
    def get_materials_by_category(self, category):
        """Get all materials in a specific category."""
        return {name: mat for name, mat in self.materials.items() 
                if mat['category'] == category}
        
    def get_categories(self):
        """Get list of all material categories."""
        categories = set(mat['category'] for mat in self.materials.values())
        return sorted(list(categories))
        
    def add_custom_material(self, name, E_func, rho_func, description="Custom material", 
                           category="Custom", thickness_range=(0.001, 0.1), color="#888888"):
        """Add a custom material to the database."""
        self.materials[name] = {
            'name': name,
            'category': category,
            'E_func': E_func,
            'rho_func': rho_func,
            'description': description,
            'typical_thickness_range': thickness_range,
            'color': color
        }
        
    def get_material_properties_at_temperature(self, material_name, temperature):
        """Get material properties evaluated at a specific temperature."""
        material = self.get_material(material_name)
        if material is None:
            return None
            
        return {
            'E': material['E_func'](temperature),
            'rho': material['rho_func'](temperature),
            'name': material['name'],
            'description': material['description']
        }
        
    def create_temperature_profile(self, material_name, temp_range=(-50, 200), num_points=100):
        """Create a temperature vs. properties profile for a material."""
        material = self.get_material(material_name)
        if material is None:
            return None
            
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
        E_values = [material['E_func'](T) for T in temperatures]
        rho_values = [material['rho_func'](T) for T in temperatures]
        
        return {
            'temperatures': temperatures,
            'E_values': E_values,
            'rho_values': rho_values,
            'material_name': material['name']
        }
        
    def suggest_material_combination(self, application="general"):
        """Suggest material combinations for specific applications."""
        suggestions = {
            'aerospace': [
                {
                    'description': 'Carbon Fiber Sandwich Panel',
                    'layers': [
                        ('Carbon Fiber', 0.001),
                        ('Honeycomb Core', 0.02),
                        ('Carbon Fiber', 0.001)
                    ]
                },
                {
                    'description': 'Titanium-Aluminum Hybrid',
                    'layers': [
                        ('Titanium', 0.002),
                        ('Aluminum', 0.005),
                        ('Titanium', 0.002)
                    ]
                }
            ],
            'automotive': [
                {
                    'description': 'Steel-Aluminum Composite',
                    'layers': [
                        ('Steel', 0.003),
                        ('Aluminum', 0.004),
                        ('Steel', 0.003)
                    ]
                },
                {
                    'description': 'Glass Fiber Reinforced Panel',
                    'layers': [
                        ('Glass Fiber', 0.002),
                        ('Foam Core', 0.015),
                        ('Glass Fiber', 0.002)
                    ]
                }
            ],
            'marine': [
                {
                    'description': 'Stainless Steel Sandwich',
                    'layers': [
                        ('Stainless Steel', 0.002),
                        ('Foam Core', 0.025),
                        ('Stainless Steel', 0.002)
                    ]
                }
            ],
            'general': [
                {
                    'description': 'Basic Steel-Aluminum',
                    'layers': [
                        ('Steel', 0.005),
                        ('Aluminum', 0.003),
                        ('Steel', 0.005)
                    ]
                }
            ]
        }
        
        return suggestions.get(application, suggestions['general'])
        
    def validate_material_combination(self, layers):
        """Validate a material combination for potential issues."""
        warnings = []
        
        # Check for thermal expansion mismatch
        if len(layers) > 1:
            # Simplified check - in reality, would need thermal expansion coefficients
            E_values = []
            for layer in layers:
                material_name = layer.get('material_type', 'Custom')
                material = self.get_material(material_name)
                if material:
                    E_values.append(material['E_func'](20))  # At room temperature
                    
            if E_values:
                max_E = max(E_values)
                min_E = min(E_values)
                if max_E / min_E > 100:  # Large stiffness mismatch
                    warnings.append("Large stiffness mismatch detected - consider intermediate layers")
                    
        # Check layer thickness ratios
        thicknesses = [layer.get('thickness', 0.001) for layer in layers]
        if thicknesses:
            max_thickness = max(thicknesses)
            min_thickness = min(thicknesses)
            if max_thickness / min_thickness > 50:
                warnings.append("Large thickness variation - verify manufacturing feasibility")
                
        return warnings
        
    def get_material_info_string(self, material_name, temperature=20):
        """Get a formatted string with material information."""
        material = self.get_material(material_name)
        if material is None:
            return "Material not found"
            
        E = material['E_func'](temperature)
        rho = material['rho_func'](temperature)
        
        info = f"{material['name']}\n"
        info += f"Category: {material['category']}\n"
        info += f"Young's Modulus: {E/1e9:.1f} GPa (at {temperature}°C)\n"
        info += f"Density: {rho:.0f} kg/m³ (at {temperature}°C)\n"
        info += f"Description: {material['description']}\n"
        info += f"Typical thickness: {material['typical_thickness_range'][0]*1000:.1f}-{material['typical_thickness_range'][1]*1000:.1f} mm"
        
        return info


# Example usage and testing functions
def create_example_composites():
    """Create some example composite configurations."""
    db = MaterialDatabase()
    
    examples = {
        'aerospace_panel': {
            'name': 'Aerospace Carbon Fiber Panel',
            'layers': [
                {'material': 'Carbon Fiber', 'thickness': 0.0005},
                {'material': 'Honeycomb Core', 'thickness': 0.02},
                {'material': 'Carbon Fiber', 'thickness': 0.0005}
            ]
        },
        'automotive_body': {
            'name': 'Automotive Body Panel',
            'layers': [
                {'material': 'Steel', 'thickness': 0.001},
                {'material': 'Foam Core', 'thickness': 0.01},
                {'material': 'Aluminum', 'thickness': 0.002}
            ]
        },
        'marine_hull': {
            'name': 'Marine Hull Section',
            'layers': [
                {'material': 'Glass Fiber', 'thickness': 0.003},
                {'material': 'Foam Core', 'thickness': 0.025},
                {'material': 'Glass Fiber', 'thickness': 0.003}
            ]
        }
    }
    
    return examples


if __name__ == "__main__":
    # Test the material database
    db = MaterialDatabase()
    
    print("Available materials:")
    for name in db.get_material_names():
        print(f"- {name}")
        
    print("\nMaterial categories:")
    for category in db.get_categories():
        print(f"- {category}")
        
    print(f"\nSteel properties at 20°C:")
    steel_props = db.get_material_properties_at_temperature('Steel', 20)
    if steel_props:
        print(f"E = {steel_props['E']/1e9:.1f} GPa")
        print(f"ρ = {steel_props['rho']:.0f} kg/m³")
        
    print(f"\nCarbon Fiber info:")
    print(db.get_material_info_string('Carbon Fiber')) 