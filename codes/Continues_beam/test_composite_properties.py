"""
Test script for composite beam properties calculation

This script demonstrates and validates the composite beam properties calculation
functionality, showing how effective properties are computed for multi-layer
composite structures.
"""

import numpy as np
from beam.properties import calc_composite_properties


def test_single_layer():
    """Test with a single homogeneous layer"""
    print("=== Single Layer Test ===")
    
    width = 0.05  # 50mm width
    layers = [
        {
            'thickness': 0.01,  # 10mm thickness
            'E_func': lambda T: 210e9,  # Steel E
            'rho_func': lambda T: 7800   # Steel density
        }
    ]
    
    EI_eff, rhoA_eff = calc_composite_properties(width, layers)
    
    # Calculate expected values
    E = 210e9
    rho = 7800
    h = 0.01
    b = width
    
    # For a single layer, neutral axis is at h/2
    I_expected = (b * h**3) / 12  # Second moment of area
    EI_expected = E * I_expected
    rhoA_expected = rho * b * h
    
    print(f"Width: {width*1000:.1f} mm")
    print(f"Thickness: {h*1000:.1f} mm")
    print(f"Material: Steel (E={E/1e9:.0f} GPa, ρ={rho} kg/m³)")
    print(f"Expected EI: {EI_expected:.2e} N·m²")
    print(f"Calculated EI: {EI_eff:.2e} N·m²")
    print(f"Expected ρA: {rhoA_expected:.2f} kg/m")
    print(f"Calculated ρA: {rhoA_eff:.2f} kg/m")
    print(f"EI Error: {abs(EI_eff - EI_expected)/EI_expected*100:.6f}%")
    print(f"ρA Error: {abs(rhoA_eff - rhoA_expected)/rhoA_expected*100:.6f}%")
    print()


def test_two_layer_composite():
    """Test with a two-layer composite (steel-aluminum)"""
    print("=== Two Layer Composite Test ===")
    
    width = 0.05  # 50mm width
    layers = [
        {
            'thickness': 0.005,  # 5mm steel layer (bottom)
            'E_func': lambda T: 210e9,  # Steel
            'rho_func': lambda T: 7800
        },
        {
            'thickness': 0.005,  # 5mm aluminum layer (top)
            'E_func': lambda T: 70e9,   # Aluminum
            'rho_func': lambda T: 2700
        }
    ]
    
    EI_eff, rhoA_eff = calc_composite_properties(width, layers)
    
    # Manual calculation for verification
    E1, E2 = 210e9, 70e9
    rho1, rho2 = 7800, 2700
    h1, h2 = 0.005, 0.005
    b = width
    
    # Areas
    A1 = b * h1
    A2 = b * h2
    
    # Neutral axis calculation
    y1_centroid = h1 / 2  # Centroid of layer 1 from bottom
    y2_centroid = h1 + h2 / 2  # Centroid of layer 2 from bottom
    
    neutral_axis = (E1 * A1 * y1_centroid + E2 * A2 * y2_centroid) / (E1 * A1 + E2 * A2)
    
    # EI calculation
    I1 = (b * h1**3) / 12
    I2 = (b * h2**3) / 12
    d1 = y1_centroid - neutral_axis
    d2 = y2_centroid - neutral_axis
    
    EI_expected = E1 * (I1 + A1 * d1**2) + E2 * (I2 + A2 * d2**2)
    rhoA_expected = rho1 * A1 + rho2 * A2
    
    print(f"Layer 1 (Steel): {h1*1000:.1f}mm, E={E1/1e9:.0f} GPa, ρ={rho1} kg/m³")
    print(f"Layer 2 (Aluminum): {h2*1000:.1f}mm, E={E2/1e9:.0f} GPa, ρ={rho2} kg/m³")
    print(f"Neutral axis: {neutral_axis*1000:.2f} mm from bottom")
    print(f"Expected EI: {EI_expected:.2e} N·m²")
    print(f"Calculated EI: {EI_eff:.2e} N·m²")
    print(f"Expected ρA: {rhoA_expected:.2f} kg/m")
    print(f"Calculated ρA: {rhoA_eff:.2f} kg/m")
    print(f"EI Error: {abs(EI_eff - EI_expected)/EI_expected*100:.6f}%")
    print(f"ρA Error: {abs(rhoA_eff - rhoA_expected)/rhoA_expected*100:.6f}%")
    print()


def test_sandwich_structure():
    """Test with a sandwich structure (carbon fiber - foam - carbon fiber)"""
    print("=== Sandwich Structure Test ===")
    
    width = 0.05  # 50mm width
    layers = [
        {
            'thickness': 0.001,  # 1mm carbon fiber (bottom)
            'E_func': lambda T: 150e9,  # Carbon fiber
            'rho_func': lambda T: 1600
        },
        {
            'thickness': 0.015,  # 15mm foam core
            'E_func': lambda T: 0.1e9,  # Foam
            'rho_func': lambda T: 100
        },
        {
            'thickness': 0.001,  # 1mm carbon fiber (top)
            'E_func': lambda T: 150e9,  # Carbon fiber
            'rho_func': lambda T: 1600
        }
    ]
    
    EI_eff, rhoA_eff = calc_composite_properties(width, layers)
    
    # Calculate total properties
    total_thickness = sum(layer['thickness'] for layer in layers)
    total_mass = sum(layer['rho_func'](0) * width * layer['thickness'] for layer in layers)
    
    print(f"Sandwich structure:")
    print(f"  Bottom CF: {layers[0]['thickness']*1000:.1f}mm")
    print(f"  Foam core: {layers[1]['thickness']*1000:.1f}mm")
    print(f"  Top CF: {layers[2]['thickness']*1000:.1f}mm")
    print(f"Total thickness: {total_thickness*1000:.1f}mm")
    print(f"Total mass per unit length: {total_mass:.2f} kg/m")
    print(f"Calculated EI: {EI_eff:.2e} N·m²")
    print(f"Calculated ρA: {rhoA_eff:.2f} kg/m")
    
    # Compare with individual face sheets
    cf_thickness = layers[0]['thickness']
    cf_E = layers[0]['E_func'](0)
    face_sheet_EI = 2 * cf_E * width * cf_thickness * ((total_thickness - cf_thickness) / 2)**2
    
    print(f"Face sheet contribution to EI: {face_sheet_EI:.2e} N·m²")
    print(f"Face sheet EI / Total EI: {face_sheet_EI/EI_eff:.2f}")
    print()


def test_temperature_dependent():
    """Test with temperature-dependent properties"""
    print("=== Temperature-Dependent Properties Test ===")
    
    width = 0.05  # 50mm width
    layers = [
        {
            'thickness': 0.008,  # 8mm thickness
            'E_func': lambda T: 210e9 * (1 - 0.0001 * T),  # E decreases with T
            'rho_func': lambda T: 7800 * (1 + 0.00001 * T)  # ρ increases with T
        }
    ]
    
    # Test at different temperatures
    temperatures = [0, 50, 100, 200]
    
    print("Temperature effects on effective properties:")
    print("T (°C) | E (GPa) | ρ (kg/m³) | EI (N·m²) | ρA (kg/m)")
    print("-" * 60)
    
    for T in temperatures:
        # Update layer functions to evaluate at specific temperature
        temp_layers = [
            {
                'thickness': layers[0]['thickness'],
                'E_func': lambda T_val: layers[0]['E_func'](T),
                'rho_func': lambda T_val: layers[0]['rho_func'](T)
            }
        ]
        
        EI_eff, rhoA_eff = calc_composite_properties(width, temp_layers)
        E_val = layers[0]['E_func'](T)
        rho_val = layers[0]['rho_func'](T)
        
        print(f"{T:5.0f}  | {E_val/1e9:5.1f} | {rho_val:8.0f} | {EI_eff:.2e} | {rhoA_eff:.2f}")
    
    print()


def test_multi_layer_complex():
    """Test with a complex multi-layer structure"""
    print("=== Complex Multi-Layer Test ===")
    
    width = 0.05  # 50mm width
    layers = [
        {
            'thickness': 0.002,  # 2mm steel (bottom)
            'E_func': lambda T: 210e9,
            'rho_func': lambda T: 7800
        },
        {
            'thickness': 0.003,  # 3mm aluminum
            'E_func': lambda T: 70e9,
            'rho_func': lambda T: 2700
        },
        {
            'thickness': 0.001,  # 1mm carbon fiber
            'E_func': lambda T: 150e9,
            'rho_func': lambda T: 1600
        },
        {
            'thickness': 0.004,  # 4mm aluminum
            'E_func': lambda T: 70e9,
            'rho_func': lambda T: 2700
        },
        {
            'thickness': 0.002,  # 2mm steel (top)
            'E_func': lambda T: 210e9,
            'rho_func': lambda T: 7800
        }
    ]
    
    EI_eff, rhoA_eff = calc_composite_properties(width, layers)
    
    print("Multi-layer structure (bottom to top):")
    materials = ['Steel', 'Aluminum', 'Carbon Fiber', 'Aluminum', 'Steel']
    for i, (layer, material) in enumerate(zip(layers, materials)):
        print(f"  Layer {i+1}: {layer['thickness']*1000:.1f}mm {material}")
    
    total_thickness = sum(layer['thickness'] for layer in layers)
    print(f"Total thickness: {total_thickness*1000:.1f}mm")
    print(f"Effective EI: {EI_eff:.2e} N·m²")
    print(f"Effective ρA: {rhoA_eff:.2f} kg/m")
    
    # Compare with equivalent steel beam
    steel_thickness = total_thickness
    steel_EI = 210e9 * (width * steel_thickness**3) / 12
    steel_rhoA = 7800 * width * steel_thickness
    
    print(f"Equivalent steel beam:")
    print(f"  EI: {steel_EI:.2e} N·m²")
    print(f"  ρA: {steel_rhoA:.2f} kg/m")
    print(f"Stiffness ratio (composite/steel): {EI_eff/steel_EI:.2f}")
    print(f"Mass ratio (composite/steel): {rhoA_eff/steel_rhoA:.2f}")
    print()


def main():
    """Run all tests"""
    print("Composite Beam Properties Calculation Tests")
    print("=" * 50)
    print()
    
    test_single_layer()
    test_two_layer_composite()
    test_sandwich_structure()
    test_temperature_dependent()
    test_multi_layer_complex()
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    main() 