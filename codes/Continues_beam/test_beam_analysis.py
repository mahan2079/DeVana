"""
Test script for beam analysis implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from beam.solver import solve_beam_vibration

def test_beam_analysis():
    """Test the beam analysis with a simple example"""
    print("Testing beam analysis...")
    
    # Define beam parameters
    width = 0.05  # 5 cm width
    length = 1.0  # 1 meter length
    thickness = 0.01  # 1 cm thickness
    E = 210e9  # Steel Young's modulus (Pa)
    rho = 7800  # Steel density (kg/m³)
    
    # Create layer definition
    layers = [
        {
            'height': thickness,
            'E': lambda T=0: E,
            'rho': lambda T=0: rho
        }
    ]
    
    # Test 1: Free vibration (no external force)
    print("Test 1: Free vibration analysis...")
    try:
        results = solve_beam_vibration(
            width=width,
            layers=layers,
            L=length,
            k_spring=0,
            num_elems=20,
            f_profile=lambda x, t: 0.0,
            t_span=(0, 1),
            num_time_points=100
        )
        
        print(f"Natural frequencies (Hz): {results['natural_frequencies_hz'][:5]}")
        print("✓ Free vibration test passed")
        
    except Exception as e:
        print(f"✗ Free vibration test failed: {e}")
        return False
    
    # Test 2: Forced vibration with harmonic excitation
    print("\nTest 2: Forced vibration analysis...")
    try:
        def harmonic_force(x, t):
            # Apply force at 90% of the beam length
            if x > 0.9 * length:
                return 1000 * np.sin(2 * np.pi * 20 * t)  # 20 Hz, 1000 N
            return 0.0
        
        results = solve_beam_vibration(
            width=width,
            layers=layers,
            L=length,
            k_spring=0,
            num_elems=20,
            f_profile=harmonic_force,
            t_span=(0, 2),
            num_time_points=200
        )
        
        max_tip_displacement = np.max(np.abs(results['tip_displacement']))
        print(f"Maximum tip displacement: {max_tip_displacement:.6f} m")
        print("✓ Forced vibration test passed")
        
    except Exception as e:
        print(f"✗ Forced vibration test failed: {e}")
        return False
    
    # Test 3: Verify theoretical natural frequency for cantilever beam
    print("\nTest 3: Theoretical validation...")
    try:
        # For a cantilever beam, first natural frequency is:
        # f1 = (1.875²/2π) * sqrt(EI/ρAL⁴)
        
        # Calculate theoretical properties
        I = width * thickness**3 / 12  # Second moment of area
        EI = E * I
        rhoA = rho * width * thickness
        
        # Theoretical first natural frequency
        f1_theory = (1.875**2 / (2 * np.pi)) * np.sqrt(EI / (rhoA * length**4))
        f1_numerical = results['natural_frequencies_hz'][0]
        
        error_percent = abs(f1_numerical - f1_theory) / f1_theory * 100
        
        print(f"Theoretical first frequency: {f1_theory:.2f} Hz")
        print(f"Numerical first frequency: {f1_numerical:.2f} Hz")
        print(f"Error: {error_percent:.2f}%")
        
        if error_percent < 5:  # Accept 5% error
            print("✓ Theoretical validation passed")
        else:
            print("✗ Theoretical validation failed - large error")
            return False
            
    except Exception as e:
        print(f"✗ Theoretical validation failed: {e}")
        return False
    
    print("\n✓ All tests passed! Beam analysis implementation is working correctly.")
    return True

if __name__ == "__main__":
    test_beam_analysis() 