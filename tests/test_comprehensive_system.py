import sys
import os

# Add the current directory to path to ensure devana is found
sys.path.append(os.getcwd())

from devana import DVASystem

def test_dva_system_comprehensive():
    print("\n--- Testing Comprehensive DVA System ---")
    try:
        # 1. Initialize the system with characteristic frequency and damping
        system = DVASystem(omega_dc=5000.0, zeta_dc=0.01)
        
        # 2. Set Main System Parameters
        # MU (Ratio M2/M1), Stiffness Ratios (Lambda1-5), Damping Ratios (Nu1-5)
        system.set_primary_mass_ratio(1.0)
        system.set_stiffness_ratios([1.0, 1.0, 1.0, 0.5, 0.5])
        system.set_damping_ratios([0.05, 0.05, 0.05, 0.05, 0.05])
        
        # 3. Configure DVAs
        # Setting mu_1, mu_2, mu_3 (mass ratios of DVAs)
        system.set_dva_parameters(mass_ratios=[0.1, 0.1, 0.1])
        # Setting stiffness (lambda_1-15) and damping (nu_1-15)
        system.set_dva_parameters(stiffness=[1.1, 1.2, 1.3], damping=[0.02, 0.02, 0.02])

        # 4. Define Targets for masses
        # We want to monitor Peak Value for Mass 1
        targets = {
            1: {"peak_value": 0.001} # Target 0.001 for peak
        }

        # 5. Calculate Response
        print("Calculating Frequency Response...")
        results = system.calculate_response(
            omega_start=0.1, 
            omega_end=2.0, 
            points=500,
            target_masses=targets
        )

        # 6. Verify Results
        print(f"System Response calculated for: {list(results.keys())}")
        if "mass_1" in results:
            peak = results["mass_1"]["peak_values"].get("peak_value_1", 0)
            print(f"Mass 1 Primary Peak: {peak:.6e}")
            print(f"Composite Measure (Singular Response): {results['singular_response']:.4f}")

        print("Comprehensive DVA System Test: SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Comprehensive DVA System Test: FAILED - {e}")

if __name__ == "__main__":
    test_dva_system_comprehensive()
