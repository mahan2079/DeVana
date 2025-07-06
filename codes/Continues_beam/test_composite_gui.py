"""
Test script for the Composite Beam GUI Integration

This script tests the new composite beam interface to ensure all components
work correctly together.
"""

import sys
import os

# Add the parent directory to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test main module imports
        from Continues_beam import (
            CompositeBeamInterface,
            EnhancedCrossSectionVisualizer,
            BeamSideViewWidget,
            ForceVisualizationWidget,
            MaterialDatabase,
            create_composite_beam_interface,
            get_available_materials,
            create_example_composite
        )
        print("✓ Main module imports successful")
        
        # Test UI module imports
        from Continues_beam.ui import (
            CompositeBeamInterface as UICompositeBeamInterface,
            EnhancedCrossSectionVisualizer as UIEnhancedCrossSectionVisualizer,
            BeamSideViewWidget as UIBeamSideViewWidget,
            ForceVisualizationWidget as UIForceVisualizationWidget,
            MaterialDatabase as UIMaterialDatabase
        )
        print("✓ UI module imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_material_database():
    """Test the material database functionality."""
    print("\nTesting material database...")
    
    try:
        from Continues_beam import MaterialDatabase, get_available_materials
        
        # Test database creation
        db = MaterialDatabase()
        print("✓ Material database created")
        
        # Test getting materials
        materials = get_available_materials()
        print(f"✓ Found {len(materials)} materials: {materials[:3]}...")
        
        # Test material properties
        steel_props = db.get_material_properties_at_temperature('Steel', 20)
        if steel_props:
            print(f"✓ Steel properties: E={steel_props['E']/1e9:.1f} GPa, ρ={steel_props['rho']:.0f} kg/m³")
        
        return True
        
    except Exception as e:
        print(f"✗ Material database error: {e}")
        return False

def test_example_composite():
    """Test the example composite creation."""
    print("\nTesting example composite...")
    
    try:
        from Continues_beam import create_example_composite
        
        example = create_example_composite()
        print(f"✓ Example composite created: {example['name']}")
        print(f"✓ Number of layers: {len(example['layers'])}")
        print(f"✓ Beam dimensions: {example['beam_width']*1000:.1f}mm × {example['beam_length']*1000:.0f}mm")
        
        return True
        
    except Exception as e:
        print(f"✗ Example composite error: {e}")
        return False

def test_gui_creation():
    """Test GUI component creation (without showing)."""
    print("\nTesting GUI component creation...")
    
    try:
        # Import PyQt5 (this will fail if not available)
        from PyQt5.QtWidgets import QApplication
        
        # Create minimal application
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        from Continues_beam import (
            create_composite_beam_interface,
            EnhancedCrossSectionVisualizer,
            BeamSideViewWidget,
            ForceVisualizationWidget,
            MaterialDatabase
        )
        
        # Test main interface creation
        interface = create_composite_beam_interface()
        print("✓ Composite beam interface created")
        
        # Test individual widget creation
        cross_section = EnhancedCrossSectionVisualizer()
        print("✓ Enhanced cross-section visualizer created")
        
        side_view = BeamSideViewWidget()
        print("✓ Beam side view widget created")
        
        force_viz = ForceVisualizationWidget()
        print("✓ Force visualization widget created")
        
        # Test setting some data
        db = MaterialDatabase()
        example_layers = [
            {
                'name': 'Test Layer',
                'thickness': 0.005,
                'material_type': 'Steel',
                'E_func': lambda T: 210e9,
                'rho_func': lambda T: 7800
            }
        ]
        
        cross_section.set_layers(example_layers, 0.05)
        print("✓ Cross-section data set successfully")
        
        side_view.set_beam_geometry(1.0, example_layers)
        print("✓ Side view data set successfully")
        
        force_viz.set_beam_and_forces(1.0, example_layers, [])
        print("✓ Force visualization data set successfully")
        
        return True
        
    except ImportError:
        print("! PyQt5 not available - skipping GUI tests")
        return True
        
    except Exception as e:
        print(f"✗ GUI creation error: {e}")
        return False

def test_analysis_integration():
    """Test the analysis integration."""
    print("\nTesting analysis integration...")
    
    try:
        from Continues_beam import solve_beam_vibration, calc_composite_properties
        
        # Create test layers
        layers = [
            {
                'thickness': 0.005,
                'E_func': lambda T: 210e9,
                'rho_func': lambda T: 7800
            }
        ]
        
        # Test effective properties calculation
        EI_eff, rhoA_eff = calc_composite_properties(0.05, layers)
        print(f"✓ Effective properties: EI={EI_eff:.2e} N·m², ρA={rhoA_eff:.2f} kg/m")
        
        # Test beam analysis (simplified)
        solver_layers = [
            {
                'height': 0.005,
                'E': lambda T=0: 210e9,
                'rho': lambda T=0: 7800
            }
        ]
        
        results = solve_beam_vibration(
            width=0.05,
            layers=solver_layers,
            L=1.0,
            k_spring=0.0,
            num_elems=10,  # Small number for quick test
            t_span=(0, 1.0),
            num_time_points=50
        )
        
        print(f"✓ Analysis completed - first natural frequency: {results['natural_frequencies_hz'][0]:.2f} Hz")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis integration error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("COMPOSITE BEAM GUI INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Material Database", test_material_database),
        ("Example Composite", test_example_composite),
        ("GUI Creation", test_gui_creation),
        ("Analysis Integration", test_analysis_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Composite beam GUI is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 