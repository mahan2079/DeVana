"""Test the visualization adapters"""

try:
    from beam_animation_adapter import BeamAnimationAdapter
    print("✓ BeamAnimationAdapter imported successfully")
except Exception as e:
    print(f"✗ BeamAnimationAdapter import failed: {e}")

try:
    from mode_shape_adapter import ModeShapeAdapter
    print("✓ ModeShapeAdapter imported successfully")
except Exception as e:
    print(f"✗ ModeShapeAdapter import failed: {e}")

print("Adapter tests completed") 