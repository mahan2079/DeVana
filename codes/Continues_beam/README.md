# Composite Beam Analysis Module for DeVana

This module provides advanced composite beam analysis capabilities for the DeVana application, supporting multi-layered composite structures with temperature-dependent material properties.

## Features

### Composite Layer Modeling
- Multi-layer composite beam analysis
- Temperature-dependent material properties (E, ρ)
- Automatic calculation of effective flexural rigidity (EI) and mass per unit length (ρA)
- Support for both constant and function-based material properties

### Advanced Finite Element Analysis
- Euler-Bernoulli beam elements with cubic shape functions
- Consistent mass matrix formulation
- Automatic assembly of global stiffness and mass matrices
- Support for various boundary conditions and tip springs

### Dynamic Analysis
- Natural frequency and mode shape calculation
- Time-domain vibration analysis
- Forced vibration with arbitrary load profiles
- Modal analysis with proper mode shape extraction

### Interactive Visualization
- Real-time beam vibration animation
- Mode shape visualization with amplitude control
- Interactive controls for animation playback
- Results dashboard with frequency analysis

## Composite Layer Structure

### Layer Definition Format
```python
layers = [
    {
        'thickness': 0.001,  # Layer thickness in meters
        'E_func': lambda T: 210e9 * (1 - 0.0001 * T),  # Temperature-dependent Young's modulus
        'rho_func': lambda T: 7800 * (1 - 0.00001 * T)  # Temperature-dependent density
    },
    {
        'thickness': 0.002,
        'E_func': lambda T: 70e9 * (1 - 0.0002 * T),  # Different material (e.g., aluminum)
        'rho_func': lambda T: 2700 * (1 - 0.00002 * T)
    }
]
```

### Effective Properties Calculation
The module automatically calculates effective properties using composite beam theory:
- **Neutral axis position**: Weighted by stiffness (EA)
- **Effective flexural rigidity**: EI_eff = Σ E_i * (I_i + A_i * d_i²)
- **Effective mass per unit length**: ρA_eff = Σ ρ_i * A_i

## Integration into DeVana

### Method 1: Direct Solver Usage
```python
from Continues_beam.beam.solver import solve_beam_vibration

# Define composite layers
layers = [
    {
        'height': 0.005,  # 5mm layer
        'E': lambda: 210e9,  # Steel
        'rho': lambda: 7800
    },
    {
        'height': 0.003,  # 3mm layer
        'E': lambda: 70e9,   # Aluminum
        'rho': lambda: 2700
    }
]

# Run analysis
results = solve_beam_vibration(
    width=0.05,          # 50mm width
    layers=layers,
    L=1.0,              # 1m length
    k_spring=0.0,       # No tip spring
    num_elems=50,       # 50 elements for accuracy
    f_profile=force_function,  # Custom force profile
    t_span=(0, 5),      # 5 second analysis
    num_time_points=500
)
```

### Method 2: Using the Animation Adapters
```python
from Continues_beam.beam_animation_adapter import BeamAnimationAdapter
from Continues_beam.mode_shape_adapter import ModeShapeAdapter

# Create visualization widgets
self.animation_adapter = BeamAnimationAdapter()
self.mode_shape_adapter = ModeShapeAdapter()

# Add to your UI layout
layout.addWidget(self.animation_adapter)
layout.addWidget(self.mode_shape_adapter)

# Update with results
self.animation_adapter.update_animation(results)
self.mode_shape_adapter.update_results(results)
```

### Method 3: Using the Layer Management UI
```python
from Continues_beam.ui.layer_dialog import LayerDialog

# Create layer editing dialog
dialog = LayerDialog(parent=self)
if dialog.exec_() == QDialog.Accepted:
    layer_data = dialog.get_layer_data()
    # Use layer_data in your analysis
```

## Analysis Results

The solver returns comprehensive results:
```python
{
    'time': array,                    # Time points
    'displacement': array,            # Full displacement history (ndof × ntime)
    'velocity': array,                # Velocity history
    'acceleration': array,            # Acceleration history
    'coords': array,                  # Node coordinates
    'natural_frequencies_hz': array,  # Natural frequencies in Hz
    'mode_shapes': array,             # Mode shapes (nnodes × nmodes)
    'tip_displacement': array,        # Tip displacement time history
    'EI_eff': float,                 # Effective flexural rigidity
    'rhoA_eff': float                # Effective mass per unit length
}
```

## Example Applications

### 1. Composite Wind Turbine Blade Analysis
```python
# Carbon fiber/foam sandwich construction
layers = [
    {'thickness': 0.001, 'E_func': lambda T: 150e9, 'rho_func': lambda T: 1600},  # Carbon fiber
    {'thickness': 0.020, 'E_func': lambda T: 0.1e9, 'rho_func': lambda T: 100},   # Foam core
    {'thickness': 0.001, 'E_func': lambda T: 150e9, 'rho_func': lambda T: 1600}   # Carbon fiber
]
```

### 2. Temperature-Dependent Analysis
```python
# Steel with temperature effects
layers = [
    {
        'thickness': 0.010,
        'E_func': lambda T: 210e9 * (1 - 0.0001 * T),    # E decreases with temperature
        'rho_func': lambda T: 7800 * (1 + 0.00001 * T)    # ρ increases with temperature
    }
]
```

### 3. Multi-Material Composite
```python
# Steel-aluminum-steel sandwich
layers = [
    {'thickness': 0.002, 'E_func': lambda T: 210e9, 'rho_func': lambda T: 7800},  # Steel
    {'thickness': 0.005, 'E_func': lambda T: 70e9,  'rho_func': lambda T: 2700},  # Aluminum
    {'thickness': 0.002, 'E_func': lambda T: 210e9, 'rho_func': lambda T: 7800}   # Steel
]
```

## Running Examples

### Complete Analysis Example
```bash
python integration_example.py
```

### Layer Dialog Test
```bash
python test_layer_dialog.py
```

### Composite Properties Validation
```bash
python test_composite_properties.py
```

## Technical Details

### Finite Element Implementation
- **Element Type**: Euler-Bernoulli beam elements
- **Shape Functions**: Cubic Hermite interpolation
- **DOF per Node**: 2 (vertical displacement, rotation)
- **Mass Matrix**: Consistent formulation
- **Integration**: Gauss quadrature for distributed loads

### Composite Theory
- **Neutral Axis**: Calculated using transformed section method
- **Effective Properties**: Homogenized using classical lamination theory
- **Temperature Effects**: Integrated through material property functions

### Boundary Conditions
- **Left End**: Fixed (cantilever)
- **Right End**: Free or spring-supported
- **Distributed Loads**: Arbitrary spatial and temporal profiles

## Performance Considerations

- **Recommended Elements**: 20-100 elements for typical problems
- **Time Points**: 200-1000 points for smooth animations
- **Layer Count**: No practical limit, but 2-10 layers typical
- **Computation Time**: Linear with number of elements and time points 