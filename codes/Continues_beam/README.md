# Beam Animation for DeVana

This module provides beam vibration animation capabilities for the DeVana application.

## Features

### Time-Domain Beam Animation
- Real-time visualization of beam vibration over time
- Interactive controls (play, pause, reset)
- Adjustable animation speed
- Time display during animation

### Mode Shape Animation
- Visualization of each natural vibration mode
- Mode selection dropdown to switch between modes
- Adjustable amplitude control
- Frequency display for each mode

## Integration into DeVana

### Option 1: Using the Adapters

1. Import the adapters:
```python
from beam_animation_adapter import BeamAnimationAdapter
from mode_shape_adapter import ModeShapeAdapter
```

2. Create instances in your UI:
```python
# For time-domain animation
self.animation_adapter = BeamAnimationAdapter()
layout.addWidget(self.animation_adapter)

# For mode shape animation
self.mode_shape_adapter = ModeShapeAdapter()
layout.addWidget(self.mode_shape_adapter)
```

3. Update the animations with simulation results:
```python
# After running a simulation
self.animation_adapter.update_animation(results)
self.mode_shape_adapter.update_results(results)
```

4. Reset the animations if needed:
```python
self.animation_adapter.reset()
self.mode_shape_adapter.reset()
```

### Option 2: Using the Widgets directly

1. Import the widgets:
```python
from ui.beam_animation import BeamAnimationWidget
from ui.mode_shape_animation import ModeShapeAnimationWidget
```

2. Create instances in your UI:
```python
# For time-domain animation
self.animation_widget = BeamAnimationWidget()
layout.addWidget(self.animation_widget)

# For mode shape animation
self.mode_shape_widget = ModeShapeAnimationWidget()
layout.addWidget(self.mode_shape_widget)
```

3. Update the animations with simulation results:
```python
# After running a simulation
self.animation_widget.update_animation(results)
self.mode_shape_widget.update_results(results)
```

## Example Usage

See `integration_example.py` for a complete working example that demonstrates:
- Creating the animation widget
- Running a simulation
- Updating the animation with results
- Controlling the animation

To run the example:
```
python integration_example.py
```

## Required Data Format

The animation expects simulation results in the following format:
```python
{
    'coords': [x1, x2, x3, ...],  # X coordinates of nodes
    'displacement': array,  # 2D array with shape (ndof, ntimes)
    'times': [t1, t2, t3, ...]  # Time points
}
```

This is the standard format returned by the `solve_beam_vibration()` function. 