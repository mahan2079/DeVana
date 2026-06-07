# FRFWorker

## Purpose
Core implementation of FRFWorker logic.

## Internal Logic Flow: `run`
```mermaid
graph TD
    Start["Start: run"] --> Step0["Error Handling Block"]
    Step0["Error Handling Block"] --> End["End: run"]
```

### Flowchart Pseudo-code
```python
FUNCTION run(self):
    DO "Error Handling Block"
END FUNCTION
```

## Methods & Functions

### `__init__`
- **Arguments**: `self, main_params, dva_params, omega_start, omega_end, omega_points, target_values_dict, weights_dict, plot_figure, show_peaks, show_slopes, interpolation_method, interpolation_points`
- **Returns**: `None`
- **Logic**: Assigns self.main_params; Assigns self.dva_params; Assigns self.omega_start; Assigns self.omega_end; Assigns self.omega_points...

### `run`
- **Arguments**: `self`
- **Returns**: `None`
- **Logic**: Simple function logic.

