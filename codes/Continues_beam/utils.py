"""
Utility functions for the VIBRAOPT application
"""

import numpy as np
import math

def parse_expression(expr_str):
    """
    Parse a string expression to a callable function.
    
    Args:
        expr_str (str): A string containing a mathematical expression or a simple number.
        
    Returns:
        callable: A function that evaluates the expression.
        
    Raises:
        ValueError: If the expression is invalid or contains unsafe operations.
    """
    # Define a safe dictionary of allowed functions
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'pi': np.pi,
        'e': np.e,
        'abs': np.abs,
        'sign': np.sign,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'arcsinh': np.arcsinh,
        'arccosh': np.arccosh,
        'arctanh': np.arctanh,
        'degrees': np.degrees,
        'radians': np.radians
    }
    
    # Check if the input string is just a number
    try:
        value = float(expr_str)
        return lambda t: value
    except ValueError:
        # Not a simple number, assume it's an expression
        pass
    
    # Check for unsafe operations
    if any(keyword in expr_str for keyword in ['__', 'import', 'eval', 'exec', 'compile', 'open']):
        raise ValueError(f"Expression contains unsafe operations: {expr_str}")
    
    try:
        # Create a lambda function for the expression
        # The function takes 't' as its parameter (time)
        code = compile(expr_str, "<string>", "eval")
        
        def func(t):
            # Copy locals to avoid modifying the original
            local_dict = safe_dict.copy()
            local_dict['t'] = t
            return eval(code, {"__builtins__": {}}, local_dict)
        
        # Test the function to verify it works
        func(0.0)  # Test with a dummy value
        return func
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr_str}. Error: {str(e)}")

class ForceRegion:
    """
    Represents a region where a specific type of force is applied.
    
    Attributes:
        name (str): Display name for the force region
        force_type (str): Type of force ('harmonic', 'step', 'impulse', 'custom')
        params (dict): Parameters specific to the force type
        spatial_type (str): How the force is applied spatially ('point' or 'distributed')
        locations (list): List of locations to apply the force
    """
    
    def __init__(self, name, force_type, params, spatial_type, locations):
        self.name = name
        self.force_type = force_type.lower()
        self.params = params
        self.spatial_type = spatial_type.lower()
        self.locations = locations
        
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'force_type': self.force_type,
            'params': self.params,
            'spatial_type': self.spatial_type,
            'locations': self.locations
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary for deserialization"""
        return cls(
            data.get('name', 'Unnamed Region'),
            data.get('force_type', 'harmonic'),
            data.get('params', {}),
            data.get('spatial_type', 'point'),
            data.get('locations', [])
        )


class ForceRegionManager:
    """
    Manages multiple force regions and creates a combined force function.
    
    Attributes:
        regions (list): List of ForceRegion objects
    """
    
    def __init__(self):
        self.regions = []
        
    def add_region(self, region):
        """Add a force region"""
        self.regions.append(region)
        
    def remove_region(self, index):
        """Remove a force region by index"""
        if 0 <= index < len(self.regions):
            del self.regions[index]
            
    def clear_regions(self):
        """Remove all force regions"""
        self.regions = []
        
    def create_force_function(self, force_generators):
        """
        Create a combined force function from all regions.
        
        Args:
            force_generators (dict): Dictionary of force generator functions
                                    keyed by force type.
                                    
        Returns:
            callable: A function f(x, t) representing the combined force 
                     at position x and time t.
        """
        if not self.regions:
            return lambda x, t: 0.0  # No forces
        
        def combined_force(x, t):
            """Combined force function at position x and time t"""
            total_force = 0.0
            
            for region in self.regions:
                # Get the force function for this type
                if region.force_type not in force_generators:
                    continue  # Skip unknown force types
                    
                force_gen = force_generators[region.force_type]
                force_func = force_gen(region.params)
                
                # Apply the force based on spatial type
                if region.spatial_type == 'point':
                    # Point forces
                    for loc in region.locations:
                        position = loc.get('position', 0.0)
                        scale = loc.get('scale', 1.0)
                        
                        # Dirac delta approximation for point force
                        # Use a narrow Gaussian centered at position
                        sigma = 0.01  # Width of the Gaussian (adjust as needed)
                        delta = np.exp(-0.5 * ((x - position) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
                        
                        total_force += scale * delta * force_func(t)
                        
                elif region.spatial_type == 'distributed':
                    # Distributed forces
                    for loc in region.locations:
                        start = loc.get('start', 0.0)
                        end = loc.get('end', 1.0)
                        scale = loc.get('scale', 1.0)
                        
                        # Check if x is in the range [start, end]
                        if start <= x <= end:
                            total_force += scale * force_func(t)
                            
            return total_force
            
        return combined_force
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'regions': [region.to_dict() for region in self.regions]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary for deserialization"""
        manager = cls()
        regions_data = data.get('regions', [])
        
        for region_data in regions_data:
            manager.add_region(ForceRegion.from_dict(region_data))
            
        return manager


# Force generator functions

def create_harmonic_force(params):
    """
    Create a harmonic force function.
    
    Args:
        params (dict): Parameters with keys 'amplitude', 'frequency', and 'phase'
        
    Returns:
        callable: Function f(t) that returns force at time t
    """
    amplitude = float(params.get('amplitude', 1.0))
    frequency = float(params.get('frequency', 1.0))
    phase_deg = float(params.get('phase', 0.0))
    phase_rad = np.radians(phase_deg)
    
    return lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)


def create_step_force(params):
    """
    Create a step force function with optional ramp time.
    
    Args:
        params (dict): Parameters with keys 'amplitude', 'start_time', and 'ramp_time'
        
    Returns:
        callable: Function f(t) that returns force at time t
    """
    amplitude = float(params.get('amplitude', 1.0))
    start_time = float(params.get('start_time', 0.0))
    ramp_time = float(params.get('ramp_time', 0.0))
    
    def force_function(t):
        if t < start_time:
            return 0.0
        elif ramp_time > 0 and t < start_time + ramp_time:
            # Linear ramp
            progress = (t - start_time) / ramp_time
            return amplitude * progress
        else:
            return amplitude
            
    return force_function


def create_impulse_force(params):
    """
    Create an impulse force function.
    
    Args:
        params (dict): Parameters with keys 'amplitude', 'start_time', and 'duration'
        
    Returns:
        callable: Function f(t) that returns force at time t
    """
    amplitude = float(params.get('amplitude', 1.0))
    start_time = float(params.get('start_time', 0.0))
    duration = float(params.get('duration', 0.01))
    
    def force_function(t):
        if start_time <= t <= start_time + duration:
            # Rectangular pulse
            return amplitude
        else:
            return 0.0
            
    return force_function


def create_custom_force(params):
    """
    Create a custom force function from a user-provided expression.
    
    Args:
        params (dict): Parameters with key 'expression' containing a math expression
                      in terms of 't'
        
    Returns:
        callable: Function f(t) that returns force at time t
    """
    expression = params.get('expression', '0')
    
    try:
        return parse_expression(expression)
    except ValueError as e:
        # If there's an error in the expression, return a zero force
        print(f"Error parsing custom force expression: {e}")
        return lambda t: 0.0


def get_force_generators():
    """
    Get a dictionary of all force generator functions.
    
    Returns:
        dict: Dictionary of force generator functions keyed by type name
    """
    return {
        'harmonic': create_harmonic_force,
        'step': create_step_force,
        'impulse': create_impulse_force,
        'custom': create_custom_force
    } 