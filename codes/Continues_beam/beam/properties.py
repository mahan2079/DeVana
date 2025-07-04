"""
Module for calculating beam section properties.
"""

import numpy as np


def calc_composite_properties(width, layers):
    """
    Calculate effective properties for a composite beam.
    
    The beam is assumed to be a multi-layered composite with variable
    material properties. This function computes the effective flexural
    rigidity (EI) and mass per unit length (rhoA).
    
    Parameters:
    -----------
    width : float
        Beam width
    layers : list of dict
        List of layer definitions. Each layer is a dictionary with keys:
        - thickness: Layer thickness
        - E_func: Function that returns Young's modulus
        - rho_func: Function that returns density
        
    Returns:
    --------
    EI_eff : float
        Effective flexural rigidity
    rhoA_eff : float
        Effective mass per unit length
    """
    if not layers:
        raise ValueError("No layers provided")
    
    # Sort layers by their position from bottom to top
    # (assuming first layer is at the bottom)
    layers_sorted = sorted(layers, key=lambda x: x.get('position', 0))
    
    # Calculate total thickness and neutral axis position
    total_thickness = sum(layer['thickness'] for layer in layers_sorted)
    
    # Calculate position of bottom of each layer from the bottom of the beam
    position = 0
    for layer in layers_sorted:
        layer['bottom_pos'] = position
        position += layer['thickness']
    
    # Calculate position of neutral axis (centroid)
    # For simple homogeneous beams, this is half the height
    # For composite beams, weighted by stiffness
    sum_EAy = 0
    sum_EA = 0
    
    for layer in layers_sorted:
        # Get material properties (at reference temperature)
        if callable(layer['E_func']):
            E = layer['E_func'](0)  # Evaluate at reference temperature (T=0)
        else:
            E = layer['E_func']  # Constant value
            
        # Calculate area
        A = width * layer['thickness']
        
        # Calculate centroid of this layer
        y = layer['bottom_pos'] + layer['thickness'] / 2
        
        # Sum for neutral axis calculation
        sum_EAy += E * A * y
        sum_EA += E * A
    
    # Neutral axis position from bottom
    if sum_EA > 0:
        neutral_axis = sum_EAy / sum_EA
    else:
        neutral_axis = total_thickness / 2
    
    # Calculate effective EI and rhoA
    EI_eff = 0
    rhoA_eff = 0
    
    for layer in layers_sorted:
        # Get material properties
        if callable(layer['E_func']):
            E = layer['E_func'](0)
        else:
            E = layer['E_func']
            
        if callable(layer['rho_func']):
            rho = layer['rho_func'](0)
        else:
            rho = layer['rho_func']
        
        # Calculate geometric properties
        h = layer['thickness']  # Layer thickness
        b = width  # Layer width
        y1 = layer['bottom_pos'] - neutral_axis  # Distance from neutral axis to bottom of layer
        y2 = y1 + h  # Distance from neutral axis to top of layer
        
        # Add contribution to EI (using parallel axis theorem)
        I_layer = (b * h**3) / 12  # Second moment of area about layer centroid
        A_layer = b * h  # Layer area
        d = (y1 + y2) / 2  # Distance from neutral axis to layer centroid
        
        EI_eff += E * (I_layer + A_layer * d**2)
        
        # Add contribution to rhoA
        rhoA_eff += rho * A_layer
    
    return EI_eff, rhoA_eff 