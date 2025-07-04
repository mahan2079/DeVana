"""
Module for solving beam vibration problems using numerical integration.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .properties import calc_composite_properties
from .fem import BeamAssembler


class BeamVibrationSolver:
    """Class for solving beam vibration problems."""
    
    def __init__(self, width, layers, L, k_spring, num_elems):
        """
        Initialize the beam vibration solver.
        
        Parameters:
        -----------
        width : float
            Beam width
        layers : list of dicts
            Layer definitions
        L : float
            Beam length
        k_spring : float
            Tip spring stiffness
        num_elems : int
            Number of finite elements
        """
        # Calculate effective properties
        self.EI_eff, self.rhoA_eff = calc_composite_properties(width, layers)
        
        # Create FEM assembler
        self.assembler = BeamAssembler(
            self.EI_eff, self.rhoA_eff, L, k_spring, num_elems
        )
        
        # Get global matrices
        self.M_global, self.K_global = self.assembler.assemble_matrices()
        
        # Extract active DOF matrices
        self.M_a = self.M_global[np.ix_(self.assembler.active_dof, self.assembler.active_dof)]
        self.K_a = self.K_global[np.ix_(self.assembler.active_dof, self.assembler.active_dof)]
        
        # Compute natural frequencies
        self._compute_natural_frequencies()
    
    def _compute_natural_frequencies(self):
        """Compute natural frequencies and mode shapes."""
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(self.M_a) @ self.K_a)
        omegas = np.sqrt(np.real(eigvals))
        idx_sort = np.argsort(omegas)
        
        self.natural_frequencies_rad_s = omegas[idx_sort]
        self.natural_frequencies_hz = self.natural_frequencies_rad_s / (2 * np.pi)
        self.mode_shapes = eigvecs[:, idx_sort]
        
        # Normalize mode shapes to unit mass
        for i in range(self.mode_shapes.shape[1]):
            modal_mass = self.mode_shapes[:, i].T @ self.M_a @ self.mode_shapes[:, i]
            self.mode_shapes[:, i] /= np.sqrt(modal_mass)
    
    def extract_vertical_mode_shapes(self):
        """
        Extract vertical components of mode shapes for visualization.
        
        Returns:
        --------
        numpy.ndarray : Mode shapes array with shape (num_nodes, num_modes)
                        containing only vertical displacements
        """
        if not hasattr(self, 'mode_shapes'):
            return None
            
        num_nodes = self.assembler.nnode
        num_modes = self.mode_shapes.shape[1]
        vertical_modes = np.zeros((num_nodes, num_modes))
        
        # For each node, extract the vertical DOF (odd indices)
        for i in range(num_nodes):
            node_dof = 2 * i + 1  # Vertical DOF index
            if node_dof in self.assembler.active_dof:
                # Find the index in the active DOF array
                active_idx = np.where(self.assembler.active_dof == node_dof)[0][0]
                # Extract mode shape values for this DOF
                vertical_modes[i, :] = self.mode_shapes[active_idx, :]
        
        return vertical_modes
    
    def solve(self, f_profile=None, t_span=(0, 3), num_time_points=300,
             initial_displacement=None, initial_velocity=None):
        """
        Solve the beam vibration problem.
        
        Parameters:
        -----------
        f_profile : callable, optional
            Distributed load function f(x, t)
        t_span : tuple, optional
            Time span (start, end)
        num_time_points : int, optional
            Number of time points
        initial_displacement : array-like, optional
            Initial displacement for each DOF
        initial_velocity : array-like, optional
            Initial velocity for each DOF
            
        Returns:
        --------
        dict : Results containing time history and modal information
        """
        # Update assembler with force profile
        self.assembler.f_profile = f_profile if f_profile is not None else lambda x, t: 0.0
        F_time_func = self.assembler.force_function()
        
        # Set initial conditions
        ndof = self.M_global.shape[0]
        if initial_displacement is None:
            initial_displacement = np.zeros(ndof)
        if initial_velocity is None:
            initial_velocity = np.zeros(ndof)
        
        # Extract active DOF initial conditions
        half_active = len(self.assembler.active_dof)
        u0_active = initial_displacement[self.assembler.active_dof]
        v0_active = initial_velocity[self.assembler.active_dof]
        y0 = np.concatenate([u0_active, v0_active])
        
        # Create ODE system
        M_a_inv = np.linalg.inv(self.M_a)
        def ode_system(t, y):
            u_a = y[:half_active]
            v_a = y[half_active:]
            F_full = F_time_func(t)
            F_a = F_full[self.assembler.active_dof]
            du_dt = v_a
            dv_dt = M_a_inv @ (F_a - self.K_a @ u_a)
            return np.concatenate([du_dt, dv_dt])
        
        # Solve ODE system
        t_eval = np.linspace(t_span[0], t_span[1], num_time_points)
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)
        
        # Extract solution components
        t_vals = sol.t
        Y = sol.y
        U_active = Y[:half_active, :]
        V_active = Y[half_active:, :]
        
        # Reconstruct full solution vectors
        U_full = np.zeros((ndof, len(t_vals)))
        V_full = np.zeros((ndof, len(t_vals)))
        U_full[self.assembler.active_dof, :] = U_active
        V_full[self.assembler.active_dof, :] = V_active
        
        # Compute accelerations
        A_full = np.zeros((ndof, len(t_vals)))
        M_inv = np.linalg.inv(self.M_global)
        for i_time in range(len(t_vals)):
            F_cur = F_time_func(t_vals[i_time])
            Uc = U_full[:, i_time]
            A_full[:, i_time] = M_inv @ (F_cur - self.K_global @ Uc)
        
        # Extract vertical mode shapes for visualization
        vertical_mode_shapes = self.extract_vertical_mode_shapes()
        
        # Return results
        return {
            'time': t_vals,
            'displacement': U_full,
            'velocity': V_full,
            'acceleration': A_full,
            'coords': self.assembler.coords,
            'natural_frequencies_rad_s': self.natural_frequencies_rad_s,
            'natural_frequencies_hz': self.natural_frequencies_hz,
            'mode_shapes_active': self.mode_shapes,
            'mode_shapes': vertical_mode_shapes,  # Add the vertical mode shapes
            'natural_frequencies': self.natural_frequencies_hz  # Alias for UI compatibility
        }


def solve_beam_vibration(width, layers, L, k_spring, num_elems, **kwargs):
    """
    Convenience function to solve beam vibration problem.
    
    This is a wrapper around BeamVibrationSolver for backward compatibility
    and ease of use.
    
    Parameters:
    -----------
    width : float
        Beam width in meters
    layers : list of dicts
        Layer definitions. Each dict should contain:
        - 'height': float, layer thickness in meters
        - 'E': callable, Young's modulus function that can be called as E(T) or E()
        - 'rho': callable, density function that can be called as rho(T) or rho()
    L : float
        Beam length in meters
    k_spring : float
        Tip spring stiffness in N/m
    num_elems : int
        Number of finite elements
    **kwargs : dict
        Additional arguments to pass to the solver:
        - f_profile: callable, force profile as function of x and t
        - t_span: tuple, time span (start, end) in seconds
        - num_time_points: int, number of time points
        
    Returns:
    --------
    dict : Results containing:
        - times: array of time points
        - displacement: array of displacements for each DOF
        - velocity: array of velocities for each DOF
        - acceleration: array of accelerations for each DOF
        - natural_frequencies_hz: array of natural frequencies in Hz
        - tip_displacement: array of tip displacements over time
    """
    # Convert layer format if needed
    prepared_layers = []
    for layer in layers:
        # Check if the layer is in the expected format
        if 'height' in layer and 'E' in layer and 'rho' in layer:
            # Format the layer correctly
            prepared_layers.append({
                'thickness': layer['height'],
                'E_func': layer['E'],
                'rho_func': layer['rho']
            })
        else:
            raise ValueError(f"Layer format invalid: {layer}")
    
    # Create solver instance
    solver = BeamVibrationSolver(width, prepared_layers, L, k_spring, num_elems)
    
    # Solve the system
    results = solver.solve(**kwargs)
    
    # Add tip displacement data
    tip_dof = 2 * (solver.assembler.nnode - 1)  # Vertical displacement DOF at tip
    results['tip_displacement'] = results['displacement'][tip_dof, :]
    results['times'] = results['time']  # Alias for consistency
    
    return results