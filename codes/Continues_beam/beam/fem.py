"""
Finite element analysis module for beam vibration.
"""

import numpy as np


class BeamAssembler:
    """
    Class to assemble global stiffness and mass matrices for beam elements.
    
    Implements a simple Euler-Bernoulli beam model with cubic shape functions.
    The beam is fixed at one end and may have a spring at the other end.
    """
    
    def __init__(self, EI, rhoA, L, k_spring=0, num_elems=10):
        """
        Initialize the beam assembler.
        
        Parameters:
        -----------
        EI : float
            Flexural rigidity (Young's modulus * moment of inertia)
        rhoA : float
            Mass per unit length (density * cross-sectional area)
        L : float
            Beam length
        k_spring : float
            Spring stiffness at the right end
        num_elems : int
            Number of finite elements
        """
        self.EI = EI
        self.rhoA = rhoA
        self.L = L
        self.k_spring = k_spring
        self.nelem = num_elems
        self.nnode = num_elems + 1
        self.ndof = 2 * self.nnode  # 2 DOFs per node (displacement and rotation)
        
        # Element length
        self.Le = L / num_elems
        
        # Node coordinates
        self.coords = np.linspace(0, L, self.nnode)
        
        # Define active DOFs (all except first node)
        self.active_dof = np.arange(2, self.ndof)
        
        # Default force profile
        self.f_profile = lambda x, t: 0.0
    
    def element_matrices(self):
        """
        Compute element stiffness and mass matrices.
        
        Returns:
        --------
        Ke : ndarray
            Element stiffness matrix (4x4)
        Me : ndarray
            Element mass matrix (4x4)
        """
        # Element stiffness matrix for Euler-Bernoulli beam
        Ke = np.array([
            [12, 6*self.Le, -12, 6*self.Le],
            [6*self.Le, 4*self.Le**2, -6*self.Le, 2*self.Le**2],
            [-12, -6*self.Le, 12, -6*self.Le],
            [6*self.Le, 2*self.Le**2, -6*self.Le, 4*self.Le**2]
        ]) * (self.EI / self.Le**3)
        
        # Element mass matrix for Euler-Bernoulli beam (consistent formulation)
        Me = np.array([
            [156, 22*self.Le, 54, -13*self.Le],
            [22*self.Le, 4*self.Le**2, 13*self.Le, -3*self.Le**2],
            [54, 13*self.Le, 156, -22*self.Le],
            [-13*self.Le, -3*self.Le**2, -22*self.Le, 4*self.Le**2]
        ]) * (self.rhoA * self.Le / 420)
        
        return Ke, Me
    
    def assemble_matrices(self):
        """
        Assemble global stiffness and mass matrices.
        
        Returns:
        --------
        M_global : ndarray
            Global mass matrix
        K_global : ndarray
            Global stiffness matrix
        """
        # Initialize global matrices
        K_global = np.zeros((self.ndof, self.ndof))
        M_global = np.zeros((self.ndof, self.ndof))
        
        # Get element matrices
        Ke, Me = self.element_matrices()
        
        # Assemble global matrices
        for i in range(self.nelem):
            # Global indices for this element
            idx1 = 2*i      # left node, displacement
            idx2 = 2*i + 1  # left node, rotation
            idx3 = 2*i + 2  # right node, displacement
            idx4 = 2*i + 3  # right node, rotation
            
            # Global indices vector
            indices = np.array([idx1, idx2, idx3, idx4])
            
            # Add element contribution to global matrices
            for i_loc, i_glob in enumerate(indices):
                for j_loc, j_glob in enumerate(indices):
                    K_global[i_glob, j_glob] += Ke[i_loc, j_loc]
                    M_global[i_glob, j_glob] += Me[i_loc, j_loc]
        
        # Add spring at the tip if needed
        if self.k_spring > 0:
            # Spring acts on the vertical displacement of the last node
            last_disp_idx = 2 * (self.nnode - 1)
            K_global[last_disp_idx, last_disp_idx] += self.k_spring
            
        return M_global, K_global
    
    def element_force_vector(self, elem_idx, t):
        """
        Compute element force vector for distributed load.
        
        Parameters:
        -----------
        elem_idx : int
            Element index
        t : float
            Time
            
        Returns:
        --------
        Fe : ndarray
            Element force vector (4,)
        """
        # Node coordinates for this element
        x1 = elem_idx * self.Le
        x2 = (elem_idx + 1) * self.Le
        
        # Gauss quadrature points and weights (2-point rule)
        xi_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])
        
        # Initialize element force vector
        Fe = np.zeros(4)
        
        # Numerical integration using Gauss quadrature
        for xi, w in zip(xi_points, weights):
            # Map xi from [-1,1] to physical coordinate x in [x1,x2]
            x = 0.5 * (x1 + x2 + xi * (x2 - x1))
            
            # Shape functions for displacement (cubic)
            N1 = 0.25 * (1 - xi)**2 * (2 + xi)
            N2 = 0.25 * (1 - xi)**2 * (1 + xi) * self.Le
            N3 = 0.25 * (1 + xi)**2 * (2 - xi)
            N4 = -0.25 * (1 + xi)**2 * (1 - xi) * self.Le
            
            # Shape functions vector
            N = np.array([N1, N2, N3, N4])
            
            # Force intensity at this point and time
            f = self.f_profile(x, t)
            
            # Contribute to element force vector (Jacobian = Le/2)
            Fe += N * f * (self.Le / 2) * w
            
        return Fe
    
    def force_function(self):
        """
        Create a function that returns the global force vector at any time.
        
        Returns:
        --------
        callable : Function that takes time as input and returns global force vector
        """
        def F_global(t):
            """Global force vector at time t"""
            # Initialize global force vector
            F = np.zeros(self.ndof)
            
            # Add contributions from all elements
            for i in range(self.nelem):
                # Get element force vector
                Fe = self.element_force_vector(i, t)
                
                # Global indices for this element
                idx1 = 2*i      # left node, displacement
                idx2 = 2*i + 1  # left node, rotation
                idx3 = 2*i + 2  # right node, displacement
                idx4 = 2*i + 3  # right node, rotation
                
                # Global indices vector
                indices = np.array([idx1, idx2, idx3, idx4])
                
                # Assemble to global vector
                F[indices] += Fe
                
            return F
            
        return F_global 