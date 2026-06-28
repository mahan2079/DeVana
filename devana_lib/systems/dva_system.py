from typing import List, Dict, Optional, Tuple
from ..physics.frf import frf as raw_frf

class DVASystem:
    """
    High-level interface for a Discrete Multi-Degree-of-Freedom system with DVAs.
    
    This class handles the mapping of human-readable parameters (Main Mass, Stiffness, 
    Damping, and DVA ratios) to the raw 17-parameter main system and 48-parameter 
    DVA vectors required by the physics engine.
    """
    
    def __init__(self, omega_dc: float = 1.0, zeta_dc: float = 0.01):
        # Default Primary System Parameters (17 slots)
        # [MU, Landa1-5, Nu1-5, A_LOW, A_UPP, F1, F2, OMEGA_DC, ZETA_DC]
        self.main_params = [
            1.0,  # MU (Mass Ratio of Mass 2 to Mass 1)
            1.0, 1.0, 1.0, 0.5, 0.5, # LANDA_1 to LANDA_5 (Stiffness Ratios)
            0.05, 0.05, 0.05, 0.05, 0.05, # NU_1 to NU_5 (Damping Ratios)
            0.75, 0.75, # A_LOW, A_UPP (Base excitation amplitudes)
            100.0, 100.0, # F_1, F_2 (Force amplitudes)
            omega_dc, zeta_dc # Characteristic Frequency and Damping
        ]
        
        # Default DVA Parameters (48 values)
        # beta_1-15, lambda_1-15, mu_1-3, nu_1-15
        self.dva_params = [0.0]*15 + [1.0]*15 + [0.1]*3 + [0.01]*15

    def set_primary_mass_ratio(self, mu: float):
        """Set the ratio of Mass 2 to Mass 1."""
        self.main_params[0] = mu

    def set_stiffness_ratios(self, landas: List[float]):
        """Set LANDA_1 to LANDA_5."""
        for i, val in enumerate(landas[:5]):
            self.main_params[1 + i] = val

    def set_damping_ratios(self, nus: List[float]):
        """Set NU_1 to NU_5."""
        for i, val in enumerate(nus[:5]):
            self.main_params[6 + i] = val

    def set_excitation(self, base_amps: Tuple[float, float], force_amps: Tuple[float, float]):
        """Set excitation levels."""
        self.main_params[11] = base_amps[0]
        self.main_params[12] = base_amps[1]
        self.main_params[13] = force_amps[0]
        self.main_params[14] = force_amps[1]

    def set_dva_parameters(self, 
                           stiffness: Optional[List[float]] = None, 
                           mass_ratios: Optional[List[float]] = None,
                           damping: Optional[List[float]] = None,
                           inerter: Optional[List[float]] = None):
        """
        Configure the DVAs.
        - inerter: beta_1 to beta_15
        - stiffness: lambda_1 to lambda_15
        - mass_ratios: mu_1 to mu_3
        - damping: nu_1 to nu_15
        """
        if inerter:
            for i, v in enumerate(inerter[:15]): self.dva_params[i] = v
        if stiffness:
            for i, v in enumerate(stiffness[:15]): self.dva_params[15 + i] = v
        if mass_ratios:
            for i, v in enumerate(mass_ratios[:3]): self.dva_params[30 + i] = v
        if damping:
            for i, v in enumerate(damping[:15]): self.dva_params[33 + i] = v

    def calculate_response(self, 
                           omega_start: float = 0.1, 
                           omega_end: float = 10.0, 
                           points: int = 1000,
                           target_masses: Optional[Dict[int, Dict[str, float]]] = None):
        """
        Calculate the Frequency Response Function.
        
        target_masses: Dict mapping mass index (1-5) to weights/targets, e.g.:
                       {1: {"peak_value": 1.0, "area_under_curve": 10.0}}
        """
        
        # Default targets if none provided
        def_targets = [{} for _ in range(5)]
        def_weights = [{} for _ in range(5)]
        
        if target_masses:
            for m_idx, values in target_masses.items():
                if 1 <= m_idx <= 5:
                    def_targets[m_idx-1] = values
                    # Default weight of 1.0 for each specified target
                    def_weights[m_idx-1] = {k: 1.0 for k in values.keys()}

        return raw_frf(
            self.main_params,
            self.dva_params,
            omega_start,
            omega_end,
            points,
            def_targets[0], def_weights[0],
            def_targets[1], def_weights[1],
            def_targets[2], def_weights[2],
            def_targets[3], def_weights[3],
            def_targets[4], def_weights[4]
        )
