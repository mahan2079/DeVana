from pydantic import BaseModel, Field
from typing import List, Dict

class DVAConfiguration(BaseModel):
    """Configuration for Dynamic Vibration Absorbers (DVA)."""
    mu_1: float = Field(0.1, description="Mass ratio of DVA 1")
    mu_2: float = Field(0.1, description="Mass ratio of DVA 2")
    mu_3: float = Field(0.1, description="Mass ratio of DVA 3")
    lambda_1_15: List[float] = Field([1.0]*15, description="Stiffness parameters (15 values)")
    nu_1_15: List[float] = Field([0.01]*15, description="Damping parameters (15 values)")
    beta_1_15: List[float] = Field([0.0]*15, description="Inerter parameters (15 values)")

class FRFRequest(BaseModel):
    """Request parameters for Frequency Response Function calculation."""
    dva_params: DVAConfiguration
    main_system_params: List[float] = Field(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.05, 0.05, 100.0, 100.0, 5000.0, 0.01],
        description="[MU, Landa1-5, Nu1-5, A_LOW, A_UPP, F1, F2, OMEGA_DC, ZETA_DC] (17 values)"
    )
    omega_range: List[float] = Field([0.1, 10.0, 1000], description="[start, end, points]")
    target_masses: List[int] = Field([1, 2, 3, 4, 5], description="Masses to monitor")
    use_interpolation: bool = True
    interpolation_method: str = "cubic"

class OptimizationRequest(BaseModel):
    """General request for optimization algorithms."""
    algorithm: str = Field(..., description="GA, PSO, DE, SA, etc.")
    pop_size: int = 100
    generations: int = 50
    dva_bounds: Dict[str, List[float]] = Field(..., description="Bounds for optimization")
    fixed_parameters: List[int] = Field([], description="Indices of fixed parameters")
    target_masses: List[int] = Field([1, 2], description="Masses to optimize for")
    use_pinn_acceleration: bool = False
    omega_range: List[float] = Field([0.1, 10.0, 100], description="[start, end, points]")

class PINNRequest(BaseModel):
    """Request for PINN-based system identification."""
    csv_data_path: str = Field(..., description="Path to vibration CSV data")
    num_epochs: int = 5000
    learning_rate: float = 0.001
    topology: str = "dense"
