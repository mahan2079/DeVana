from .de_mixin import DEOptimizationMixin
from .sa_mixin import SAOptimizationMixin
from .cmaes_mixin import CMAESOptimizationMixin
from .omega_sensitivity_mixin import OmegaSensitivityMixin

class ExtraOptimizationMixin(DEOptimizationMixin, SAOptimizationMixin,
                             CMAESOptimizationMixin, OmegaSensitivityMixin):
    """Aggregate mixin combining all extra optimization features."""
    pass
