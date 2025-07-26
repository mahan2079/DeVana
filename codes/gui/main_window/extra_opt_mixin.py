from .de_mixin import DEOptimizationMixin
from .sa_mixin import SAOptimizationMixin
from .cmaes_mixin import CMAESOptimizationMixin
from .omega_sensitivity_mixin import OmegaSensitivityMixin
from .rl_mixin import RLOptimizationMixin

class ExtraOptimizationMixin(DEOptimizationMixin, SAOptimizationMixin,
                             CMAESOptimizationMixin, OmegaSensitivityMixin,
                             RLOptimizationMixin):
    """Aggregate mixin combining all extra optimization features."""
    pass
