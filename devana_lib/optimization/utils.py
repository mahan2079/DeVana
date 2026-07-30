from typing import Callable, Any
from deap import creator

def safe_deap_operation(func: Callable) -> Callable:
    """Decorator to handle DEAP creator issues with retries."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt < max_retries - 1:
                    try:
                        if hasattr(creator, "FitnessMulti"):
                            delattr(creator, "FitnessMulti")
                        if hasattr(creator, "Individual"):
                            delattr(creator, "Individual")
                    except Exception:
                        pass
                else:
                    raise
    return wrapper
