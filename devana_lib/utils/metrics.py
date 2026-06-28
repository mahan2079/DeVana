"""
Headless computational metrics for DeVana.
Extracted from codes/computational_metrics_new.py.
"""

import psutil
import platform
import time
from typing import Any, Dict


def get_hardware_profile() -> Dict[str, Any]:
    """Get the hardware profile of the system."""
    try:
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "total_memory": psutil.virtual_memory().total,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    except Exception:
        return {"cpu_count": 1, "error": "Could not collect hardware profile"}


def get_resource_usage() -> Dict[str, Any]:
    """Get the current resource usage of the process."""
    try:
        process = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_rss": process.memory_info().rss,
            "memory_vms": process.memory_info().vms,
            "num_threads": process.num_threads(),
            "timestamp": time.time()
        }
    except Exception:
        return {"cpu_percent": 0.0, "memory_rss": 0, "error": "Could not collect resource usage"}
