"""
coping-dynamics-sequencing source modules.

Modules:
    config: Centralized configuration and path management
    io_utils: Data loading and saving utilities
    metrics: Behavioral metrics computation
    plotting: Visualization utilities
    stats_utils: Statistical testing utilities
    transition_utils: Transition matrix analysis (BFL scores, bootstrap tests)
"""

from . import config
from . import io_utils
from . import metrics
from . import plotting
from . import transition_utils

__all__ = [
    "config",
    "io_utils", 
    "metrics",
    "plotting",
    "transition_utils",
]
