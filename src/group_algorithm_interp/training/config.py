"""Compatibility imports for scientific analysis modules.

The authoritative schema lives in :mod:`group_algorithm_interp.config` so
the experiment lifecycle and analysis code validate exactly the same object.
"""

from ..config import (
    DataConfig,
    ExperimentConfig,
    GeneralizationConfig,
    ModelConfig,
    OptimConfig,
    ProjectConfig,
    SnapshotConfig,
)

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "GeneralizationConfig",
    "ModelConfig",
    "OptimConfig",
    "ProjectConfig",
    "SnapshotConfig",
]
