"""LVAE core components: autoencoder architectures, pruning, constraints, and shared layers.

This module provides the dimension-agnostic building blocks for Least Volume Autoencoder (LVAE)
implementations. It consolidates shared functionality used across both 1D and 2D LVAE models.

Modules:
    aes: Autoencoder architectures with volume regularization and dynamic pruning
    constraint_handlers: Constrained optimization methods for multi-objective training
    components: Shared neural network components, utilities, and schedules

Example:
    >>> from engiopt.lvae_core import (
    ...     LeastVolumeAE_DynamicPruning,
    ...     polynomial_schedule,
    ...     create_constraint_handler,
    ... )
"""

from .aes import AutoEncoder
from .aes import ConstrainedDesignLeastVolumeAE_DP
from .aes import DesignLeastVolumeAE_DP
from .aes import InterpretableDesignLeastVolumeAE_DP
from .aes import LeastVolumeAE
from .aes import LeastVolumeAE_DynamicPruning
from .aes import PruningPolicy
from .aes import VAE
from .components import MLP
from .components import Normalizer
from .components import polynomial_schedule
from .components import Scale
from .components import SNLinearCombo
from .components import SNMLP
from .components import spectral_norm_conv
from .components import TrueSNDeconv2DCombo
from .constraint_handlers import ConstraintHandler
from .constraint_handlers import ConstraintLosses
from .constraint_handlers import ConstraintThresholds
from .constraint_handlers import create_constraint_handler

__all__ = [
    # Autoencoder models
    "AutoEncoder",
    "VAE",
    "LeastVolumeAE",
    "LeastVolumeAE_DynamicPruning",
    "DesignLeastVolumeAE_DP",
    "InterpretableDesignLeastVolumeAE_DP",
    "ConstrainedDesignLeastVolumeAE_DP",
    "PruningPolicy",
    # Constraint handling
    "ConstraintHandler",
    "ConstraintLosses",
    "ConstraintThresholds",
    "create_constraint_handler",
    # Utilities and components
    "polynomial_schedule",
    "spectral_norm_conv",
    "Scale",
    "Normalizer",
    "MLP",
    "SNMLP",
    "SNLinearCombo",
    "TrueSNDeconv2DCombo",
]
