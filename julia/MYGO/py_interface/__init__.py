"""
MYGO Julia Python Interface Package

This package  provides Python wrappers for the MYGO.jl Julia package.
"""

from .mygo_julia import (
    MolecularDescriptorJulia,
    ADMETPredictorJulia,
    JULIA_AVAILABLE
)

__all__ = [
    "MolecularDescriptorJulia",
    "ADMETPredictorJulia",
    "JULIA_AVAILABLE"
]

__version__ = "0.1.0"
