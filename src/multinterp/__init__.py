"""
Copyright (c) 2025 Alan Lujan. All rights reserved.

multinterp: Multivariate Interpolation.
"""

from __future__ import annotations

from ._version import version as __version__
from .curvilinear import (
    PiecewiseAffineInterp,
    PipelineCurvilinearInterp,
    RegressionCurvilinearInterp,
    Warped2DInterp,
)
from .rectilinear._multi import MultivariateInterp
from .unstructured import (
    PipelineUnstructuredInterp,
    RegressionUnstructuredInterp,
    UnstructuredInterp,
)

__all__ = [
    "__version__",
    "MultivariateInterp",
    "Warped2DInterp",
    "PiecewiseAffineInterp",
    "UnstructuredInterp",
    "PipelineCurvilinearInterp",
    "PipelineUnstructuredInterp",
    "RegressionCurvilinearInterp",
    "RegressionUnstructuredInterp",
]
