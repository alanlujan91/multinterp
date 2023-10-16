"""
Copyright (c) 2023 Alan Lujan. All rights reserved.

multinterp: A great package.
"""


from __future__ import annotations

from ._version import version as __version__

__all__ = (
    "__version__",
    "MultivariateInterp",
    "Warped2DInterp",
    "PiecewiseAffineInterp",
    "UnstructuredInterp",
    "PipelineCurvilinearInterp",
    "PipelineUnstructuredInterp",
    "RegressionCurvilinearInterp",
    "RegressionUnstructuredInterp",
)


from .curvilinear import PiecewiseAffineInterp, Warped2DInterp
from .regression import (
    PipelineCurvilinearInterp,
    PipelineUnstructuredInterp,
    RegressionCurvilinearInterp,
    RegressionUnstructuredInterp,
)
from .regular import MultivariateInterp
from .unstructured import UnstructuredInterp
