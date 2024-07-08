"""Copyright (c) 2024 Alan Lujan. All rights reserved.

multinterp: Multivariate Interpolation.
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
