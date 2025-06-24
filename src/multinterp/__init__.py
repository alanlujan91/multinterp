"""Copyright (c) 2024 Alan Lujan. All rights reserved.

multinterp: Multivariate Interpolation.
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = (
    "MultivariateInterp",
    "PiecewiseAffineInterp",
    "PipelineCurvilinearInterp",
    "PipelineUnstructuredInterp",
    "RegressionCurvilinearInterp",
    "RegressionUnstructuredInterp",
    "UnstructuredInterp",
    "Warped2DInterp",
    "__version__",
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
