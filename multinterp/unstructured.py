from __future__ import annotations

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from multinterp.core import _UnstructuredGridInterp

LNDI_KWARGS = {"fill_value": np.nan, "rescale": False}  # linear
NNDI_KWARGS = {"rescale": False, "tree_options": None}  # nearest
CT2DI_KWARGS = {  # cubic
    "fill_value": np.nan,
    "tol": 1e-06,
    "maxiter": 400,
    "rescale": False,
}
RBFI_KWARGS = {  # rbf (radial basis function)
    "neighbors": None,
    "smoothing": 0.0,
    "kernel": "linear",
    "epsilon": None,
    "degree": None,
}

AVAILABLE_METHODS = ["nearest", "linear", "cubic", "rbf"]


class UnstructuredInterp(_UnstructuredGridInterp):
    """
    Multivariate interpolation on an unstructured grid.
    This class wraps various scipy unstructured interpolation
    methods to provide a common interface. Additionally, it
    can be used with mesh-grids and returns mesh-grids, which
    are not supported by scipy but are the default used in
    HARK.
    """

    def __init__(
        self,
        values,
        grids,
        method="linear",
        options=None,
    ):
        """
        Initialize an Unstructured Grid Interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on an unstructured grid.
        grids : np.ndarray
            Points on an unstructured grid.
        method : str, optional
            One of "nearest", "linear", "cubic", "rbf". Determines
            which scipy interpolation method to use. The interpolators
            are "nearest" for NearestNDInterpolator, "linear" for
            LinearNDInterpolator, "cubic" for CloughTocher2DInterpolator
            and "rbf" for RBFInterpolator. The default is "linear".

        Raises
        ------
        ValueError
            The interpolation method is not valid.
        """

        super().__init__(values, grids, backend="scipy")

        # Check for valid interpolation method
        if method not in AVAILABLE_METHODS:
            msg = "Invalid interpolation method."
            raise ValueError(msg)

        self.method = method

        interpolator_mapping = {
            "nearest": (NNDI_KWARGS, NearestNDInterpolator),
            "linear": (LNDI_KWARGS, LinearNDInterpolator),
            "cubic": (CT2DI_KWARGS, CloughTocher2DInterpolator)
            if self.ndim == 2
            else (None, None),
            "rbf": (RBFI_KWARGS, RBFInterpolator),
        }

        interp_kwargs, interpolator_class = interpolator_mapping.get(
            method, (None, None)
        )

        if not interp_kwargs:
            msg = f"Unknown interpolation method {method} for {self.ndim} dimensional data."
            raise ValueError(msg)

        self.interp_kwargs = interp_kwargs
        if options:
            self.interp_kwargs.copy()
            intersection = interp_kwargs.keys() & options.keys()
            self.interp_kwargs.update({key: options[key] for key in intersection})

        self.interpolator = interpolator_class(
            np.moveaxis(self.grids, -1, 0), self.values, **self.interp_kwargs
        )

    def __call__(self, *args):
        """
        Interpolates function on arguments.

        Returns
        -------
        np.ndarray
            Interpolated values.
        """

        if self.method == "rbf":
            coords = np.asarray(args).reshape(self.ndim, -1).T
            return self.interpolator(coords).reshape(args[0].shape)

        return self.interpolator(*args)
