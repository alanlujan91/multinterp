from __future__ import annotations

import numpy as np
from numba import typed


def import_backends():
    backends = ["scipy", "numba"]
    modules = {"scipy": np, "numba": np}

    try:
        import cupy as cp

        backends.append("cupy")
        modules["cupy"] = cp
    except ImportError:
        pass

    try:
        import jax.numpy as jnp

        backends.append("jax")
        modules["jax"] = jnp
    except ImportError:
        pass

    try:
        import torch

        backends.append("torch")
        modules["torch"] = torch
    except ImportError:
        pass

    return backends, modules


AVAILABLE_BACKENDS, BACKEND_MODULES = import_backends()


class _AbstractGrid:
    def __init__(self, values, backend="scipy"):
        """
        Initialize a regular grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        backend : str, optional
            Determines which backend to use for interpolation.
            Options are "scipy", "numba", and "cupy".
            If "scipy", uses numpy and scipy.
            If "numba", uses numba and scipy.
            If "cupy", uses cupy.
            If "jax", uses jax.

        Raises
        ------
        ValueError
            Backend is invalid.
        """
        if backend not in AVAILABLE_BACKENDS:
            msg = "Invalid backend."
            raise ValueError(msg)
        self.backend = backend

        self.values = BACKEND_MODULES[backend].asarray(values)

        self.ndim = self.values.ndim  # should match number of grids
        self.shape = self.values.shape  # should match points in each grid


class _RegularGrid(_AbstractGrid):
    """
    Abstract class for interpolating on a regular grid. Sets up
    structure for using different backends (scipy, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    def __init__(self, values, grids, backend="scipy"):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", or "cupy". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, backend=backend)

        if backend == "numba":
            self.grids = typed.List([np.asarray(grid) for grid in grids])
        else:
            self.grids = [BACKEND_MODULES[backend].asarray(grid) for grid in grids]

        if self.ndim != len(self.grids):
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)

        if any(self.shape[i] != grid.size for i, grid in enumerate(self.grids)):
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)


class _CurvilinearGrid(_AbstractGrid):
    """
    Abstract class for interpolating on a curvilinear grid.
    """

    def __init__(self, values, grids, backend="scipy"):
        """
        Initialize a curvilinear grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            ND curvilinear grids for each dimension
        backend : str, optional
            One of "scipy", "numba", or "cupy".
        """

        _AbstractGrid.__init__(self, values, backend=backend)

        self.grids = BACKEND_MODULES[backend].asarray(grids)

        if self.grids.ndim == 1:
            self.grids = self.grids.reshape((1, -1))

        if self.ndim != self.grids[0].ndim:
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)
        if self.shape != self.grids[0].shape:
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)


class _UnstructuredGrid(_CurvilinearGrid):
    """
    Abstract class for interpolation on unstructured grids.
    """

    def __init__(self, values, grids, backend="scipy"):
        """
        Initialize interpolation on unstructured grids.

        Parameters
        ----------
        values : np.ndarray
            Functional values on an unstructured grid.
        grids : np.ndarray
            ND unstructured grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", or "cupy".
        """

        values = values.flatten()
        grids = [grid.flatten() for grid in grids]

        super().__init__(values, grids, backend=backend)
        # remove non-finite values that might result from
        # sequential endogenous grid method


class _MultivaluedGrid:
    def __init__(self, values, backend="scipy"):
        """
        Initialize a regular grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        backend : str, optional
            Determines which backend to use for interpolation.
            Options are "scipy", "numba", and "cupy".
            If "scipy", uses numpy and scipy.
            If "numba", uses numba and scipy.
            If "cupy", uses cupy.
            If "jax", uses jax.

        Raises
        ------
        ValueError
            Backend is invalid.
        """

        if backend not in AVAILABLE_BACKENDS:
            msg = "Invalid backend."
            raise ValueError(msg)
        self.backend = backend

        self.values = BACKEND_MODULES[backend].asarray(values)

        self.ndim = self.values.ndim - 1
        self.nval = self.values.shape[0]
        self.shape = self.values.shape[1:]


class _MultivaluedRegularGrid(_MultivaluedGrid):
    """
    Abstract class for interpolating on a regular grid. Sets up
    structure for using different backends (scipy, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    def __init__(self, values, grids, backend="scipy"):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", or "cupy". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, backend=backend)

        if backend == "numba":
            self.grids = typed.List([np.asarray(grid) for grid in grids])
        else:
            self.grids = [BACKEND_MODULES[backend].asarray(grid) for grid in grids]

        if self.ndim != len(self.grids):
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)

        if any(self.shape[i] != grid.size for i, grid in enumerate(self.grids)):
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)
