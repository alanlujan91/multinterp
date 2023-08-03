import numpy as np
from numba import typed


def import_backends():
    AVAILABLE_BACKENDS = ["scipy", "numba"]
    BACKEND_MODULES = {"scipy": np, "numba": np}

    try:
        import cupy as cp

        AVAILABLE_BACKENDS.append("cupy")
        BACKEND_MODULES["cupy"] = cp
    except ImportError:
        pass

    try:
        import jax.numpy as jnp

        AVAILABLE_BACKENDS.append("jax")
        BACKEND_MODULES["jax"] = jnp
    except ImportError:
        pass

    return AVAILABLE_BACKENDS, BACKEND_MODULES


AVAILABLE_BACKENDS, BACKEND_MODULES = import_backends()

JAX_MC_KWARGS = {
    "order": 1,  # order of interpolation, default to linear
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
}

MC_KWARGS = {
    **JAX_MC_KWARGS,
    "output": None,  # output array or dtype
    "prefilter": False,  # whether to prefilter input
}


class _AbstractMultInterp:
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
            raise ValueError("Invalid backend.")
        self.backend = backend

        self.values = BACKEND_MODULES[backend].asarray(values)

        self.ndim = self.values.ndim  # should match number of grids
        self.shape = self.values.shape  # should match points in each grid

    def __call__(self, *args):
        """
        Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of argumets does not match number of dimensions.
        """

        args = BACKEND_MODULES[self.backend].asarray(args)

        if args.shape[0] != self.ndim:
            raise ValueError("Number of arguments must match number of dimensions.")

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """
        Abstract method for getting coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def _map_coordinates(self, coords):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on backend.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """
        raise NotImplementedError("Must be implemented by subclass.")


class _RegularGridInterp(_AbstractMultInterp):
    """
    Abstract class for interpolating on a regular grid. Sets up
    structure for using different backends (cpu, parallel, gpu).
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

        if not (self.ndim == len(self.grids)):
            raise ValueError("Number of grids must match number of dimensions.")

        if not all(self.shape[i] == grid.size for i, grid in enumerate(self.grids)):
            raise ValueError("Values shape must match points in each grid.")


class _CurvilinearGridInterp(_AbstractMultInterp):
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

        super().__init__(values, backend=backend)

        self.grids = BACKEND_MODULES[backend].asarray(grids)

        if not self.ndim == self.grids[0].ndim:
            raise ValueError("Number of grids must match number of dimensions.")
        if not self.shape == self.grids[0].shape:
            raise ValueError("Values shape must match points in each grid.")


class _UnstructuredGridInterp(_CurvilinearGridInterp):
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

        super().__init__(values, grids, backend=backend)
        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in self.grids])
        condition = np.logical_and(condition, np.isfinite(self.values))
        self.values = self.values[condition]
        self.grids = self.grids[:, condition]
        self.ndim = self.grids.shape[0]
