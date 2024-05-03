from __future__ import annotations

from multinterp.utilities import asarray, aslist


class _AbstractGrid:
    def __init__(self, values, backend="scipy"):
        """Initialize a regular grid interpolator.

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
        self.values = asarray(values, backend=backend)
        self.backend = backend

        self.ndim = self.values.ndim  # should match number of grids
        self.shape = self.values.shape  # should match points in each grid


class _StructuredGrid(_AbstractGrid):
    def __init__(self, values, grids, backend="scipy"):
        super().__init__(values, backend=backend)

        self.grids = aslist(grids, backend=backend)


class _RegularGrid(_StructuredGrid):
    """Abstract class for interpolating on a regular grid. Sets up
    structure for using different backends (scipy, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    def __init__(self, values, grids, backend="scipy"):
        """Initialize a multivariate interpolator.

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
        super().__init__(values, grids, backend=backend)

        if self.ndim != len(self.grids):
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)

        if any(self.shape[i] != grid.size for i, grid in enumerate(self.grids)):
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)


class _CurvilinearGrid(_AbstractGrid):
    """Abstract class for interpolating on a curvilinear grid."""

    def __init__(self, values, grids, backend="scipy"):
        """Initialize a curvilinear grid interpolator.

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
        self.grids = asarray(grids, backend=self.backend)

        if self.ndim != self.grids[0].ndim:
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)
        if self.shape != self.grids[0].shape:
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)


class _UnstructuredGrid(_StructuredGrid):
    """Abstract class for interpolation on unstructured grids."""

    def __init__(self, values, grids, backend="scipy"):
        """Initialize interpolation on unstructured grids.

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

        self.ndim = len(grids)
        self.shape = values.shape


class _MultivaluedGrid(_AbstractGrid):
    def __init__(self, values, backend="scipy"):
        """Initialize a regular grid interpolator.

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
        super().__init__(values, backend=backend)

        self.ndim = self.values.ndim - 1
        self.nval = self.values.shape[0]
        self.shape = self.values.shape[1:]


class _MultivaluedRegularGrid(_MultivaluedGrid):
    """Abstract class for interpolating on a regular grid. Sets up
    structure for using different backends (scipy, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    def __init__(self, values, grids, backend="scipy"):
        """Initialize a multivariate interpolator.

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

        self.grids = aslist(grids, backend=backend)

        if self.ndim != len(self.grids):
            msg = "Number of grids must match number of dimensions."
            raise ValueError(msg)

        if any(self.shape[i] != grid.size for i, grid in enumerate(self.grids)):
            msg = "Values shape must match points in each grid."
            raise ValueError(msg)
