import numpy as np
from numba import typed


AVAILABLE_TARGETS = ["cpu", "parallel"]

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    AVAILABLE_TARGETS.append("gpu")
except ImportError:
    CUPY_AVAILABLE = False


class _AbstractMultInterp:
    def __init__(self, values, target="cpu"):
        """
        Initialize a regular grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        target : str, optional
            Determines which target to use for interpolation.
            Options are "cpu", "parallel", and "gpu".
            If "cpu", uses numpy and scipy.
            If "parallel", uses numba and scipy.
            If "gpu", uses cupy.

        Raises
        ------
        ValueError
            Target is invalid.
        """
        if target not in AVAILABLE_TARGETS:
            raise ValueError("Invalid target.")
        self.target = target

        if target in ["cpu", "parallel"]:
            self.values = np.asarray(values)
        elif target == "gpu":
            self.values = cp.asarray(values)

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
        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

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
        `map_coordinates` from scipy or cupy, depending on target.

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
    structure for using different targets (cpu, parallel, gpu).
    Takes in arguments to be used by `map_coordinates`.
    """

    def __init__(self, values, grids, target="cpu"):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        target : str, optional
            One of "cpu", "parallel", or "gpu". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, target=target)

        if target == "cpu":
            self.grids = [np.asarray(grid) for grid in grids]
        elif target == "parallel":
            self.grids = typed.List(grids)
        elif target == "gpu":
            self.grids = [cp.asarray(grid) for grid in grids]

        if not (self.ndim == len(self.grids)):
            raise ValueError("Number of grids must match number of dimensions.")

        if not all(self.shape[i] == grid.size for i, grid in enumerate(self.grids)):
            raise ValueError("Values shape must match points in each grid.")


class _CurvilinearGridInterp(_AbstractMultInterp):
    """
    Abstract class for interpolating on a curvilinear grid.
    """

    def __init__(self, values, grids, target="cpu"):
        """
        Initialize a curvilinear grid interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            ND curvilinear grids for each dimension
        target : str, optional
            One of "cpu", "parallel", or "gpu".
        """

        super().__init__(values, target=target)

        if target in ["cpu", "parallel"]:
            self.grids = np.asarray(grids)
        elif target == "gpu":
            self.grids = cp.asarray(grids)

        if not self.ndim == self.grids[0].ndim:
            raise ValueError("Number of grids must match number of dimensions.")
        if not self.shape == self.grids[0].shape:
            raise ValueError("Values shape must match points in each grid.")


class _UnstructuredGridInterp(_CurvilinearGridInterp):
    """
    Abstract class for interpolation on unstructured grids.
    """

    def __init__(self, values, grids, target="cpu"):
        """
        Initialize interpolation on unstructured grids.

        Parameters
        ----------
        values : np.ndarray
            Functional values on an unstructured grid.
        grids : np.ndarray
            ND unstructured grids for each dimension.
        target : str, optional
            One of "cpu", "parallel", or "gpu".
        """

        super().__init__(values, grids, target=target)
        # remove non finite values that might result from
        # sequential endogenous grid method
        condition = np.logical_and.reduce([np.isfinite(grid) for grid in self.grids])
        condition = np.logical_and(condition, np.isfinite(self.values))
        self.values = self.values[condition]
        self.grids = self.grids[:, condition]
        self.ndim = self.grids.shape[0]
