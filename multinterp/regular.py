import numpy as np
from numba import njit, prange, typed
from multinterp.core import _RegularGridInterp


AVAILABLE_TARGETS = ["cpu", "parallel"]

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    CUPY_AVAILABLE = True
    AVAILABLE_TARGETS.append("gpu")
except ImportError:
    CUPY_AVAILABLE = False


MC_KWARGS = {
    "order": 1,  # order of interpolation
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
    "prefilter": False,  # whether to prefilter input
}


class MultivariateInterp(_RegularGridInterp):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy or cupy.
    """

    def __init__(self, values, grids, target="cpu", **kwargs):
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

        super().__init__(values, target=target, **kwargs)

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

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        if self.target == "cpu":
            coordinates = np.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = np.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, np.arange(self.shape[dim])
                )
        elif self.target == "parallel":
            coordinates = _nb_interp(self.grids, args)
        elif self.target == "gpu":
            coordinates = cp.empty_like(args)
            for dim, grid in enumerate(self.grids):  # for each dimension
                coordinates[dim] = cp.interp(  # x, xp, fp (new x, x points, y values)
                    args[dim], grid, cp.arange(self.shape[dim])
                )

        return coordinates


@njit(parallel=True, cache=True, fastmath=True)
def _nb_interp(grids, args):
    """
    Just-in-time compiled function for interpolating on a regular grid.

    Parameters
    ----------
    grids : np.ndarray
        1D grids for each dimension.
    args : np.ndarray
        Arguments to be interpolated.

    Returns
    -------
    np.ndarray
        Index coordinates for each argument.
    """

    coordinates = np.empty_like(args)
    for dim in prange(args.shape[0]):
        coordinates[dim] = np.interp(args[dim], grids[dim], np.arange(grids[dim].size))

    return coordinates
