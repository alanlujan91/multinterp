import numpy as np
from numba import njit, prange
from multinterp.core import _CurvilinearGridInterp


AVAILABLE_BACKENDS = ["cpu", "parallel"]

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates

    CUPY_AVAILABLE = True
    AVAILABLE_BACKENDS.append("gpu")
except ImportError:
    CUPY_AVAILABLE = False


MC_KWARGS = {
    "order": 1,  # order of interpolation
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
    "prefilter": False,  # whether to prefilter input
}


class WarpedInterpOnInterp2D(_CurvilinearGridInterp):
    """
    Warped Grid Interpolation on a 2D grid.
    """

    def __call__(self, *args, axis=1):
        """
        Interpolate on a warped grid using the Warped Grid Interpolation
        method described in `EGM$^n$`.

        Parameters
        ----------
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.
            Setting to 0 may fix some issues where interpolation fails.

        Returns
        -------
        np.ndarray
            Interpolated values on a warped grid.

        Raises
        ------
        ValueError
            Number of arguments doesn't match number of dimensions.
        """

        if self.target in ["cpu", "parallel"]:
            args = np.asarray(args)
        elif self.target == "gpu":
            args = cp.asarray(args)

        if args.shape[0] != self.ndim:
            raise ValueError("Number of arguments must match number of dimensions.")

        if self.target == "cpu":
            output = self._target_cpu(args, axis)
        elif self.target == "parallel":
            output = self._target_parallel(args, axis)
        elif self.target == "gpu":
            output = self._target_gpu(args, axis)

        return output

    def _target_cpu(self, args, axis):
        """
        Uses numpy to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        shape = args[0].shape  # original shape of arguments
        size = args[0].size  # number of points in arguments
        shape_axis = self.shape[axis]  # number of points in axis

        # flatten arguments by dimension
        args = args.reshape((self.ndim, -1))

        y_intermed = np.empty((shape_axis, size))
        z_intermed = np.empty((shape_axis, size))

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = np.take(self.grids[0], i, axis=axis)
            grids1 = np.take(self.grids[1], i, axis=axis)
            values = np.take(self.values, i, axis=axis)
            y_intermed[i] = np.interp(args[0], grids0, grids1)
            z_intermed[i] = np.interp(args[0], grids0, values)

        output = np.empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = np.interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    def _target_parallel(self, args, axis):
        """
        Uses numba to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        return nb_interp_piecewise(args, self.grids, self.values, axis)

    def _target_gpu(self, args, axis):
        """
        Uses cupy to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.
        """

        shape = args[0].shape  # original shape of arguments
        size = args[0].size  # number of points in arguments
        shape_axis = self.shape[axis]  # number of points in axis

        args = args.reshape((self.ndim, -1))

        y_intermed = cp.empty((shape_axis, size))
        z_intermed = cp.empty((shape_axis, size))

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = cp.take(self.grids[0], i, axis=axis)
            grids1 = cp.take(self.grids[1], i, axis=axis)
            values = cp.take(self.values, i, axis=axis)
            y_intermed[i] = cp.interp(args[0], grids0, grids1)
            z_intermed[i] = cp.interp(args[0], grids0, values)

        output = cp.empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = cp.interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    def warmup(self):
        """
        Warms up the JIT compiler.
        """
        self(*self.grids)

        return None


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp_piecewise(args, grids, values, axis):
    """
    Just-in-time compiled function to interpolate on a warped grid.

    Parameters
    ----------
    args : np.ndarray
        Arguments to be interpolated.
    grids : np.ndarray
        Curvilinear grids for each dimension.
    values : np.ndarray
        Functional values on a curvilinear grid.
    axis : int, 0 or 1
        See `WarpedInterpOnInterp2D.__call__`.


    Returns
    -------
    np.ndarray
        Interpolated values on arguments.
    """

    shape = args[0].shape  # original shape of arguments
    size = args[0].size  # number of points in arguments
    shape_axis = values.shape[axis]  # number of points in axis

    # flatten arguments by dimension
    args = args.reshape((values.ndim, -1))

    y_intermed = np.empty((shape_axis, size))
    z_intermed = np.empty((shape_axis, size))

    for i in prange(shape_axis):
        # for each dimension, interpolate the first argument
        grids0 = grids[0][i] if axis == 0 else grids[0][:, i]
        grids1 = grids[1][i] if axis == 0 else grids[1][:, i]
        vals = values[i] if axis == 0 else values[:, i]
        y_intermed[i] = np.interp(args[0], grids0, grids1)
        z_intermed[i] = np.interp(args[0], grids0, vals)

    output = np.empty_like(args[0])

    for j in prange(size):
        y_temp = y_intermed[:, j]
        z_temp = z_intermed[:, j]

        if y_temp[0] > y_temp[-1]:
            # reverse
            y_temp = y_temp[::-1]
            z_temp = z_temp[::-1]

        output[j] = np.interp(args[1][j], y_temp, z_temp)

    return output.reshape(shape)
