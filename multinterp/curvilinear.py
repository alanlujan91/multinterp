from __future__ import annotations

from skimage.transform import PiecewiseAffineTransform

from multinterp.backend._numba import nb_interp_piecewise
from multinterp.core import _CurvilinearGridInterp, import_backends
from multinterp.regular import MultivariateInterp

AVAILABLE_BACKENDS, BACKEND_MODULES = import_backends()


class Warped2DInterp(_CurvilinearGridInterp):
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

        args = BACKEND_MODULES[self.backend].asarray(args)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        if self.backend in ["scipy", "cupy"]:
            output = self._interp_piecewise(args, axis)
        elif self.backend == "numba":
            output = self._backend_numba(args, axis)

        return output

    def _interp_piecewise(self, args, axis):
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

        y_intermed = BACKEND_MODULES[self.backend].empty((shape_axis, size))
        z_intermed = BACKEND_MODULES[self.backend].empty((shape_axis, size))

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = BACKEND_MODULES[self.backend].take(self.grids[0], i, axis=axis)
            grids1 = BACKEND_MODULES[self.backend].take(self.grids[1], i, axis=axis)
            values = BACKEND_MODULES[self.backend].take(self.values, i, axis=axis)
            y_intermed[i] = BACKEND_MODULES[self.backend].interp(
                args[0], grids0, grids1
            )
            z_intermed[i] = BACKEND_MODULES[self.backend].interp(
                args[0], grids0, values
            )

        output = BACKEND_MODULES[self.backend].empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = BACKEND_MODULES[self.backend].interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    def _backend_numba(self, args, axis):
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

    def warmup(self):
        """
        Warms up the JIT compiler.
        """
        self(*self.grids)

        return


class PiecewiseAffineInterp(_CurvilinearGridInterp, MultivariateInterp):
    def __init__(self, values, grids, options=None):
        super().__init__(values, grids, backend="scipy")
        self._parse_mc_options(options)

        source = self.grids.reshape((self.ndim, -1)).T
        coordinates = BACKEND_MODULES[self.backend].mgrid[
            tuple(slice(0, dim) for dim in self.shape)
        ]
        destination = coordinates.reshape((self.ndim, -1)).T

        interpolator = PiecewiseAffineTransform()
        interpolator.estimate(source, destination)

        self.interpolator = interpolator

    def _get_coordinates(self, args):
        _input = args.reshape((self.ndim, -1)).T
        output = self.interpolator(_input).T.copy()
        return output.reshape(args.shape)
