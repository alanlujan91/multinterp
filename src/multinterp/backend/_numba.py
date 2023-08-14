from __future__ import annotations

import numpy as np
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates

from multinterp.core import MC_KWARGS


def numba_multinterp(grids, values, args, options=None):
    mc_kwargs = MC_KWARGS
    if options:
        mc_kwargs = MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})

    args = np.asarray(args)
    values = np.asarray(values)
    grids = typed.List([np.asarray(grid) for grid in grids])

    coords = numba_get_coordinates(grids, args)
    return numba_map_coordinates(values, coords, **mc_kwargs)


@njit(parallel=True, cache=True, fastmath=True)
def numba_get_coordinates(grids, args):
    coords = np.empty_like(args)
    for dim in prange(len(grids)):
        grid_size = np.arange(grids[dim].size)
        coords[dim] = np.interp(args[dim], grids[dim], grid_size)

    return coords


def numba_map_coordinates(values, coords, **kwargs):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)


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
            y_temp = y_temp[::-1].copy()
            z_temp = z_temp[::-1].copy()

        output[j] = np.interp(args[1][j], y_temp, z_temp)

    return output.reshape(shape)
