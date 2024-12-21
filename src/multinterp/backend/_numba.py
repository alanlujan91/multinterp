from __future__ import annotations

import numpy as np
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs


def numba_multinterp(grids, values, args, options=None):
    """Perform multivariate interpolation using JIT-compiled functions with Numba.

    Parameters
    ----------
    grids : array-like
        Grid points in the domain.
    values: array-like
        Functional values at the grid points.
    args: array-like
        Points at which to interpolate data.
    options: dict, optional
        Additional options for interpolation.

    Returns
    -------
    array-like
        Interpolated values of the function.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(grids, (list, typed.List)):
        raise ValueError("grids should be a list or typed.List of arrays.")
    if not isinstance(values, np.ndarray):
        raise ValueError("values should be a numpy array.")
    if not isinstance(args, np.ndarray):
        raise ValueError("args should be a numpy array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")

    mc_kwargs = update_mc_kwargs(options)

    args = np.asarray(args)
    values = np.asarray(values)
    grids = typed.List([np.asarray(grid) for grid in grids])

    coords = numba_get_coordinates(grids, args)
    return numba_map_coordinates(values, coords, **mc_kwargs)


@njit(parallel=True, cache=True, fastmath=True)
def numba_get_coordinates(grids, args):
    """Converts input arguments to coordinates with respect to the specified grid. JIT-compiled using Numba.

    Parameters
    ----------
    grids : typed.List
        Curvilinear grids for each dimension.
    args : np.ndarray
        Values in the domain at which the function is to be interpolated.

    Returns
    -------
    np.ndarray
        Coordinates of the input arguments.

    """
    coords = np.empty_like(args)
    for dim in prange(len(grids)):
        grid_size = np.arange(grids[dim].size)
        coords[dim] = np.interp(args[dim], grids[dim], grid_size)

    return coords


# same as scipy map_coordinates until replacement is found
def numba_map_coordinates(values, coords, **kwargs):
    """Identical to scipy_map_coordinates until a replacement is found. See documentation for scipy_map_coordinates."""
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp_piecewise(args, grids, values, axis):
    """Just-in-time compiled function to interpolate on a warped grid.

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

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(args, np.ndarray):
        raise ValueError("args should be a numpy array.")
    if not isinstance(grids, np.ndarray):
        raise ValueError("grids should be a numpy array.")
    if not isinstance(values, np.ndarray):
        raise ValueError("values should be a numpy array.")
    if not isinstance(axis, int):
        raise ValueError("axis should be an integer.")

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
