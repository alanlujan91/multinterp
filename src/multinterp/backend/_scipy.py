from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs


def scipy_multinterp(grids, values, args, options=None):
    """Perform multivariate interpolation using SciPy.

    Parameters
    ----------
    grids : list of array-like
        Grid points in the domain.
    values : array-like
        Functional values at the grid points.
    args : array-like
        Points at which to interpolate data.
    options : dict, optional
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
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(values, np.ndarray):
        raise ValueError("values should be a numpy array.")
    if not isinstance(args, np.ndarray):
        raise ValueError("args should be a numpy array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")

    mc_kwargs = update_mc_kwargs(options)

    args = np.asarray(args)
    values = np.asarray(values)
    grids = [np.asarray(grid) for grid in grids]

    coords = scipy_get_coordinates(grids, args)
    return scipy_map_coordinates(values, coords, **mc_kwargs)


def scipy_gradinterp(grids, values, args, axis=None, options=None):
    """Computes the interpolated value of the gradient evaluated at specified points using SciPy.

    Parameters
    ----------
    grids : list of array-like
        Grid points in the domain.
    values : array-like
        Functional values at the grid points.
    args : array-like
        Points at which to interpolate data.
    axis : int, optional
        Axis along which to compute the gradient.
    options : dict, optional
        Additional options for interpolation.

    Returns
    -------
    array-like
        Interpolated values of the gradient.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types or if the axis parameter is not an integer.

    """
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(values, np.ndarray):
        raise ValueError("values should be a numpy array.")
    if not isinstance(args, np.ndarray):
        raise ValueError("args should be a numpy array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")
    if axis is not None and not isinstance(axis, int):
        raise ValueError("Axis should be an integer.")

    mc_kwargs = update_mc_kwargs(options)
    eo = options.get("edge_order", 1) if options else 1

    args = np.asarray(args)
    values = np.asarray(values)
    grids = [np.asarray(grid) for grid in grids]

    coords = scipy_get_coordinates(grids, args)

    if axis is not None:
        gradient = np.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return scipy_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = np.gradient(values, *grids, edge_order=eo)
    return np.asarray(
        [scipy_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient],
    )


def scipy_get_coordinates(grids, args):
    """Takes input values and converts them to coordinates with respect to the specified grid.

    Parameters
    ----------
    grids : np.array
        Grid points for each dimension.
    args : np.array
        Points at which to interpolate data.

    Returns
    -------
    np.array
        Coordinates with respect to the grid.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(args, np.ndarray):
        raise ValueError("args should be a numpy array.")

    coords = np.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = np.arange(grid.size)
        coords[dim] = np.interp(args[dim], grid, grid_size)

    return coords


def scipy_map_coordinates(values, coords, **kwargs):
    """Run the map_coordinates function from the scipy.ndimage module on the specified values.

    Parameters
    ----------
    values : np.array
        Functional values from which to interpolate.
    coords : np.array
        Coordinates at which to interpolate values.

    Returns
    -------
    np.array
        Interpolated values of the function.

    """
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
