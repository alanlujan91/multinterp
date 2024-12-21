from __future__ import annotations

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs


def cupy_multinterp(grids, values, args, options=None):
    """Perform multivariate interpolation using CuPy.

    Parameters
    ----------
    grids : array-like
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
    if not isinstance(values, cp.ndarray):
        raise ValueError("values should be a cupy array.")
    if not isinstance(args, cp.ndarray):
        raise ValueError("args should be a cupy array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")

    mc_kwargs = update_mc_kwargs(options)

    args = cp.asarray(args)
    values = cp.asarray(values)
    grids = [cp.asarray(grid) for grid in grids]

    coords = cupy_get_coordinates(grids, args)
    return cupy_map_coordinates(values, coords, **mc_kwargs)


def cupy_gradinterp(grids, values, args, axis=None, options=None):
    """Computes the interpolated value of the gradient evaluated at specified points using CuPy.

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
    if not isinstance(values, cp.ndarray):
        raise ValueError("values should be a cupy array.")
    if not isinstance(args, cp.ndarray):
        raise ValueError("args should be a cupy array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")
    if axis is not None and not isinstance(axis, int):
        raise ValueError("Axis should be an integer.")

    mc_kwargs = update_mc_kwargs(options)
    eo = options.get("edge_order", 1) if options else 1

    args = cp.asarray(args)
    values = cp.asarray(values)
    grids = [cp.asarray(grid) for grid in grids]

    coords = cupy_get_coordinates(grids, args)

    if axis is not None:
        gradient = cp.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return cupy_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = cp.gradient(values, *grids, edge_order=eo)
    return cp.asarray(
        [cupy_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient],
    )


def cupy_get_coordinates(grids, args):
    """Takes input values and converts them to coordinates with respect to the specified grid.

    Parameters
    ----------
    grids : cp.array
        Grid points for each dimension.
    args : cp.array
        Points at which to interpolate data.

    Returns
    -------
    cp.array
        Coordinates with respect to the grid.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(args, cp.ndarray):
        raise ValueError("args should be a cupy array.")

    coords = cp.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = cp.arange(grid.size)
        coords[dim] = cp.interp(args[dim], grid, grid_size)

    return coords


def cupy_map_coordinates(values, coords, **kwargs):
    """Run the map_coordinates function from the cupyx.scipy.ndimage module on the specified values.

    Parameters
    ----------
    values : cp.array
        Functional values from which to interpolate.
    coords : cp.array
        Coordinates at which to interpolate values.

    Returns
    -------
    cp.array
        Interpolated values.

    """
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
