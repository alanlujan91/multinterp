"""SciPy backend for multinterp."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs

__all__ = [
    "scipy_get_coordinates",
    "scipy_gradinterp",
    "scipy_map_coordinates",
    "scipy_multinterp",
]


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

    """
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

    """
    mc_kwargs = update_mc_kwargs(options)
    eo = options.get("edge_order", 1) if options else 1

    args = np.asarray(args)
    values = np.asarray(values)
    grids = [np.asarray(grid) for grid in grids]

    coords = scipy_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            msg = "Axis should be an integer."
            raise ValueError(msg)
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

    """
    coords = np.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = np.arange(grid.size)
        coords[dim] = np.interp(args[dim], grid, grid_size)

    return coords


def scipy_map_coordinates(values, coords, **kwargs):
    """Run the map_coordinates function from scipy.ndimage.

    Parameters
    ----------
    values : np.ndarray
        Functional values from which to interpolate.
    coords : np.ndarray
        Coordinates at which to interpolate values.
    **kwargs : dict
        Additional keyword arguments passed to scipy.ndimage.map_coordinates:
        - order : int (0-5, default 1)
        - mode : str ('constant', 'nearest', 'wrap', 'mirror', 'reflect')
        - cval : float (default 0.0)
        - output : array or dtype (optional)
        - prefilter : bool (default True for order > 1)

    Returns
    -------
    np.ndarray
        Interpolated values of the function.

    """
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
