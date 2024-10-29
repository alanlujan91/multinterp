from __future__ import annotations

import functools

import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs


@jit
def jax_multinterp(grids, values, args, options=None):
    """Perform multidimensional interpolation using JAX.

    Parameters
    ----------
    grids : list of array-like
        List of grid points for each dimension.
    values : array-like
        Values at each point in the grid.
    args : array-like
        Points at which to interpolate data.
    options : dict, optional
        Additional options for interpolation.

    Returns
    -------
    array-like
        Interpolated values.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(values, jnp.ndarray):
        raise ValueError("values should be a jax array.")
    if not isinstance(args, jnp.ndarray):
        raise ValueError("args should be a jax array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")

    mc_kwargs = update_mc_kwargs(options, jax=True)

    args = jnp.asarray(args)
    values = jnp.asarray(values)
    grids = [jnp.asarray(grid) for grid in grids]

    coords = jax_get_coordinates(grids, args)
    return jax_map_coordinates(values, coords, **mc_kwargs)


def jax_gradinterp(grids, values, args, axis=None, options=None):
    """Computes the interpolated value of the gradient evaluated at specified points using JAX.

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
    if not isinstance(values, jnp.ndarray):
        raise ValueError("values should be a jax array.")
    if not isinstance(args, jnp.ndarray):
        raise ValueError("args should be a jax array.")
    if options is not None and not isinstance(options, dict):
        raise ValueError("options should be a dictionary.")
    if axis is not None and not isinstance(axis, int):
        raise ValueError("Axis should be an integer.")

    mc_kwargs = update_mc_kwargs(options, jax=True)
    eo = options.get("edge_order", 1) if options else 1

    args = jnp.asarray(args)
    values = jnp.asarray(values)
    grids = [jnp.asarray(grid) for grid in grids]

    coords = jax_get_coordinates(grids, args)

    if axis is not None:
        gradient = jnp.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return jax_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = jnp.gradient(values, *grids, edge_order=eo)
    return jnp.asarray(
        [jax_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient],
    )


@jit
def jax_get_coordinates(grids, args):
    """Takes input values and converts them to coordinates with respect to the specified grid.

    Parameters
    ----------
    grids : jnp.array
        Grid points for each dimension.
    args : jnp.array
        Points at which to interpolate data.

    Returns
    -------
    jnp.array
        Coordinates of the specified input points with respect to the grid.

    Raises
    ------
    ValueError
        If the input parameters are not of the expected types.

    """
    if not isinstance(grids, list):
        raise ValueError("grids should be a list of arrays.")
    if not isinstance(args, jnp.ndarray):
        raise ValueError("args should be a jax array.")

    grid_sizes = [jnp.arange(grid.size) for grid in grids]
    return jnp.array(
        [
            jnp.interp(arg, grid, grid_size)
            for arg, grid, grid_size in zip(args, grids, grid_sizes)
        ],
    )


@functools.partial(jit, static_argnums=(2, 3, 4))
def jax_map_coordinates(values, coords, order=None, mode=None, cval=None):
    """Run the map_coordinates function from the jax.scipy.ndimage module on the specified values.

    Parameters
    ----------
    values : jnp.array
        The functional values from which to interpolate.
    coords : jnp.array
        The coordinates at which to interpolate the values.
    order : int, optional
        The order of interpolation, 0 for Nearest-Neighbour, 1 for Linear.
    mode : str, optional
        Method to handle extrapolation. See JAX documentation for options.
    cval : float, optional
        Value to use for extrapolation under 'constant' method.

    Returns
    -------
    jnp.array
        Interpolated values at specified coordinates.

    """
    if not isinstance(values, jnp.ndarray):
        raise ValueError("values should be a jax array.")
    if not isinstance(coords, jnp.ndarray):
        raise ValueError("coords should be a jax array.")
    if order is not None and not isinstance(order, int):
        raise ValueError("order should be an integer.")
    if mode is not None and not isinstance(mode, str):
        raise ValueError("mode should be a string.")
    if cval is not None and not isinstance(cval, (int, float)):
        raise ValueError("cval should be a number.")

    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, order, mode, cval)
    return output.reshape(original_shape)
