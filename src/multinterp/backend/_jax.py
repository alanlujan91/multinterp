from __future__ import annotations

import functools

import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates

from multinterp.utilities import update_mc_kwargs


@jit
def jax_multinterp(grids, values, args, options=None):
    """
    Perform multidimensional interpolation using JAX.

    Parameters:
    ----------
    grids : list of array-like
        List of grid points for each dimension.
    values : array-like
        Values at each point in the grid.
    args : array-like
        Points at which to interpolate data.
    options : dict, optional
        Additional options for interpolation.

    Returns:
    -------
    array-like
        Interpolated values.
    """

    mc_kwargs = update_mc_kwargs(options, jax=True)

    args = jnp.asarray(args)
    values = jnp.asarray(values)
    grids = [jnp.asarray(grid) for grid in grids]

    coords = jax_get_coordinates(grids, args)
    return jax_map_coordinates(values, coords, **mc_kwargs)


def jax_gradinterp(grids, values, args, axis=None, options=None):
    mc_kwargs = update_mc_kwargs(options, jax=True)
    eo = options.get("edge_order", 1) if options else 1

    args = jnp.asarray(args)
    values = jnp.asarray(values)
    grids = [jnp.asarray(grid) for grid in grids]

    coords = jax_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            msg = "Axis should be an integer."
            raise ValueError(msg)
        gradient = jnp.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return jax_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = jnp.gradient(values, *grids, edge_order=eo)
    return jnp.asarray(
        [jax_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient]
    )


@jit
def jax_get_coordinates(grids, args):
    grid_sizes = [jnp.arange(grid.size) for grid in grids]
    return jnp.array(
        [
            jnp.interp(arg, grid, grid_size)
            for arg, grid, grid_size in zip(args, grids, grid_sizes)
        ]
    )


@functools.partial(jit, static_argnums=(2, 3, 4))
def jax_map_coordinates(values, coords, order=None, mode=None, cval=None):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, order, mode, cval)
    return output.reshape(original_shape)
