from __future__ import annotations

import functools

import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates

from multinterp.core import JAX_MC_KWARGS


@jit
def jax_multinterp(grids, values, args, options=None):
    mc_kwargs = JAX_MC_KWARGS
    if options:
        mc_kwargs = JAX_MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})

    args = jnp.asarray(args)
    values = jnp.asarray(values)
    grids = [jnp.asarray(grid) for grid in grids]

    coords = jax_get_coordinates(grids, args)
    return jax_map_coordinates(values, coords, **mc_kwargs)


@jit
def jax_get_coordinates(grids, args):
    coords = jnp.empty_like(args)
    for dim in range(len(grids)):
        grid_size = jnp.arange(grids[dim].size)
        coords = coords.at[dim].set(jnp.interp(args[dim], grids[dim], grid_size))

    return coords


@functools.partial(jit, static_argnums=(2, 3, 4))
def jax_map_coordinates(values, coords, order=None, mode=None, cval=None):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, order, mode, cval)
    return output.reshape(original_shape)
