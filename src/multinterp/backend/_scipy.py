from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from multinterp.core import update_mc_kwargs


def scipy_multinterp(grids, values, args, options=None):
    mc_kwargs = update_mc_kwargs(options)

    args = np.asarray(args)
    values = np.asarray(values)
    grids = [np.asarray(grid) for grid in grids]

    coords = scipy_get_coordinates(grids, args)
    return scipy_map_coordinates(values, coords, **mc_kwargs)


def scipy_gradinterp(grids, values, args, axis=None, options=None):
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
        [scipy_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient]
    )


def scipy_get_coordinates(grids, args):
    coords = np.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = np.arange(grid.size)
        coords[dim] = np.interp(args[dim], grid, grid_size)

    return coords


def scipy_map_coordinates(values, coords, **kwargs):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
