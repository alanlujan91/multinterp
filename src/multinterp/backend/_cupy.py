from __future__ import annotations

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

from multinterp.core import update_mc_kwargs


def cupy_multinterp(grids, values, args, options=None):
    mc_kwargs = update_mc_kwargs(options)

    args = cp.asarray(args)
    values = cp.asarray(values)
    grids = [cp.asarray(grid) for grid in grids]

    coords = cupy_get_coordinates(grids, args)
    return cupy_map_coordinates(values, coords, **mc_kwargs)


def cupy_gradinterp(grids, values, args, axis=None, options=None):
    mc_kwargs = update_mc_kwargs(options)
    eo = options.get("edge_order", 1) if options else 1

    args = cp.asarray(args)
    values = cp.asarray(values)
    grids = [cp.asarray(grid) for grid in grids]

    coords = cupy_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            raise ValueError("Axis should be an integer.")
        gradient = cp.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return cupy_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = cp.gradient(values, *grids, edge_order=eo)
    return cp.asarray(
        [cupy_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient]
    )


def cupy_get_coordinates(grids, args):
    coords = cp.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = cp.arange(grid.size)
        coords[dim] = cp.interp(args[dim], grid, grid_size)

    return coords


def cupy_map_coordinates(values, coords, **kwargs):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
