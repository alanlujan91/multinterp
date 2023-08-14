from __future__ import annotations

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

from multinterp.core import MC_KWARGS


def cupy_multinterp(grids, values, args, options=None):
    mc_kwargs = MC_KWARGS
    if options:
        mc_kwargs = MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})

    args = cp.asarray(args)
    values = cp.asarray(values)
    grids = [cp.asarray(grid) for grid in grids]

    coords = cupy_get_coordinates(grids, args)
    return cupy_map_coordinates(values, coords, **mc_kwargs)


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
