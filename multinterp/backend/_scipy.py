from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from multinterp.core import MC_KWARGS


def scipy_multinterp(grids, values, args, options=None):
    mc_kwargs = MC_KWARGS
    if options:
        mc_kwargs = MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})

    args = np.asarray(args)
    values = np.asarray(values)
    grids = [np.asarray(grid) for grid in grids]

    coords = scipy_get_coordinates(grids, args)
    return scipy_map_coordinates(values, coords, **mc_kwargs)


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
