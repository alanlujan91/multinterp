import numpy as np
from numba import njit, prange, typed
from scipy.ndimage import map_coordinates
from multinterp.core import MC_KWARGS


def numba_multinterp(grids, values, args, options=None):
    mc_kwargs = MC_KWARGS
    if options:
        mc_kwargs = MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})

    args = np.asarray(args)
    values = np.asarray(values)
    grids = typed.List([np.asarray(grid) for grid in grids])

    coords = numba_get_coordinates(grids, args)
    output = numba_map_coordinates(values, coords, **mc_kwargs)
    return output


@njit(parallel=True, cache=True, fastmath=True)
def numba_get_coordinates(grids, args):
    coords = np.empty_like(args)
    for dim in prange(len(grids)):
        grid_size = np.arange(grids[dim].size)
        coords[dim] = np.interp(args[dim], grids[dim], grid_size)

    return coords


def numba_map_coordinates(values, coords, **kwargs):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)
