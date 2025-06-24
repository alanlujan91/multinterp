from __future__ import annotations

import numpy as np
from numba import njit, prange, typed

from multinterp.utilities import update_mc_kwargs


def numba_multinterp(grids, values, args, options=None):
    """Perform multivariate interpolation using JIT-compiled functions with Numba.

    Parameters
    ----------
    grids : array-like
        Grid points in the domain.
    values: array-like
        Functional values at the grid points.
    args: array-like
        Points at which to interpolate data.
    options: dict, optional
        Additional options for interpolation.

    Returns
    -------
    array-like
        Interpolated values of the function.

    """
    mc_kwargs = update_mc_kwargs(options)

    args = np.asarray(args)
    values = np.asarray(values)
    grids = typed.List([np.asarray(grid) for grid in grids])

    coords = numba_get_coordinates(grids, args)
    return numba_map_coordinates_wrapper(values, coords, **mc_kwargs)


@njit(parallel=True, cache=True, fastmath=True)
def numba_get_coordinates(grids, args):
    """Converts input arguments to coordinates with respect to the specified grid. JIT-compiled using Numba.

    Parameters
    ----------
    grids : typed.List
        Curvilinear grids for each dimension.
    args : np.ndarray
        Values in the domain at which the function is to be interpolated.

    Returns
    -------
    np.ndarray
        Coordinates of the input arguments.

    """
    coords = np.empty_like(args)
    for dim in prange(len(grids)):
        # Create grid indices directly to avoid uint64 to int64 conversion issues
        grid_len = np.int64(len(grids[dim]))
        grid_size = np.empty(grid_len, dtype=np.float64)
        for i in range(grid_len):
            grid_size[i] = float(i)
        coords[dim] = np.interp(args[dim], grids[dim], grid_size)

    return coords


@njit(cache=True, fastmath=True)
def _round_half_away_from_zero(a):
    """Round half away from zero for nearest neighbor interpolation."""
    return np.round(a)


@njit(cache=True, fastmath=True)
def _mirror_index_fixer(index, size):
    """Mirror boundary condition index fixer."""
    s = size - 1  # Half-wavelength of triangular wave
    return np.abs((index + s) % (2 * s) - s)


@njit(cache=True, fastmath=True)
def _reflect_index_fixer(index, size):
    """Reflect boundary condition index fixer."""
    return np.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


@njit(cache=True, fastmath=True)
def _fix_index(index, size, mode):
    """Apply boundary condition to indices."""
    if mode == 0:  # constant
        return index
    if mode == 1:  # nearest
        return np.clip(index, 0, size - 1)
    if mode == 2:  # wrap
        return index % size
    if mode == 3:  # mirror
        return _mirror_index_fixer(index, size)
    if mode == 4:  # reflect
        return _reflect_index_fixer(index, size)
    return index


@njit(cache=True, fastmath=True)
def _is_valid_index(index, size, mode):
    """Check if index is valid for constant mode."""
    if mode == 0:  # constant mode
        return (index >= 0) & (index < size)
    return True


@njit(parallel=True, cache=True, fastmath=True)
def numba_map_coordinates(input_arr, coordinates, order=1, mode="nearest", cval=0.0):
    """Numba implementation of map_coordinates with full interpolation support.

    Parameters
    ----------
    input_arr : np.ndarray
        N-dimensional input array from which values are interpolated.
    coordinates : np.ndarray
        Array of shape (ndim, ...) specifying coordinates at which to evaluate.
    order : int
        Interpolation order. 0 for nearest, 1 for linear.
    mode : str
        Boundary mode: 'constant', 'nearest', 'wrap', 'mirror', 'reflect'.
    cval : float
        Value for constant mode outside boundaries.

    Returns
    -------
    np.ndarray
        Interpolated values at the specified coordinates.
    """
    # Convert mode string to integer for Numba compatibility
    if mode == "constant":
        mode_int = 0
    elif mode == "nearest":
        mode_int = 1
    elif mode == "wrap":
        mode_int = 2
    elif mode == "mirror":
        mode_int = 3
    elif mode == "reflect":
        mode_int = 4
    else:
        mode_int = 1  # default to nearest

    # Get coordinate shape and flatten for processing
    coord_shape = coordinates.shape[1:]
    total_points = coordinates[0].size
    ndim = coordinates.shape[0]

    # Flatten coordinates for easier indexing
    coords_flat = coordinates.reshape(ndim, total_points)

    # Output array
    output = np.zeros(total_points, dtype=input_arr.dtype)

    # Process each point
    for point_idx in prange(total_points):
        point_coords = coords_flat[:, point_idx]

        if order == 0:  # Nearest neighbor
            indices = np.empty(ndim, dtype=np.int64)
            for dim in range(ndim):
                coord = point_coords[dim]
                nearest_idx = _round_half_away_from_zero(coord).astype(np.int64)
                fixed_idx = _fix_index(nearest_idx, input_arr.shape[dim], mode_int)
                indices[dim] = fixed_idx

            # Check validity for constant mode
            valid = True
            if mode_int == 0:  # constant mode
                for dim in range(ndim):
                    original_idx = _round_half_away_from_zero(point_coords[dim]).astype(
                        np.int64
                    )
                    if not _is_valid_index(
                        original_idx, input_arr.shape[dim], mode_int
                    ):
                        valid = False
                        break

            if valid:
                output[point_idx] = input_arr[tuple(indices)]
            else:
                output[point_idx] = cval

        else:  # Linear interpolation (order == 1)
            # Get interpolation weights and indices for each dimension
            result = 0.0

            # For linear interpolation, we need 2^ndim combinations
            num_combinations = 2**ndim

            for combo in range(num_combinations):
                # Extract which corner we're using for each dimension
                weight = 1.0
                indices = np.empty(ndim, dtype=np.int64)
                valid = True

                for dim in range(ndim):
                    coord = point_coords[dim]
                    lower = np.floor(coord)
                    upper_weight = coord - lower
                    lower_weight = 1 - upper_weight

                    # Determine if we use lower or upper index for this dimension
                    use_upper = bool((combo >> dim) & 1)

                    if use_upper:
                        idx = (lower + 1).astype(np.int64)
                        weight *= upper_weight
                    else:
                        idx = lower.astype(np.int64)
                        weight *= lower_weight

                    # Apply boundary conditions
                    fixed_idx = _fix_index(idx, input_arr.shape[dim], mode_int)
                    indices[dim] = fixed_idx

                    # Check validity for constant mode
                    if mode_int == 0 and not _is_valid_index(
                        idx, input_arr.shape[dim], mode_int
                    ):  # constant mode
                        valid = False

                # Add contribution from this corner
                if valid:
                    result += weight * input_arr[tuple(indices)]
                else:
                    result += weight * cval

            output[point_idx] = result

    # Reshape back to original coordinate shape
    return output.reshape(coord_shape)


# Wrapper function that maintains scipy interface for backward compatibility
def numba_map_coordinates_wrapper(
    values,
    coords,
    order=1,
    mode="nearest",
    cval=0.0,
    **kwargs,  # noqa: ARG001
):
    """Wrapper to maintain compatibility with scipy map_coordinates interface."""
    original_shape = coords[0].shape
    coords_array = np.array(coords).reshape(len(values.shape), -1)
    output = numba_map_coordinates(values, coords_array, order, mode, cval)
    return output.reshape(original_shape)


@njit(parallel=True, cache=True, fastmath=True)
def nb_interp_piecewise(args, grids, values, axis):
    """Just-in-time compiled function to interpolate on a warped grid.

    Parameters
    ----------
    args : np.ndarray
        Arguments to be interpolated.
    grids : np.ndarray
        Curvilinear grids for each dimension.
    values : np.ndarray
        Functional values on a curvilinear grid.
    axis : int, 0 or 1
        See `WarpedInterpOnInterp2D.__call__`.

    Returns
    -------
    np.ndarray
        Interpolated values on arguments.

    """
    shape = args[0].shape  # original shape of arguments
    size = args[0].size  # number of points in arguments
    shape_axis = values.shape[axis]  # number of points in axis

    # flatten arguments by dimension
    args = args.reshape((values.ndim, -1))

    y_intermed = np.empty((shape_axis, size))
    z_intermed = np.empty((shape_axis, size))

    for i in prange(shape_axis):
        # for each dimension, interpolate the first argument
        grids0 = grids[0][i] if axis == 0 else grids[0][:, i]
        grids1 = grids[1][i] if axis == 0 else grids[1][:, i]
        vals = values[i] if axis == 0 else values[:, i]
        y_intermed[i] = np.interp(args[0], grids0, grids1)
        z_intermed[i] = np.interp(args[0], grids0, vals)

    output = np.empty_like(args[0])

    for j in prange(size):
        y_temp = y_intermed[:, j]
        z_temp = z_intermed[:, j]

        if y_temp[0] > y_temp[-1]:
            # reverse
            y_temp = y_temp[::-1].copy()
            z_temp = z_temp[::-1].copy()

        output[j] = np.interp(args[1][j], y_temp, z_temp)

    return output.reshape(shape)
