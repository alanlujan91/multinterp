"""Numba JIT-compiled backend for multinterp."""

from __future__ import annotations

import numpy as np
from numba import njit, prange, typed

from multinterp.utilities import update_mc_kwargs

__all__ = [
    "nb_interp_piecewise",
    "numba_get_coordinates",
    "numba_gradinterp",
    "numba_map_coordinates",
    "numba_multinterp",
]


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

    # Extract and convert kwargs for JIT
    order = mc_kwargs.get("order", 1)
    mode = mc_kwargs.get("mode", "nearest")
    cval = mc_kwargs.get("cval", 0.0)
    mode_int = _get_mode_int(mode)

    return _numba_multinterp_impl(grids, values, args, order, mode_int, cval)


def _get_mode_int(mode):
    """Convert mode string to integer for numba JIT functions."""
    mode_map = {"constant": 0, "nearest": 1, "wrap": 2, "mirror": 3, "reflect": 4}
    mode_int = mode_map.get(mode)
    if mode_int is None:
        msg = f"Mode {mode!r} not supported. Use one of: {set(mode_map.keys())}"
        raise NotImplementedError(msg)
    return mode_int


@njit(cache=True, fastmath=True)
def _numba_multinterp_impl(grids, values, args, order, mode, cval):
    """JIT-compiled implementation of multivariate interpolation."""
    coords = numba_get_coordinates(grids, args)
    return _numba_map_coordinates_impl(values, coords, order, mode, cval)


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
        grid_size = np.arange(grids[dim].size)
        coords[dim] = np.interp(args[dim], grids[dim], grid_size)

    return coords


def numba_gradinterp(grids, values, args, axis=None, options=None):
    """Computes the interpolated value of the gradient using JIT-compiled Numba functions.

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
    grids_list = [np.asarray(grid) for grid in grids]
    grids = typed.List(grids_list)

    # Extract kwargs once for JIT calls
    order = mc_kwargs.get("order", 1)
    mode = mc_kwargs.get("mode", "nearest")
    cval = mc_kwargs.get("cval", 0.0)
    mode_int = _get_mode_int(mode)

    coords = numba_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            msg = "Axis should be an integer."
            raise ValueError(msg)
        gradient = np.gradient(values, grids_list[axis], axis=axis, edge_order=eo)
        return _numba_map_coordinates_impl(gradient, coords, order, mode_int, cval)

    gradient = np.gradient(values, *grids_list, edge_order=eo)
    return _numba_gradinterp_multi(gradient, coords, order, mode_int, cval)


@njit(cache=True, fastmath=True)
def _numba_gradinterp_multi(gradients, coords, order, mode, cval):
    """JIT-compiled multi-gradient interpolation."""
    ndim = len(gradients)
    result_shape = (ndim,) + coords[0].shape  # noqa: RUF005 - numba doesn't support unpacking
    output = np.empty(result_shape, dtype=np.float64)
    for i in range(ndim):
        output[i] = _numba_map_coordinates_impl(gradients[i], coords, order, mode, cval)
    return output


def numba_map_coordinates(values, coords, **kwargs):
    """Run map_coordinates using JIT-compiled Numba functions.

    Parameters
    ----------
    values : np.ndarray
        Functional values from which to interpolate.
    coords : np.ndarray
        Coordinates at which to interpolate values.
    **kwargs : dict
        Additional keyword arguments:
        - order : int (0 or 1, default 1)
        - mode : str ('constant', 'nearest', 'wrap', 'mirror', 'reflect')
        - cval : float (default 0.0)

    Returns
    -------
    np.ndarray
        Interpolated values of the function.

    """
    order = kwargs.get("order", 1)
    mode = kwargs.get("mode", "nearest")
    cval = kwargs.get("cval", 0.0)

    if order not in (0, 1):
        msg = f"Order {order} not supported. Use order=0 or order=1."
        raise NotImplementedError(msg)

    mode_int = _get_mode_int(mode)
    return _numba_map_coordinates_impl(values, coords, order, mode_int, cval)


@njit(cache=True, fastmath=True)
def _numba_map_coordinates_impl(values, coords, order, mode, cval):
    """JIT-compiled implementation of map_coordinates."""
    original_shape = coords[0].shape
    coords_flat = coords.reshape(len(values.shape), -1)

    if order == 0:
        output = _numba_map_coordinates_nearest(values, coords_flat, mode, cval)
    else:
        output = _numba_map_coordinates_linear(values, coords_flat, mode, cval)

    return output.reshape(original_shape)


@njit(cache=True, fastmath=True)
def _fix_index(index, size, mode):
    """Fix index based on boundary mode.

    Parameters
    ----------
    index : int
        The index to fix.
    size : int
        The size of the dimension.
    mode : int
        Boundary mode: 0=constant, 1=nearest, 2=wrap, 3=mirror, 4=reflect

    Returns
    -------
    tuple
        (fixed_index, is_valid) where is_valid indicates if the point is in bounds.

    """
    if 0 <= index < size:
        return index, True

    if mode == 0:  # constant
        return 0, False
    if mode == 1:  # nearest
        if index < 0:
            return 0, True
        return size - 1, True
    if mode == 2:  # wrap
        return index % size, True
    if mode == 3:  # mirror
        # Mirror: size=5: -1->0, -2->1, 5->4, 6->3
        if size == 1:
            return 0, True
        period = 2 * size - 2
        index = index % period
        if index >= size:
            index = period - index
        return index, True
    if mode == 4:  # reflect
        # Reflect: size=5: -1->-1->0, 5->5->4 (reflects at -0.5 and size-0.5)
        if size == 1:
            return 0, True
        period = 2 * size
        index = index % period
        if index >= size:
            index = period - 1 - index
        return index, True
    return 0, False


@njit(parallel=True, cache=True, fastmath=True)
def _numba_map_coordinates_nearest(values, coords, mode, cval):
    """Nearest neighbor interpolation using Numba.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    coords : np.ndarray
        Coordinates with shape (ndim, npoints).
    mode : int
        Boundary mode.
    cval : float
        Constant value for mode=0.

    Returns
    -------
    np.ndarray
        Interpolated values.

    """
    ndim = coords.shape[0]
    npoints = coords.shape[1]
    shape = values.shape
    output = np.empty(npoints, dtype=values.dtype)

    for i in prange(npoints):
        valid = True
        idx = np.empty(ndim, dtype=np.int64)

        for d in range(ndim):
            # Round to nearest integer
            raw_idx = int(np.round(coords[d, i]))
            fixed_idx, is_valid = _fix_index(raw_idx, shape[d], mode)
            idx[d] = fixed_idx
            if not is_valid:
                valid = False

        if valid:
            # Access the value at the multi-dimensional index
            flat_idx = 0
            stride = 1
            for d in range(ndim - 1, -1, -1):
                flat_idx += idx[d] * stride
                stride *= shape[d]
            output[i] = values.flat[flat_idx]
        else:
            output[i] = cval

    return output


@njit(parallel=True, cache=True, fastmath=True)
def _numba_map_coordinates_linear(values, coords, mode, cval):
    """Linear interpolation using Numba.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    coords : np.ndarray
        Coordinates with shape (ndim, npoints).
    mode : int
        Boundary mode.
    cval : float
        Constant value for mode=0.

    Returns
    -------
    np.ndarray
        Interpolated values.

    """
    ndim = coords.shape[0]
    npoints = coords.shape[1]
    shape = values.shape
    output = np.empty(npoints, dtype=np.float64)

    # Number of corners in n-dimensional hypercube
    ncorners = 1 << ndim  # 2^ndim

    for i in prange(npoints):
        result = 0.0

        # For each corner of the hypercube
        for corner in range(ncorners):
            weight = 1.0
            valid = True
            flat_idx = 0
            stride = 1

            # Process dimensions in reverse order for correct flat indexing
            for d in range(ndim - 1, -1, -1):
                coord_val = coords[d, i]
                lower = int(np.floor(coord_val))
                frac = coord_val - lower

                # Determine if we use lower or upper index for this corner
                use_upper = (corner >> d) & 1

                if use_upper:
                    idx = lower + 1
                    w = frac
                else:
                    idx = lower
                    w = 1.0 - frac

                weight *= w

                fixed_idx, is_valid = _fix_index(idx, shape[d], mode)
                if not is_valid:
                    valid = False

                flat_idx += fixed_idx * stride
                stride *= shape[d]

            if valid and weight > 0:
                result += weight * values.flat[flat_idx]
            elif not valid and weight > 0:
                result += weight * cval

        output[i] = result

    return output


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
