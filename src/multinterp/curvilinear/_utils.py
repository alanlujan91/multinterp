"""Utility functions for curvilinear grid interpolation."""

from __future__ import annotations

import contextlib

import numpy as np

__all__ = [
    "interp_piecewise",
    "INTERP_PIECEWISE",
]


def _scipy_interp_piecewise(args, grids, values, axis):
    """Interpolate on a warped 2D grid using numpy/scipy.

    Parameters
    ----------
    args : np.ndarray
        Coordinates to be interpolated, shape (2, ...).
    grids : np.ndarray
        Curvilinear grids for each dimension, shape (2, shape[0], shape[1]).
    values : np.ndarray
        Functional values on a curvilinear grid, shape (shape[0], shape[1]).
    axis : int, 0 or 1
        Determines which axis to use for linear interpolators.

    Returns
    -------
    np.ndarray
        Interpolated values on arguments.

    """
    shape = args[0].shape
    size = args[0].size
    shape_axis = values.shape[axis]

    args = args.reshape((values.ndim, -1))

    y_intermed = np.empty((shape_axis, size))
    z_intermed = np.empty((shape_axis, size))

    for i in range(shape_axis):
        grids0 = np.take(grids[0], i, axis=axis)
        grids1 = np.take(grids[1], i, axis=axis)
        vals = np.take(values, i, axis=axis)
        y_intermed[i] = np.interp(args[0], grids0, grids1)
        z_intermed[i] = np.interp(args[0], grids0, vals)

    output = np.empty_like(args[0])

    for j in range(size):
        y_temp = y_intermed[:, j]
        z_temp = z_intermed[:, j]

        if y_temp[0] > y_temp[-1]:
            y_temp = y_temp[::-1]
            z_temp = z_temp[::-1]

        output[j] = np.interp(args[1][j], y_temp, z_temp)

    return output.reshape(shape)


# Backend registry - scipy is always available
INTERP_PIECEWISE = {
    "scipy": _scipy_interp_piecewise,
}


# Optional numba backend
with contextlib.suppress(ImportError):
    from multinterp.backend._numba import nb_interp_piecewise

    INTERP_PIECEWISE["numba"] = nb_interp_piecewise


# Optional backend imports
with contextlib.suppress(ImportError):
    import cupy as cp

    def _cupy_interp_piecewise(args, grids, values, axis):
        """Interpolate on a warped 2D grid using cupy.

        Parameters
        ----------
        args : cp.ndarray
            Coordinates to be interpolated.
        grids : cp.ndarray
            Curvilinear grids for each dimension.
        values : cp.ndarray
            Functional values on a curvilinear grid.
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.

        Returns
        -------
        cp.ndarray
            Interpolated values on arguments.

        """
        shape = args[0].shape
        size = args[0].size
        shape_axis = values.shape[axis]

        args = args.reshape((values.ndim, -1))

        y_intermed = cp.empty((shape_axis, size))
        z_intermed = cp.empty((shape_axis, size))

        for i in range(shape_axis):
            grids0 = cp.take(grids[0], i, axis=axis)
            grids1 = cp.take(grids[1], i, axis=axis)
            vals = cp.take(values, i, axis=axis)
            y_intermed[i] = cp.interp(args[0], grids0, grids1)
            z_intermed[i] = cp.interp(args[0], grids0, vals)

        output = cp.empty_like(args[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            # Note: float() forces GPU->CPU transfer, but is necessary for control flow
            if float(y_temp[0]) > float(y_temp[-1]):
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = cp.interp(args[1][j], y_temp, z_temp)

        return output.reshape(shape)

    INTERP_PIECEWISE["cupy"] = _cupy_interp_piecewise


with contextlib.suppress(ImportError):
    import jax.numpy as jnp

    def _jax_interp_piecewise(args, grids, values, axis):
        """Interpolate on a warped 2D grid using JAX.

        Parameters
        ----------
        args : jnp.ndarray
            Coordinates to be interpolated.
        grids : jnp.ndarray
            Curvilinear grids for each dimension.
        values : jnp.ndarray
            Functional values on a curvilinear grid.
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.

        Returns
        -------
        jnp.ndarray
            Interpolated values on arguments.

        """
        shape = args[0].shape
        size = args[0].size
        shape_axis = values.shape[axis]

        args_flat = args.reshape((values.ndim, -1))

        # First interpolation pass
        y_intermed = jnp.empty((shape_axis, size))
        z_intermed = jnp.empty((shape_axis, size))

        for i in range(shape_axis):
            grids0 = jnp.take(grids[0], i, axis=axis)
            grids1 = jnp.take(grids[1], i, axis=axis)
            vals = jnp.take(values, i, axis=axis)
            y_intermed = y_intermed.at[i].set(jnp.interp(args_flat[0], grids0, grids1))
            z_intermed = z_intermed.at[i].set(jnp.interp(args_flat[0], grids0, vals))

        # Second interpolation pass
        output = jnp.empty_like(args_flat[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            # Conditional reverse based on ordering
            needs_reverse = y_temp[0] > y_temp[-1]
            y_sorted = jnp.where(needs_reverse, y_temp[::-1], y_temp)
            z_sorted = jnp.where(needs_reverse, z_temp[::-1], z_temp)

            output = output.at[j].set(jnp.interp(args_flat[1, j], y_sorted, z_sorted))

        return output.reshape(shape)

    INTERP_PIECEWISE["jax"] = _jax_interp_piecewise


with contextlib.suppress(ImportError):
    import torch

    def _torch_interp_piecewise(args, grids, values, axis):
        """Interpolate on a warped 2D grid using PyTorch.

        Parameters
        ----------
        args : torch.Tensor
            Coordinates to be interpolated.
        grids : torch.Tensor
            Curvilinear grids for each dimension.
        values : torch.Tensor
            Functional values on a curvilinear grid.
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.

        Returns
        -------
        torch.Tensor
            Interpolated values on arguments.

        """
        from multinterp.backend._torch import torch_interp

        shape = args[0].shape
        size = args[0].numel()
        shape_axis = values.shape[axis]
        device = args.device

        args_flat = args.reshape((values.ndim, -1))

        y_intermed = torch.empty((shape_axis, size), device=device)
        z_intermed = torch.empty((shape_axis, size), device=device)

        for i in range(shape_axis):
            grids0 = torch.select(grids[0], axis, i)
            grids1 = torch.select(grids[1], axis, i)
            vals = torch.select(values, axis, i)
            y_intermed[i] = torch_interp(args_flat[0], grids0, grids1)
            z_intermed[i] = torch_interp(args_flat[0], grids0, vals)

        output = torch.empty_like(args_flat[0])

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                y_temp = y_temp.flip(0)
                z_temp = z_temp.flip(0)

            output[j] = torch_interp(args_flat[1, j : j + 1], y_temp, z_temp)[0]

        return output.reshape(shape)

    INTERP_PIECEWISE["torch"] = _torch_interp_piecewise


def interp_piecewise(args, grids, values, axis, backend="scipy"):
    """Wrapper function for piecewise interpolation using the chosen backend.

    Parameters
    ----------
    args : array-like
        Coordinates to be interpolated.
    grids : array-like
        Curvilinear grids for each dimension.
    values : array-like
        Functional values on a curvilinear grid.
    axis : int, 0 or 1
        Determines which axis to use for linear interpolators.
    backend : str, optional
        Backend to use. One of 'scipy', 'numba', 'cupy', 'jax', 'torch'.

    Returns
    -------
    array-like
        Interpolated values.

    Raises
    ------
    NotImplementedError
        If the backend is not supported.

    """
    if backend not in INTERP_PIECEWISE:
        available = list(INTERP_PIECEWISE.keys())
        msg = f"Backend {backend!r} not supported. Available: {available}"
        raise NotImplementedError(msg)
    return INTERP_PIECEWISE[backend](args, grids, values, axis)
