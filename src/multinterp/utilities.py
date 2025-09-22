from __future__ import annotations

import numpy as np
from numba import typed

BACKENDS = ["scipy", "numba"]
MODULES = {"scipy": np, "numba": np}

try:
    import cupy as cp

    BACKENDS.append("cupy")
    MODULES["cupy"] = cp
except ImportError:
    pass

try:
    import jax.numpy as jnp

    BACKENDS.append("jax")
    MODULES["jax"] = jnp
except ImportError:
    pass

try:
    import torch

    BACKENDS.append("torch")
    MODULES["torch"] = torch
except ImportError:
    pass

SHORT_MC_KWARGS = {
    "order": 1,  # order of interpolation, default to linear
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
}

LONG_MC_KWARGS = {
    **SHORT_MC_KWARGS,
    "output": None,  # output array or dtype
    "prefilter": False,  # whether to prefilter input
}


def update_mc_kwargs(options: dict | None = None, jax: bool = False) -> dict:
    """
    Update the keyword arguments for the map_coordinates function based on the provided options.

    Parameters
    ----------
    options : dict, optional
        Additional options for interpolation.
    jax : bool, optional
        Flag indicating whether to use JAX-specific options.

    Returns
    -------
    dict
        Updated keyword arguments for the map_coordinates function.

    """
    mc_kwargs = SHORT_MC_KWARGS if jax else LONG_MC_KWARGS
    if options:
        mc_kwargs = SHORT_MC_KWARGS.copy() if jax else LONG_MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})
    return mc_kwargs


def asarray(values: np.ndarray, backend: str) -> np.ndarray:
    """
    Convert the input values to an array using the specified backend.

    Parameters
    ----------
    values : np.ndarray
        Input values to be converted.
    backend : str
        Backend to use for conversion. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        Converted array.

    Raises
    ------
    ValueError
        If the specified backend is not valid.

    """
    if backend not in BACKENDS:
        msg = f"Invalid backend. Must be one of: {BACKENDS}"
        raise ValueError(msg)

    return MODULES[backend].asarray(values)


def aslist(grids: list[np.ndarray], backend: str) -> list[np.ndarray]:
    """
    Convert the input grids to a list of arrays using the specified backend.

    Parameters
    ----------
    grids : list of np.ndarray
        Input grids to be converted.
    backend : str
        Backend to use for conversion. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    list of np.ndarray
        Converted list of arrays.

    """
    if backend == "numba":
        grids = typed.List([np.asarray(grid) for grid in grids])
    else:
        grids = [MODULES[backend].asarray(grid) for grid in grids]

    return grids


def empty(shape: tuple[int, ...], backend: str) -> np.ndarray:
    """
    Create an empty array with the specified shape using the specified backend.

    Parameters
    ----------
    shape : tuple of int
        Shape of the empty array.
    backend : str
        Backend to use for creating the array. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        Empty array with the specified shape.

    """
    return MODULES[backend].empty(shape)


def empty_like(values: np.ndarray, backend: str) -> np.ndarray:
    """
    Create an empty array with the same shape and type as the input values using the specified backend.

    Parameters
    ----------
    values : np.ndarray
        Input values to determine the shape and type of the empty array.
    backend : str
        Backend to use for creating the array. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        Empty array with the same shape and type as the input values.

    """
    return MODULES[backend].empty_like(values)


def interp(x: np.ndarray, y: np.ndarray, z: np.ndarray, backend: str) -> np.ndarray:
    """
    Perform one-dimensional linear interpolation using the specified backend.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the interpolated values.
    y : np.ndarray
        The x-coordinates of the data points.
    z : np.ndarray
        The y-coordinates of the data points, same length as y.
    backend : str
        Backend to use for interpolation. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        The interpolated values, same shape as x.

    """
    return MODULES[backend].interp(x, y, z)


def take(arr: np.ndarray, indices: int, axis: int, backend: str) -> np.ndarray:
    """
    Take elements from an array along an axis using the specified backend.

    Parameters
    ----------
    arr : np.ndarray
        Input array from which to take elements.
    indices : int
        Indices of elements to take.
    axis : int
        Axis along which to take elements.
    backend : str
        Backend to use for taking elements. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        Array of taken elements.

    """
    return MODULES[backend].take(arr, indices, axis=axis)


def mgrid(args: tuple[slice, ...], backend: str) -> np.ndarray:
    """
    Return coordinate matrices from coordinate vectors using the specified backend.

    Parameters
    ----------
    args : tuple of slice
        Coordinate vectors.
    backend : str
        Backend to use for creating coordinate matrices. Must be one of "scipy", "numba", "cupy", "jax", or "torch".

    Returns
    -------
    np.ndarray
        Coordinate matrices.

    """
    return MODULES[backend].mgrid[args]
