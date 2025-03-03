from __future__ import annotations

import contextlib
import numpy as np
from typing import Any, Callable

from multinterp.backend._numba import numba_get_coordinates, numba_map_coordinates
from multinterp.backend._scipy import scipy_get_coordinates, scipy_map_coordinates

GET_COORDS: dict[str, Callable[..., Any]] = {
    "scipy": scipy_get_coordinates,
    "numba": numba_get_coordinates,
}
MAP_COORDS: dict[str, Callable[..., Any]] = {
    "scipy": scipy_map_coordinates,
    "numba": numba_map_coordinates,
}
GET_GRAD: dict[str, Callable[..., Any]] = {
    "scipy": np.gradient,
    "numba": np.gradient,
}

with contextlib.suppress(ImportError):
    import cupy as cp
    from multinterp.backend._cupy import cupy_get_coordinates, cupy_map_coordinates

    GET_COORDS["cupy"] = cupy_get_coordinates
    MAP_COORDS["cupy"] = cupy_map_coordinates
    GET_GRAD["cupy"] = cp.gradient

with contextlib.suppress(ImportError):
    import jax.numpy as jnp
    from multinterp.backend._jax import jax_get_coordinates, jax_map_coordinates

    GET_COORDS["jax"] = jax_get_coordinates
    MAP_COORDS["jax"] = jax_map_coordinates
    GET_GRAD["jax"] = jnp.gradient


def get_coords(grids: list[np.ndarray], args: np.ndarray, backend: str = "scipy") -> np.ndarray:
    """Wrapper function for the get_coordinates function from the chosen backend.

    Parameters
    ----------
    grids : list of np.ndarray
        Grid points in the domain.
    args : np.ndarray
        Points at which to interpolate data.
    backend : str, optional
        Backend to use for interpolation. Default is "scipy".

    Returns
    -------
    np.ndarray
        Coordinates with respect to the grid.

    """
    return GET_COORDS[backend](grids, args)


def map_coords(values: np.ndarray, coords: np.ndarray, backend: str = "scipy", **kwargs: Any) -> np.ndarray:
    """Wrapper function for the map_coordinates function from the chosen backend.

    Parameters
    ----------
    values : np.ndarray
        Functional values from which to interpolate.
    coords : np.ndarray
        Coordinates at which to interpolate values.
    backend : str, optional
        Backend to use for interpolation. Default is "scipy".
    **kwargs : dict
        Additional keyword arguments for the map_coordinates function.

    Returns
    -------
    np.ndarray
        Interpolated values of the function.

    """
    return MAP_COORDS[backend](values, coords, **kwargs)


def get_grad(values: np.ndarray, grids: np.ndarray, axis: int | None = None, edge_order: int | None = None, backend: str = "scipy") -> np.ndarray:
    """Wrapper function for the gradient function from the chosen backend.

    Parameters
    ----------
    values : np.ndarray
        Functional values at the grid points.
    grids : np.ndarray
        Grid points in the domain.
    axis : int, optional
        Axis along which to compute the gradient.
    edge_order : int, optional
        Order of the finite difference approximation used to compute the gradient.
    backend : str, optional
        Backend to use for interpolation. Default is "scipy".

    Returns
    -------
    np.ndarray
        Gradient of the function.

    """
    return GET_GRAD[backend](values, grids, axis=axis, edge_order=edge_order)
