"""Utility functions for rectilinear grid interpolation."""

from __future__ import annotations

import contextlib

import numpy as np

from multinterp.backend._numba import numba_get_coordinates, numba_map_coordinates
from multinterp.backend._scipy import scipy_get_coordinates, scipy_map_coordinates

__all__ = ["get_coords", "get_grad", "map_coords"]

GET_COORDS = {
    "scipy": scipy_get_coordinates,
    "numba": numba_get_coordinates,
}
MAP_COORDS = {
    "scipy": scipy_map_coordinates,
    "numba": numba_map_coordinates,
}
GET_GRAD = {
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

with contextlib.suppress(ImportError):
    import torch

    from multinterp.backend._torch import torch_get_coordinates, torch_map_coordinates

    GET_COORDS["torch"] = torch_get_coordinates
    MAP_COORDS["torch"] = torch_map_coordinates
    GET_GRAD["torch"] = torch.gradient


def get_coords(grids, args, backend="scipy"):
    """Wrapper function for the get_coordinates function from the chosen backend."""
    return GET_COORDS[backend](grids, args)


def map_coords(values, coords, backend="scipy", **kwargs):
    """Wrapper function for the map_coordinates function from the chosen backend."""
    return MAP_COORDS[backend](values, coords, **kwargs)


def get_grad(values, grids, axis=None, edge_order=None, backend="scipy"):
    """Wrapper function for the gradient function from the chosen backend.

    TODO: use appropriate gradient functions from each backend.
    """
    return GET_GRAD[backend](values, grids, axis=axis, edge_order=edge_order)
