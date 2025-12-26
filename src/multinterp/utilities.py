"""Utility functions and backend management for multinterp."""

from __future__ import annotations

import numpy as np
from numba import typed

__all__ = [
    "BACKENDS",
    "LONG_MC_KWARGS",
    "MODULES",
    "SHORT_MC_KWARGS",
    "asarray",
    "aslist",
    "empty",
    "empty_like",
    "interp",
    "mgrid",
    "take",
    "update_mc_kwargs",
]

# Backend registry - scipy and numba are always available
BACKENDS: list[str] = ["scipy", "numba"]
MODULES: dict[str, object] = {"scipy": np, "numba": np}

# Optional backend imports
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


def update_mc_kwargs(options=None, jax=False):
    mc_kwargs = SHORT_MC_KWARGS if jax else LONG_MC_KWARGS
    if options:
        mc_kwargs = SHORT_MC_KWARGS.copy() if jax else LONG_MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})
    return mc_kwargs


def asarray(values, backend):
    if backend not in BACKENDS:
        msg = f"Invalid backend. Must be one of: {BACKENDS}"
        raise ValueError(msg)

    return MODULES[backend].asarray(values)


def aslist(grids, backend):
    if backend == "numba":
        grids = typed.List([np.asarray(grid) for grid in grids])
    else:
        grids = [MODULES[backend].asarray(grid) for grid in grids]

    return grids


def empty(shape, backend):
    return MODULES[backend].empty(shape)


def empty_like(values, backend):
    return MODULES[backend].empty_like(values)


def interp(x, y, z, backend):
    return MODULES[backend].interp(x, y, z)


def take(arr, indices, axis, backend):
    return MODULES[backend].take(arr, indices, axis=axis)


def mgrid(args, backend):
    return MODULES[backend].mgrid[args]
