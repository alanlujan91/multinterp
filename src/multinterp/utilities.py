from __future__ import annotations

import numpy as np

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


def import_backends():
    backends = ["scipy", "numba"]
    modules = {"scipy": np, "numba": np}

    try:
        import cupy as cp

        backends.append("cupy")
        modules["cupy"] = cp
    except ImportError:
        pass

    try:
        import jax.numpy as jnp

        backends.append("jax")
        modules["jax"] = jnp
    except ImportError:
        pass

    try:
        import torch

        backends.append("torch")
        modules["torch"] = torch
    except ImportError:
        pass

    return backends, modules
