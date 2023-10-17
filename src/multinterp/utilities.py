from __future__ import annotations

JAX_MC_KWARGS = {
    "order": 1,  # order of interpolation, default to linear
    "mode": "nearest",  # how to handle extrapolation
    "cval": 0.0,  # value to use for extrapolation
}

MC_KWARGS = {
    **JAX_MC_KWARGS,
    "output": None,  # output array or dtype
    "prefilter": False,  # whether to prefilter input
}


def update_mc_kwargs(options=None, jax=False):
    mc_kwargs = JAX_MC_KWARGS if jax else MC_KWARGS
    if options:
        mc_kwargs = JAX_MC_KWARGS.copy() if jax else MC_KWARGS.copy()
        intersection = mc_kwargs.keys() & options.keys()
        mc_kwargs.update({key: options[key] for key in intersection})
    return mc_kwargs
