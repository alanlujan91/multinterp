from __future__ import annotations

backend_functions = {}
try:
    from multinterp.backend._cupy import cupy_multinterp

    backend_functions["cupy"] = cupy_multinterp
except ImportError:
    pass

try:
    from multinterp.backend._cupy import jax_multinterp

    backend_functions["jax"] = jax_multinterp
except ImportError:
    pass

try:
    from multinterp.backend._cupy import numba_multinterp

    backend_functions["numba"] = numba_multinterp
except ImportError:
    pass

try:
    from multinterp.backend._cupy import scipy_multinterp

    backend_functions["scipy"] = scipy_multinterp
except ImportError:
    pass


def multinterp(grids, values, args, backend="numba"):
    if backend in backend_functions:
        return backend_functions[backend](grids, values, args)
    msg = f"Invalid backend: {backend}"
    raise ValueError(msg)
