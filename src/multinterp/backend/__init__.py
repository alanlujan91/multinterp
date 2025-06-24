from __future__ import annotations

import contextlib

backend_functions = {}
backends = ["cupy", "jax", "numba", "scipy"]

for backend in backends:
    with contextlib.suppress(ImportError):
        backend_functions[backend] = getattr(
            __import__(f"multinterp.backend._{backend}", fromlist=[backend]),
            f"{backend}_multinterp",
        )


def multinterp(grids, values, args, backend="numba"):
    if backend not in backends:
        msg = f"Invalid backend: {backend}"
        raise ValueError(msg)

    return backend_functions[backend](grids, values, args)
