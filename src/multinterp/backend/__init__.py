from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Union

backend_functions: Dict[str, Callable] = {}
backends: List[str] = ["cupy", "jax", "numba", "scipy"]

for backend in backends:
    with contextlib.suppress(ImportError):
        backend_functions[backend] = getattr(
            __import__(f"multinterp.backend._{backend}", fromlist=[backend]),
            f"{backend}_multinterp",
        )


def multinterp(
    grids: List[Union[List[float], List[int]]],
    values: Union[List[float], List[int]],
    args: Union[List[float], List[int]],
    backend: str = "numba",
) -> Union[List[float], List[int]]:
    """
    Perform multivariate interpolation using the specified backend.

    Parameters
    ----------
    grids : list of list of float or int
        Grid points in the domain.
    values : list of float or int
        Functional values at the grid points.
    args : list of float or int
        Points at which to interpolate data.
    backend : str, optional
        Backend to use for interpolation. Default is "numba".

    Returns
    -------
    list of float or int
        Interpolated values of the function.

    Raises
    ------
    ValueError
        If the specified backend is not valid.

    """
    if backend not in backends:
        msg = f"Invalid backend: {backend}"
        raise ValueError(msg)

    return backend_functions[backend](grids, values, args)
