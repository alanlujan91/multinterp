from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

try:
    import jax.numpy as jnp
except ImportError:
    pass

from multinterp.backend._numba import numba_get_coordinates, numba_map_coordinates
from multinterp.backend._scipy import scipy_get_coordinates, scipy_map_coordinates
from multinterp.core import (
    JAX_MC_KWARGS,
    MC_KWARGS,
    _MultivaluedRegularGrid,
    _RegularGrid,
    import_backends,
)


def get_methods():
    get_coords = {
        "scipy": scipy_get_coordinates,
        "numba": numba_get_coordinates,
    }
    map_coords = {
        "scipy": scipy_map_coordinates,
        "numba": numba_map_coordinates,
    }
    get_grad = {
        "scipy": np.gradient,
        "numba": np.gradient,
    }

    try:
        from multinterp.backend._cupy import cupy_get_coordinates, cupy_map_coordinates

        get_coords["cupy"] = cupy_get_coordinates
        map_coords["cupy"] = cupy_map_coordinates
        get_grad["cupy"] = cp.gradient
    except ImportError:
        pass

    try:
        from multinterp.backend._jax import jax_get_coordinates, jax_map_coordinates

        get_coords["jax"] = jax_get_coordinates
        map_coords["jax"] = jax_map_coordinates
        get_grad["jax"] = jnp.gradient
    except ImportError:
        pass

    return get_coords, map_coords, get_grad


GET_COORDS, MAP_COORDS, GET_GRAD = get_methods()

AVAILABLE_BACKENDS, BACKEND_MODULES = import_backends()


class MultivariateInterp(_RegularGrid):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", "cupy", or "jax". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, grids, backend=backend)
        self._parse_mc_options(options)
        self._gradient = {}

    def _parse_mc_options(self, options):
        self.mc_kwargs = MC_KWARGS if self.backend != "jax" else JAX_MC_KWARGS
        if options:
            self.mc_kwargs = self.mc_kwargs.copy()
            intersection = self.mc_kwargs.keys() & options.keys()
            self.mc_kwargs.update({key: options[key] for key in intersection})

    def compile(self):
        self(*self.grids)

    def __call__(self, *args):
        """
        Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of arguments does not match number of dimensions.
        """

        args = BACKEND_MODULES[self.backend].asarray(args)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        return GET_COORDS[self.backend](self.grids, args)

    def _map_coordinates(self, coords):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on backend.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """

        return MAP_COORDS[self.backend](self.values, coords, **self.mc_kwargs)

    def diff(self, axis=None, edge_order=1):
        # if axis is not an integer less than or equal to the number
        # of dimensions of the input array, then a ValueError is raised.
        if axis is None:
            # return MultivaluedInterp
            for ax in range(self.ndim):
                if ax not in self._gradient:
                    self._gradient[ax] = GET_GRAD[self.backend](
                        self.values, self.grids[ax], axis=ax, edge_order=edge_order
                    )
            return MultivaluedInterp(
                np.asarray(list(self._gradient.items())),
                self.grids,
                backend=self.backend,
                options=self.mc_kwargs,
            )

        if axis >= self.ndim:
            msg = "Axis must be less than number of dimensions."
            raise ValueError(msg)

        grad = self._gradient.get(axis)
        if grad is None:
            self._gradient[axis] = GET_GRAD[self.backend](
                self.values, self.grids[axis], axis=axis, edge_order=edge_order
            )

        return MultivariateInterp(
            grad, self.grids, backend=self.backend, options=self.mc_kwargs
        )


class MultivaluedInterp(_MultivaluedRegularGrid):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", "cupy", or "jax". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, grids, backend=backend)
        self._parse_mc_options(options)
        self._gradient = {}

    def _parse_mc_options(self, options):
        self.mc_kwargs = MC_KWARGS if self.backend != "jax" else JAX_MC_KWARGS
        if options:
            self.mc_kwargs = self.mc_kwargs.copy()
            intersection = self.mc_kwargs.keys() & options.keys()
            self.mc_kwargs.update({key: options[key] for key in intersection})

    def compile(self):
        self(*self.grids)

    def __call__(self, *args):
        """
        Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of arguments does not match number of dimensions.
        """

        args = BACKEND_MODULES[self.backend].asarray(args)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        return GET_COORDS[self.backend](self.grids, args)

    def _map_coordinates(self, coords):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on backend.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """

        fvals = [
            MAP_COORDS[self.backend](self.values[i], coords, **self.mc_kwargs)
            for i in range(self.nval)
        ]

        return BACKEND_MODULES[self.backend].asarray(fvals)

    def diff(self, axis=None, argnum=None, edge_order=1):
        # if axis is not an integer less than or equal to the number
        # of dimensions of the input array, then a ValueError is raised.
        if axis is None:
            msg = "Must specify axis (function) to differentiate."
            raise ValueError(msg)

        if axis >= self.nval:
            msg = "Axis must be less than number of functions."
            raise ValueError(msg)

        if argnum is None:
            # return MultivaluedInterp
            for arg in range(self.ndim):
                if (axis, arg) not in self._gradient:
                    self._gradient[(axis, arg)] = GET_GRAD[self.backend](
                        self.values, self.grids[arg], axis=arg, edge_order=edge_order
                    )
            return MultivaluedInterp(
                np.asarray(list(self._gradient.items())),
                self.grids,
                backend=self.backend,
                options=self.mc_kwargs,
            )

        grad = self._gradient.get((axis, arg))
        if grad is None:
            self._gradient[(axis, arg)] = GET_GRAD[self.backend](
                self.values, self.grids[axis], axis=axis, edge_order=edge_order
            )

        return MultivariateInterp(
            grad, self.grids, backend=self.backend, options=self.mc_kwargs
        )


class MultivaluedInterp(_MultivaluedRegularGrid):
    """
    Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """
        Initialize a multivariate interpolator.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a regular grid.
        grids : _type_
            1D grids for each dimension.
        backend : str, optional
            One of "scipy", "numba", "cupy", or "jax". Determines
            hardware to use for interpolation.
        """

        super().__init__(values, grids, backend=backend)
        self._parse_mc_options(options)
        self._gradient = {}

    def _parse_mc_options(self, options):
        self.mc_kwargs = MC_KWARGS if self.backend != "jax" else JAX_MC_KWARGS
        if options:
            self.mc_kwargs = self.mc_kwargs.copy()
            intersection = self.mc_kwargs.keys() & options.keys()
            self.mc_kwargs.update({key: options[key] for key in intersection})

    def compile(self):
        self(*self.grids)

    def __call__(self, *args):
        """
        Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of arguments does not match number of dimensions.
        """

        args = BACKEND_MODULES[self.backend].asarray(args)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """
        For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.
        """

        return GET_COORDS[self.backend](self.grids, args)

    def _map_coordinates(self, coords):
        """
        Uses coordinates to interpolate on the regular grid with
        `map_coordinates` from scipy or cupy, depending on backend.

        Parameters
        ----------
        coordinates : np.ndarray
            Index coordinates for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each coordinate.
        """

        fvals = [
            MAP_COORDS[self.backend](self.values[i], coords, **self.mc_kwargs)
            for i in range(self.nval)
        ]

        return BACKEND_MODULES[self.backend].asarray(fvals)

    # def diff(self, axis=(None, None), edge_order=1):
    #     # if axis is not an integer less than or equal to the number
    #     # of dimensions of the input array, then a ValueError is raised.
    #     if axis is (None, None):
    #         grad = []
    #     if axis >= self.ndim:
    #         msg = "Axis must be less than number of dimensions."
    #         raise ValueError(msg)

    #     grad = self._gradient.get(axis)
    #     if grad is None:
    #         grad = GET_GRAD[self.backend](
    #             self.values, self.grids[axis], axis=axis, edge_order=edge_order
    #         )

    #     return MultivariateInterp(
    #         grad, self.grids, backend=self.backend, options=self.mc_kwargs
    #     )
