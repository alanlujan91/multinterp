from __future__ import annotations

import numpy as np

from multinterp.grids import _MultivaluedRegularGrid, _RegularGrid
from multinterp.utilities import asarray, update_mc_kwargs

from ._utils import get_coords, get_grad, map_coords


class MultivariateInterp(_RegularGrid):
    """Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """Initialize a multivariate interpolator.

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
        self.mc_kwargs = update_mc_kwargs(options, jax=self.backend == "jax")
        self._gradient = {}

    def compile(self):
        self(*self.grids)

    def __call__(self, *args):
        """Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of arguments does not match number of dimensions.

        """
        args = asarray(args, backend=self.backend)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.

        """
        return get_coords(self.grids, args, backend=self.backend)

    def _map_coordinates(self, coords):
        """Uses coordinates to interpolate on the regular grid with
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
        return map_coords(self.values, coords, **self.mc_kwargs, backend=self.backend)

    def diff(self, axis=None, edge_order=1):
        """Differentiates the interpolator along the specified axis. If axis is None, then returns a MultivaluedInterp object that approximates the partial derivative of the function across all axes. Otherwise, returns a MultivariateInterp object that approximates the partial derivative of the function along the specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to differentiate the function.
        edge_order : int, optional
            TODO: Add description

        Returns
        -------
        MultivaluedInterp or MultivariateInterp
            Interpolator object that approximates the partial derivative(s) of the function.

        """
        # if axis is not an integer less than or equal to the number
        # of dimensions of the input array, then a ValueError is raised.
        if axis is None:
            # return MultivaluedInterp
            for ax in range(self.ndim):
                if ax not in self._gradient:
                    self._gradient[ax] = get_grad(
                        self.values,
                        self.grids[ax],
                        axis=ax,
                        edge_order=edge_order,
                        backend=self.backend,
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
            self._gradient[axis] = get_grad(
                self.values,
                self.grids[axis],
                axis=axis,
                edge_order=edge_order,
                backend=self.backend,
            )
            grad = self._gradient[axis]

        return MultivariateInterp(
            grad,
            self.grids,
            backend=self.backend,
            options=self.mc_kwargs,
        )


class MultivaluedInterp(_MultivaluedRegularGrid):
    """Multivariate Interpolator on a regular grid. Maps functional coordinates
    to index coordinates and uses `map_coordinates` from scipy, cupy, or jax.
    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """Initialize a multivariate interpolator.

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
        self.mc_kwargs = update_mc_kwargs(options)
        self._gradient = {}

    def compile(self):
        self(*self.grids)

    def __call__(self, *args):
        """Interpolates arguments on the regular grid.

        Returns
        -------
        np.ndarray
            Interpolated functional values for each argument.

        Raises
        ------
        ValueError
            Number of arguments does not match number of dimensions.

        """
        args = asarray(args, backend=self.backend)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        coords = self._get_coordinates(args)
        return self._map_coordinates(coords)

    def _get_coordinates(self, args):
        """For each argument, finds the index coordinates for interpolation.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for interpolation.

        """
        return get_coords(self.grids, args, self.backend)

    def _map_coordinates(self, coords):
        """Uses coordinates to interpolate on the regular grid with
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
            map_coords(self.values[i], coords, **self.mc_kwargs, backend=self.backend)
            for i in range(self.nval)
        ]

        return asarray(fvals, backend=self.backend)

    def diff(self, axis=None, argnum=None, edge_order=1):
        """Differentiates the interpolator along the specified axis. If both axis and argnum are specified, then returns the partial derivative of the specified function argument along the specified axis. If axis is None, then returns a MultivaluedInterp object that approximates the partial derivatives of the specified function argument along each axis. If argnum is None, then returns a MultivaluedInterp object that approximates the partial derivatives of all arguments of the function along the specified axes.

        Parameters
        ----------
        axis : int, optional
            Axis along which to differentiate the function.
        edge_order : int, optional
            TODO: Add description

        Returns
        -------
        MultivaluedInterp or MultivariateInterp
            Interpolator object that approximates the partial derivative(s) of the function.

        """
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
                    self._gradient[(axis, arg)] = get_grad(
                        self.values,
                        self.grids[arg],
                        axis=arg,
                        edge_order=edge_order,
                        backend=self.backend,
                    )
            return MultivaluedInterp(
                np.asarray(list(self._gradient.items())),
                self.grids,
                backend=self.backend,
                options=self.mc_kwargs,
            )

        grad = self._gradient.get((axis, arg))
        if grad is None:
            self._gradient[(axis, arg)] = get_grad(
                self.values,
                self.grids[axis],
                axis=axis,
                edge_order=edge_order,
                backend=self.backend,
            )

        return MultivariateInterp(
            grad,
            self.grids,
            backend=self.backend,
            options=self.mc_kwargs,
        )
