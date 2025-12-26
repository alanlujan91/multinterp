"""Warped and curvilinear grid interpolation."""

from __future__ import annotations

import numpy as np

from multinterp.grids import _CurvilinearGrid
from multinterp.rectilinear._utils import MAP_COORDS, map_coords
from multinterp.utilities import asarray, update_mc_kwargs

from ._utils import INTERP_PIECEWISE, interp_piecewise

__all__ = ["Curvilinear2DInterp", "Warped2DInterp"]


class Warped2DInterp(_CurvilinearGrid):
    """Warped Grid Interpolation on a 2D grid.

    This interpolator uses piecewise linear interpolation on warped grids,
    suitable for models solved with the endogenous grid method (EGM).

    Parameters
    ----------
    values : np.ndarray
        Functional values on a curvilinear grid, shape (n, m).
    grids : np.ndarray
        Curvilinear grids for each dimension, shape (2, n, m).
    backend : str, optional
        Backend to use for interpolation. One of 'scipy', 'numba', 'cupy',
        'jax', 'torch'. Default is 'scipy'.

    """

    def __init__(self, values, grids, backend="scipy"):
        """Initialize a Warped2DInterp object."""
        if backend not in INTERP_PIECEWISE:
            available = list(INTERP_PIECEWISE.keys())
            msg = f"Backend {backend!r} not supported for Warped2DInterp. Available: {available}"
            raise NotImplementedError(msg)
        super().__init__(values, grids, backend=backend)

    def __call__(self, *args, axis=1):
        """Interpolate on a warped grid.

        Uses the Warped Grid Interpolation method for piecewise linear
        interpolation on non-rectilinear grids.

        Parameters
        ----------
        *args : array-like
            Coordinate arrays for each dimension.
        axis : int, 0 or 1, optional
            Determines which axis to use for linear interpolators.
            Setting to 0 may fix some issues where interpolation fails.
            Default is 1.

        Returns
        -------
        np.ndarray
            Interpolated values at the specified coordinates.

        Raises
        ------
        ValueError
            If the number of arguments doesn't match the number of dimensions.

        """
        args = asarray(args, backend=self.backend)

        if args.shape[0] != self.ndim:
            msg = f"Number of arguments ({args.shape[0]}) must match number of dimensions ({self.ndim})."
            raise ValueError(msg)

        return interp_piecewise(args, self.grids, self.values, axis, self.backend)

    def warmup(self):
        """Warm up the JIT compiler by running interpolation once."""
        self(*self.grids)


class Curvilinear2DInterp(_CurvilinearGrid):
    """A 2D interpolation method for curvilinear or "warped grid" interpolation.

    Implements the method described in White (2015) for models with two
    endogenous states solved with the endogenous grid method.

    This interpolator uses bilinear interpolation within quadrilateral sectors,
    with automatic sector identification and polarity tracking. After finding
    the fractional index coordinates, it uses `map_coordinates` for interpolation,
    enabling multi-backend support.

    Parameters
    ----------
    values : np.ndarray
        A 2D array of function values such that values[i,j] =
        f(x_values[i,j], y_values[i,j]).
    grids : np.ndarray
        Shape (2, n, m) array where grids[0] contains x-coordinates
        and grids[1] contains y-coordinates.
    backend : str, optional
        Backend to use for interpolation. One of 'scipy', 'numba', 'cupy',
        'jax', 'torch'. Default is 'scipy'.
    options : dict, optional
        Additional options for map_coordinates (e.g., order, mode, cval).

    Attributes
    ----------
    values : np.ndarray
        Functional values on the grid.
    grids : np.ndarray
        Curvilinear grids for each dimension.
    polarity : np.ndarray
        Boolean array indicating which solution polarity to use per sector.

    """

    def __init__(self, values, grids, backend="scipy", options=None):
        """Initialize a Curvilinear2DInterp object."""
        if backend not in MAP_COORDS:
            available = list(MAP_COORDS.keys())
            msg = f"Backend {backend!r} not supported for Curvilinear2DInterp. Available: {available}"
            raise NotImplementedError(msg)
        # Store grids as numpy for internal coordinate finding algorithms
        # The parent class will convert values to the backend type
        super().__init__(values, grids, backend=backend)
        # Keep numpy copy of grids for internal use
        self._grids_np = np.asarray(grids)
        # Use short kwargs for JAX (doesn't support output/prefilter)
        self.mc_kwargs = update_mc_kwargs(options, jax=(backend == "jax"))
        self._update_polarity()

    def _update_polarity(self):
        """Determine the polarity for each sector.

        Determines whether the "plus" (True) or "minus" (False) solution
        of the quadratic system of equations should be used for each sector.

        The polarity is determined by testing the midpoint of each sector
        and checking if the resulting (alpha, beta) coordinates fall within
        the unit square [0, 1] x [0, 1].
        """
        g0 = self._grids_np[0]
        g1 = self._grids_np[1]
        s0m1 = self.shape[0] - 1
        s1m1 = self.shape[1] - 1
        size = s0m1 * s1m1

        # Grab midpoint of each sector
        x_temp = np.reshape(0.5 * (g0[:-1, :-1] + g0[1:, 1:]), size)
        y_temp = np.reshape(0.5 * (g1[:-1, :-1] + g1[1:, 1:]), size)
        y_pos = np.tile(np.arange(s1m1), s0m1)
        x_pos = np.reshape(np.tile(np.arange(s0m1), (s1m1, 1)).T, size)

        # Test with "plus" polarity first
        self.polarity = np.ones((s0m1, s1m1), dtype=bool)
        alpha, beta = self._find_coords(x_temp, y_temp, x_pos, y_pos)

        # Update: use "minus" where (alpha, beta) not in unit square
        polarity = (alpha > 0) & (alpha < 1) & (beta > 0) & (beta < 1)
        self.polarity = polarity.reshape(s0m1, s1m1)

    def _find_sector(self, x, y):
        """Find the quadrilateral sector for each (x, y) point.

        Uses an iterative search algorithm that moves toward the correct
        sector by checking boundary violations.

        Parameters
        ----------
        x : np.ndarray
            X-coordinates of query points.
        y : np.ndarray
            Y-coordinates of query points. Same shape as x.

        Returns
        -------
        x_pos : np.ndarray
            Sector x-indices for each point.
        y_pos : np.ndarray
            Sector y-indices for each point.

        """
        m = x.size
        x_pos_guess = np.full(m, self.shape[0] // 2, dtype=int)
        y_pos_guess = np.full(m, self.shape[1] // 2, dtype=int)

        # Iterative search
        these = np.full(m, True)
        max_loops = sum(self.shape)
        loops = 0

        while np.any(these) and loops < max_loops:
            x_temp = x[these]
            y_temp = y[these]

            # Get vertex coordinates: A(0,0), B(1,0), C(0,1), D(1,1)
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
            x_coords = [
                self._grids_np[0][x_pos_guess[these] + dx, y_pos_guess[these] + dy]
                for dx, dy in offsets
            ]
            y_coords = [
                self._grids_np[1][x_pos_guess[these] + dx, y_pos_guess[these] + dy]
                for dx, dy in offsets
            ]

            # Check boundary violations (down, up, left, right)
            comps = [
                (y_temp, np.less, np.minimum, [y_coords[0], y_coords[1]]),
                (y_temp, np.greater, np.maximum, [y_coords[2], y_coords[3]]),
                (x_temp, np.less, np.minimum, [x_coords[0], x_coords[2]]),
                (x_temp, np.greater, np.maximum, [x_coords[1], x_coords[3]]),
            ]
            moves = [op(vec, func(*coords)) for vec, op, func, coords in comps]

            # Refined check for points passing simple tests
            c = np.sum(moves) == 0
            vertex_pairs = [(0, 0, 1, 1), (3, 3, 2, 2), (2, 2, 0, 0), (1, 1, 3, 3)]

            for i, (xi1, yi1, xi2, yi2) in enumerate(vertex_pairs):
                moves[i][c] = _violation_check(
                    x_temp[c],
                    y_temp[c],
                    x_coords[xi1][c],
                    y_coords[yi1][c],
                    x_coords[xi2][c],
                    y_coords[yi2][c],
                )

            # Update sector guess
            x_pos_next = x_pos_guess[these] - moves[2] + moves[3]
            x_pos_next = np.clip(x_pos_next, 0, self.shape[0] - 2)
            y_pos_next = y_pos_guess[these] - moves[0] + moves[1]
            y_pos_next = np.clip(y_pos_next, 0, self.shape[1] - 2)

            # Check convergence
            no_move = (x_pos_guess[these] == x_pos_next) & (
                y_pos_guess[these] == y_pos_next
            )
            x_pos_guess[these] = x_pos_next
            y_pos_guess[these] = y_pos_next
            temp = these.nonzero()
            these[temp[0][no_move]] = False
            loops += 1

        return x_pos_guess, y_pos_guess

    def _find_coords(self, x, y, x_pos, y_pos):
        """Calculate relative coordinates (alpha, beta) within sectors.

        Solves a system of bilinear equations to find where each point
        lies within its quadrilateral sector.

        Parameters
        ----------
        x : np.ndarray
            X-coordinates of query points.
        y : np.ndarray
            Y-coordinates of query points.
        x_pos : np.ndarray
            Sector x-indices.
        y_pos : np.ndarray
            Sector y-indices.

        Returns
        -------
        alpha : np.ndarray
            Relative horizontal position in [0, 1].
        beta : np.ndarray
            Relative vertical position in [0, 1].

        """
        # Get vertex coordinates
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        x_coords = [self._grids_np[0][x_pos + dx, y_pos + dy] for dx, dy in offsets]
        y_coords = [self._grids_np[1][x_pos + dx, y_pos + dy] for dx, dy in offsets]

        polarity = 2.0 * self.polarity[x_pos, y_pos] - 1.0

        # Bilinear coefficients
        a = x_coords[0]
        b = x_coords[1] - x_coords[0]
        c = x_coords[2] - x_coords[0]
        d = x_coords[0] - x_coords[1] - x_coords[2] + x_coords[3]
        e = y_coords[0]
        f = y_coords[1] - y_coords[0]
        g = y_coords[2] - y_coords[0]
        h = y_coords[0] - y_coords[1] - y_coords[2] + y_coords[3]

        # Solve quadratic system
        denom = d * g - h * c
        mu = (h * b - d * f) / denom
        tau = (h * (a - x) - d * (e - y)) / denom
        zeta = a - x + c * tau
        eta = b + c * mu + d * tau
        theta = d * mu

        alpha = (-eta + polarity * np.sqrt(eta**2 - 4.0 * zeta * theta)) / (
            2.0 * theta
        )
        beta = mu * alpha + tau

        # Handle degenerate sectors where the quadratic solution fails
        # This happens when:
        # 1. denom ≈ 0 (degenerate quadrilateral)
        # 2. Negative discriminant (shouldn't happen for valid interior points)
        # 3. theta ≈ 0 (parallel iso-beta lines, need linear solve instead)
        nan_mask = np.isnan(alpha) | np.isnan(beta)
        if np.any(nan_mask):
            # Check for parallel iso-beta lines (same slope on AB and CD edges)
            # Only apply fallback to NaN results that have this property
            with np.errstate(divide="ignore", invalid="ignore"):
                slope_ab = f / b  # slope of edge AB
                slope_cd = (y_coords[3] - y_coords[2]) / (x_coords[3] - x_coords[2])
            parallel_mask = nan_mask & np.isclose(slope_ab, slope_cd, equal_nan=True)

            if np.any(parallel_mask):
                # For parallel edges, use linear interpolation along iso-β lines
                kappa = f[parallel_mask] / b[parallel_mask]
                # y-intercepts of bottom (AB) and top (CD) edges
                int_bot = y_coords[0][parallel_mask] - kappa * x_coords[0][parallel_mask]
                int_top = y_coords[2][parallel_mask] - kappa * x_coords[2][parallel_mask]
                int_query = y[parallel_mask] - kappa * x[parallel_mask]
                # β from linear interpolation between edge intercepts
                beta_temp = (int_query - int_bot) / (int_top - int_bot)
                # x-coordinates of left and right boundaries at this β
                x_left = (
                    beta_temp * x_coords[2][parallel_mask]
                    + (1.0 - beta_temp) * x_coords[0][parallel_mask]
                )
                x_right = (
                    beta_temp * x_coords[3][parallel_mask]
                    + (1.0 - beta_temp) * x_coords[1][parallel_mask]
                )
                # alpha from linear interpolation between left and right boundaries
                alpha_temp = (x[parallel_mask] - x_left) / (x_right - x_left)
                beta[parallel_mask] = beta_temp
                alpha[parallel_mask] = alpha_temp

        return alpha, beta

    def __call__(self, x, y):
        """Evaluate the interpolated function at specified points.

        Parameters
        ----------
        x : array-like
            X-coordinates of query points.
        y : array-like
            Y-coordinates of query points. Must be broadcastable with x.

        Returns
        -------
        np.ndarray
            Interpolated function values at each (x, y) point.
            Shape matches the broadcast shape of x and y.

        """
        original_shape = np.asarray(x).shape
        xa = np.asarray(x).flatten()
        ya = np.asarray(y).flatten()

        x_pos, y_pos = self._find_sector(xa, ya)
        alpha, beta = self._find_coords(xa, ya, x_pos, y_pos)

        # Convert to fractional index coordinates for map_coordinates
        # The bilinear formula f(a,b) = sum(w_ij * f[i,j]) is equivalent to
        # map_coordinates with order=1 at coordinates (x_pos + a, y_pos + b)
        coords = np.array([x_pos + alpha, y_pos + beta])

        # Convert coords to backend-specific array type
        coords = asarray(coords, backend=self.backend)

        # Use map_coordinates for backend-agnostic bilinear interpolation
        result = map_coords(
            self.values, coords, backend=self.backend, **self.mc_kwargs
        )

        return result.reshape(original_shape)


def _violation_check(x, y, x1, y1, x2, y2):
    """Check if points violate a linear boundary.

    The boundary is defined by points (x1, y1) and (x2, y2), where the
    second point is counter-clockwise from the first.

    Parameters
    ----------
    x, y : np.ndarray
        Points to check.
    x1, y1 : np.ndarray
        First boundary point.
    x2, y2 : np.ndarray
        Second boundary point (counter-clockwise from first).

    Returns
    -------
    np.ndarray
        1 if point is outside boundary, 0 otherwise.

    """
    return ((y2 - y1) * x - (x2 - x1) * y > x1 * y2 - y1 * x2) + 0
