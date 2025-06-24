from __future__ import annotations

import numpy as np

from multinterp.backend._numba import nb_interp_piecewise
from multinterp.grids import _CurvilinearGrid
from multinterp.utilities import asarray, empty, empty_like, interp, take


class Warped2DInterp(_CurvilinearGrid):
    """Warped Grid Interpolation on a 2D grid."""

    def __call__(self, *args, axis=1):
        """Interpolate on a warped grid using the Warped Grid Interpolation
        method described in `EGM$^n$`.

        Parameters
        ----------
        axis : int, 0 or 1
            Determines which axis to use for linear interpolators.
            Setting to 0 may fix some issues where interpolation fails.

        Returns
        -------
        np.ndarray
            Interpolated values on a warped grid.

        Raises
        ------
        ValueError
            Number of arguments doesn't match number of dimensions.

        """
        args = asarray(args, backend=self.backend)

        if args.shape[0] != self.ndim:
            msg = "Number of arguments must match number of dimensions."
            raise ValueError(msg)

        if self.backend in ["scipy", "cupy"]:
            output = self._interp_piecewise(args, axis)
        elif self.backend == "numba":
            output = self._backend_numba(args, axis)

        return output

    def _interp_piecewise(self, args, axis):
        """Uses numpy to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.

        """
        shape = args[0].shape  # original shape of arguments
        size = args[0].size  # number of points in arguments
        shape_axis = self.shape[axis]  # number of points in axis

        # flatten arguments by dimension
        args = args.reshape((self.ndim, -1))

        y_intermed = empty((shape_axis, size), self.backend)
        z_intermed = empty((shape_axis, size), self.backend)

        for i in range(shape_axis):
            # for each dimension, interpolate the first argument
            grids0 = take(self.grids[0], i, axis=axis, backend=self.backend)
            grids1 = take(self.grids[1], i, axis=axis, backend=self.backend)
            values = take(self.values, i, axis=axis, backend=self.backend)
            y_intermed[i] = interp(args[0], grids0, grids1, backend=self.backend)
            z_intermed[i] = interp(args[0], grids0, values, backend=self.backend)

        output = empty_like(args[0], self.backend)

        for j in range(size):
            y_temp = y_intermed[:, j]
            z_temp = z_intermed[:, j]

            if y_temp[0] > y_temp[-1]:
                # reverse
                y_temp = y_temp[::-1]
                z_temp = z_temp[::-1]

            output[j] = interp(args[1][j], y_temp, z_temp, backend=self.backend)

        return output.reshape(shape)

    def _backend_numba(self, args, axis):
        """Uses numba to interpolate on a warped grid.

        Parameters
        ----------
        args : np.ndarray
            Coordinates to be interpolated.
        axis : int, 0 or 1
            See `WarpedInterpOnInterp2D.__call__`.

        Returns
        -------
        np.ndarray
            Interpolated values on arguments.

        """
        return nb_interp_piecewise(args, self.grids, self.values, axis)

    def warmup(self) -> None:
        """Warms up the JIT compiler."""
        self(*self.grids)


class Curvilinear2DInterp(_CurvilinearGrid):
    """A 2D interpolation method for curvilinear or "warped grid" interpolation, as
    in White (2015).  Used for models with two endogenous states that are solved
    with the endogenous grid method.

    Parameters
    ----------
    values: numpy.array
        A 2D array of function values such that values[i,j] =
        f(x_values[i,j],y_values[i,j]).
    x_values: numpy.array
        A 2D array of x values of the same size as values.
    y_values: numpy.array
        A 2D array of y values of the same size as values.

    """

    def __init__(self, values, grids, backend="scipy") -> None:
        super().__init__(values, grids, backend=backend)
        self.update_polarity()

    def update_polarity(self) -> None:
        """Fills in the polarity attribute of the interpolation, determining whether
        the "plus" (True) or "minus" (False) solution of the system of equations
        should be used for each sector.  Needs to be called in __init__.

        Parameters
        ----------
        none

        Returns
        -------
        none

        """
        # Grab a point known to be inside each sector: the midway point between
        # the lower left and upper right vertex of each sector
        g0 = self.grids[0]
        g1 = self.grids[1]
        s0m1 = self.shape[0] - 1
        s1m1 = self.shape[1] - 1
        size = s0m1 * s1m1

        x_temp = np.reshape(0.5 * (g0[:-1, :-1] + g0[1:, 1:]), size)
        y_temp = np.reshape(0.5 * (g1[:-1, :-1] + g1[1:, 1:]), size)
        y_pos = np.tile(np.arange(s1m1), s0m1)
        x_pos = np.reshape(np.tile(np.arange(s0m1), (s1m1, 1)).T, size)

        # Set the polarity of all sectors to "plus", then test each sector
        self.polarity = np.ones((s0m1, s1m1), dtype=bool)
        alpha, beta = self.find_coords(x_temp, y_temp, x_pos, y_pos)
        polarity = (alpha > 0) & (alpha < 1) & (beta > 0) & (beta < 1)

        # Update polarity: if (alpha,beta) not in the unit square, then that
        # sector must use the "minus" solution instead
        self.polarity = polarity.reshape(s0m1, s1m1)

    def find_sector(self, x, y):
        """Finds the quadrilateral "sector" for each (x,y) point in the input.
        Only called as a subroutine of _evaluate().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.

        Returns
        -------
        x_pos : np.array
            Sector x-coordinates for each point of the input, of the same size.
        y_pos : np.array
            Sector y-coordinates for each point of the input, of the same size.

        """
        # Initialize the sector guess
        m = x.size
        x_pos_guess = np.full(m, self.shape[0] // 2, dtype=int)
        y_pos_guess = np.full(m, self.shape[1] // 2, dtype=int)

        # Define a function that checks whether a set of points violates a linear
        # boundary defined by (x_bound_1,y_bound_1) and (x_bound_2,y_bound_2),
        # where the latter is *COUNTER CLOCKWISE* from the former.  Returns
        # 1 if the point is outside the boundary and 0 otherwise.

        # Identify the correct sector for each point to be evaluated
        these = np.full(m, True)
        max_loops = sum(self.shape)
        loops = 0
        while np.any(these) and loops < max_loops:
            # Get coordinates for the four vertices: (x_coords[0],y_coords[0]),...,(x_coords[3],y_coords[3])
            x_temp = x[these]
            y_temp = y[these]

            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # A, B, C, D
            x_coords = [
                self.grids[0][x_pos_guess[these] + dx, y_pos_guess[these] + dy]
                for dx, dy in offsets
            ]
            y_coords = [
                self.grids[1][x_pos_guess[these] + dx, y_pos_guess[these] + dy]
                for dx, dy in offsets
            ]

            # Define checks list
            comps = [
                (y_temp, np.less, np.minimum, [y_coords[0], y_coords[1]]),  # down
                (y_temp, np.greater, np.maximum, [y_coords[2], y_coords[3]]),  # up
                (x_temp, np.less, np.minimum, [x_coords[0], x_coords[2]]),  # left
                (x_temp, np.greater, np.maximum, [x_coords[1], x_coords[3]]),  # right
            ]

            # Generate moves list using list comprehension
            moves = [op(vec, func(*coords)) for vec, op, func, coords in comps]

            # Check which boundaries are violated (and thus where to look next)
            c = np.sum(moves) == 0

            comps = [(0, 0, 1, 1), (3, 3, 2, 2), (2, 2, 0, 0), (1, 1, 3, 3)]

            for i in range(4):
                moves[i][c] = violation_check(
                    x_temp[c],
                    y_temp[c],
                    x_coords[comps[i][0]][c],
                    y_coords[comps[i][1]][c],
                    x_coords[comps[i][2]][c],
                    y_coords[comps[i][3]][c],
                )

            # Update the sector guess based on the violations
            x_pos_next = x_pos_guess[these] - moves[2] + moves[3]
            x_pos_next[x_pos_next < 0] = 0
            x_pos_next[x_pos_next > (self.shape[0] - 2)] = self.shape[0] - 2
            y_pos_next = y_pos_guess[these] - moves[0] + moves[1]
            y_pos_next[y_pos_next < 0] = 0
            y_pos_next[y_pos_next > (self.shape[1] - 2)] = self.shape[1] - 2

            # Check which sectors have not changed, and mark them as complete
            no_move = np.array(
                np.logical_and(
                    x_pos_guess[these] == x_pos_next,
                    y_pos_guess[these] == y_pos_next,
                ),
            )
            x_pos_guess[these] = x_pos_next
            y_pos_guess[these] = y_pos_next
            temp = these.nonzero()
            these[temp[0][no_move]] = False

            # Move to the next iteration of the search
            loops += 1

        # Return the output
        x_pos = x_pos_guess
        y_pos = y_pos_guess
        return x_pos, y_pos

    def find_coords(self, x, y, x_pos, y_pos):
        """Calculates the relative coordinates (alpha,beta) for each point (x,y),
        given the sectors (x_pos,y_pos) in which they reside.  Only called as
        a subroutine of __call__().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.
        x_pos : np.array
            Sector x-coordinates for each point in (x,y), of the same size.
        y_pos : np.array
            Sector y-coordinates for each point in (x,y), of the same size.

        Returns
        -------
        alpha : np.array
            Relative "horizontal" position of the input in their respective sectors.
        beta : np.array
            Relative "vertical" position of the input in their respective sectors.

        """
        # Calculate relative coordinates in the sector for each point

        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]  # A, B, C, D
        x_coords = [self.grids[0][x_pos + dx, y_pos + dy] for dx, dy in offsets]
        y_coords = [self.grids[1][x_pos + dx, y_pos + dy] for dx, dy in offsets]

        polarity = 2.0 * self.polarity[x_pos, y_pos] - 1.0
        a = x_coords[0]
        b = x_coords[1] - x_coords[0]
        c = x_coords[2] - x_coords[0]
        d = x_coords[0] - x_coords[1] - x_coords[2] + x_coords[3]
        e = y_coords[0]
        f = y_coords[1] - y_coords[0]
        g = y_coords[2] - y_coords[0]
        h = y_coords[0] - y_coords[1] - y_coords[2] + y_coords[3]

        denom = d * g - h * c
        mu = (h * b - d * f) / denom
        tau = (h * (a - x) - d * (e - y)) / denom
        zeta = a - x + c * tau
        eta = b + c * mu + d * tau
        theta = d * mu
        alpha = (-eta + polarity * np.sqrt(eta**2.0 - 4.0 * zeta * theta)) / (
            2.0 * theta
        )
        beta = mu * alpha + tau

        # Alternate method if there are sectors that are "too regular"
        z = np.logical_or(np.isnan(alpha), np.isnan(beta))
        # These points weren't able to identify coordinates
        if np.any(z):
            these = np.isclose(
                f / b,
                (y_coords[3] - y_coords[2]) / (x_coords[3] - x_coords[2]),
            )
            # iso-beta lines have equal slope
            if np.any(these):
                kappa = f[these] / b[these]
                int_bot = y_coords[0][these] - kappa * x_coords[0][these]
                int_top = y_coords[2][these] - kappa * x_coords[2][these]
                int_these = y[these] - kappa * x[these]
                beta_temp = (int_these - int_bot) / (int_top - int_bot)
                x_left = (
                    beta_temp * x_coords[2][these]
                    + (1.0 - beta_temp) * x_coords[0][these]
                )
                x_right = (
                    beta_temp * x_coords[3][these]
                    + (1.0 - beta_temp) * x_coords[1][these]
                )
                alpha_temp = (x[these] - x_left) / (x_right - x_left)
                beta[these] = beta_temp
                alpha[these] = alpha_temp

        return alpha, beta

    def __call__(self, x, y):
        """Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        xa = np.asarray(x).flatten()
        ya = np.asarray(y).flatten()

        x_pos, y_pos = self.find_sector(xa, ya)
        alpha, beta = self.find_coords(xa, ya, x_pos, y_pos)

        # Calculate the function at each point using bilinear interpolation
        f = (
            (1 - alpha) * (1 - beta) * self.values[x_pos, y_pos]
            + (1 - alpha) * beta * self.values[x_pos, y_pos + 1]
            + alpha * (1 - beta) * self.values[x_pos + 1, y_pos]
            + alpha * beta * self.values[x_pos + 1, y_pos + 1]
        )
        return f.reshape(np.asarray(x).shape)


def violation_check(x, y, x1, y1, x2, y2):
    return ((y2 - y1) * x - (x2 - x1) * y > x1 * y2 - y1 * x2) + 0
