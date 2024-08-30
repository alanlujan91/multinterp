from __future__ import annotations

from skimage.transform import PiecewiseAffineTransform

from multinterp.grids import _CurvilinearGrid
from multinterp.rectilinear._multi import MultivariateInterp
from multinterp.utilities import mgrid, update_mc_kwargs


class PiecewiseAffineInterp(_CurvilinearGrid, MultivariateInterp):
    """Curvilinear interpolator that uses the PiecewiseAffineTransform from scikit-image."""

    def __init__(self, values, grids, options=None):
        """Initialize a PiecewiseAffineInterp object.

        Parameters
        ----------
        values : np.ndarray
            Functional values on a curvilinear grid.
        grids : np.ndarray
            Coordinates of the points in the curvilinear grid.
        options : dict, optional
            Additional keyword arguments to pass to the map_coordinates backend.

        """
        super().__init__(values, grids, backend="scipy")
        self.mc_kwargs = update_mc_kwargs(options)

        source = self.grids.reshape((self.ndim, -1)).T
        coordinates = mgrid(
            tuple(slice(0, dim) for dim in self.shape),
            backend=self.backend,
        )
        destination = coordinates.reshape((self.ndim, -1)).T

        interpolator = PiecewiseAffineTransform()
        interpolator.estimate(source, destination)

        self.interpolator = interpolator

    def _get_coordinates(self, args):
        """Obtain the index coordinates for each of the arguments.

        Parameters
        ----------
        args : np.ndarray
            Arguments to be interpolated.

        Returns
        -------
        np.ndarray
            Index coordinates for each of the arguments.

        """
        _input = args.reshape((self.ndim, -1)).T
        output = self.interpolator(_input).T.copy()
        return output.reshape(args.shape)
