from __future__ import annotations

from skimage.transform import PiecewiseAffineTransform

from multinterp.grids import _CurvilinearGrid
from multinterp.rectilinear._multi import MultivariateInterp
from multinterp.utilities import mgrid, update_mc_kwargs


class PiecewiseAffineInterp(_CurvilinearGrid, MultivariateInterp):
    def __init__(self, values, grids, options=None):
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
        _input = args.reshape((self.ndim, -1)).T
        output = self.interpolator(_input).T.copy()
        return output.reshape(args.shape)
