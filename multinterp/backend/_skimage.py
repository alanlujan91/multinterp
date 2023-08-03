import numpy as np
from multinterp.core import _CurvilinearGridInterp
from skimage.transform import PiecewiseAffineTransform

DIM_MESSAGE = "Dimension mismatch."


class PiecewiseAffineInterp(_CurvilinearGridInterp):
    def __init__(self, values, grids, **kwargs):
        super().__init__(values, grids, target="cpu", **kwargs)

        source = np.reshape(self.grids, (self.ndim, -1)).T
        coordinates = np.mgrid[tuple(slice(0, dim) for dim in self.shape)]
        destination = np.reshape(coordinates, (self.ndim, -1)).T

        interpolator = PiecewiseAffineTransform()
        interpolator.estimate(source, destination)

        self.interpolator = interpolator

    def _get_coordinates(self, args):
        input = np.reshape(args, (self.ndim, -1)).T
        output = self.interpolator(input).T
        return output.reshape(args.shape)
