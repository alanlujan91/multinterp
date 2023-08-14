from __future__ import annotations

import numpy as np

from multinterp import RegularMultInterp


def function(*args):
    mats = np.meshgrid(*args, indexing="ij")

    return np.sum(mats, axis=0)


class TestMultivariateInterp:
    def setUp(self):
        # create test data

        self.grids = [
            np.linspace(0, 1, 10),
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 12),
        ]

        self.args = [
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 12),
            np.linspace(0, 1, 13),
        ]

    def test_interpolation_values(self):
        # check that interpolation values match expected values

        interpolator2D_scipy = RegularMultInterp(
            function(*self.grids[0:2]), self.grids[0:2], backend="scipy"
        )
        interpolator2D_parallel = RegularMultInterp(
            function(*self.grids[0:2]), self.grids[0:2], backend="parallel"
        )
        interpolator3D_scipy = RegularMultInterp(
            function(*self.grids), self.grids, backend="scipy"
        )
        interpolator3D_parallel = RegularMultInterp(
            function(*self.grids), self.grids, backend="parallel"
        )

        val2D_scipy = interpolator2D_scipy(*np.meshgrid(*self.args[0:2], indexing="ij"))
        val2D_parallel = interpolator2D_parallel(
            *np.meshgrid(*self.args[0:2], indexing="ij")
        )
        val3D_scipy = interpolator3D_scipy(*np.meshgrid(*self.args, indexing="ij"))
        val3D_parallel = interpolator3D_parallel(
            *np.meshgrid(*self.args, indexing="ij")
        )

        assert np.allclose(val2D_scipy, function(*self.args[0:2]))
        assert np.allclose(val2D_parallel, function(*self.args[0:2]))
        assert np.allclose(val3D_scipy, function(*self.args))
        assert np.allclose(val3D_parallel, function(*self.args))
