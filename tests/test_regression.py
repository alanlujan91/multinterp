from __future__ import annotations

import numpy as np

from multinterp import RegressionUnstructuredInterp


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

        interpolator2D = RegressionUnstructuredInterp(
            function(*self.grids[0:2]),
            [*np.meshgrid(*self.grids[0:2], indexing="ij")],
            model_kwargs={"fit_intercept": False},
        )

        interpolator3D = RegressionUnstructuredInterp(
            function(*self.grids),
            [*np.meshgrid(*self.grids, indexing="ij")],
            model_kwargs={"fit_intercept": False},
        )

        val2D = interpolator2D(*np.meshgrid(*self.args[0:2], indexing="ij"))

        val3D = interpolator3D(*np.meshgrid(*self.args, indexing="ij"))

        assert np.allclose(val2D, function(*self.args[0:2]), rtol=0.01)
        assert np.allclose(val3D, function(*self.args), rtol=0.01)
