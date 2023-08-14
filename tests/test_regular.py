from __future__ import annotations

import numpy as np
import pytest

from multinterp import MultivariateInterp


def sum_first_axis(*args):
    mats = np.meshgrid(*args, indexing="ij")

    return np.sum(mats, axis=0)


@pytest.fixture()
def setup_data():
    # create test data
    grids = [
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 12),
    ]

    args = [
        np.linspace(0, 1, 11),
        np.linspace(0, 1, 12),
        np.linspace(0, 1, 13),
    ]

    return grids, args


def test_interpolation_values(setup_data):
    # check that interpolation values match expected values
    grids, args = setup_data

    interpolator2D_scipy = MultivariateInterp(
        sum_first_axis(*grids[0:2]), grids[0:2], backend="scipy"
    )
    interpolator2D_numba = MultivariateInterp(
        sum_first_axis(*grids[0:2]), grids[0:2], backend="numba"
    )
    interpolator3D_scipy = MultivariateInterp(
        sum_first_axis(*grids), grids, backend="scipy"
    )
    interpolator3D_numba = MultivariateInterp(
        sum_first_axis(*grids), grids, backend="numba"
    )

    val2D_scipy = interpolator2D_scipy(*np.meshgrid(*args[0:2], indexing="ij"))
    val2D_numba = interpolator2D_numba(*np.meshgrid(*args[0:2], indexing="ij"))
    val3D_scipy = interpolator3D_scipy(*np.meshgrid(*args, indexing="ij"))
    val3D_numba = interpolator3D_numba(*np.meshgrid(*args, indexing="ij"))

    assert np.allclose(val2D_scipy, sum_first_axis(*args[0:2]))
    assert np.allclose(val2D_numba, sum_first_axis(*args[0:2]))
    assert np.allclose(val3D_scipy, sum_first_axis(*args))
    assert np.allclose(val3D_numba, sum_first_axis(*args))
