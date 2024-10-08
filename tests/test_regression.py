from __future__ import annotations

import numpy as np
import pytest

from multinterp import RegressionUnstructuredInterp


def sum_first_axis(*args):
    """
    Sum the first axis of meshgrid arrays.

    Args:
        *args: Input arrays for meshgrid.

    Returns:
        numpy.ndarray: Sum of meshgrid arrays along the first axis.
    """
    mats = np.meshgrid(*args, indexing="ij")

    return np.sum(mats, axis=0)


@pytest.fixture()
def setup_data():
    """
    Fixture to set up test data for regression interpolation.

    Returns:
        tuple: grids, args
    """
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
    """
    Test regression interpolation values for 2D and 3D cases.

    Compares interpolated values with expected values using
    RegressionUnstructuredInterp.
    """
    # check that interpolation values match expected values

    grids, args = setup_data

    interpolator2D = RegressionUnstructuredInterp(
        sum_first_axis(*grids[:2]),
        [*np.meshgrid(*grids[:2], indexing="ij")],
        mod_options={"fit_intercept": False},
    )

    interpolator3D = RegressionUnstructuredInterp(
        sum_first_axis(*grids),
        [*np.meshgrid(*grids, indexing="ij")],
        mod_options={"fit_intercept": False},
    )

    val2D = interpolator2D(*np.meshgrid(*args[:2], indexing="ij"))

    val3D = interpolator3D(*np.meshgrid(*args, indexing="ij"))

    assert np.allclose(val2D, sum_first_axis(*args[:2]), rtol=0.01)
    assert np.allclose(val3D, sum_first_axis(*args), rtol=0.01)
