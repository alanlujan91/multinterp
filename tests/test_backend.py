from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interpn

from multinterp.backend._cupy import cupy_multinterp
from multinterp.backend._jax import jax_multinterp
from multinterp.backend._numba import numba_multinterp
from multinterp.backend._scipy import scipy_multinterp
from multinterp.backend._torch import torch_multinterp


def f_2d(u, v):
    """2D test function: u * cos(u * v) + v * sin(u * v)."""
    return u * np.cos(u * v) + v * np.sin(u * v)


def f_3d(x, y, z):
    """3D test function: 2 * x^3 + 3 * y^2 - z."""
    return 2 * x**3 + 3 * y**2 - z


@pytest.fixture
def setup_data_2d():
    """Fixture to set up 2D test data for interpolation.

    Returns:
        tuple: grids, values, args, true_values

    """
    grids = [np.linspace(0, 3, 8), np.linspace(0, 3, 11)]
    values = f_2d(*np.meshgrid(*grids, indexing="ij"))
    args = np.meshgrid(np.linspace(0, 3, 50), np.linspace(0, 3, 50), indexing="ij")
    rgi_args = np.array([args[0].ravel(), args[1].ravel()]).T

    result_interpn = interpn(grids, values, rgi_args)
    true_values = result_interpn.reshape(50, 50)

    return grids, values, args, true_values


def test_scipy_2d(setup_data_2d) -> None:
    """Test scipy backend for 2D interpolation."""
    grids, values, args, true_values = setup_data_2d
    result_multinterp = scipy_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_numba_2d(setup_data_2d) -> None:
    """Test numba backend for 2D interpolation."""
    grids, values, args, true_values = setup_data_2d
    result_multinterp = numba_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_cupy_2d(setup_data_2d) -> None:
    """Test cupy backend for 2D interpolation."""
    grids, values, args, true_values = setup_data_2d
    result_multinterp = cupy_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_jax_2d(setup_data_2d) -> None:
    """Test jax backend for 2D interpolation."""
    grids, values, args, true_values = setup_data_2d
    result_multinterp = jax_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_torch_2d(setup_data_2d) -> None:
    """Test torch backend for 2D interpolation."""
    grids, values, args, true_values = setup_data_2d
    result_multinterp = torch_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp.cpu(), atol=1e-05)


@pytest.fixture
def setup_data_3d():
    """Fixture to set up 3D test data for interpolation.

    Returns:
        tuple: grids, values, args, true_values

    """
    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    z = np.linspace(7, 9, 33)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)

    grids = [x, y, z]
    values = f_3d(xg, yg, zg)
    args = np.meshgrid(
        np.linspace(1, 4, 50),
        np.linspace(4, 7, 50),
        np.linspace(7, 9, 50),
        indexing="ij",
    )

    rgi_args = np.array([args[0].ravel(), args[1].ravel(), args[2].ravel()]).T
    result_interp = interpn(grids, values, rgi_args)
    true_values = result_interp.reshape(50, 50, 50)

    return grids, values, args, true_values


def test_scipy_3d(setup_data_3d) -> None:
    """Test scipy backend for 3D interpolation."""
    grids, values, args, true_values = setup_data_3d
    result_multinterp = scipy_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_numba_3d(setup_data_3d) -> None:
    """Test numba backend for 3D interpolation."""
    grids, values, args, true_values = setup_data_3d
    result_multinterp = numba_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_cupy_3d(setup_data_3d) -> None:
    """Test cupy backend for 3D interpolation."""
    grids, values, args, true_values = setup_data_3d
    result_multinterp = cupy_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_jax_3d(setup_data_3d) -> None:
    """Test jax backend for 3D interpolation."""
    grids, values, args, true_values = setup_data_3d
    result_multinterp = jax_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp, atol=1e-05)


def test_torch_3d(setup_data_3d) -> None:
    """Test torch backend for 3D interpolation."""
    grids, values, args, true_values = setup_data_3d
    result_multinterp = torch_multinterp(grids, values, args)
    assert np.allclose(true_values, result_multinterp.cpu(), atol=1e-05)
