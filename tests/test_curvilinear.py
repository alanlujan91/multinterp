"""Tests for curvilinear grid interpolation backends."""

from __future__ import annotations

import numpy as np
import pytest


def f_2d(x, y):
    """2D test function: x^2 + y^2."""
    return x**2 + y**2


@pytest.fixture
def setup_warped_2d():
    """Set up test data for 2D warped grid interpolation.

    Creates a warped grid by applying a nonlinear transformation
    to a regular grid.

    Returns
    -------
    tuple
        (grids, values, test_args, reference_values)
    """
    # Create a regular grid
    n, m = 15, 20
    x_reg = np.linspace(0, 2, n)
    y_reg = np.linspace(0, 3, m)
    x_mesh, y_mesh = np.meshgrid(x_reg, y_reg, indexing="ij")

    # Apply a warping transformation
    x_warped = x_mesh + 0.1 * np.sin(y_mesh * np.pi / 3)
    y_warped = y_mesh + 0.1 * np.cos(x_mesh * np.pi / 2)

    # Create grids array (2, n, m)
    grids = np.array([x_warped, y_warped])

    # Evaluate function on the warped grid
    values = f_2d(x_warped, y_warped)

    # Test points (within the grid interior)
    test_x = np.linspace(0.2, 1.8, 10)
    test_y = np.linspace(0.3, 2.7, 10)
    test_args = (test_x, test_y)

    return grids, values, test_args


def test_warped2d_scipy(setup_warped_2d):
    """Test scipy backend for Warped2DInterp."""
    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d
    interp = Warped2DInterp(values, grids, backend="scipy")
    result = interp(*test_args)

    # Result should have the same shape as test points
    assert result.shape == test_args[0].shape
    # Results should be finite
    assert np.all(np.isfinite(result))


def test_warped2d_numba(setup_warped_2d):
    """Test numba backend for Warped2DInterp."""
    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d

    # Scipy result as reference
    scipy_interp = Warped2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(*test_args)

    # Numba result
    numba_interp = Warped2DInterp(values, grids, backend="numba")
    numba_result = numba_interp(*test_args)

    assert np.allclose(scipy_result, numba_result, rtol=1e-10)


def test_warped2d_cupy(setup_warped_2d):
    """Test cupy backend for Warped2DInterp."""
    cp = pytest.importorskip("cupy")
    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d

    # Scipy result as reference
    scipy_interp = Warped2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(*test_args)

    # CuPy result
    cupy_interp = Warped2DInterp(values, grids, backend="cupy")
    cupy_result = cupy_interp(*test_args)

    assert np.allclose(scipy_result, cp.asnumpy(cupy_result), rtol=1e-10)


def test_warped2d_jax(setup_warped_2d):
    """Test jax backend for Warped2DInterp."""
    pytest.importorskip("jax")

    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d

    # Scipy result as reference
    scipy_interp = Warped2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(*test_args)

    # JAX result
    jax_interp = Warped2DInterp(values, grids, backend="jax")
    jax_result = jax_interp(*test_args)

    assert np.allclose(scipy_result, np.asarray(jax_result), rtol=1e-5)


def test_warped2d_torch(setup_warped_2d):
    """Test torch backend for Warped2DInterp."""
    import warnings

    pytest.importorskip("torch")
    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d

    # Scipy result as reference
    scipy_interp = Warped2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(*test_args)

    # Torch result - convert test_args to array first to avoid warning
    test_args_array = np.array(test_args)
    torch_interp = Warped2DInterp(values, grids, backend="torch")

    # Suppress UserWarning from torch about tensor creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch_result = torch_interp(*test_args_array)

    assert np.allclose(scipy_result, torch_result.cpu().numpy(), rtol=1e-5)


def test_warped2d_invalid_backend():
    """Test that invalid backend raises error."""
    from multinterp.curvilinear import Warped2DInterp

    grids = np.random.rand(2, 5, 5)
    values = np.random.rand(5, 5)

    with pytest.raises(NotImplementedError, match="not supported"):
        Warped2DInterp(values, grids, backend="invalid")


def test_warped2d_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    from multinterp.curvilinear import Warped2DInterp

    grids = np.random.rand(2, 5, 5)
    values = np.random.rand(5, 5)
    interp = Warped2DInterp(values, grids)

    # Try to interpolate with wrong number of arguments
    with pytest.raises(ValueError, match="Number of arguments"):
        interp(np.array([1.0]))  # Only one argument instead of two


@pytest.fixture
def setup_curvilinear_2d():
    """Set up test data for Curvilinear2DInterp.

    Returns
    -------
    tuple
        (grids, values, test_x, test_y)
    """
    # Create a warped grid with sufficient deformation
    # to avoid degenerate sectors (division by zero in bilinear solve)
    n, m = 10, 12
    x_reg = np.linspace(0, 2, n)
    y_reg = np.linspace(0, 3, m)
    x_mesh, y_mesh = np.meshgrid(x_reg, y_reg, indexing="ij")

    # Stronger warping that varies in both dimensions to create non-degenerate sectors
    x_warped = x_mesh + 0.15 * np.sin(y_mesh * np.pi / 3) * np.cos(x_mesh * np.pi / 4)
    y_warped = y_mesh + 0.15 * np.cos(x_mesh * np.pi / 2) * np.sin(y_mesh * np.pi / 6)

    grids = np.array([x_warped, y_warped])
    values = f_2d(x_warped, y_warped)

    # Test points in the interior
    test_x = np.linspace(0.3, 1.7, 8)
    test_y = np.linspace(0.4, 2.6, 8)

    return grids, values, test_x, test_y


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_scipy(setup_curvilinear_2d):
    """Test Curvilinear2DInterp with scipy backend."""
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, test_x, test_y = setup_curvilinear_2d
    interp = Curvilinear2DInterp(values, grids, backend="scipy")
    result = interp(test_x, test_y)

    # Result should have the correct shape
    assert result.shape == test_x.shape
    # Results should be finite
    assert np.all(np.isfinite(result))


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_polarity(setup_curvilinear_2d):
    """Test that polarity is correctly computed."""
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, _, _ = setup_curvilinear_2d
    n, m = values.shape

    interp = Curvilinear2DInterp(values, grids)

    # Polarity should be a boolean array with shape (n-1, m-1)
    assert interp.polarity.shape == (n - 1, m - 1)
    assert interp.polarity.dtype == bool


def test_warped2d_axis_parameter(setup_warped_2d):
    """Test that axis parameter works correctly."""
    from multinterp.curvilinear import Warped2DInterp

    grids, values, test_args = setup_warped_2d

    interp = Warped2DInterp(values, grids, backend="scipy")

    # Test with axis=0
    result_axis0 = interp(*test_args, axis=0)
    # Test with axis=1 (default)
    result_axis1 = interp(*test_args, axis=1)

    # Both should return finite values
    assert np.all(np.isfinite(result_axis0))
    assert np.all(np.isfinite(result_axis1))


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_numba(setup_curvilinear_2d):
    """Test numba backend for Curvilinear2DInterp."""
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, test_x, test_y = setup_curvilinear_2d

    # Scipy result as reference
    scipy_interp = Curvilinear2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(test_x, test_y)

    # Numba result
    numba_interp = Curvilinear2DInterp(values, grids, backend="numba")
    numba_result = numba_interp(test_x, test_y)

    assert np.allclose(scipy_result, numba_result, rtol=1e-10)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_cupy(setup_curvilinear_2d):
    """Test cupy backend for Curvilinear2DInterp."""
    cp = pytest.importorskip("cupy")
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, test_x, test_y = setup_curvilinear_2d

    # Scipy result as reference
    scipy_interp = Curvilinear2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(test_x, test_y)

    # CuPy result
    cupy_interp = Curvilinear2DInterp(values, grids, backend="cupy")
    cupy_result = cupy_interp(test_x, test_y)

    assert np.allclose(scipy_result, cp.asnumpy(cupy_result), rtol=1e-10)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_jax(setup_curvilinear_2d):
    """Test jax backend for Curvilinear2DInterp."""
    pytest.importorskip("jax")
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, test_x, test_y = setup_curvilinear_2d

    # Scipy result as reference
    scipy_interp = Curvilinear2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(test_x, test_y)

    # JAX result
    jax_interp = Curvilinear2DInterp(values, grids, backend="jax")
    jax_result = jax_interp(test_x, test_y)

    assert np.allclose(scipy_result, np.asarray(jax_result), rtol=1e-5)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_curvilinear2d_torch(setup_curvilinear_2d):
    """Test torch backend for Curvilinear2DInterp."""
    pytest.importorskip("torch")
    from multinterp.curvilinear import Curvilinear2DInterp

    grids, values, test_x, test_y = setup_curvilinear_2d

    # Scipy result as reference
    scipy_interp = Curvilinear2DInterp(values, grids, backend="scipy")
    scipy_result = scipy_interp(test_x, test_y)

    # Torch result
    torch_interp = Curvilinear2DInterp(values, grids, backend="torch")
    torch_result = torch_interp(test_x, test_y)

    assert np.allclose(scipy_result, torch_result.cpu().numpy(), rtol=1e-5)


def test_curvilinear2d_invalid_backend():
    """Test that invalid backend raises error for Curvilinear2DInterp."""
    from multinterp.curvilinear import Curvilinear2DInterp

    grids = np.random.rand(2, 5, 5)
    values = np.random.rand(5, 5)

    with pytest.raises(NotImplementedError, match="not supported"):
        Curvilinear2DInterp(values, grids, backend="invalid")
