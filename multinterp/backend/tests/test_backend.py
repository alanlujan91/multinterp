from __future__ import annotations

import unittest

import numpy as np
from multinterp.backend._cupy import cupy_multinterp
from multinterp.backend._jax import jax_multinterp
from multinterp.backend._numba import numba_multinterp
from multinterp.backend._scipy import scipy_multinterp
from scipy.interpolate import interpn


def f_2d(u, v):
    return u * np.cos(u * v) + v * np.sin(u * v)


def f_3d(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


class Test2DInterpolation(unittest.TestCase):
    def setUp(self):
        self.grids = [np.linspace(0, 3, 8), np.linspace(0, 3, 11)]
        self.values = f_2d(*np.meshgrid(*self.grids, indexing="ij"))
        self.args = np.meshgrid(
            np.linspace(0, 3, 50), np.linspace(0, 3, 50), indexing="ij"
        )
        rgi_args = np.array([self.args[0].ravel(), self.args[1].ravel()]).T

        result_interpn = interpn(self.grids, self.values, rgi_args)
        self.true_values = result_interpn.reshape(50, 50)

    def test_scipy(self):
        result_multinterp = scipy_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_numba(self):
        result_multinterp = numba_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_cupy(self):
        result_multinterp = cupy_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_jax(self):
        result_multinterp = jax_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)


class Test3DInterpolation(unittest.TestCase):
    def setUp(self):
        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)
        z = np.linspace(7, 9, 33)
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)

        self.grids = [x, y, z]
        self.values = f_3d(xg, yg, zg)
        self.args = np.meshgrid(
            np.linspace(1, 4, 50),
            np.linspace(4, 7, 50),
            np.linspace(7, 9, 50),
            indexing="ij",
        )

        rgi_args = np.array(
            [self.args[0].ravel(), self.args[1].ravel(), self.args[2].ravel()]
        ).T
        result_interp = interpn(self.grids, self.values, rgi_args)
        self.true_values = result_interp.reshape(50, 50, 50)

    def test_scipy(self):
        result_multinterp = scipy_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_numba(self):
        result_multinterp = numba_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_cupy(self):
        result_multinterp = cupy_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)

    def test_jax(self):
        result_multinterp = jax_multinterp(self.grids, self.values, self.args)
        assert np.allclose(self.true_values, result_multinterp, atol=1e-05)
