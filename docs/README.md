# About

**multinterp** is a Python library for multivariate interpolation, providing efficient tools for interpolating data on regular, curvilinear, and unstructured grids.

## Features

- Multiple backend support: scipy, numba, cupy, jax, torch
- Regular grid interpolation with `MultivariateInterp`
- Curvilinear grid interpolation with `PiecewiseAffineInterp`, `Warped2DInterp`
- Unstructured grid interpolation with `UnstructuredInterp`
- Pipeline-based interpolation with sklearn models

## Quick Start

```python
import numpy as np
from multinterp import MultivariateInterp

# Create grids and values
grids = [np.linspace(0, 1, 10), np.linspace(0, 1, 10)]
values = np.random.rand(10, 10)

# Create interpolator
interp = MultivariateInterp(values, grids)

# Interpolate at new points
x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
result = interp(x, y)
```
