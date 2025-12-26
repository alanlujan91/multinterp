from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from multinterp.utilities import update_mc_kwargs

if TYPE_CHECKING:
    from collections.abc import Sequence


def as_tensor(arrs, device="cpu"):
    target_device = torch.device(device)

    if isinstance(arrs, (torch.Tensor, np.ndarray)):
        return torch.as_tensor(arrs, device=target_device)
    if isinstance(arrs, (list, tuple)) and isinstance(arrs[0], np.ndarray):
        arrs = np.asarray(arrs)
        return torch.as_tensor(arrs, device=target_device)
    msg = "arrs must be a numpy array, a torch tensor, or a list of these."
    raise TypeError(msg)


def torch_multinterp(grids, values, args, options=None):
    """Perform multivariate interpolation using PyTorch.

    Parameters
    ----------
    grids : list of array-like
        Grid points in the domain.
    values : array-like
        Functional values at the grid points.
    args : array-like
        Points at which to interpolate data.
    options : dict, optional
        Additional options for interpolation. Supports 'device' for GPU placement.

    Returns
    -------
    torch.Tensor
        Interpolated values of the function.

    """
    mc_kwargs = update_mc_kwargs(options)
    target_device = options.get("device", "cpu") if options else "cpu"

    args = as_tensor(args, device=target_device)
    values = as_tensor(values, device=target_device)
    grids = [as_tensor(grid, device=target_device) for grid in grids]

    coords = torch_get_coordinates(grids, args)
    return torch_map_coordinates(values, coords, **mc_kwargs)


def torch_gradinterp(grids, values, args, axis=None, options=None):
    """Computes the interpolated value of the gradient using PyTorch.

    Parameters
    ----------
    grids : list of array-like
        Grid points in the domain.
    values : array-like
        Functional values at the grid points.
    args : array-like
        Points at which to interpolate data.
    axis : int, optional
        Axis along which to compute the gradient.
    options : dict, optional
        Additional options for interpolation. Supports 'device' for GPU placement.

    Returns
    -------
    torch.Tensor
        Interpolated values of the gradient.

    """
    mc_kwargs = update_mc_kwargs(options)
    target_device = options.get("device", "cpu") if options else "cpu"

    args = as_tensor(args, device=target_device)
    values = as_tensor(values, device=target_device)
    grids = [as_tensor(grid, device=target_device) for grid in grids]

    coords = torch_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            msg = "Axis should be an integer."
            raise ValueError(msg)
        gradient = torch.gradient(values, spacing=(grids[axis],), dim=axis)[0]
        return torch_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = torch.gradient(values, spacing=tuple(grids))
    return torch.stack(
        [torch_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient],
    )


def torch_get_coordinates(grids, args):
    coords = torch.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = torch.arange(grid.numel(), device=grid.device)
        coords[dim] = torch_interp(args[dim], grid, grid_size)

    return coords


def torch_map_coordinates(values, coords, **kwargs):
    """Run map_coordinates using PyTorch tensors.

    Parameters
    ----------
    values : torch.Tensor
        Functional values from which to interpolate.
    coords : torch.Tensor
        Coordinates at which to interpolate values.
    **kwargs : dict
        Additional keyword arguments:
        - order : int (0 or 1, default 1)
        - mode : str ('constant', 'nearest', 'wrap', 'mirror', 'reflect')
        - cval : float (default 0.0)

    Returns
    -------
    torch.Tensor
        Interpolated values of the function.

    Note
    ----
    The 'output' and 'prefilter' kwargs from scipy are accepted but ignored.

    """
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    # Filter to only supported kwargs for torch map_coordinates
    supported_kwargs = {"order", "mode", "cval"}
    ignored_kwargs = {"output", "prefilter"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
    # Warn about truly unsupported kwargs (not just ignored ones)
    unsupported = set(kwargs.keys()) - supported_kwargs - ignored_kwargs
    if unsupported:
        warnings.warn(
            f"Unsupported kwargs for torch backend: {unsupported}. "
            f"Supported: {supported_kwargs}",
            stacklevel=2,
        )
    output = map_coordinates(values, coords, **filtered_kwargs)
    return output.reshape(original_shape)


def _mirror_index_fixer(index: torch.Tensor, size: int) -> torch.Tensor:
    s = size - 1
    return torch.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.floor((_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1) / 2)


_INDEX_FIXERS = {
    "constant": lambda index, _size: index,
    "nearest": lambda index, size: torch.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _round_half_away_from_zero(a: torch.Tensor) -> torch.Tensor:
    return torch.round(a) if torch.is_floating_point(a) else a


def _nearest_indices_and_weights(coordinate: torch.Tensor) -> list:
    index = torch.round(coordinate).to(torch.int32)
    weight = coordinate.new_ones(())
    return [(index, weight)]


def _linear_indices_and_weights(coordinate: torch.Tensor) -> list:
    lower = torch.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.to(torch.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _map_coordinates(
    input: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    order: int,
    mode: str,
    cval: float,
) -> torch.Tensor:
    if len(coordinates) != input.ndim:
        msg = f"coordinates must be a sequence of length {input.ndim}"
        raise ValueError(msg)

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        msg = f"Mode {mode} is not yet supported. Supported modes are {set(_INDEX_FIXERS.keys())}."
        raise NotImplementedError(msg)

    if mode == "constant":

        def is_valid(index, size):
            return (index >= 0) & (index < size)

    else:

        def is_valid(_index, _size):
            return True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        msg = "Currently requires order<=1"
        raise NotImplementedError(msg)

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinates, input.shape, strict=False):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items, strict=False)
        if all(valid is True for valid in validities):
            contribution = input[indices]
        else:
            all_valid = torch.all(torch.stack(validities), dim=0)
            contribution = torch.where(all_valid, input[indices], cval)
        outputs.append(torch.prod(torch.stack(weights), dim=0) * contribution)
    result = torch.sum(torch.stack(outputs), dim=0)
    if input.dtype == torch.int:
        result = _round_half_away_from_zero(result)
    return result.to(input.dtype)


def map_coordinates(
    input: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    order: int,
    mode: str = "nearest",
    cval: float = 0.0,
    _output=None,
    _prefilter=None,
) -> torch.Tensor:
    """Map coordinates using PyTorch tensors.

    Parameters
    ----------
    input : torch.Tensor
        Input array from which to interpolate.
    coordinates : Sequence[torch.Tensor]
        Coordinates at which to evaluate the input.
    order : int
        Order of interpolation (0=nearest, 1=linear).
    mode : str, optional
        Boundary mode: 'constant', 'nearest', 'wrap', 'mirror', 'reflect'.
        Default is 'nearest'.
    cval : float, optional
        Value for points outside boundaries when mode='constant'. Default 0.0.
    _output : None
        Unused, for API compatibility with scipy.
    _prefilter : None
        Unused, for API compatibility with scipy.

    Returns
    -------
    torch.Tensor
        Interpolated values.

    """
    return _map_coordinates(input, coordinates, order, mode, cval)


def torch_interp(x, xp, fp):
    """One-dimensional linear interpolation in PyTorch.

    Parameters
    ----------
        x: array_like
            The x-coordinates of the interpolated values.
        xp: 1-D sequence of floats
            The x-coordinates of the data points.
        fp: 1-D sequence of floats
            The y-coordinates of the data points, same length as xp.

    Returns
    -------
        array_like
            The interpolated values, same shape as x.

    """
    # Sort and get sorted indices
    sort_idx = torch.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[sort_idx]

    # Find bin indices: searchsorted returns insertion point, subtract 1 for left boundary
    # Then clamp to valid range [0, len(xp) - 2] for interpolation between idx and idx+1
    bin_indices = torch.clamp(torch.searchsorted(xp, x, right=True) - 1, 0, len(xp) - 2)

    # Compute weights and interpolate
    bin_diff = xp[bin_indices + 1] - xp[bin_indices]
    w2 = (x - xp[bin_indices]) / bin_diff

    # Clip weights to [0, 1] to match np.interp behavior (no extrapolation)
    w2 = torch.clamp(w2, 0.0, 1.0)

    return (1 - w2) * fp[bin_indices] + w2 * fp[bin_indices + 1]
