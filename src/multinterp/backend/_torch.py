from __future__ import annotations

import functools
import itertools
import operator
from typing import Sequence as SequenceType

import torch

from multinterp.utilities import update_mc_kwargs


def torch_multinterp(grids, values, args, options=None):
    mc_kwargs = update_mc_kwargs(options)

    args = torch.asarray(args)
    values = torch.asarray(values)
    grids = [torch.asarray(grid) for grid in grids]

    coords = torch_get_coordinates(grids, args)
    return torch_map_coordinates(values, coords, **mc_kwargs)


def torch_gradinterp(grids, values, args, axis=None, options=None):
    mc_kwargs = update_mc_kwargs(options)
    eo = options.get("edge_order", 1) if options else 1

    args = torch.asarray(args)
    values = torch.asarray(values)
    grids = [torch.asarray(grid) for grid in grids]

    coords = torch_get_coordinates(grids, args)

    if axis is not None:
        if not isinstance(axis, int):
            msg = "Axis should be an integer."
            raise ValueError(msg)
        gradient = torch.gradient(values, grids[axis], axis=axis, edge_order=eo)
        return torch_map_coordinates(gradient, coords, **mc_kwargs)
    gradient = torch.gradient(values, *grids, edge_order=eo)
    return torch.asarray(
        [torch_map_coordinates(grad, coords, **mc_kwargs) for grad in gradient]
    )


def torch_get_coordinates(grids, args):
    coords = torch.empty_like(args)
    for dim, grid in enumerate(grids):
        grid_size = torch.arange(grid.size)
        coords[dim] = torch.interp(args[dim], grid, grid_size)

    return coords


def torch_map_coordinates(values, coords, **kwargs):
    original_shape = coords[0].shape
    coords = coords.reshape(len(values.shape), -1)
    output = map_coordinates(values, coords, **kwargs)
    return output.reshape(original_shape)


def _nonempty_prod(arrs: SequenceType[torch.Tensor]) -> torch.Tensor:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: SequenceType[torch.Tensor]) -> torch.Tensor:
    return functools.reduce(operator.add, arrs)


def _mirror_index_fixer(index: torch.Tensor, size: int) -> torch.Tensor:
    s = size - 1
    return torch.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.floor((_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1) / 2)


_INDEX_FIXERS = {
    "constant": lambda index, size: index,
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
    coordinates: SequenceType[torch.Tensor],
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

        def is_valid(index, size):
            return True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        msg = "Currently requires order<=1"
        raise NotImplementedError(msg)

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinates, input.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        if all(valid is True for valid in validities):
            contribution = input[indices]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = torch.where(all_valid, input[indices], cval)
        outputs.append(_nonempty_prod(weights) * contribution)
    result = _nonempty_sum(outputs)
    if input.dtype == torch.int:
        result = _round_half_away_from_zero(result)
    return result.to(input.dtype)


def map_coordinates(
    input: torch.Tensor,
    coordinates: SequenceType[torch.Tensor],
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
) -> torch.Tensor:
    return _map_coordinates(input, coordinates, order, mode, cval)
