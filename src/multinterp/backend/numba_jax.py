from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
from jax._src.typing import Array, ArrayLike


def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return np.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: Array, size: int) -> Array:
    return np.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


_INDEX_FIXERS: dict[str, Callable[[Array, int], Array]] = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: np.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
    return a if np.issubdtype(a.dtype, np.integer) else np.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
    index = _round_half_away_from_zero(coordinate).astype(np.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
    lower = np.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(np.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str,
    cval: ArrayLike,
) -> Array:
    input_arr = np.asarray(input)
    coordinate_arrs = [np.asarray(c) for c in coordinates]
    cval = np.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        msg = (
            "coordinates must be a sequence of length input.ndim, but {} != {}".format(
                len(coordinates), input_arr.ndim
            )
        )
        raise ValueError(msg)

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        msg = "jax.scipy.ndimage.map_coordinates does not yet support mode {}. Currently supported modes are {}.".format(
            mode, set(_INDEX_FIXERS)
        )
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
        msg = "jax.scipy.ndimage.map_coordinates currently requires order<=1"
        raise NotImplementedError(msg)

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
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
            # fast path
            contribution = input_arr[indices]
        else:
            all_valid = np.all(validities)
            contribution = np.where(all_valid, input_arr[indices], cval)
        outputs.append(np.prod(weights) * contribution)
    result = np.sum(outputs)
    if np.issubdtype(input_arr.dtype, np.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    order: int,
    mode: str = "constant",
    cval: ArrayLike = 0.0,
):
    return _map_coordinates(input, coordinates, order, mode, cval)
