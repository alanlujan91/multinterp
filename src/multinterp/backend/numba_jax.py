from __future__ import annotations

import itertools

import numpy as np
from numba import njit

_INDEX_FIXERS = ("constant", "nearest", "wrap", "mirror", "reflect")


@njit
def _mirror_index_fixer(index: np.ndarray, size: int) -> np.ndarray:
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return np.abs((index + s) % (2 * s) - s)


@njit
def _reflect_index_fixer(index: np.ndarray, size: int) -> np.ndarray:
    return np.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


@njit
def _round_half_away_from_zero(a: np.ndarray) -> np.ndarray:
    return a if a.dtype.kind in "iu" else np.round(a)


@njit
def _nearest_indices_and_weights(coordinate: np.ndarray) -> np.ndarray:
    index = _round_half_away_from_zero(coordinate).astype(np.int32)
    weight = coordinate.dtype.type(1)
    return ((index, weight),)


@njit
def _linear_indices_and_weights(coordinate: np.ndarray) -> np.ndarray:
    lower = np.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(np.int32)
    return ((index, lower_weight), (index + 1, upper_weight))


def _map_coordinates(
    input: np.ndarray,
    coordinates: np.ndarray,
    order: int,
    mode: str,
    cval: np.ndarray,
) -> np.ndarray:
    input_arr = np.asarray(input)
    coordinate_arrs = np.asarray(coordinates)
    cval = np.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        msg = (
            f"coordinates must be a sequence of length input.ndim,"
            f"but {len(coordinates)} != {input_arr.ndim}"
        )
        raise ValueError(msg)

    if mode == "constant":

        def is_valid(index, size):
            return (index >= 0) & (index < size)

        def index_fixer(index, size):
            return index

    else:

        def is_valid(index, size):
            return True

        if mode == "nearest":

            def index_fixer(index, size):
                return np.clip(index, 0, size - 1)

        elif mode == "wrap":

            def index_fixer(index, size):
                return index % size

        elif mode == "mirror":
            index_fixer = _mirror_index_fixer

        elif mode == "reflect":
            index_fixer = _reflect_index_fixer

        else:
            msg = (
                f"map_coordinates does not yet support mode {mode}."
                f"Currently supported modes are {_INDEX_FIXERS}."
            )
            raise NotImplementedError(msg)

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        msg = "map_coordinates currently requires order<=1"
        raise NotImplementedError(msg)

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape, strict=True):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []

    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items, strict=True)
        if all(valid is True for valid in validities):
            # fast path
            contribution = input_arr[indices]
        else:
            all_valid = np.all(validities, axis=0)
            contribution = np.where(all_valid, input_arr[indices], cval)
        outputs.append(np.prod(weights, axis=0) * contribution)

    result = np.sum(outputs, axis=0)
    if np.issubdtype(input_arr.dtype, np.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_coordinates(
    input: np.ndarray,
    coordinates: np.ndarray,
    order: int,
    output: None,
    prefilter: None,
    mode: str = "constant",
    cval: np.ndarray = 0.0,
):
    return _map_coordinates(input, coordinates, order, mode, cval)
