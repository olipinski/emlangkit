"""Utilities for array operations."""
# Adapted from https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size

import numpy as np


def pad_jagged(array: np.ndarray, fill: int = 0) -> np.ndarray:
    """
    Append the minimal required amount of a given integer at the end of each array, such that it looses its jagedness.

    Parameters
    ----------
    array : np.ndarray
        Input array to be padded.
    fill : int
        Integer to pad the array with.

    Returns
    -------
    padded : np.ndarray
        Padded array.

    """
    maxlen = max(len(r) for r in array)
    padded = np.full((len(array), maxlen), fill_value=fill)
    for enu, row in enumerate(array):
        padded[enu, : len(row)] += row
    return padded
