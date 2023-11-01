from typing import Union

import numpy as np
from scipy.stats import entropy


def compute_entropy(x: np.ndarray, base: int = 2):
    """
    Calculate the entropy of the given input.

    Parameters
    ----------
    x: Union[list, np.ndarray]
        Input to calculate the entropy for.
    base: int
        Base to use for the entropy.

    Returns
    -------
    entropy: float
        Entropy of the input.
    """
    x_s = [str(y) for y in x]
    _, count = np.unique(x_s, return_counts=True)
    return entropy(count, base=base)
