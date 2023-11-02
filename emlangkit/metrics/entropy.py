"""Calculate entropy for a given input."""

import numpy as np
from scipy.stats import entropy


def compute_entropy(x: np.ndarray, base: int = 2):
    """
    Calculate the entropy of the given input.

    This function assumes that the segmentation has already been performed.
    For example,

    Parameters
    ----------
    x : np.ndarray
        Input to calculate the entropy for.
    base : int, default=2
        Base to use for the entropy. The default is 2, for entropy measured in bits.

    Returns
    -------
    entropy : float
        Entropy measure.
    """
    x_s = [str(y) for y in x]
    _, count = np.unique(x_s, return_counts=True)
    return entropy(count, base=base)
