"""
Functions for Positional/BagOfWords Disentanglement.

Adapted from https://proceedings.neurips.cc/paper/2021/hash/c2839bed26321da8b466c80a032e4714-Abstract.html
"""

import numpy as np

from emlangkit.metrics.entropy import compute_entropy
from emlangkit.metrics.mutual_information import compute_mutual_information


def compute_posdis(messages: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Positional Disentanglement between the given messages and observations.

    Parameters
    ----------
    messages : np.ndarray
        Messages to calculate positional disentanglement for.
    observations : np.ndarray
        Observations to calculate positional disentanglement for.

    Returns
    -------
    posdis : float
        Positional disentanglement score.
    """
    disentanglement_scores = []
    non_constant_positions = 0

    for j in range(len(messages[0])):
        symbols_j = [message[j] for message in messages]
        symbol_mutual_info = []
        symbol_entropy = compute_entropy(np.array(symbols_j))
        for i in range(len(observations[0])):
            concepts_i = [observation[i] for observation in observations]
            mutual_info = compute_mutual_information(
                np.array([concepts_i]).T, np.array([symbols_j]).T
            )
            symbol_mutual_info.append(mutual_info)
        symbol_mutual_info.sort(reverse=True)

        if symbol_entropy > 0:
            disentanglement_score = (
                symbol_mutual_info[0] - symbol_mutual_info[1]
            ) / symbol_entropy
            disentanglement_scores.append(disentanglement_score)
            non_constant_positions += 1
    if non_constant_positions > 0:
        return sum(disentanglement_scores) / non_constant_positions
    else:
        return float("nan")
