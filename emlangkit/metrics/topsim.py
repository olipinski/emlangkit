"""Calculate topographic similarity for a given language."""
from typing import Tuple

import editdistance
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr


def compute_topographic_similarity(
    messages: np.ndarray, observations: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate the topographic similarity between the given messages and observations.

    Parameters
    ----------
    messages : np.ndarray
        Messages to calculate the topographic similarity for.
    observations : np.ndarray
        Observations to calculate the topographic similarity for.

    Returns
    -------
    topsim_value : np.ndarray
        Topographic similarity score.
    """
    observations_dist = distance.pdist(observations, "hamming")
    # Even though they are ints treat as text
    messages_dist = distance.pdist(
        messages,
        lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
    )
    topsim, pvalue = spearmanr(observations_dist, messages_dist, nan_policy="raise")
    return topsim, pvalue
