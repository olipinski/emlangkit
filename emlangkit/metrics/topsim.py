"""Calculate topographic similarity for a given language."""
from typing import Tuple

import editdistance
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr


def compute_topographic_similarity(
    messages: np.ndarray,
    observations: np.ndarray,
    observations_dist_metric: str = "hamming",
    message_dist_metric: str = "editdistance",
) -> Tuple[float, float]:
    """
    Calculate the topographic similarity between the given messages and observations.

    Parameters
    ----------
    messages : np.ndarray
        Messages to calculate the topographic similarity for.
    observations : np.ndarray
        Observations to calculate the topographic similarity for.
    observations_dist_metric: Literal["editdistance", "cosine", "hamming", "jaccard", "euclidean"]
        Metric to use to calculate the distances between observations.
    message_dist_metric: Literal["editdistance", "cosine", "hamming", "jaccard", "euclidean"]
        Metric to use to calculate the distances between messages.

    Returns
    -------
    topsim_value : np.ndarray
        Topographic similarity score.
    """
    if message_dist_metric == "editdistance":

        def msg_metric(x, y):
            return editdistance.eval(x, y) / ((len(x) + len(y)) / 2)

    else:
        msg_metric = message_dist_metric

    # noinspection PyTypeChecker
    observations_dist = distance.pdist(observations, observations_dist_metric)
    # Even though they are ints treat as text
    messages_dist = distance.pdist(
        messages,
        msg_metric,
    )
    # noinspection PyTypeChecker
    topsim, pvalue = spearmanr(observations_dist, messages_dist, nan_policy="raise")
    return topsim, pvalue
