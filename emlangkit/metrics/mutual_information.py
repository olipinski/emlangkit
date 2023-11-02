"""
Functions for Mutual information.

Adapted from https://proceedings.neurips.cc/paper/2021/hash/c2839bed26321da8b466c80a032e4714-Abstract.html
"""
from typing import Optional, Tuple

import numpy as np

from emlangkit.metrics.entropy import compute_entropy


def compute_mutual_information(
    messages: np.ndarray,
    observations: np.ndarray,
    entropies: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Compute mutual information between the given messages and observations.

    Parameters
    ----------
    messages : np.ndarray
        Messages to calculate the mutual information for.
    observations : np.ndarray
        Observations to calculate the mutual information for.
    entropies : Tuple[np.ndarray, np.ndarray], optional
        Pre-calculated entropies for messages and observations.

    Returns
    -------
    mi : np.ndarray
        Mutual information score.
    """
    if not entropies:
        message_entropy = compute_entropy(messages)
        observations_entropy = compute_entropy(observations)
    else:
        message_entropy = entropies[0]
        observations_entropy = entropies[1]
    messages_and_observations = np.concatenate(
        (np.array(observations), np.array(messages)), axis=1
    )
    messages_and_observations_joint_entropy = compute_entropy(messages_and_observations)
    return (
        observations_entropy + message_entropy - messages_and_observations_joint_entropy
    )
