"""Function for calculating the non-compositional NPMI."""
from collections import defaultdict

import numpy as np


def compute_nc_npmi(messages: np.ndarray, observations: np.ndarray) -> dict:
    """
    Calculate the non-compositional NPMI.

    This will find the correlations between messages and observations.
    It also allows for interpretation of the emergent language.

    Parameters
    ----------
    messages: np.ndarray
        The array of messages.
    observations: np.ndarray
        The array of observations.

    Returns
    -------
    non_compositional_npmi_dict: dict
        Dictionary of non-compositional messages, observations, and their respective NPMI values.
        The format is non_compositional_npmi_dict[msg][obs] = npmi_value.

    """
    msgs, _, msg_counts = np.unique(
        messages, return_counts=True, return_inverse=True, axis=0
    )
    msg_counts_dict = {f"{msg}": msg_counts[idx] for idx, msg in enumerate(msgs)}

    obs, _, obs_counts = np.unique(
        observations, return_counts=True, return_inverse=True, axis=0
    )
    obs_counts_dict = {
        f"{observation}": obs_counts[idx] for idx, observation in enumerate(obs)
    }

    joint_occurences_msg_obs = defaultdict(lambda: defaultdict(int))

    for msg, obs in zip(messages, observations):
        joint_occurences_msg_obs[f"{msg}"][f"{obs}"] += 1

    total_messages = messages.shape[0]
    total_observations = observations.shape[0]

    non_compositional_npmi_dict = defaultdict(dict)

    for msg in msg_counts_dict:
        msg_occurrences = msg_counts_dict[msg]
        msg_prob = msg_occurrences / total_messages
        for obs in obs_counts_dict:
            prob_obs = obs_counts_dict[obs] / total_observations
            joint_occurrences = joint_occurences_msg_obs[msg][obs]
            joint_prob = joint_occurrences / total_observations
            joint_self_inf = -np.log2(joint_prob)
            npmi = np.log2(joint_prob / (msg_prob * prob_obs)) / joint_self_inf
            non_compositional_npmi_dict[msg][obs] = npmi

    return non_compositional_npmi_dict
