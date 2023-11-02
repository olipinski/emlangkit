"""
Functions for Positional/BagOfWords Disentanglement.

Adapted from https://proceedings.neurips.cc/paper/2021/hash/c2839bed26321da8b466c80a032e4714-Abstract.html
"""

import numpy as np

from emlangkit.metrics.posdis import compute_posdis


def compute_bosdis(messages: np.ndarray, observations: np.ndarray) -> float:
    """
    Compute Bag-of-Words Disentanglement between the given messages and observations.

    Parameters
    ----------
    messages : np.ndarray
        Messages to calculate bag-of-words disentanglement for.
    observations : np.ndarray
        Observations to calculate bag-of-words disentanglement for.

    Returns
    -------
    bosdis : float
        Bag-of-words disentanglement score.
    """
    character_set = list(c for message in messages for c in message)
    vocab = {char: idx for idx, char in enumerate(character_set)}
    num_symbols = len(vocab)
    bow_message = []
    bow_observation = []
    for observation, message in zip(observations, messages):
        message_bow = [0 for _ in range(num_symbols)]
        for symbol in message:
            message_bow[list(vocab.keys()).index(symbol)] += 1
        message_bow = [str(symbol) for symbol in message_bow]
        bow_message.append(message_bow)
        bow_observation.append(observation)
    return compute_posdis(
        messages=np.array(bow_message), observations=np.array(bow_observation)
    )
