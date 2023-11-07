"""
Zipf's Law statistics.

Adapted from https://openreview.net/forum?id=b4t9_XASt6G
"""

import itertools
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np


def zla(words: np.ndarray) -> Tuple[list, list]:
    """
    Compute Zipf's Law of Abbreviation (ZLA) statistics.

    Returns the mean word lengths and their frequencies, and just the raw frequencies.

    Parameters
    ----------
    words : numpy.ndarray
        A numpy array of words.

    Returns
    -------
    tuple : (list, list)
        The first element contains the mean length of words that have
        the same frequency of occurrence in the given words array.

        The second element of the tuple contains a list of frequencies, where
        each frequency represents the number of occurrences of a word in the given words array.
    """
    frequencies = []
    freq_to_lens = defaultdict(list)
    for word, freq in Counter(itertools.chain.from_iterable(words)).most_common():
        frequencies.append(freq)
        freq_to_lens[freq].append(len(words))
    zla_stats = [np.mean(freq_to_lens[freq]) for freq in frequencies]

    return zla_stats, frequencies
