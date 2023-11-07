"""
The Harris' Articulation Scheme based segmentation.

Adapted from https://openreview.net/forum?id=b4t9_XASt6G
"""

import itertools
from collections import Counter
from typing import List, Tuple

import numpy as np


def has_init(messages: np.ndarray) -> Tuple[set, Counter]:
    """
    Compute initial values used by the other HAS functions.

    Parameters
    ----------
    messages : numpy.ndarray
        The array of messages.

    Returns
    -------
    alpha : set
        The set of unique characters present in the messages.
    freq : Counter
        A Counter containing all sequences and their corresponding frequencies.

    """
    # Create the alphabet
    alpha = set(np.unique(messages))
    # Count all subsequences
    freq = Counter(
        tuple(s[i:j])
        for s in messages
        for i in range(len(s))
        for j in range(i + 1, len(s) + 1)
    )
    # The frequency of empty sequence is defined as follows.
    # This is just for the convenience.
    freq[tuple()] = sum(len(s) for s in messages)

    return alpha, freq


def compute_branching_entropy(alpha, freq):
    """
    Calculate the branching entropy for a given alphabet, with given frequencies of each item.

    Parameters
    ----------
    alpha : set
        The set of unique characters present in the messages.
    freq : Counter
        A dictionary containing sequences as keys and their corresponding frequencies as values.

    Returns
    -------
    branching_entropy : dict
        Dictionary mapping contexts to their corresponding branching entropy.
    """
    branching_entropy = dict()
    for context, context_freq in freq.items():
        succ_freq_list = [freq[context + (a,)] for a in alpha]
        branching_entropy[context] = (
            -1
            * sum(
                succ_freq * (np.log2(succ_freq) - np.log2(context_freq))
                for succ_freq in succ_freq_list
                if succ_freq > 0
            )
            / context_freq
        )
    return branching_entropy


def compute_conditional_entropy(branching_entropy, freq) -> dict:
    """
    Compute conditional entropy of a given alphabet, given the branching entropy and the character frequencies.

    Parameters
    ----------
    branching_entropy : dict
        A dictionary containing sequences as keys and their corresponding branching entropy values as values.
    freq : dict
        A dictionary containing sequences as keys and their corresponding frequencies as values.


    Returns
    -------
    dict
        A dictionary containing the conditional entropy for each sequence length.
        The keys are sequence lengths and the values are the corresponding conditional entropy values.
    """
    conditional_entropy = dict()
    length_to_total_freq = dict()
    for seq, ent in branching_entropy.items():
        seq_len = len(seq)
        if seq_len not in conditional_entropy:
            conditional_entropy[seq_len] = 0
        if seq_len not in length_to_total_freq:
            length_to_total_freq[seq_len] = 0
        conditional_entropy[seq_len] += freq[seq] * ent
        length_to_total_freq[seq_len] += freq[seq]
    for length, total_freq in length_to_total_freq.items():
        conditional_entropy[length] /= total_freq
    return conditional_entropy


def compute_boundaries(
    messages: np.ndarray, branching_entropy: dict, threshold: float
) -> List[set]:
    """
    Compute the boundaries of a language, given its pre-computed branching entropy and a threshold value.

    Parameters
    ----------
    messages : numpy.ndarray
        A numpy array containing the input messages.
    branching_entropy : dict
        The branching entropy for each context in the messages.
    threshold : float
        The threshold value used for determining the boundaries.

    Returns
    -------
    boundaries : List[set]
        A list of sets, where each set represents the boundary positions in each message.

    Notes
    -----
    This method computes the boundaries in a list of messages based on the branching entropy and a threshold value.
    The boundaries are determined by comparing the branching entropy of each context with the previous context.
    If the difference is greater than the threshold, a boundary is added at the position.
    The algorithm starts with a width of 2, assuming that the branching entropy has already been computed.

    """
    boundaries = []
    for d in messages:
        boundaries.append(set())
        start: int = 0
        width: int = 2
        """
        We begin with width=2, while the algorithm in the paper begins with width=1.
        It is because this code block assumes that self.branching_entropy is already computed.
        """
        while start < len(d):
            context = tuple(d[start : start + width])
            if branching_entropy[context] - branching_entropy[context[:-1]] > threshold:
                boundaries[-1].add(start + width)
            if start + width + 1 < len(d):
                width += 1
            else:
                start += 1
                width = 2
    return boundaries


def compute_segments(
    messages: np.ndarray, boundaries: List[set]
) -> Tuple[list, dict, list]:
    """
    Compute language segments given the pre-computed boundaries.

    Parameters
    ----------
    messages : numpy.ndarray
        An array containing the messages to be segmented.

    boundaries : List[set]
        A list representing the boundaries for segmentation. Each element of
        this iterable represents the positions where the messages will be split.

    Returns
    -------
    segments : list
        A list of tuples containing the segmented messages. Each tuple represents
        a segment of the message.

    segment_ids : dict
        A dictionary mapping each unique segment to its corresponding ID. The ID
        is calculated based on the order of occurrence in the segments list.

    hashed_segments : list
        A list of tuples containing the hashed versions of the segmented messages.
        Each element in the tuple represents the ID of a segment.

    """
    segs = []
    for data, boundaries in zip(messages, boundaries):
        segs.append([])
        bot = 0
        for top in sorted(boundaries | {len(data)}):
            word = tuple(data[bot:top])
            bot = top
            segs[-1].append(word)
    segments = [tuple(x) for x in segs]
    segment_ids = {
        s: i + 1
        for i, s in enumerate(
            {tuple(x) for x in itertools.chain.from_iterable(segments)}
        )
    }
    hashed_segments = [tuple(segment_ids[x] for x in s) for s in segments]
    return segments, segment_ids, hashed_segments


def compute_random_boundaries(
    messages: np.ndarray, boundaries, rng: np.random.Generator
) -> List[set]:
    """
    Compute random boundaries for a language, given pre-computed boundaries and a random number generator instance.

    Parameters
    ----------
    messages : np.ndarray
        The input array of messages.

    boundaries : list
        The input list of boundaries.

    rng : np.random.Generator
        The random number generator object.

    Returns
    -------
    random_boundaries : List[set]
        The list of randomly computed boundaries.
    """
    random_boundaries = [
        set(rng.choice(np.arange(1, len(data), dtype=np.int32), size=len(boundaries)))
        for data, boundaries in zip(messages, boundaries)
    ]
    return random_boundaries
