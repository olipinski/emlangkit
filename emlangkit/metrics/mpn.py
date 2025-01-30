"""
The M_previous^n metric.

Adapted from https://arxiv.org/abs/2310.06555
"""

import numpy as np


def compute_mpn(
    messages: np.ndarray,
    observations: np.ndarray,
    prev_horizon: int,
    return_stats: bool = False,
):
    """
    Calculate the M_previous^n metric.

    This function assumes that the messages and observations are temporally ordered (i.e., index 0 is first timestep,
    last index is last timestep).

    The metric will be computed for all horizons up to and including prev_horizon, i.e., [1,prev_horizon].

    Parameters
    ----------
    messages : np.ndarray
        The temporally ordered messages.
    observations : np.ndarray
        The temporally ordered observations.
    prev_horizon : int
        The horizon up to which to calculate the metric.

    Returns
    -------
        mpn : np.ndarray
            The highest M_previous^n value for each horizon.
        msg_stats : dict
            The stats for each unique message. Only returned if `return_stats` is True.
    """
    msgs, inverse, msg_counts = np.unique(
        messages, return_counts=True, return_inverse=True, axis=0
    )

    msg_stats = {
        f"{msg}": {
            "count": msg_counts[idx],
            "same_as_previous_obj": np.zeros(shape=prev_horizon + 1, dtype=np.int32),
            "prev_use_percentage": np.zeros(shape=prev_horizon + 1, dtype=np.float32),
        }
        for idx, msg in enumerate(msgs)
    }

    # Times that the object was the same as the previous object
    # first we go from first to last observation
    for i in range(len(observations)):
        # Then starting at a given observation we look to the future to see if it repeats
        for horizon in range(1, prev_horizon + 1):
            # We cannot look beyond the end of the array
            if horizon + i >= len(observations):
                break
            # If it repeats, then we count it as a possible temporal reference for a given message
            if np.array_equal(observations[i], observations[horizon + i]):
                # We get the message index from the inverse of the messages
                # This looks complicated, but it's not too bad
                msg_stats[f"{msgs[inverse[horizon + i]]}"]["same_as_previous_obj"][
                    horizon
                ] += 1
                # Break, otherwise if there are multiple repeats in a horizon
                # They could get labelled twice, and incorrectly
                break

    mpn = np.zeros(shape=prev_horizon, dtype=np.float32)

    for msg in msg_stats:
        if msg_stats[f"{msg}"]["count"] > 0:
            for horizon in range(1, prev_horizon + 1):
                if msg_stats[f"{msg}"]["same_as_previous_obj"][horizon] != 0:
                    msg_stats[f"{msg}"]["prev_use_percentage"][horizon] = (
                        round(
                            msg_stats[f"{msg}"]["same_as_previous_obj"][horizon]
                            / msg_stats[f"{msg}"]["count"],
                            3,
                        )
                        * 100
                    )
                    if (
                        msg_stats[f"{msg}"]["prev_use_percentage"][horizon]
                        > mpn[horizon]
                    ):
                        mpn[horizon] = msg_stats[f"{msg}"]["prev_use_percentage"][
                            horizon
                        ]
                else:
                    msg_stats[f"{msg}"]["prev_use_percentage"][horizon] = 0

    if return_stats:
        return mpn, msg_stats
    else:
        return mpn
