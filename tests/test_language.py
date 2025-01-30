"""
Language tests.

Contains a suite of tests to evaluate the main Language class.
"""
import numpy as np
import pytest

from emlangkit import Language


def test_instantiations():
    # Check error for not numpy array
    with pytest.raises(ValueError, match=r".* numpy .*"):
        # noinspection PyTypeChecker
        Language(messages=[])

    # Check error for empty messages
    with pytest.raises(ValueError, match=r".* messages .*"):
        Language(messages=np.array([]))

    # Check error for not numpy array
    with pytest.raises(ValueError, match=r".* numpy .*"):
        # noinspection PyTypeChecker
        Language(messages=np.array([1, 1, 1]), observations=[])

    # Check error for empty observations, when provided
    with pytest.raises(ValueError, match=r".* observations .*"):
        Language(messages=np.array([1, 1, 1]), observations=np.array([]))


def test_language_metrics():
    test_msgs = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 0, 3],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 1, 3],
        ]
    )
    test_obs = np.array([[x, y] for x in range(4) for y in range(4)])

    lang = Language(messages=test_msgs, observations=test_obs)
    lang_no_obs = Language(messages=test_msgs)

    # Test no observation language
    with pytest.raises(ValueError):
        lang_no_obs.topsim()
    with pytest.raises(ValueError):
        lang_no_obs.mutual_information()
    with pytest.raises(ValueError):
        lang_no_obs.posdis()
    with pytest.raises(ValueError):
        lang_no_obs.bosdis()
    with pytest.raises(ValueError):
        lang_no_obs.observation_entropy()

    lang_no_obs.language_entropy()

    # Test language with observations provided
    lang.topsim()
    lang.mutual_information()
    lang.posdis()
    lang.bosdis()
    lang.observation_entropy()
    lang.language_entropy()

    # MPN
    lang.mpn()

    # NPMI
    lang.nc_npmi()

    # HAS
    lang.branching_entropy()
    lang.conditional_entropy()
    lang.boundaries(return_count=True, return_mean=True)
    lang.random_boundaries(return_count=True, return_mean=True)
    lang.segments(return_ids=True, return_hashed_segments=True)
    lang.random_segments(return_ids=True, return_hashed_segments=True)
    lang.has_stats(compute_topsim=True)

    # Test recomputing random stats
    lang.random_boundaries(recompute=True)
    lang.random_segments(recompute=True)
