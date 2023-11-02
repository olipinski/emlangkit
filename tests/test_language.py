"""
Language tests.

Contains a suite of tests to evaluate the main Language class.
"""
import numpy as np
import pytest

from emlangkit import Language


def test_instantiations():
    with pytest.raises(ValueError, match=r".* numpy .*"):
        Language(messages=[])

    with pytest.raises(ValueError, match=r".* messages .*"):
        Language(messages=np.array([]))

    with pytest.raises(ValueError, match=r".* observations .*"):
        Language(messages=np.array([1, 1, 1]), observations=np.array([]))


def test_metrics():
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

    # Test no observation langauge
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

    # Test langauge with observations provided

    lang.topsim()
    lang.mutual_information()
    lang.posdis()
    lang.bosdis()
    lang.observation_entropy()
    lang.language_entropy()