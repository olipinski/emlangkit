"""
Metrics tests.

Contains a suite of tests to evaluate the correctness of the calculations done in metrics.py.
"""
import numpy as np

from emlangkit import metrics


def test_entropy():
    """Tests to check if the message entropy is calculated correctly."""
    np.testing.assert_almost_equal(
        metrics.compute_entropy(
            x=np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        1.0,
        5,
    )
    np.testing.assert_almost_equal(
        metrics.compute_entropy(
            x=np.array(
                [
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 0],
                    [1, 0, 0, 1, 2],
                ]
            )
        ),
        1.5,
        5,
    )
    np.testing.assert_almost_equal(
        metrics.compute_entropy(
            x=np.array(
                [
                    [1, 2, 0, 3, 4],
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 0],
                    [1, 0, 0, 1, 2],
                ]
            ),
        ),
        2.0,
        5,
    )


def test_topsim():
    """Tests to see if topographic similarity is calculated correctly."""
    test_obs = np.array([[x, y] for x in range(4) for y in range(4)])

    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [1, 2, 0, 3, 4],
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 0],
                    [1, 0, 0, 1, 2],
                ]
            ),
            observations=np.array([[4, 4], [4, 3], [3, 2], [1, 4]]),
        )[0],
        0.32,
        2,
    )

    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [1, 2, 0, 3, 4],
                    [1, 2, 0, 3, 4],
                    [1, 2, 0, 3, 4],
                    [1, 2, 0, 3, 4],
                ]
            ),
            observations=np.array([[4, 4], [4, 3], [3, 2], [1, 4]]),
        )[0],
        np.NAN,
        2,
    )

    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [1, 2, 0, 3, 4],
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 0],
                    [1, 0, 0, 1, 2],
                ]
            ),
            observations=np.array([[4, 4], [4, 3], [4, 4], [1, 4]]),
        )[0],
        0.43,
        2,
    )

    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [1, 2, 0, 3, 4],
                    [1, 2, 2, 3, 4],
                    [1, 2, 2, 3, 0],
                    [1, 0, 0, 1, 2],
                ]
            ),
            observations=np.array([[4, 4], [4, 4], [4, 4], [1, 4]]),
        )[0],
        0.90,
        2,
    )

    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
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
            ),
            observations=test_obs,
        )[0],
        0.82,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [2, 3, 0],
                    [2, 3, 1],
                    [2, 3, 2],
                    [2, 3, 3],
                    [3, 1, 0],
                    [3, 1, 1],
                    [3, 2, 1],
                    [3, 2, 1],
                ]
            ),
            observations=test_obs,
        )[0],
        0.75,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_topographic_similarity(
            messages=np.array(
                [
                    [0, 0, 4],
                    [0, 0, 5],
                    [0, 0, 6],
                    [0, 0, 7],
                    [1, 4, 1],
                    [1, 5, 1],
                    [1, 6, 1],
                    [1, 7, 1],
                    [2, 4, 2],
                    [2, 5, 2],
                    [2, 6, 2],
                    [2, 7, 2],
                    [3, 4, 3],
                    [3, 3, 5],
                    [3, 3, 6],
                    [3, 3, 7],
                ]
            ),
            observations=test_obs,
        )[0],
        0.75,
        2,
    )


def test_mi():
    """Tests to see if mutual information is calculated correctly."""
    test_obs = np.array([[x, y] for x in range(4) for y in range(4)])

    np.testing.assert_almost_equal(
        metrics.compute_mutual_information(
            messages=np.array(
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
            ),
            observations=test_obs,
        ),
        4.0,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_mutual_information(
            messages=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [2, 3, 0],
                    [2, 3, 1],
                    [2, 3, 2],
                    [2, 3, 3],
                    [3, 1, 0],
                    [3, 1, 1],
                    [3, 2, 1],
                    [3, 2, 1],
                ]
            ),
            observations=test_obs,
        ),
        3.875,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_mutual_information(
            messages=np.array(
                [
                    [0, 0, 4],
                    [0, 0, 5],
                    [0, 0, 6],
                    [0, 0, 7],
                    [1, 4, 1],
                    [1, 5, 1],
                    [1, 6, 1],
                    [1, 7, 1],
                    [2, 4, 2],
                    [2, 5, 2],
                    [2, 6, 2],
                    [2, 7, 2],
                    [3, 4, 3],
                    [3, 3, 5],
                    [3, 3, 6],
                    [3, 3, 7],
                ]
            ),
            observations=test_obs,
        ),
        4.0,
        2,
    )


def test_posdis():
    """Tests to see if positional disentanglement is calculated correctly."""
    test_obs = np.array([[x, y] for x in range(4) for y in range(4)])

    np.testing.assert_almost_equal(
        metrics.compute_posdis(
            messages=np.array(
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
            ),
            observations=test_obs,
        ),
        1.0,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_posdis(
            messages=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [2, 3, 0],
                    [2, 3, 1],
                    [2, 3, 2],
                    [2, 3, 3],
                    [3, 1, 0],
                    [3, 1, 1],
                    [3, 2, 1],
                    [3, 2, 1],
                ]
            ),
            observations=test_obs,
        ),
        0.81,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_posdis(
            messages=np.array(
                [
                    [0, 0, 4],
                    [0, 0, 5],
                    [0, 0, 6],
                    [0, 0, 7],
                    [1, 4, 1],
                    [1, 5, 1],
                    [1, 6, 1],
                    [1, 7, 1],
                    [2, 4, 2],
                    [2, 5, 2],
                    [2, 6, 2],
                    [2, 7, 2],
                    [3, 4, 3],
                    [3, 3, 5],
                    [3, 3, 6],
                    [3, 3, 7],
                ]
            ),
            observations=test_obs,
        ),
        0.43,
        2,
    )


def test_bosdis():
    """Tests to see if bag-of-words disentanglement is calculated correctly."""
    test_obs = np.array([[x, y] for x in range(4) for y in range(4)])

    np.testing.assert_almost_equal(
        metrics.compute_bosdis(
            messages=np.array(
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
            ),
            observations=test_obs,
        ),
        0.42,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_bosdis(
            messages=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [1, 2, 3],
                    [2, 3, 0],
                    [2, 3, 1],
                    [2, 3, 2],
                    [2, 3, 3],
                    [3, 1, 0],
                    [3, 1, 1],
                    [3, 2, 1],
                    [3, 2, 1],
                ]
            ),
            observations=test_obs,
        ),
        0.13,
        2,
    )
    np.testing.assert_almost_equal(
        metrics.compute_bosdis(
            messages=np.array(
                [
                    [0, 0, 4],
                    [0, 0, 5],
                    [0, 0, 6],
                    [0, 0, 7],
                    [1, 4, 1],
                    [1, 5, 1],
                    [1, 6, 1],
                    [1, 7, 1],
                    [2, 4, 2],
                    [2, 5, 2],
                    [2, 6, 2],
                    [2, 7, 2],
                    [3, 4, 3],
                    [3, 3, 5],
                    [3, 3, 6],
                    [3, 3, 7],
                ]
            ),
            observations=test_obs,
        ),
        1.0,
        2,
    )