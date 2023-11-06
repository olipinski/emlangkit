"""The Language class implementation."""
from typing import Optional

import numpy as np

import emlangkit.metrics as metrics


class Language:
    """
    The Language class makes calculations of the most commonly used EC metrics easier.

    It takes the messages and observations for an emergent language, and
    allows calculations of the most commonly used metrics.

    Parameters
    ----------
    messages : numpy.ndarray
        Numpy array containing the messages.
    observations : numpy.ndarray, optional
        Numpy array containing the observations. Default is None.
    seed : int, optional
        Seed value for random number generation. Default is 42.

    Examples
    --------
    Create a Language object with messages and observations:
    >>> messages = np.array([1, 2, 3, 4, 5])
    >>> observations = np.array([6, 7, 8, 9, 10])
    >>> lang = Language(messages, observations)

    Create a Language object with only messages and default seed:
    >>> messages = np.array([1, 2, 3, 4, 5])
    >>> lang = Language(messages)
    """

    def __init__(
        self,
        messages: np.ndarray,
        observations: Optional[np.ndarray] = None,
        prev_horizon: int = 8,
        seed: int = 42,
    ):
        if not isinstance(messages, np.ndarray):
            raise ValueError("Language only accepts numpy arrays!")

        if np.size(messages) == 0:
            raise ValueError("Empty messages passed!")

        if observations is not None:
            if not isinstance(observations, np.ndarray):
                raise ValueError("Language only accepts numpy arrays!")
            if np.size(observations) == 0:
                raise ValueError("Empty observations passed!")

        self.messages = messages
        self.observations = observations

        self.__rng = np.random.default_rng(seed=seed)

        # Placeholders
        self.__topsim_value = None
        self.__posdis_value = None
        self.__bosdis_value = None
        self.__langauge_entropy_value = None
        self.__observation_entropy_value = None
        self.__mutual_information_value = None

        # M_previous^n placeholders
        self.__mpn_value = None
        self.prev_horizon = prev_horizon
    def topsim(self) -> tuple[float, float]:
        """
        Calculate the topographic similarity score for the language.

        This method requires observations to be set in the class.

        Returns
        -------
            tuple of floats: The topographic similarity value, and the p-value.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError(
                "Observations are needed to calculate topographic similarity."
            )

        if self.__topsim_value is None:
            self.__topsim_value = metrics.compute_topographic_similarity(
                self.messages, self.observations
            )

        return self.__topsim_value

    def posdis(self):
        """
        Calculate the positional disentanglement score for the language.

        This method requires observations to be set.

        Returns
        -------
            float: The positional disentanglement score.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError(
                "Observations are needed to calculate positional disentanglement!"
            )
        if self.__posdis_value is None:
            self.__posdis_value = metrics.compute_posdis(
                self.messages, self.observations
            )

        return self.__posdis_value

    def bosdis(self):
        """
        Calculate the Bag-of-Words disentanglement score for the language.

        This method requires observations to be set.

        Returns
        -------
            float: The positional disentanglement score.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError(
                "Observations are needed to calculate bag-of-words disentanglement!"
            )
        if self.__bosdis_value is None:
            self.__bosdis_value = metrics.compute_bosdis(
                self.messages, self.observations
            )

        return self.__bosdis_value

    def language_entropy(self):
        """
        Calculate the entropy value for the language.

        This method requires observations to be set for calculating bag-of-words disentanglement.

        Returns
        -------
            float: The positional disentanglement value.

        Raises
        ------
            ValueError: If observations are not set.
        """
        # This may have been calculated previously
        if self.__langauge_entropy_value is None:
            self.__langauge_entropy_value = metrics.compute_entropy(self.messages)

        return self.__langauge_entropy_value

    def observation_entropy(self):
        """
        Calculate the entropy value for the observations.

        This method requires observations to be set.

        Returns
        -------
            float: The positional disentanglement value.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError(
                "Observations are needed to calculate observation entropy!"
            )
        # This may have been calculated previously
        if self.__observation_entropy_value is None:
            self.__observation_entropy_value = metrics.compute_entropy(
                self.observations
            )

        return self.__observation_entropy_value

    def mutual_information(self):
        """
        Calculate the mutual information value.

        This method requires observations to be set.

        Returns
        -------
            float: The mutual information value.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError("Observations are needed to calculate mutual information!")

        if self.__observation_entropy_value is None:
            self.observation_entropy()
        if self.__langauge_entropy_value is None:
            self.language_entropy()

        if self.__mutual_information_value is None:
            self.__mutual_information_value = metrics.compute_mutual_information(
                self.messages,
                self.observations,
                (self.__langauge_entropy_value, self.__observation_entropy_value),
            )

        return self.__mutual_information_value

    # M_previous_n metric

    def mpn(self):
        """
        Calculate the M_previous^n score for the language.

        This method requires observations to be set in the class.

        Returns
        -------
            float: The highest M_previous^n value.

        Raises
        ------
            ValueError: If observations are not set.
        """
        if self.observations is None:
            raise ValueError("Observations are needed to calculate M_previous^n.")

        if self.__mpn_value is None:
            self.__mpn_value = metrics.compute_mpn(
                self.messages, self.observations, self.prev_horizon
            )

        return self.__mpn_value
