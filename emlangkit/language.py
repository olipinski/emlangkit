"""The Language class implementation."""
from typing import Optional

import numpy as np

import emlangkit.metrics as metrics
import emlangkit.utils as utils


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
        has_threshold: float = 0.8,
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

        # NPMI placeholders
        self.__nc_npmi_dict = None

        # HAS placeholders
        self.has_threshold = has_threshold
        self.__alpha = None
        self.__freq = None
        self.__branching_entropy = None
        self.__conditional_entropy = None
        self.__boundaries = None
        self.__segments = None
        self.__segment_ids = None
        self.__hashed_segments = None
        self.__random_boundaries = None
        self.__random_segments = None
        self.__random_segment_ids = None
        self.__random_hashed_segments = None
        self.__has_stats = None

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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.observations is None:
            raise ValueError("Observations are needed to calculate mutual information!")

        if self.__mutual_information_value is None:
            if self.__observation_entropy_value is None:
                self.observation_entropy()
            if self.__langauge_entropy_value is None:
                self.language_entropy()
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

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.observations is None:
            raise ValueError("Observations are needed to calculate M_previous^n.")

        if self.__mpn_value is None:
            self.__mpn_value = metrics.compute_mpn(
                self.messages, self.observations, self.prev_horizon
            )

        return self.__mpn_value

    def nc_npmi(self) -> dict:
        """
        Calculate the non-compositional NPMI correlations for the language.

        This method requires observations to be set in the class.

        Returns
        -------
            dict: Dictionary of non-compositional messages, observations, and their respective NPMI values.

        Raises
        ------
            ValueError: If observations are not set.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.observations is None:
            raise ValueError("Observations are needed to calculate M_previous^n.")

        if self.__nc_npmi_dict is None:
            self.__nc_npmi_dict = metrics.compute_nc_npmi(
                self.messages, self.observations
            )

        return self.__nc_npmi_dict

    # Harris' Articulation Scheme metrics
    def branching_entropy(self):
        """
        Calculate the branching entropy for a given language.

        Returns
        -------
            float: The calculated branching entropy value.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.__branching_entropy is None:
            if self.__freq is None:
                self.__alpha, self.__freq = metrics.has_init(self.messages)
            self.__branching_entropy = metrics.compute_branching_entropy(
                self.__alpha, self.__freq
            )

        return self.__branching_entropy

    def conditional_entropy(self):
        """
        Calculate the conditional entropy for a given language.

        Returns
        -------
        float
            The calculated conditional entropy value.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        # No need to even check for __freq as branching entropy already requires that
        if self.__conditional_entropy is None:
            if self.__branching_entropy is None:
                self.branching_entropy()
            self.__conditional_entropy = metrics.compute_conditional_entropy(
                self.__branching_entropy, self.__freq
            )

        return self.__conditional_entropy

    def boundaries(self, return_count: bool = False, return_mean: bool = False):
        """
        Calculate the HAS boundaries for a given language.

        Parameters
        ----------
        return_count : bool, optional
            If True, the method will return the boundaries and the count of each boundary.
            Default is False.

        return_mean : bool, optional
            If True, the method will return the boundaries, the count of each boundary,
            and the mean count. Default is False.

        Returns
        -------
        boundaries : list of lists
            A list of boundary lists for each message in the language.

        Optional Returns:
            If `return_count` is True, the method will also return `nb`, which is a list
            containing the count of each boundary.

            If `return_mean` is True, the method will also return `nb` and `mean`. `nb` is
            a list containing the count of each boundary, and `mean` is the mean count.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.__boundaries is None:
            if self.__branching_entropy is None:
                self.branching_entropy()
            self.__boundaries = metrics.compute_boundaries(
                self.messages, self.__branching_entropy, threshold=self.has_threshold
            )

        if return_count:
            nb = [len(b) for b in self.__boundaries]
            return self.__boundaries, nb

        if return_mean:
            nb = [len(b) for b in self.__boundaries]
            mean = np.mean(nb)
            return self.__boundaries, nb, mean

        return self.__boundaries

    def random_boundaries(
        self,
        return_count: bool = False,
        return_mean: bool = False,
        recompute: bool = False,
    ):
        """
        Calculate the random HAS boundaries for a given language.

        Parameters
        ----------
        return_count : bool, optional
            If True, returns the random boundaries along with the number of boundary items for each boundary.
            Default is False.
        return_mean : bool, optional
            If True, returns the random boundaries along with the number of boundary items for each boundary,
            as well as the mean number of boundary items across all boundaries.
            Default is False.
        recompute : bool, optional
            If True, forces the recomputation of the random boundaries.
            Default is False.

        Returns
        -------
        boundaries : list of lists
            A list of random boundary lists for each message in the language.

        Optional Returns:
            If `return_count` is True, the method will also return `nb`, which is a list
            containing the count of each boundary.

            If `return_mean` is True, the method will also return `nb` and `mean`. `nb` is
            a list containing the count of each boundary, and `mean` is the mean count.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.__random_boundaries is None and not recompute:
            if self.__boundaries is None:
                self.boundaries()
            self.__random_boundaries = metrics.compute_random_boundaries(
                self.messages, self.__boundaries, self.__rng
            )

        if return_count:
            nb = [len(b) for b in self.__random_boundaries]
            return self.__random_boundaries, nb

        if return_mean:
            nb = [len(b) for b in self.__random_boundaries]
            mean = np.mean(nb)
            return self.__random_boundaries, nb, mean

        return self.__random_boundaries

    def segments(self, return_ids: bool = False, return_hashed_segments: bool = False):
        """
        Calculate the HAS segments for a given language.

        Parameters
        ----------
        return_ids : bool, optional
            If True, returns the segments along with their corresponding segment ids.
            Default is False.

        return_hashed_segments : bool, optional
            If True, returns the segments along with their hashed versions.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of segments.

        Optional Returns:
            If `return_ids` is True, the method will also return segment_ids.
            If `return_hashed_segments` is True, the method will also return the hashed segments.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.

        """
        if self.__segments is None:
            if self.__boundaries is None:
                self.boundaries()
            (
                self.__segments,
                self.__segment_ids,
                self.__hashed_segments,
            ) = metrics.compute_segments(self.messages, self.__boundaries)

        if return_ids:
            return self.__segments, self.__segment_ids

        if return_hashed_segments:
            return self.__segments, self.__hashed_segments

        if return_ids and return_hashed_segments:
            return self.__segments, self.__segment_ids, self.__hashed_segments

        return self.__segments

    def random_segments(
        self,
        return_ids: bool = False,
        return_hashed_segments: bool = False,
        recompute: bool = False,
    ):
        """
        Calculate the random HAS segments for a given language.

        Parameters
        ----------
        return_ids : bool, optional
            Specifies whether to return segment IDs along with the segments. Default is False.
        return_hashed_segments : bool, optional
            Specifies whether to return hashed segments along with the segments. Default is False.
        recompute : bool, optional
            Specifies whether to recompute the random segments. Default is False.

        Returns
        -------
        numpy.ndarray
            Array of segments.

        Optional Returns:
            If `return_ids` is True, the method will also return segment_ids.
            If `return_hashed_segments` is True, the method will also return the hashed segments.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.
        """
        if self.__random_segments is None and not recompute:
            if self.__random_boundaries is None and not recompute:
                self.random_boundaries()
            (
                self.__random_segments,
                self.__random_segment_ids,
                self.__random_hashed_segments,
            ) = metrics.compute_segments(self.messages, self.__random_boundaries)

        if return_ids:
            return self.__random_segments, self.__random_segment_ids

        if return_hashed_segments:
            return self.__random_segments, self.__random_hashed_segments

        if return_ids and return_hashed_segments:
            return (
                self.__random_segments,
                self.__random_segment_ids,
                self.__random_hashed_segments,
            )

        return self.__random_segments

    def has_stats(self, compute_topsim: bool = False) -> dict:
        """
        Calculate the HAS statistics for a given language.

        Parameters
        ----------
        compute_topsim : bool, optional
            Flag indicating whether to compute topographic similarity. Default is False.

        Returns
        -------
        dict
            A dictionary containing various statistics related to the language.

        Raises
        ------
        ValueError
            If observations are None and compute_topsim is True.

        Notes
        -----
            The result is cached and will only be computed once.
            Subsequent calls to this method will return the cached value.

        """
        if self.__has_stats is None:
            if self.observations is None and compute_topsim:
                raise ValueError(
                    "Observations are needed to calculate topographic similarity."
                )

            zla, freq = metrics.zla(self.segments())
            random_zla, random_freq = metrics.zla(self.random_segments())

            # Pad the segments for topsim computation
            # We use 0 as it is not used in the has table
            # and has no effect on the distance measurement
            if compute_topsim:
                padded_hashed_segments = utils.pad_jagged(self.__hashed_segments)
                padded_random_hashed_segments = utils.pad_jagged(
                    self.__random_hashed_segments
                )

            self.__has_stats = {
                "vocab_size": len(self.__segment_ids),
                "zla": zla,
                "zipf": freq,
                # We use hamming here, as the segments could contain multiple characters
                # So editdistance would give us a worse estimate
                "topographic_similarity": metrics.compute_topographic_similarity(
                    padded_hashed_segments,
                    self.observations,
                    message_dist_metric="hamming",
                )
                if compute_topsim
                else None,
                "random_vocab_size": len(self.__random_segment_ids),
                "random_zla": random_zla,
                "random_zipf": random_freq,
                "random_topographic_similarity": metrics.compute_topographic_similarity(
                    padded_random_hashed_segments,
                    self.observations,
                    message_dist_metric="hamming",
                )
                if compute_topsim
                else None,
            }

        return self.__has_stats
