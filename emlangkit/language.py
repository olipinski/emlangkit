from typing import Optional

import numpy as np

import emlangkit.metrics as metrics


class Language:
    def __init__(
        self,
        messages: np.ndarray,
        observations: Optional[np.ndarray],
    ):
        if len(messages) == 0 or len(messages[0]) == 0:
            raise ValueError("Empty messages passed!")

        if observations is not None:
            if len(observations) == 0 or len(observations[0]) == 0:
                raise ValueError("Empty messages passed!")

        self.messages = messages
        self.observations = observations

        # Placeholders
        self.topsim_value = None
        self.posdis_value = None
        self.bosdis_value = None
        self.langauge_entropy_value = None
        self.observation_entropy_value = None
        self.mutual_information_value = None
        self.has = None

    def topsim(self):
        if not self.observations:
            raise ValueError("Observations needed to calculate topographic similarity.")

        if not self.topsim_value:
            self.topsim_value = metrics.compute_topographic_similarity(
                self.messages, self.observations
            )

        return self.topsim_value

    def posdis(self):
        if not self.observations:
            raise ValueError(
                "Observations needed to calculate positional disentanglement!"
            )
        if not self.posdis_value:
            self.posdis_value = metrics.compute_posdis(self.messages, self.observations)

        return self.posdis_value

    def bosdis(self):
        if not self.observations:
            raise ValueError(
                "Observations needed to calculate bag-of-words disentanglement."
            )
        if not self.bosdis_value:
            self.bosdis_value = metrics.compute_bosdis(self.messages, self.observations)

        return self.bosdis_value

    def language_entropy(self):
        # This may have been calculated previously
        if not self.langauge_entropy_value:
            self.langauge_entropy_value = metrics.compute_entropy(self.messages)

        return self.langauge_entropy_value

    def observation_entropy(self):
        # This may have been calculated previously
        if not self.observation_entropy_value:
            self.observation_entropy_value = metrics.compute_entropy(self.observations)

        return self.observation_entropy_value

    def mutual_information(self):
        if not self.observation_entropy_value:
            self.observation_entropy()
        if not self.langauge_entropy_value:
            self.langauge_entropy()

        if not self.mutual_information_value:
            self.mutual_information_value = metrics.compute_mutual_information(
                self.messages,
                self.observations,
                (self.langauge_entropy_value, self.observation_entropy_value),
            )

        return self.mutual_information_value
