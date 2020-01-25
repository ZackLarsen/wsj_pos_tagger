"""
Define hmm (hidden markov model) class

This file defines a class to represent hidden markov models.

"""

# Standard library imports
import os
import sys
import json
from typing import Any, List
from dataclasses import dataclass, make_dataclass, field, fields
from dataclasses_json import dataclass_json

# 3rd party imports
import numpy as np

# local machine imports (if any)


@dataclass_json
@dataclass
class hmm:
    """
    This class is meant to represent a particular hidden markov model.
    It can be initialized using a string that uniquely identifies the model
    as well as training data in the form of a list of (observation, state) tuples
    or a generator that allows us to read the training data from a file without
    consuming high amounts of memory.

    Attributes:
        hmm_id
        observation_set
        n_obs
        obs_map
        state_set
        n_states
        state_map
        transitions_matrix_log
        emissions_matrix_log
        Pi_log

    Methods:
        __post_init__:
        make_transitions:
        make_emissions:
        make_pi:

    """
    hmm_id: str
    training_data_nt: namedtuple
    observation_set: set = None
    state_set: set = None
    n_obs: int = field(init=False)
    n_states: int = field(init=False)
    transitions_matrix: np.array = field(init=False)
    transitions_matrix_log: np.array = field(init=False)

    def __post_init__(self):
        """
        This function automatically initializes certain class attributes
         based on what is provided to the __init__ method.
        """
        self.observations = training_data_nt.obs
        self.states = training_data_nt.states
        self.observation_set = set(self.observations)
        self.n_obs = len(set(self.observations))
        self.state_set = set(self.states)
        self.n_states = len(set(self.states))

    def make_transitions(self, log_space = True):
        """
        Create the transitions matrix.
        :return: transitions_log: log10-based matrix of transitions.
        """
        transitions_matrix = np.zeros((n_states, n_states))
        for i, state_i in enumerate(unique_integer_states):
            for j, state_j in enumerate(unique_integer_states):
                bigram = (state_i, state_j)
                bigram_count = bigram_counts[bigram]
                unigram_count = state_counts[state_i]
                # a = bigram_count / (unigram_count + 0.1) # With smoothing
                a = bigram_count / (unigram_count)  # Without smoothing
                transitions_matrix[i, j] = a
        if log_space:
            self.transitions_matrix_log = np.log10(transitions_matrix)
        elif not log_space:
            self.transitions_matrix = transitions_matrix

    def make_emissions(self, log_space = True):
        """
        Create the emissions matrix.
        :return: emissions_log: log10-based matrix of emissions.
        """
        emissions_matrix = np.zeros((self.n_states, self.n_obs))
        for i, state in enumerate(self.states_set):
            for j, observation in enumerate(observations):
                tuple_count = tuple_counts[(observation, state)]
                state_count = state_counts[states[i]]
                b = tuple_count / state_count
                emissions_matrix[i, j] = b
        if log_space:
            self.emissions_matrix_log = np.log10(emissions_matrix)
        elif not log_space:
            self.emissions_matrix = emissions_matrix

    def make_pi(self, log_space = True):
        """
        Create the pi matrix.
        :return: pi_log: log10-based matrix of initial probabilities.
        """
        pi_matrix = np.zeros(self.n_states)
        for key, value in self.state_counts.items():
            pi_matrix[key] = value / sum(self.state_counts.values())
        if log_space:
            self.pi_matrix_log = np.log10(pi_matrix)
        elif not log_space:
            self.pi_matrix = pi_matrix

