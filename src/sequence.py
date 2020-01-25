"""
Define sequence class

This file defines a class to represent sequence instances, which are
the individual units of the training data.

"""

# Standard library imports
import os
import sys
import json
from typing import Any, List
from dataclasses import dataclass, make_dataclass, field, fields

# 3rd party imports
import numpy as np
from sklearn.metrics import cohen_kappa_score
from dataclasses_json import dataclass_json

# local machine imports (if any)
home_dir = '/Users/zacklarsen/Zack_Master/Projects/WSJ_POS'
data_dir = os.path.join(home_dir, 'data')
sys.path.append(os.path.join(home_dir, 'src'))
from hmm_utils import viterbi

@dataclass_json
@dataclass
class sequence:
    """
    This class is meant to facilitate bookkeeping of model output for
     the hidden markov model.
    """
    seq_id: int
    sequence: List[int] = field(default_factory=list)
    actual_hidden_states: List[int] = field(default_factory=list)
    best_path_prob: float = None
    predicted_hidden_states: List[int] = field(default_factory=list)
    sequence_length: int = field(init=False)
    linear_kappa: float = None
    quad_kappa: float = None

    def __post_init__(self):
        """
        This function is simply here as a convenience so we can store
         the length of a sequence as a class instance attribute
         without having to enter the length as an argument in the
         instance creation call.
        :return: sequence_length: length of this sequence
        """
        self.sequence_length = len(self.sequence)

    def __repr__(self):
        # using the !r conversion flag to make sure the output
        # string uses repr instead of str:
        return f'sequence #({self.seq_id!r}, length = {self.sequence_length!r})'

    def __str__(self):
        return f'sequence #{self.seq_id}, length = {self.sequence_length}'

    def predict(self, test=False):
        """
        Perform inference on the hidden markov model for this test sequence.
        :return: predicted_hidden_states: a list of predicted hidden
        states for the observed states in this sequence.
        """
        if test:
            best_path_prob, predicted_hidden_states = viterbi_tester(self.sequence)
        elif not test:
            best_path_prob, predicted_hidden_states = viterbi(self.sequence)
        self.best_path_prob = best_path_prob
        self.predicted_hidden_states = predicted_hidden_states

    def evaluate(self):
        """
        Calculate the kappa score between this sequence's actual hidden
         states and the hidden states predicted by the hidden markov model.
        :return:
        """
        if self.predicted_hidden_states:
            self.linear_kappa = cohen_kappa_score(self.actual_hidden_states, self.predicted_hidden_states, weights = "linear")
            self.quad_kappa = cohen_kappa_score(self.actual_hidden_states, self.predicted_hidden_states, weights = "quadratic")
        elif not self.predicted_hidden_states:
            print("You have not provided any predicted states yet! Please run sequence through model before trying"
                  " to calculate kappa score.")

    def viterbi_tester(sequence):
        """
        This is a placeholder for now. In the future this will be the
         actual viterbi decoding algorithm which takes an event sequence
          and returns the probability of the best path of hidden states
          along with the path of hidden states itself.
        :param sequence: List of events
        :return: best_path_prob, predicted_hidden_states
        """
        best_path_prob = 0.87
        predicted_hidden_states = [i ** 2 for i in sequence]
        return (best_path_prob, predicted_hidden_states)

