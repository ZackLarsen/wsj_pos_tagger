"""

Script purpose: train hidden markov model (HMM) on hand-tagged Wall Street Journal corpus for
    part-of-speech (POS) tagging application.
Author: Zack Larsen
Date: January 10, 2020

Input:
- Preprocessed data, already pre-split into train and test sequence sets
-     and with <START> and <EOS> placeholders inserted to demarcate sequence
-     boundaries.

Output:
- compressed numpy array with multiple elements

Workflow:
- Read in the preprocessed dataset
-     Ensure <START> and <EOS> boundary tags have been added
-     Ensure <OOV> observation (token) has been applied
- Convert string observations and states into integers with mapping dictionary (SAVE)
- Create probability matrices (potentially with Decimal 28 precision)
- Convert probability matrices to log10 space
- Save log10 matrices to compressed numpy array

Details:
- Turn train_prepped into observation, state, and observation_state lists
-

"""

# Standard library imports
import os
import sys
import json
import pickle
import cProfile
from collections import Counter, defaultdict
from decimal import *

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
# Define paths
home_dir = os.path.dirname('WSJ_POS_TAGGING')
data_dir = os.path.join(home_dir, 'data')
sys.path.append(os.path.join(home_dir, 'src'))
from hmm_utils import *

# Data i/o
train_prepped = pd.read_csv(os.path.join(data_dir, 'train_prepped.csv'))
test_prepped = pd.read_csv(os.path.join(data_dir, 'test_prepped.csv'))

# Make sure <OOV> has already been handled in prep file:
train_prepped[train_prepped['token'] == '<OOV>']

# Begin HMM construction:
observation_list = list(train_prepped.token.values)
state_list = list(train_prepped.tag.values)
observation_state_list = list((token, tag) for token, tag in train_prepped.values)
prep_tuple = hmm_prep(observation_state_list)

# Define matrices
transitions = create_transitions(states = prep_tuple.unique_integer_states,
                                 bigram_counts = prep_tuple.bigram_counts,
                                 state_counts = prep_tuple.state_counts,
                                 n_states = prep_tuple.n_states)

emissions = create_emissions(observations = prep_tuple.unique_integer_observations,
                             states = prep_tuple.unique_integer_states,
                             observation_state_counts = prep_tuple.observation_state_counts,
                             observation_counts = prep_tuple.observation_counts,
                             n_observations = prep_tuple.n_observations,
                             n_states = prep_tuple.n_states)

Pi = create_pi(prep_tuple.state_map, transitions)

# Convert probs to log10 probs for transitions and emissions and Pi.
#  Allow negative infinity values for the zero probabilities that
#  incur divide-by-zero errors:
transitions_log = np.log10(transitions)
emissions_log = np.log10(emissions)
Pi_log = np.log10(Pi)

# Save in compressed format:
np.savez_compressed(os.path.join(data_dir, 'log_matrices'),
                    transitions=transitions_log,
                    emissions=emissions_log,
                    Pi=Pi_log)