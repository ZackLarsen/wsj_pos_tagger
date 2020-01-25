"""
Prepare test tuples for decoder
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
test = pd.read_csv(os.path.join(data_dir, 'wsj_test.csv'))

# Prepare the test sequences to feed to the decoder program:
# Turn dataframe into list of tuples:
test_tuples = list(map(tuple, test.values))

hcc_test_tuples = []
sequence_counter = 0
for a,b in test_prepped.values:
    if a == '<START>':
        sequence_counter += 1
    if a in prep_tuple.observation_map.keys():
        hcc_test_tuples.append((sequence_counter, prep_tuple.observation_map[a], prep_tuple.state_map[str(b)]))
    else:
        hcc_test_tuples.append((sequence_counter, prep_tuple.observation_map['<OOV>'], prep_tuple.state_map[str(b)]))

# Save as pickled list of tuples:
with open(os.path.join(data_dir, 'test_tuples.pickle'), 'wb') as f:
    pickle.dump(hcc_test_tuples, f)