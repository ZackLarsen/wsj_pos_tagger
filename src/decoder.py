"""
Part-of-speech tagger

This file is a command-line utility for part-of-speech tagging using
 an already trained hidden markov model.

Input data:
    state_map
    observation_map
    probability_matrices
    test_tuples

Output:
    pickle of sequence objects with predicted hidden states

Workflow:
    Read in input data
    Transform test tuples into instances of sequence class
    Call predict() method on all sequences to predict the hidden states
    Call evaluate() method on all sequences to store kappa score
    Print average kappa score to console for user to evaluate model's performance
"""

# Standard library imports
import os
import sys
import getopt
import argparse
import json
import pickle
from timeit import timeit
import cProfile
from typing import Any, List
from collections import namedtuple
from dataclasses import dataclass, make_dataclass, field, fields
import importlib

# 3rd party imports
import pandas as pd
import numpy as np
from numpy import NINF, inf
from numba import jit
from sklearn.metrics import cohen_kappa_score
from pympler import asizeof, asizeof
from tqdm import tqdm

# local machine imports (if any)
#from hmm_utils import viterbi
#from sequence import sequence

# Local package import unsuccessful. Change working directory:
home_dir = '/Users/zacklarsen/Zack_Master/Projects/WSJ_POS'
data_dir = os.path.join(home_dir, 'data')
sys.path.append(os.path.join(home_dir, 'src'))
#sys.path[-1]
#path = os.getcwd()
#print(path)
sys.path.append(os.path.join(home_dir, 'src'))
#sys.path.append('WSJ_POS_TAGGING')
#from hmm_utils import viterbi
from sequence import sequence

#importlib.reload(sequence)




# Import necessary data:
with open(os.path.join(data_dir, 'test_tuples.pickle'), 'rb') as f:
    test_sequences = pickle.load(f)

test_mids = list(set([tup[0] for tup in test_sequences]))

with open(os.path.join(data_dir, 'observation_map.json'), 'r') as fp:
    observation_map = json.load(fp)

with open(os.path.join(data_dir, 'state_map.json'), 'r') as fp:
    state_map = json.load(fp)

log_matrices = np.load(os.path.join(data_dir, 'log_matrices.npz'))
transitions_matrix_log = log_matrices['transitions']
emissions_matrix_log = log_matrices['emissions']
Pi_log = log_matrices['Pi']

# Transform test tuples into instances of sequence class:
#test_sequences
#len(test_mids) # 2012

# for seq in test_sequences[:50]:
#     print(f"Sequence #{seq[0]}: observation={seq[1]}, actual state={seq[2]}")

sequences = []
seq_id = test_sequences[0][0]
current_sequence = []
actual_hidden_states = []
for tup in test_sequences:
    if tup[0] == seq_id:
        #print(f"Current sequence: #{tup[0]}")
        current_sequence.append(tup[1])
        actual_hidden_states.append(tup[2])
    else:
        #print(f"Current sequence: #{tup[0]}")
        sequences.append(sequence(seq_id=seq_id, sequence = current_sequence, actual_hidden_states = actual_hidden_states))
        seq_id = tup[0]
# Handle last sequence:
sequences.append(sequence(seq_id=seq_id, sequence = current_sequence, actual_hidden_states = actual_hidden_states))


sequences[0]
fields(sequences[0])







# Create the parser
my_parser = argparse.ArgumentParser(prog='tagger',
                                    description='Tag observations with part-of-speech (POS) tags.',
                                    add_help=False)

my_parser.add_argument('-a', action='store', default='42')
my_parser.add_argument('-a', action='store', type=int)

# Add AT LEAST ONE value:
my_parser.add_argument('input', action='store', nargs='+')

# FLEXIBLE number of values and store them in a list:
my_parser.add_argument('input',
                       action='store',
                       nargs='*',
                       default='my default value')

my_parser.version = '1.0'
my_parser.add_argument('-a', action='store')
my_parser.add_argument('-b', action='store_const', const=42)
my_parser.add_argument('-c', action='store_true')
my_parser.add_argument('-d', action='store_false')
my_parser.add_argument('-e', action='append')
my_parser.add_argument('-f', action='append_const', const=42)
my_parser.add_argument('-g', action='count')
my_parser.add_argument('-h', action='help')
my_parser.add_argument('-j', action='version')

# Set domain of allowed values:
my_parser.add_argument('-a', action='store', choices=['head', 'tail'])

# Make argument required:
my_parser.add_argument('-a',
                       action='store',
                       choices=['head', 'tail'],
                       required=True)

# Create mutually exclusive group of arguments that can't be entered at
# the same command:
my_group = my_parser.add_mutually_exclusive_group(required=True)
my_group.add_argument('-v', '--verbose', action='store_true')
my_group.add_argument('-s', '--silent', action='store_true')

# Specify a name for the value of an argument (using metavar keyword):
my_parser.add_argument('-v',
                       '--verbosity',
                       action='store',
                       type=int,
                       metavar='LEVEL')


args = my_parser.parse_args()

print(vars(args))












sequences[0].sequence
sequences[0].seq_id


def seq_to_json(sequence):
    seq_dict = {
        "seq_id" : sequence.seq_id,
        "sequence" : sequence.sequence,
        "sequence_length": sequence.sequence_length,
        "actual_hidden_states": sequence.actual_hidden_states,
        "predicted_hidden_states": sequence.predicted_hidden_states,
        "best_path_prob": sequence.best_path_prob,
        "quad_kappa": sequence.quad_kappa,
        "linear_kappa": sequence.linear_kappa
    }
    return seq_dict

json_gen = (seq_to_json(seq) for seq in sequences)
next(json_gen)
next(json_gen)
next(json_gen)










# Convert the list of sequences to JSON array
json_array = sequence.schema().dumps(test_sequences, many=True)
json_array
sequence.schema().loads(json_array, many=True)


# Write to file:
with open(os.path.join(home_dir, 'models/model_1/model_1_sequences.json'), 'w') as outfile:
    json.dump(json_array, outfile)


with open(os.path.join(home_dir, 'models/model_1/model_1_sequences.json'), 'r') as infile:
    json_array_read = json.load(infile)

json_array_read # This should be a JSON array as a STRING

# We need to decode this into our dataclass using the same schema we
# provided to save it as a JSON array in the first place:
sequence.schema().loads(json_array_read, many=True)














def main():
    for seq in tqdm(test_sequences):
        seq.predict()
        seq.evaluate()

    # With sequences predicted and evaluated, we can transform the sequence instances
    # into JSON format to save instead of pickle, which is unreliable:
    # Convert the list of sequences to JSON array
    json_array = sequence.schema().dumps(test_sequences, many=True)
    # Write to file:
    with open(os.path.join(home_dir, 'models/model_1/model_1_sequences.json'), 'w') as outfile:
        json.dump(json_array, outfile)

    # print("Results being saved to", os.path.join(data_dir, 'results.pickle'))
    # with open(os.path.join(data_dir, 'results.pickle'), 'wb') as f:
    #     pickle.dump(results, f)
    overall_quad_kappa = np.mean([seq.quad_kappa for seq in sequences if seq.quad_kappa is not None])
    print(f"Overall quadratic kappa score for test sequences was {overall_quad_kappa}.")

# Argument, option, parameter

# Using the Python argparse library has four steps:
#
# Import the Python argparse library
# Create the parser
# Add optional and positional arguments to the parser
# Execute .parse_args()

if __name__ == '__main__':
    main()