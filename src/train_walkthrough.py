"""

Script purpose: Walk through an example of training the HMM POS tagger.
Author: Zack Larsen
Date: January 25, 2020

"""

# Standard library imports
import os
import sys
import json
import pickle
import cProfile
from collections import Counter, defaultdict
from decimal import *
from importlib import reload

# 3rd party imports
import numpy as np
import pandas as pd

# Local imports
# Define paths
home_dir = os.path.dirname('WSJ_POS_TAGGING')
data_dir = os.path.join(home_dir, 'data')
sys.path.append(os.path.join(home_dir, 'src'))
from hmm_utils import *

reload(hmm_utils)

# Data i/o
train_prepped = pd.read_csv(os.path.join(data_dir, 'train_prepped.csv'))
test_prepped = pd.read_csv(os.path.join(data_dir, 'test_prepped.csv'))

# Make sure <OOV> has already been handled in prep file:
train_prepped[train_prepped['token'] == '<OOV>']

# Begin HMM construction:
observation_list = list(train_prepped.token.values)
state_list = list(train_prepped.tag.values)
observation_state_list = list((token, tag) for token, tag in train_prepped.values)

observation_list[:10]
state_list[:10]
observation_state_list[:10]

[bigram for bigram in find_ngrams(state_list, 2)][:10]





prep_tuple = hmm_prep(observation_state_list)

prep_tuple._fields
prep_tuple.n_states # 44
prep_tuple.unique_integer_states # 0 through 43
prep_tuple.bigram_counts.most_common()
prep_tuple.state_counts.most_common()
sorted(prep_tuple.state_counts.items(), key=lambda pair: pair[1], reverse=True)
sum(prep_tuple.state_counts.values()) # 226,577



for row_state in prep_tuple.unique_integer_states[:11]:
    for column_state in prep_tuple.unique_integer_states[:11]:
        bigram = (row_state, column_state)
        bigram_count = prep_tuple.bigram_counts[bigram]
        unigram_count = prep_tuple.state_counts[row_state]
        print(f"Row {row_state}, column {column_state} is equal to {bigram_count} / {unigram_count}")

bigram = (0, 10)
bigram_count = prep_tuple.bigram_counts[bigram]
bigram_count # 36
unigram_count = prep_tuple.state_counts[0] # 36
unigram_count

state_map_reversed = reverse_integer_map(prep_tuple.state_map)
state_map_reversed[0] # '#'
state_map_reversed[10] # 'CD'

prep_tuple.state_map['<START>'] # 8







prep_tuple.unique_integer_states # 0 through 43
prep_tuple.unique_integer_observations
prep_tuple.bigram_counts.most_common() # 0 through 6,324
prep_tuple.state_counts.most_common()
sorted(prep_tuple.state_counts.items(), key=lambda pair: pair[1], reverse=True)
sum(prep_tuple.state_counts.values()) # 226,575

os_bigram_counts = Counter(observation_state_list)
os_bigram_counts

observation_counts = Counter(observation_list)
observation_counts



for row_state in list(set(state_list))[:11]:
    for column_observation in list(set(observation_list))[:11]:
        os_bigram = (column_observation, row_state)
        os_bigram_count = os_bigram_counts[os_bigram]
        observation_count = observation_counts[column_observation]
        print(f"Row {row_state}, column {column_observation} is equal to {os_bigram_count} / {observation_count}")

























# Define matrices


transitions_matrix = create_transitions(prep_tuple)
# Row sums
np.sum(transitions_matrix, axis = 0) # All ones except 7th entry
sum(np.sum(transitions_matrix, axis=0))  # 43.0
# Column sums
np.sum(transitions_matrix, axis = 1)
sum(np.sum(transitions_matrix, axis = 1)) # 43.0

min(np.sum(transitions_matrix, axis = 0)) # 0.0
max(np.sum(transitions_matrix, axis = 0)) # 1.0000000000000002

# The column pertaining to the <START> tag should sum to 1 and have a zero in 
# the <EOS> row:
transitions_matrix[:, prep_tuple.state_map['<START>']]
transitions_matrix[:, prep_tuple.state_map['<START>']].sum() # 0.9998881056282869
transitions_matrix[prep_tuple.state_map['<EOS>'], prep_tuple.state_map['<START>']] # 0.0
# Success!


prep_tuple.observation_map['<OOV>']  # 381
prep_tuple.state_map['<START>']  # 8

Pi = create_pi(prep_tuple.state_map, transitions_matrix)
Pi
sum(Pi)  # 1.0

start_index = prep_tuple.state_map['<START>']
Pi = transitions_matrix[:, start_index]
Pi
sum(Pi)  # 0.9998881056282871






emissions_matrix = create_emissions(prep_tuple)
# Rowsums
np.sum(emissions_matrix, axis = 0) # Seems like all 1's
sum(np.sum(emissions_matrix, axis = 0)) # 6325.0

sorted(np.sum(emissions_matrix, axis = 0))  # Seems like all 1's
sorted(np.sum(emissions_matrix, axis = 0), reverse=True)

min(np.sum(emissions_matrix, axis = 0)) # 0.9999999999999999
max(np.sum(emissions_matrix, axis = 0)) # 1.0

# Column sums
np.sum(emissions_matrix, axis = 1) # Was expecting these to all be 1, but clearly not the case here.
sum(np.sum(emissions_matrix, axis = 1)) # 6324.999999999999

min(np.sum(emissions_matrix, axis = 1)) # 0.3414902570078332
max(np.sum(emissions_matrix, axis = 1)) # 1422.4383557081967






# Convert probs to log10 probs for transitions and emissions and Pi. Allow negative infinity
# values for the zero probabilities that incur divide-by-zero errors:
transitions_matrix_log = np.log10(transitions_matrix)
emissions_matrix_log = np.log10(emissions_matrix)
Pi_log = np.log10(Pi)

np.savez_compressed(os.path.join(data_dir, 'log_matrices'),
                    transitions=transitions_matrix_log,
                    emissions=emissions_matrix_log,
                    Pi=Pi_log)






# Prepare the test sequences to feed to the decoder program:
# Turn dataframe into list of tuples:
test_tuples = list(map(tuple, test_prepped.values))
test_tuples



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







# At this point, we can run decoder.py































#### Extras


"""


n_states = len(set(state_list))
bigram_counts = Counter([bigram for bigram in find_ngrams(state_list, 2)])
state_counts = Counter(state_list)

transitions_matrix = np.zeros((n_states, n_states))
for row_state in states:
    for column_state in states:
        bigram = (column_state, row_state)
        bigram_count = bigram_counts[bigram]
        unigram_count = state_counts[column_state]
        try:
            transitions_matrix[row_state, column_state] = bigram_count / (unigram_count)
        except ZeroDivisionError as err:
            transitions_matrix[row_state, column_state] = 0














for i, state_i in enumerate(prep_tuple.unique_integer_states):
    for j, state_j in enumerate(prep_tuple.unique_integer_states):
        #print(i,j)
        print(state_i, state_j)


# When saving integer maps:
# Having issues with integer states not being strings for keys:
#state_map = {str(k): v for k,v in state_map.items()}

# Save integer maps
with open(os.path.join(data_dir, 'observation_map.json'), 'w') as fp:
    json.dump(prep_tuple.observation_map, fp)

with open(os.path.join(data_dir, 'state_map.json'), 'w') as fp:
    json.dump(prep_tuple.state_map, fp)







#path = os.getcwd()
#print(path)


getcontext().prec = 28
Decimal(1) / Decimal(7)


train_prepped.token.values
train_prepped.tag.values
len(set(observation_list)) # 6,325

Counter((observation_list))['<START>'] # 8,936
Counter((observation_list))['<OOV>'] # 13,542



len(set(state_list)) # 44
set(state_list)




# Ensure that the train and test sets have the same set of hidden states,
# or at least that there are no hidden states in the test set that were
# unseen at training time:
state_list_test = list(test_prepped.tag.values)
len(set(state_list_test)) # 43
set(state_list_test)


set(state_list).difference(set(state_list_test))
# {'SYM'} is in the train state set but not test, so this is okay.

# # Save lists to pickle
# with open(os.path.join(data_dir, 'observation_list.pkl'), 'wb') as f:
#     pickle.dump(observation_list, f)
# with open(os.path.join(data_dir, 'state_list.pkl'), 'wb') as f:
#     pickle.dump(state_list, f)
# with open(os.path.join(data_dir, 'observation_state_list.pkl'), 'wb') as f:
#     pickle.dump(observation_state_list, f)
#
#
# # Read pickled lists back into memory
# with open(os.path.join(data_dir, 'observation_list.pkl'), 'rb') as f:
#     observation_list = pickle.load(f)
# with open(os.path.join(data_dir, 'state_list.pkl'), 'rb') as f:
#     state_list = pickle.load(f)
# with open(os.path.join(data_dir, 'observation_state_list.pkl'), 'rb') as f:
#     observation_state_list = pickle.load(f)



observation_list[:10]
state_list[:10]
observation_state_list[:10]




# unique_observations = set(observation_list)
# n_observations = len(unique_observations) # 6325
#
# unique_states = set(state_list)
# n_states = len(unique_states) # 44
#
# observation_map = integer_map(list(unique_observations))
# observation_map_reversed = reverse_integer_map(observation_map)
# integer_observation_list = [observation_map[observation] for observation in observation_list]
#
# state_map = integer_map(list(unique_states))
# state_map_reversed = reverse_integer_map(state_map)
# integer_state_list = [state_map[str(state)] for state in state_list]
#
# integer_tuple_list = [(a,b) for a,b in zip(integer_observation_list, integer_state_list)]
#
# unique_integer_observations = list(np.unique(integer_observation_list))
# unique_integer_states = list(np.unique(integer_state_list))
#
# observation_counts = Counter(integer_observation_list)
# state_counts = Counter(integer_state_list)
# observation_state_counts = Counter(integer_tuple_list)
#
# bigrams = find_ngrams(integer_state_list, 2)
# bigram_counts = Counter(bigrams)
# n_bigrams = len(bigram_counts.keys())







observation_map = prep_tuple.observation_map
state_map = prep_tuple.state_map
n_observations = prep_tuple.n_observations
n_states = prep_tuple.n_states
unique_integer_observations = prep_tuple.unique_integer_observations
unique_integer_states = prep_tuple.unique_integer_states
state_counts = prep_tuple.state_counts
observation_state_counts = prep_tuple.observation_state_counts
bigram_counts = prep_tuple.bigram_counts
integer_tuple_list = prep_tuple.integer_tuple_list


#prep_tuple._asdict()
prep_tuple._fields




'<START>' in observation_map.keys() # True
'<END>' in observation_map.keys() # False
'<EOS>' in observation_map.keys() # True
'<OOV>' in observation_map.keys() # True

np.sum(list(observation_state_counts.values())) # 226,575
len(observation_state_list) # 226,575

observation_map['<EOS>'] # 380
observation_map['<OOV>'] # 381
observation_map['<START>'] # 382



# Find the observation/state counts for the start observations:
[(k,v) for k,v in observation_state_counts.items() if k[0] == observation_map['<START>']]
# [((382, 8), 8936)]






transitions_matrix = np.zeros((prep_tuple.n_states, prep_tuple.n_states))
for state_i in prep_tuple.unique_integer_states:
    for state_j in prep_tuple.unique_integer_states:
        bigram = (state_i, state_j)
        bigram_count = prep_tuple.bigram_counts[bigram]
        unigram_count = prep_tuple.state_counts[state_i]
        try:
            transitions_matrix[state_i, state_j] = bigram_count / (unigram_count)
        except ZeroDivisionError as err:
            transitions_matrix[state_i, state_j] = 0

transitions_matrix = create_transitions(states = prep_tuple.unique_integer_states,
                                        bigram_counts = prep_tuple.bigram_counts,
                                        state_counts = prep_tuple.state_counts,
                                        n_states = prep_tuple.n_states)


transitions_matrix.shape
transitions_matrix
np.sum(transitions_matrix) # 43.0
np.sum(transitions_matrix, axis=0) # Row sum
np.sum(transitions_matrix, axis=1) # Column sum, all 1's or at least very close
#np.sum(transitions_matrix, axis=2) # Error

sum(np.sum(transitions_matrix, axis=0)) # 42.99999999999999
sum(np.sum(transitions_matrix, axis=1)) # 43.0





prep_tuple.state_counts
prep_tuple.state_counts[19] # 30,147

reverse_observation_map = {v:k for k,v in prep_tuple.observation_map.items()}
reverse_observation_map
reverse_observation_map[19] # '-rrb-'

reverse_state_map = {v:k for k,v in prep_tuple.state_map.items()}
reverse_state_map
reverse_state_map[7] # '<EOS>'


prep_tuple.state_counts[382]
prep_tuple.state_counts.values()[382]





observation_map['<START>'] # 382
state_map['<START>'] # 8




emissions_matrix[:, observation_map['<START>']]
emissions_matrix[:, state_map['<START>']]


transitions_matrix
emissions_matrix.shape # Row sums should give us initial probs:
emissions_matrix.sum() # 44.00000000000001
emissions_matrix.sum(axis=0)
emissions_matrix.sum(axis=1) # All 1's
emissions_matrix[:, observation_map['<START>']]
emissions_matrix[:, observation_map['<START>']].sum() # 1


transitions_matrix[:, state_map['<START>']]
transitions_matrix[:, state_map['<START>']].sum() # 1
# This is telling us that the only cell of the transitions matrix that
# is nonzero for the start state is 7, which corresponds to the
# EOS state. This means we are looking backward, so we need to shift
# things one step forward so there are no EOS states in our search.



emissions_matrix[observation_map['<START>'], :].sum()
emissions_matrix[observation_map['<START>'], :]
emissions_matrix[observation_map['<START>'], :].sum(axis=0)







emissions_matrix.shape # (44, 6325)
np.sum(emissions_matrix, axis=0)# Row sums
sum(np.sum(emissions_matrix, axis=0)) # 44.000000000000114

np.sum(emissions_matrix, axis=1)# Column sums, all 1's
sum(np.sum(emissions_matrix, axis=1)) # 44.0

np.sum(emissions_matrix) == prep_tuple.n_states # False
np.isclose(np.sum(emissions_matrix), prep_tuple.n_states) # True


sum(prep_tuple.state_counts.values()) # 226,575
# Percent of train data by state:
{k: round(v / sum(prep_tuple.state_counts.values()), 4) for k,v in prep_tuple.state_counts.items()}
{reverse_integer_map(prep_tuple.state_map)[k]: round(v / sum(prep_tuple.state_counts.values()), 4) for k,v in prep_tuple.state_counts.items()}

{reverse_integer_map(prep_tuple.state_map)[k]: "{0:.0%}".format(v / sum(prep_tuple.state_counts.values())) for k,v in prep_tuple.state_counts.items()}





#transitions_matrix_log
#emissions_matrix_log
#Pi_log

# Save
#np.save('data/hcc_transitions_matrix_log', transitions_matrix_log)
#np.save('data/hcc_emissions_matrix_log', emissions_matrix_log)
#np.save('data/hcc_Pi_log', Pi_log)



# Read log probability matrices back in:
#log_matrices = np.load(os.path.join(data_dir, 'log_matrices.npz'))
#transitions_matrix_log = log_matrices['transitions']
#emissions_matrix_log = log_matrices['emissions']
#Pi_log = log_matrices['Pi']

#emissions_matrix.shape
#emissions_matrix
#np.save('data/hcc_emissions_matrix', emissions_matrix)

# Add observation names as columns:
#emissions_matrix_named = pd.DataFrame(emissions_matrix, columns=observation_map.keys())
#emissions_matrix_named



# Make it a uniform distribution over n_states:
#Pi = create_pi_uniform(n_states)

# Make the initial distribution equal to simply the count of state
# occurrences, regardless of <START> observation counts:
Pi = np.zeros(n_states)
for key, value in state_counts.items():
    Pi[key] = value / sum(state_counts.values())
Pi
Pi.sum() # Close to 1, not exact







#np.save('data/hcc_Pi', Pi)

# Pi = np.zeros(n_states)
# Pi
# Pi[0] = 1 # Everybody starts in state 0
# Pi
#emissions_matrix = np.load('data/hcc_emissions_matrix.npy')
#transitions_matrix = np.load('data/hcc_transitions_matrix.npy')
#Pi = np.load('data/hcc_Pi.npy')

#np.save('data/hcc_transitions_matrix', transitions_matrix)

"""
