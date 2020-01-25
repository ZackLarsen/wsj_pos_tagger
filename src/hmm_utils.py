
import numpy as np
from numba import jit
from numba import types
from numba.typed import Dict
import math
import re, sys, datetime, os
from collections import defaultdict, Counter, namedtuple
from scipy.sparse import csr_matrix
from sklearn.metrics import cohen_kappa_score


def find_ngrams(input_list, n):
    '''
    Generate ngrams
    :param input_list: List of observations in a sequence
    :param n: Number of observations to slide over, e.g. 1 for unigram,
    2 for bigram, 3 for trigram, etc.
    :return: A generator object with ngrams
    '''
    return zip(*[input_list[i:] for i in range(n)])


def integer_map(observations):
    '''
    Take a list of observations and map them to integers
    :param observation_list: List of observations
    :return: Dictionary mapping each observation to a unique integer
    '''
    unique_observation_list = np.unique(observations)
    unique_observation_list.sort(kind='quicksort')
    observation_map = {observation: value for observation, value in zip(unique_observation_list, range(0,len(unique_observation_list)))}
    return observation_map


def reverse_integer_map(observation_map):
    '''
    Reverse the order of the dictionary for the observations
    :param observation_map: The result of the observation_to_int(observations) function
    :return: A reversed version of the observation map dictionary
    '''
    reversed_dictionary = {value: key for key, value in observation_map.items()}
    return reversed_dictionary


def hmm_prep(observation_state_list):
    """
    Perform conditional probability calculations and integer mapping
    :param observation_state_list: List of (observation, state) tuples
    :return: pt: Instance of prep_tuple namedtuple
    """
    prep_tuple = namedtuple('prep_tuple',
                            'n_observations unique_integer_observations n_states unique_integer_states observation_map state_map state_counts observation_counts observation_state_counts bigram_counts integer_tuple_list')

    observation_list = [tup[0] for tup in observation_state_list]
    state_list = [tup[1] for tup in observation_state_list]

    unique_observations = set(observation_list)
    n_observations = len(unique_observations)

    unique_states = set(state_list)
    n_states = len(unique_states)

    observation_map = integer_map(list(unique_observations))
    observation_map_reversed = reverse_integer_map(observation_map)
    integer_observation_list = [observation_map[observation] for observation in observation_list]

    state_map = integer_map(list(unique_states))
    state_map_reversed = reverse_integer_map(state_map)
    integer_state_list = [state_map[state] for state in state_list]

    integer_tuple_list = [(a,b) for a,b in zip(integer_observation_list, integer_state_list)]

    unique_integer_observations = list(np.unique(integer_observation_list))
    unique_integer_states = list(np.unique(integer_state_list))

    observation_counts = Counter(integer_observation_list)
    state_counts = Counter(integer_state_list)
    observation_state_counts = Counter(integer_tuple_list)

    bigrams = find_ngrams(integer_state_list, 2)
    # Remove ('<EOS>', '<START>') tuple from bigrams:
    end_start = (state_map['<EOS>'], state_map['<START>'])
    start_end = (state_map['<START>'], state_map['<EOS>'])
    ignorables = [end_start, start_end]
    bigram_counts = Counter(x for x in bigrams if x not in ignorables)
    n_bigrams = len(bigram_counts.keys())

    pt = prep_tuple(n_observations, unique_integer_observations, n_states, unique_integer_states, observation_map, state_map, state_counts, observation_counts, observation_state_counts, bigram_counts, integer_tuple_list)

    return pt


def viterbi(observations, transitions_matrix, emissions_matrix, Pi):
    """
    Compute the optimal sequence of hidden states
    Note that -inf (or np.NINF) for negative infinity is being used as a default
    value to represent zero probabilities because we are computing in log10 space

    :param observations:
    :param transitions_matrix:
    :param emissions_matrix:
    :param Pi:
    :return:
    """

    # Initialization
    N = emissions_matrix.shape[0] # Number of states (hidden states)
    T = len(observations) # Number of observations (observations)
    viterbi_trellis = np.ones((N, T)) * np.NINF
    backpointer = np.zeros_like(viterbi_trellis)
    for s in range(0, N):
        viterbi_trellis[s, 0] = Pi[s] + emissions_matrix[s, observations[0]]

    # Recursion
    for time_step in range(1, T):
        for current_state in range(0, N):
            priors = np.zeros((N, 2))
            for previous_state in range(0, N):
                priors[previous_state, 0] = previous_state
                priors[previous_state, 1] = viterbi_trellis[previous_state, time_step - 1] + \
                                            transitions_matrix[previous_state, current_state] + \
                                            emissions_matrix[current_state, observations[time_step]] # Previously, we were using timestep-1 here
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1], axis=0)
            backpointer[current_state, time_step] = np.argmax(priors[:, 1], axis=0)

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_hidden_states = []
    for i, ix in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[ix, i])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer) # Add the bestpathpointer to the last entry

    return bestpathprob, viterbi_hidden_states


@jit(nopython=True)
def viterbi_numba(observations, transitions_matrix, emissions_matrix, Pi):
    '''
    Compute the optimal sequence of hidden states

    :param observations: List of observations in sequence
    :param transitions_matrix: Matrix of probability of state given previous state
    :param emissions_matrix: Matrix of probability of state given observation
    :param Pi: Initial probability vector

    :return: bestpath, bestpathprob
    '''

    N = emissions_matrix.shape[0] # Number of states (hidden states)
    T = len(observations) # Number of observations (observations)

    # Initialize
    viterbi_trellis = np.ones((N, T)) * np.NINF
    backpointer = np.zeros_like(viterbi_trellis)
    for s in range(0, N):
        viterbi_trellis[s, 0] = Pi[s] * emissions_matrix[s, observations[0]]
        backpointer[s, 0] = 0

    # Recursion
    for time_step in range(1, T):
        for current_state in range(0, N):
            priors = np.zeros((N, 2))
            for previous_state in range(0, N):
                priors[previous_state, 0] = previous_state
                priors[previous_state, 1] = viterbi_trellis[previous_state, time_step - 1] + \
                                            transitions_matrix[previous_state, current_state] + \
                                            emissions_matrix[current_state, observations[time_step]]
            #viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1], axis=0)
            #backpointer[current_state, time_step] = np.argmax(priors[:, 1], axis=0)
            # Adding these lines below because numba doesn't like the axis argument:
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1])
            backpointer[current_state, time_step] = np.argmax(priors[:, 1])

    # Termination / backtracing the most probable path of hidden states:
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    #viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_cell_idx = np.argmax(viterbi_trellis)
    bestpath = []
    for i, ix in enumerate(viterbi_cell_idx):
        bestpath.append(backpointer[ix, i])
    bestpath = np.delete(bestpath, 0)  # We don't need the first entry - it is always zero
    bestpath = np.append(bestpath, bestpathpointer)

    return bestpath, bestpathprob


def hmm_kappa_score(actual_states, predicted_states):
    """
    Compute Cohen's kappa score for the inter-rater agreement of the
    hidden states between the actual hidden states and the hidden states
    that the hidden markov model (HMM) predicts
    :param actual_states:
    :param predicted_states:
    :return:
    """
    return cohen_kappa_score(actual_states, predicted_states)


def expb(a, b):
    """
    Return b to the power of a
    :param a:
    :param b:
    :return:
    """
    return b**a


def increment_dict(dictionary, i):
    """
    Increment all values in a dictionary by i
    :param dictionary:
    :param i: Number to increment by
    :return:updated dictionary with same keys but newly incremented values
    """
    updated_dictionary = {key:value+i for key, value in dictionary.items()}
    return updated_dictionary


def unigram(state_list):
    '''
    Construct unigram model with LaPlace smoothing
    :param state_list: A list of pos states
    :return: A default dictionary of pos state counts and
    a default dictionary of pos state counts smoothed by LaPlace smoothing
    '''
    counts_dd = defaultdict(int)
    for state in state_list:
        counts_dd[state] += 1

    model = counts_dd.copy()
    for word in counts_dd:
        model[word] = model[word]/float(sum(counts_dd.values()))

    return counts_dd, model


def input_prep(data):
    """
    Given a dataframe, take the sequence and known states and
    prepare the observation list, state list, and observation_state list
    :param data:
    :return: observation_list, state_list, observation_state_list
    """
    observation_list = list(data['observation'])
    state_list = list(data['state'])
    observation_state_list = list(([a,b]) for a,b in zip(observation_list, state_list))
    return observation_list, state_list, observation_state_list


def file_prep(filename, nrows = 100, lowercase = False):
    '''
    Read file, create a list of observations, a list of parts-of-speech
    (pos), and a data dictionary of the observation: state co-occurrences
    :param filename: Name of the file being read
    :param nrows The number of rows to read in
    :param lowercase Whether or not to lowercase all the observations
    :return: observation_list, pos_list, data
    '''
    observation_list = []
    pos_list = []
    observation_pos_list = []
    sentences = 0
    with open(filename) as infile:
        head = [next(infile) for x in range(nrows)]
        observation_list.append('<START>')
        pos_list.append('<START>')
        observation_pos_list.append(tuple(('<START>','<START>')))
        for line in head:
            line = line.strip('\n')
            chars = line.split(' ')
            if len(chars) == 3:
                observation = chars[0]
                pos = chars[1]
                if lowercase:
                    observation_list.append(observation.lower())
                    observation_pos_list.append((observation.lower(), pos))
                elif not lowercase:
                    observation_list.append(observation)
                    observation_pos_list.append((observation, pos))
                pos_list.append(pos)
            elif len(chars) != 3:
                sentences += 1
                observation_list.append('<STOP>')
                observation_list.append('<START>')
                pos_list.append('<STOP>')
                pos_list.append('<START>')
                observation_pos_list.append(('<STOP>', '<STOP>'))
                observation_pos_list.append(('<START>', '<START>'))
        observation_list.append('<STOP>')
        pos_list.append('<STOP>')
        observation_pos_list.append(('<STOP>', '<STOP>'))
    print(sentences, "sentences read in.")
    return observation_list, pos_list, observation_pos_list


def create_transitions(states, bigram_counts, state_counts, n_states, add_unknown_state = False, smoothing_rate = 0.01):
    """
    Create the transitions matrix representing the probability of a state given
    the previous state

    Parameters:
    :param states: list of unique states
    :param bigram_counts: Number of times a state co-occurs in training corpus with
        the previous state
    :param state_counts: Number of times a state occurs in training corpus
    :param n_states: Number of unique states
    :param add_unknown_state: Boolean flag for including unknown state or not (changes resulting dimensions)
    :param smoothing_rate: Number by which to inflate zero-probability states

    :return: transition_matrix
    """
    if add_unknown_state:
        transitions_matrix = np.zeros((n_states + 1, n_states + 1))
        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                bigram = (state_i, state_j)
                bigram_count = bigram_counts[bigram]
                unigram_count = state_counts[state_i]
                a = bigram_count / (unigram_count + 0.01)
                transitions_matrix[i, j] = a
        # Unknown column
        transitions_matrix[:, n_states] = smoothing_rate
        # Unknown row
        transitions_matrix[n_states, :] = smoothing_rate

    elif not add_unknown_state:
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

    return transitions_matrix


def create_emissions(observations, states, observation_state_counts, observation_counts, n_observations, n_states, add_unknown_state = False, smoothing_rate = 0.01):
    """
    Create the emissions matrix representing the probability of a observation given a state.
    The MLE of the emission probability is P(observation_i|state_i) = C(state_i,observation_i) / C(state_i).
    :param observations: list of unique observations
    :param states: list of unique states
    :param tuple_counts: tuple([observation, state]) counts
    :param state_counts: POS state counts
    :param n_observations: Number of unique observations
    :param n_states: Number of unique states
    :param add_unknown_state: Boolean flag for whether or not to add a start state and observation
    :param smoothing_rate: What value to assign to unknown states
    :return: emissions_matrix of shape (n_states, n_observations) or (n_states+1, n_observations+1) if unknown state/observation added
    """
    # TODO: fix smoothing so that no observations have greater than 1 probability

    # TODO: eliminate emissions with fewer than 2 occurrences in training data, replace with unknown observation/state
    # if add_unknown_state:
    #     emissions_matrix = np.zeros((n_states+1, n_observations+1))
    #     for i, state in enumerate(states):
    #         for j, observation in enumerate(observations):
    #             tuple_count = tuple_counts[tuple((observation, state))]
    #             state_count = state_counts[states[i]]
    #             b = tuple_count / state_count
    #             emissions_matrix[i, j] = b
    #     # Unknown state
    #     emissions_matrix[n_states, :] = smoothing_rate
    #     # Unknown observation
    #     emissions_matrix[:, n_observations] = smoothing_rate

    # elif not add_unknown_state:
    #     emissions_matrix = np.zeros((n_states, n_observations))
    #     for i, state in enumerate(states):
    #         for j, observation in enumerate(observations):
    #             tuple_count = tuple_counts[tuple((observation, state))]
    #             state_count = state_counts[states[i]]
    #             b = tuple_count / state_count
    #             emissions_matrix[i, j] = b
    # emissions_matrix = np.zeros((n_states, n_observations))
    # for i, state in enumerate(states):
    #     for j, observation in enumerate(observations):
    #         tuple_count = tuple_counts[(observation, state)]
    #         state_count = state_counts[states[i]]
    #         b = tuple_count / state_count
    #         emissions_matrix[i, j] = b
    emissions_matrix = np.zeros((n_states, n_observations))
    for row_state in states:
        for column_observation in observations:
            os_bigram = (column_observation, row_state)
            os_bigram_count = observation_state_counts[os_bigram]
            observation_count = observation_counts[column_observation]
            try:
                emissions_matrix[row_state, column_observation] = os_bigram_count / (observation_count)
            except ZeroDivisionError as err:
                emissions_matrix[row_state, column_observation] = 0

    return emissions_matrix


def create_pi(state_map, transitions_matrix, start_state = '<START>'):
    """
    Create the initial probability distribution Pi, representing the
     probability of the HMM starting in state i
    :param state_map: The mapping between states and integers
    :param transitions_matrix: The matrix of probabilities of states given a previous state
    :param start_state: Which state in the vocabulary corresponds to starting a sequence
    :return: Pi, the initial probability distribution
    """
    start_index = state_map[start_state]
    Pi = transitions_matrix[:, start_index]
    assert np.isclose(Pi.sum(), 1) == True, "Does not form a valid probability distribution."
    return Pi


def create_pi_uniform(n_states):
    """
    Create the initial probability distribution Pi, representing the
     probability of the HMM starting in state i. Treat all states as
     being equally probable from starting sequence.
    :param n_states:
    :return: Pi, the (uniform) initial probability distribution
    """
    return np.full(n_states, 1/n_states)


def transitions_smoother(transitions, n_states, smoothing_factor = 1):
    """
    Smooth the transition probabilities by adding a constant factor
    to ensure that no elements have a true zero probability. This is akin
    to starting all counts at one instead of zero and then dividing by the
    vocabulary size, which in this case is n_states.
    :param transitions: Transition probabilities matrix
    :param n_states: Number of unique states
    :param smoothing_factor: Constant to smooth by
    :return: smoothed_transitions
    """
    f = lambda x: x + smoothing_factor / (x + n_states)
    smoothed_transitions = f(transitions)
    return smoothed_transitions


def emissions_smoother(emissions, n_observations, smoothing_factor = 1):
    """
    Smooth the emission probabilities by adding a constant factor
    to ensure that no elements have a true zero probability. This is akin
    to starting all counts at one instead of zero and then dividing by the
    vocabulary size, which in this case is n_observations
    :param emissions: Emissions probabilities matrix
    :param n_observations: Number of unique observations
    :param smoothing_factor: Constant to smooth by
    :return: smoothed_emissions
    """
    f = lambda x: x + smoothing_factor / (x + n_observations)
    smoothed_emissions = f(emissions)
    return smoothed_emissions


def viterbi_part1(observations, transitions_matrix, emissions_matrix, Pi):
    '''
    Compute the optimal sequence of hidden states
    Note that -10000 is being used as a default value to represent
    zero probabilities because we are computing in log10 space

    :param observations: List of observations (observations) in sequence
    :param transitions_matrix: Matrix of probability of state given previous state
    :param emissions_matrix: Matrix of probability of state given observation
    :param Pi: Initial probability vector

    :return: bestpath, bestpathprob
    '''

    N = emissions_matrix.shape[0] # Number of states (hidden states)
    T = len(observations) # Number of observations (observations)

    # Initialize
    viterbi_trellis = np.zeros((N, T))
    backpointer = np.zeros_like(viterbi_trellis)
    for s in range(0, N):
        if Pi[s] != 0 and emissions_matrix[s, observations[0]] !=0:
            viterbi_trellis[s, 0] = Pi[s] + emissions_matrix[s, observations[0]]
        else:
            viterbi_trellis[s, 0] = -10000
        backpointer[s, 0] = 0

    # Recursion
    for time_step in range(1, T):
        for current_state in range(0, N):
            priors = np.zeros((N, 2))
            for previous_state in range(0, N):
                priors[previous_state, 0] = previous_state
                if viterbi_trellis[previous_state, time_step - 1] != 0 and transitions_matrix[previous_state, current_state] != 0 and emissions_matrix[current_state, observations[time_step] - 1] != 0:
                    priors[previous_state, 1] = viterbi_trellis[previous_state, time_step - 1] + \
                                                transitions_matrix[previous_state, current_state] + \
                                                emissions_matrix[current_state, observations[time_step] - 1]
                else:
                    priors[previous_state, 1] = -10000
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1], axis=0)
            backpointer[current_state, time_step] = np.argmax(priors[:, 1], axis=0)

    return viterbi_trellis, backpointer


def backtrace(viterbi_trellis, backpointer):
    """
    Trace the most probable hidden state path backwards from
    bestpathpointer using the probabilities in viterbi_trellis
    :param viterbi_trellis:
    :param backpointer:
    :return:
    """
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_hidden_states = []
    for i, ix in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[ix, i])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer) # Add the bestpathpointer to the last entry
    return bestpathprob, viterbi_hidden_states


def backtracer(viterbi_trellis, bestpathpointer):
    """
    Trace the most probable hidden state path backwards from
    bestpathpointer using the probabilities in viterbi_trellis
    :param viterbi_trellis:
    :param bestpathpointer:
    :return:
    """
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_hidden_states = []
    for i, ix in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[ix, i])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer)
    return (bestpathprob, viterbi_hidden_states)

