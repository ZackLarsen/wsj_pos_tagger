
"""
Take text files with token and POS tag and turn into list of (token, tag) tuples to create
the input data to construct hidden markov model (HMM) POS tagger
"""

# Standard library imports
import os
import sys

# Third party libraries
import numpy as np
import pandas as pd
import csv

# Define directory structure
home_dir = os.path.dirname('WSJ_POS_TAGGING')
data_dir = os.path.join(home_dir, 'data')

filename = os.path.join(data_dir, 'wsj_train.txt')

train_tuples = []
sentences = 0
with open(filename) as infile:
    for line in infile:
        sentences += 1
        line = line.strip('\n')
        chars = line.split(' ')
        if len(chars) == 3:
            train_tuples.append((chars[0].lower(), chars[1]))
        elif len(chars) != 3:
            sentences += 1
            train_tuples.append(('<EOS>', '<EOS>'))
            train_tuples.append(('<START>', '<START>'))
    train_tuples.insert(0, ('<START>', '<START>'))
    train_tuples.append(('<EOS>', '<EOS>'))
print(sentences, "sentences read in.")

train_tuples[:5]
train_tuples[-5:]



filename = os.path.join(data_dir, 'wsj_test.txt')
test_tuples = []
sentences = 0
with open(filename) as infile:
    for line in infile:
        sentences += 1
        line = line.strip('\n')
        chars = line.split(' ')
        if len(chars) == 3:
            test_tuples.append((chars[0].lower(), chars[1]))
        elif len(chars) != 3:
            sentences += 1
            test_tuples.append(('<EOS>', '<EOS>'))
            test_tuples.append(('<START>', '<START>'))
    test_tuples.insert(0, ('<START>', '<START>'))
    test_tuples.append(('<EOS>', '<EOS>'))
print(sentences, "sentences read in.")

test_tuples[:5]
test_tuples[-5:]



train_tuples = [tup for tup in train_tuples if "''" not in tup]
train_tuples = [tup for tup in train_tuples if "``" not in tup]

test_tuples = [tup for tup in test_tuples if "''" not in tup]
test_tuples = [tup for tup in test_tuples if "``" not in tup]

# Save output:
with open(os.path.join(data_dir, 'wsj_train.csv'), 'w', newline='') as outfile:
    csv_out = csv.writer(outfile)
    csv_out.writerow(['token', 'tag'])
    csv_out.writerows(train_tuples)

with open(os.path.join(data_dir, 'wsj_test.csv'), 'w', newline='') as outfile:
    csv_out = csv.writer(outfile)
    csv_out.writerow(['token', 'tag'])
    csv_out.writerows(test_tuples)











# Data i/o
train_prepped = pd.read_csv(os.path.join(data_dir, 'wsj_train.csv'))
test_prepped = pd.read_csv(os.path.join(data_dir, 'wsj_test.csv'))

train_prepped
test_prepped

Counter(train_prepped.token)
Counter(train_prepped.tag)



# Check to see if there are any observations in the test set that
# are out-of-vocabulary:
train_tokens = set(train_prepped.token)
test_tokens = set(test_prepped.token)

train_tokens == test_tokens

train_tokens.difference(test_tokens) # What is in train but not test?
test_tokens.difference(train_tokens) # What is in test but not train?

total_tokens = len(train_prepped.token)
# # Get the percent of the vocabulary that each token represents:
token_dict = {k: round((v/total_tokens),8) for k,v in Counter(train_prepped.token).items()}
token_dict


# Common tokens:
common_tokens = {k:v for k,v in token_dict.items() if v >= 0.00001}.keys()
# Rare tokens:
rare_tokens = {k:v for k,v in token_dict.items() if v <= 0.00001}.keys()

len(common_tokens) # 6,324
len(rare_tokens) # 10,933

len(set(rare_tokens))




# Replace rare observations with the out-of-vocabulary observation:
train_prepped[train_prepped.token == 'OOV']

train_prepped['token'].replace(
    to_replace=list(set(rare_tokens)),
    value='<OOV>',
    inplace=True
)

train_prepped[train_prepped.token == '<OOV>']




test_prepped['token'].replace(
    to_replace=list(set(rare_tokens)),
    value='<OOV>',
    inplace=True
)

test_prepped[test_prepped.token == '<OOV>']




# Save preprocessed data:
train_prepped.to_csv(os.path.join('data', 'train_prepped.csv'), index=False)
test_prepped.to_csv(os.path.join('data', 'test_prepped.csv'), index=False)

