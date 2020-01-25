##################################################
## Project: Hidden Markov Models
## Script purpose: Create HMM for POS tagging example
## Date: July 21, 2019
## Author: Zack Larsen
##################################################

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, here, DataCombine, scales,
       esquisse, ggThemeAssist, hrbrthemes, conflicted, Hmisc, glue, psych, progress)

conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")
conflict_prefer("%>%", "magrittr")



# Read in text format -----------------------------------------------------

# wsj_train <- read_delim(here("pos_tagging/",'WSJ_train.txt'),
#                         delim = ' ',
#                         skip = 0,
#                         skip_empty_rows = FALSE,
#                         col_names = c('token','tag','extra'))
# 
# wsj_train
# wsj_train %<>% select(-extra)
# glimpse(wsj_train)
# typeof(wsj_train)
# is_tibble(wsj_train)
# 
# 
# View(wsj_train[0:50,])
# 
# 
# 
# wsj_train[0:50,]
# 
# 
# 
# wsj_train <- read.delim(here("pos_tagging/",'WSJ_train.txt'), header = FALSE, sep = ' ', skipNul = FALSE)
# 
# wsj_train %>% 
#   select(token)
# 
# 
# 
# 
# wsj_test <- read_delim(here("pos_tagging/",'WSJ_test.txt'),
#             delim = ' ',
#             skip = 0,
#             skip_empty_rows = FALSE,
#             col_names = c('token','tag','extra'))
# 
# wsj_test %<>% select(-extra)
# glimpse(wsj_test)
# is_tibble(wsj_test)









# 
# 
# 
# 
# # Add <START> and <EOS> tags
# wsj_train %>% 
#   select(token, tag) %>% 
#   slice(38)
# 
# 
# wsj_train %>% 
#   select(token, tag) %>% 
#   mutate(lag = lag(token)) %>% 
#   filter(lag == '.')
# 
# 
# wsj_train %>% 
#   select(token, tag) %>% 
#   filter(token == '') %>% 
#   head()
# 
# 
# 
# start_tags <- data.frame(token = "<START>", tag = "<START>", disregard = "<START>")
# start_tags
# 
# end_tags <- data.frame(token = "<END>", tag = "<END>", disregard = "<END>")
# end_tags
# 
# 
# 
# 
# 
# wsj_train <- rbind(start_tags, wsj_train)
# wsj_train
# 
# wsj_train <- rbind(wsj_train, end_tags)
# wsj_train
# 
# 
# 
# 
# 
# wsj_train %>% 
#   select(token, tag, disregard) %>% 
#   mutate(index = ifelse(token == '', 1, 0)) %>% 
#   mutate(
#     new_token = ifelse(index == 1, "<START>", token),
#     new_tag = ifelse(index == 1, "<START>", tag),
#     new_disregard = ifelse(index == 1, "<START>", disregard)
#   ) %>% 
#   mutate(
#     new_token = ifelse(index == 1, "<START>", token),
#     new_tag = ifelse(index == 1, "<START>", tag),
#     new_disregard = ifelse(index == 1, "<START>", disregard)
#   )
# 
# 
# 
# 
# 
# 
# token_tag_list <- list(length=length(wsj_train$token))
# token_tag_list
# token_tag_list[[1]]
# 
# 
# num_lines <- 0
# for(i in 0:length(wsj_train$token)){
#   if(wsj_train[i,1] == ''){
#     print("I found a blank line")
#     token_tag_list[[num_lines]] <- wsj_train[i,]
#     token_tag_list[[num_lines]] <- wsj_train[i,]
#     num_lines <- num_lines + 2
#   }else{
#     token_tag_list[[num_lines]] <- c("<START>","<START>")
#   }
#   num_lines <- num_lines + 1
# }
# 
# 
# wsj_train[39,1]
# 
# 
# 





# Read in csv format ------------------------------------------------------------

wsj_train <- fread(here("pos_tagging/", 'WSJ_train.csv'))
wsj_test <- fread(here("pos_tagging/", 'WSJ_test.csv'))

#wsj_train <- read_csv(here("pos_tagging/", 'WSJ_train.csv'))
#wsj_test <- read_csv(here("pos_tagging/", 'WSJ_test.csv'))


# colnames(wsj_train)
# glimpse(wsj_train)
# 
# wsj_train$token %>% n_distinct() # 17,257
# wsj_train$token %>% unique()
# 
# wsj_train$tag %>% n_distinct() # 44
# wsj_train$tag %>% unique()


n_tokens <- wsj_train$token %>% n_distinct()
#n_tokens








# Handle OOV words:

token_freq <- wsj_train %>% 
  group_by(token) %>% 
  tally() %>% 
  mutate(freq = n/sum(n)) %>% 
  arrange(-n)

#token_freq


# Using proportion of tokens:
vocab <- token_freq %>% head(round(0.9*n_tokens)) %>% select(token)
vocab_size <- length(vocab$token)
#vocab_size # 15,531

oov_tokens <- token_freq %>% tail(n_tokens-round(0.9*n_tokens)) %>% select(token)
#oov_tokens
#length(oov_tokens$token) # 1,726


# Using frequency threshold:
# freq_threshold <- 0.0001
# 
# token_freq %>% 
#   filter(freq > freq_threshold) %>% 
#   tally()
# 
# token_freq %>% 
#   filter(freq <= freq_threshold) %>% 
#   tally()
# 
# vocab <- wsj_train %>% 
#   group_by(token) %>% 
#   tally() %>% 
#   mutate(freq = n/sum(n)) %>% 
#   filter(freq <= freq_threshold) %>% 
#   select(token)
# 
# vocab$token
# 
# oov_words <- wsj_train %>% 
#   group_by(token) %>% 
#   tally() %>% 
#   mutate(freq = n/sum(n)) %>% 
#   filter(freq <= freq_threshold) %>% 
#   select(token)
# 
# length(oov_words$token)
# oov_words$token








wsj_train_closed <- wsj_train
wsj_train_closed[wsj_train_closed$token %in% oov_tokens$token] <- '<OOV>'
wsj_train_closed


# Do the same for test:
#wsj_test[!wsj_test$token %in% vocab$token]
#wsj_test[wsj_test$token %in% vocab$token]

wsj_test_closed <- wsj_test
wsj_test_closed[!wsj_test_closed$token %in% vocab$token] <- '<OOV>'
wsj_test_closed












# First steps are to create transitions, emissions, 
# and starting/initial probability matrices:







# Transitions probabilities (from one state to the next state):
transitions <- wsj_train_closed %>% 
        select(tag) %>% 
        mutate(next_tag = lead(tag)) %>% 
        table()

row.names(transitions)
colnames(transitions)

row.names(transitions) == colnames(transitions)

transitions

transitions_probs <- transitions / rowSums(transitions)
#transitions_probs %>% View()

rm(transitions)

transitions_probs %>% 
  as_tibble() %>% 
  pivot_wider(names_from = next_tag, values_from = n) %>% 
  column_to_rownames("tag") %>% 
  View()

dim(transitions_probs)

transitions_probs %>% sum()
transitions_probs %>% colSums()
transitions_probs %>% rowSums()












# Emission probabilities (probability of state given observation):
emissions <- wsj_train_closed %>%
        select(token, tag) %>%
        na.omit() %>%
        table()

emissions

emissions_probs <- emissions / rowSums(emissions)
emissions_probs
rm(emissions)

dim(emissions_probs)
row.names(emissions_probs)









#emissions_probs['<START>',]
#transitions_probs['<START>',]
#sum(transitions_probs['<START>',])


# Initial probabilities
Pi <- transitions_probs['<START>',]
Pi

sum(Pi)
length(Pi)




# Do the probability matrices sum to 1 per column/row?
#rowSums(emissions_probs) # all 1's
#rowSums(emissions_probs)
#colSums(emissions_probs) %>% sum()

#rowSums(transitions_probs) # all 1's
#colSums(transitions_probs)
#colSums(transitions_probs) %>% sum()

#sum(Pi) # 1









# Initialise HMM
#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)
#row.names(emissions_probs)
#colnames(emissions_probs)

#row.names(transitions_probs)
#colnames(transitions_probs)



# This function initialises a general discrete time and discrete space 
# Hidden Markov Model (HMM). A HMM consists of an alphabet of states and 
# emission symbols. A HMM assumes that the states are hidden from the observer,
# while only the emissions of the states are observable. The HMM is designed
# to make inference on the states through the observation of emissions. 
# The stochastics of the HMM is fully described by the initial starting 
# probabilities of the states, the transition probabilities between states 
# and the emission probabilities of the states.
# 
# States
#   Vector with the names of the states.
# Symbols 
#   Vector with the names of the symbols.
# startProbs 
#   Vector with the starting probabilities of the states.
# transProbs 
#   Stochastic matrix containing the transition probabilities between the states.
# emissionProbs 
#   Stochastic matrix containing the emission probabilities of the states

# These are our observations:
Symbols <- row.names(emissions_probs)
Symbols

# These are our hidden states:
States <- colnames(transitions_probs)
States

hmm <- initHMM(States, # Hidden States
               Symbols, # Symbols, or observations
               transProbs = transitions_probs,
               emissionProbs = emissions_probs %>% t(),
               startProbs = Pi)

#print(hmm)
hmm$States
hmm$Symbols
hmm$startProbs
hmm$transProbs
row.names(hmm$transProbs)
hmm$emissionProbs
row.names(hmm$emissionProbs)


# Sequence of observations
#observations <- c("<START>","confidence","in","the","<OOV>")
observations <- wsj_test_closed[2:29,]$token
observations

actual_hidden_states <- wsj_test_closed[2:29,]$tag
actual_hidden_states


# Calculate Viterbi path
viterbi <- viterbi(hmm, observations)
print(viterbi)


# Simulate from the HMM
simHMM(hmm, 100)


# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(
  viterbi_hidden_states = viterbi,
  actual_hidden_states = actual_hidden_states
)

psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)  















# Going through multiple test set sequences:

# First, split test set into sequences, removing the <START> and <END>
# tokens/tags:

test_sequences <- wsj_test_closed
test_sequences$start <- ifelse(test_sequences$token == '<START>', 1, 0)
test_sequences$seq_id <- cumsum(test_sequences$start)
test_sequences %<>% 
  filter(
    token %nin% c('<START>', '<EOS>')
  ) %>% 
  select(-start)

test_sequences


# With the prepared test sequences, loop through them and add the kappa scores to a list:

options(show.error.messages = FALSE)
#k <- 100
k <- max(test_sequences$seq_id)
viterbi_hidden_states_list <- list(length = k)
kappas <- list(length = k)
successes <- 0
#pb <- progress_bar$new(total = k)
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = k)
for(i in 1:k){
  pb$tick()
  try(
    {
      sequences <- test_sequences %>% filter(seq_id == i)
      actual_observations <- sequences$token
      actual_hidden_states <-  sequences$tag
      viterbi_hidden_states <- HMM::viterbi(hmm, actual_observations)
      viterbi_hidden_states_list[[i]] <- viterbi_hidden_states
      x <- data.frame(
        viterbi_hidden_states = viterbi_hidden_states,
        actual_hidden_states = actual_hidden_states
      )
      kappa <- psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)
      kappas[[i]] <- kappa$weighted.kappa
      successes <- successes + 1
    }
    ,silent = TRUE
  )
}
options(show.error.messages = TRUE)


successes

kappas
avg_kappa <- mean(unlist(kappas))

print(glue("Average kappa score for hidden markov model was: {percent(round(avg_kappa,4))}"))





# Compare actual hidden states to the model's predictions:
comparison_df <- data.frame(
  actual = test_sequences %>% 
    filter(seq_id == 1) %>% 
    select(tag),
  predicted = viterbi_hidden_states_list[[1]]
) %>% 
  mutate(error = ifelse(tag == predicted, 0, 1))

comparison_df











# Janet example -----------------------------------------------------------



transitions <- matrix(
  c(
    0.3777, 8e-04, 0.0322, 0.0366, 0.0096, 0.0068, 0.1147,
    0.011, 2e-04, 5e-04, 4e-04, 0.0176, 0.0102, 0.0021,
    9e-04, 0.7968, 0.005, 1e-04, 0.0014, 0.1011, 2e-04,
    0.0084, 5e-04, 0.0837, 0.0733, 0.0086, 0.1012, 0.2157,
    0.0584, 8e-04, 0.0615, 0.4509, 0.1216, 0.012, 0.4744,
    0.009, 0.1698, 0.0514, 0.0036, 0.0177, 0.0728, 0.0102,
    0.0025, 0.0041, 0.2231, 0.0036, 0.0068, 0.0479, 0.0017
  ),
  nrow=7, # number of rows 
  ncol=7, # number of columns 
  byrow = TRUE
) %>% 
  t()

transitions

transition_labels <- c("NNP", "MD", "VB", "JJ", "NN", "RB", "DT")









emissions <- matrix(
  c(
    0.000032,0,0,0.000048,0,
    0,0.308431,0,0,0,
    0,0.000028,0.000672,0,0.000028,
    0,0,0.00034,0,0,
    0,0.0002,0.000223,0,0.002337,
    0,0,0.010446,0,0,
    0,0,0,0.506099,0
  ),
  nrow=7, # number of rows 
  ncol=5, # number of columns 
  byrow = TRUE
)

emissions

emissions_labels <- c("Janet", "will", "back", "the", "bill")







Pi <- c(0.2767, 6e-04, 0.0031, 0.0453, 0.0449, 0.051, 0.2026)
Pi





# This function initialises a general discrete time and discrete space 
# Hidden Markov Model (HMM). A HMM consists of an alphabet of states and 
# emission symbols. A HMM assumes that the states are hidden from the observer,
# while only the emissions of the states are observable. The HMM is designed
# to make inference on the states through the observation of emissions. 
# The stochastics of the HMM is fully described by the initial starting 
# probabilities of the states, the transition probabilities between states 
# and the emission probabilities of the states.
# 
# States
#   Vector with the names of the states.
# Symbols 
#   Vector with the names of the symbols.
# startProbs 
#   Vector with the starting probabilities of the states.
# transProbs 
#   Stochastic matrix containing the transition probabilities between the states.
# emissionProbs 
#   Stochastic matrix containing the emission probabilities of the states

# These are our observations:
#Symbols <- row.names(emissions)
Symbols <- emissions_labels
Symbols

# These are our hidden states:
#States <- colnames(transitions)
States <- transition_labels
States

hmm <- initHMM(States, # Hidden States
               Symbols, # Symbols, or observations
               transProbs = transitions,
               emissionProbs = emissions,
               startProbs = Pi)
hmm$States
hmm$Symbols
hmm$startProbs
hmm$transProbs
hmm$emissionProbs





# Calculate Viterbi path
viterbi <- viterbi(hmm, c('Janet', 'will', 'back', 'the', 'bill'))
print(viterbi)
# Expected output:
c('NNP', 'MD', 'VB', 'DT', 'NN')



# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(viterbi_sequence = viterbi, actual_sequence = c('NNP', 'MD', 'VB', 'DT', 'NN'))
psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)  





# Simulate from the HMM
simHMM(hmm, 4)
