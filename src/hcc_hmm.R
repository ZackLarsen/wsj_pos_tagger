##################################################
## Project: HCC
## Script purpose: Construct Hidden Markov model in R
## Date: September 25, 2019
## Author: Zack Larsen
##################################################


# Setup -------------------------------------------------------------------

library(pacman)
library(tidyverse)
library(magrittr)
p_load(DBI, odbc, readxl, writexl, lubridate, here, ggthemes, ggalt, conflicted,
        ggThemeAssist, data.table, HMM, psych, scales, visNetwork, collapsibleTree)


conflict_prefer("filter", "dplyr")
conflict_prefer("here", "here")
here()



#Teraconn <- dbConnect(odbc::odbc(), "", password = rstudioapi::askForSecret("password"))
Teraconn <- dbConnect(odbc::odbc(), "")
#options(max.print=1000)
options(scipen=999)


# data i/o ----------------------------------------------------------------

medrx_surg <- dbSendQuery(Teraconn, "
SELECT TOP 36500 mrxs.*
,RANK() OVER (PARTITION BY mid ORDER BY event_date) AS event_order
FROM  mrxs
ORDER BY mid DESC, event_date;
")

medrx_surg <- dbFetch(medrx_surg)

# Sequences ---------------------------------------------------------------



medrx_surg %>% 
  select(mid, event_order, event) %>% 
  arrange(mid, event_order) %>% 
  head(100)


medrx_surg %>% 
  select(mid, event_order, event) %>% 
  arrange(mid, event_order) %>% 
  group_by(event) %>% 
  tally() %>% 
  arrange(-n)



medrx_surg %>% 
  na.omit() %>% 
  select(mid, primy_diag_cd) %>% 
  group_by(primy_diag_cd) %>% 
  tally() %>% 
  arrange(-n)


medrx_surg %>% 
  na.omit() %>% 
  select(mid, primy_proc_cd) %>% 
  group_by(primy_proc_cd) %>% 
  tally() %>% 
  arrange(-n)





medrx_surg %>% 
  summarise_all(n_distinct) %>% 
  t()








# Remove missing observations and strip whitespace:
medrx_surg_complete <- medrx_surg %>% 
  na.omit() %>%
  select(mid, event, primy_diag_cd, primy_proc_cd, cpt_lvl2) %>% 
  mutate(cpt_lvl2 = str_trim(cpt_lvl2, side = c("both", "left", "right")))

medrx_surg_complete







# HMM ---------------------------------------------------------------------



medrx_surg_complete %>% 
  head()






# Split into train/test:
mids <- medrx_surg_complete$mid %>% unique()
mids

length(mids)

train <- medrx_surg_complete %>% 
  filter(mid %in% mids[0:75])

test <- medrx_surg_complete %>% 
  filter(mid %in% mids[75:100])

length(train$mid %>% unique())






n_events <- length(train$event %>% unique())



# Handle OOV events:
event_freq <- train %>% 
  group_by(event) %>% 
  tally() %>% 
  mutate(freq = n/sum(n)) %>% 
  arrange(-n)

event_freq





# Using proportion of events:
vocab <- event_freq %>% head(round(0.9*n_events)) %>% select(event)
vocab_size <- length(vocab$event)
#vocab_size # 15,531

oov_tokens <- event_freq %>% tail(n_events-round(0.9*n_events)) %>% select(event)




medrx_surg_complete_closed <- medrx_surg_complete
medrx_surg_complete_closed[medrx_surg_complete_closed$token %in% oov_tokens$token] <- '<OOV>'

# Do the same for test:
wsj_test_closed <- wsj_test
wsj_test_closed[!wsj_test_closed$token %in% vocab$token] <- '<OOV>'









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




# Emission probabilities (probability of state given observation):
emissions <- wsj_train_closed %>%
  select(token, tag) %>%
  na.omit() %>%
  table()




# Initial probabilities
Pi <- transitions_probs['<START>',]
Pi






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






















































# First steps are to create transitions, emissions, and starting/initial probability matrices:


# Transitions:
transitions <- medrx_surg_complete %>% 
  select(event) %>% 
  mutate(next_event = lead(event)) %>% 
  na.omit() %>% 
  table()

transitions


transitions_probs <- transitions / rowSums(transitions)
transitions_probs










# Emission probabilities
emissions <- medrx_surg_complete %>% 
  select(event, cpt_lvl2) %>% 
  na.omit() %>% 
  table()

emissions

emissions_probs <- emissions / rowSums(emissions)
emissions_probs


emissions_probs %>% 
  as.data.frame() %>% 
  View()



emissions %>% 
  as.matrix() %>% 
  View()









# Do the probability matrices sum to 1 per column/row?

rowSums(emissions_probs) # all 1's
rowSums(emissions_probs)
colSums(emissions_probs) %>% sum() # 9

rowSums(transitions_probs) # all 1's
colSums(transitions_probs)
colSums(transitions_probs) %>% sum() # 9















medrx_surg_complete %>% 
  head()








# Initialise HMM
#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)
row.names(emissions_probs)
colnames(emissions_probs)

row.names(transitions_probs)
colnames(transitions_probs)




hmm <- initHMM(colnames(transitions_probs), 
              row.names(emissions_probs), 
              transProbs = transitions_probs,
              emissionProbs = emissions_probs)

print(hmm)







# Sequence of observations
observations <- c("PRO VISIT W/ NO RX","PRO VISIT W/ GENERIC RX","PRO VISIT W/ NO RX","PRO VISIT W/ NO RX")


# Calculate Viterbi path
viterbi <- viterbi(hmm, observations)
print(viterbi)



# Simulate from the HMM
simHMM(hmm, 100)




#https://www.rdocumentation.org/packages/HMM/versions/1.0/topics/baumWelch
# Baum-Welch training to estimate HMM parameters given initial HMM parameters and observations.
# Think of this as updating the model given more observations. Viterbi training is also possible here
# and converges faster but is less theoretically justified.
# Caution: may get stuck in local optimum.
bw = baumWelch(hmm, observations, 10)
print(bw$hmm)











# Evaluation --------------------------------------------------------------

#https://www.rdocumentation.org/packages/psych/versions/1.8.12/topics/cohen.kappa

psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)  







# Example -----------------------------------------------------------------



# Sample data
dat <- data.frame(replicate(20,sample(c("A", "B", "C","D"), size = 100, replace=TRUE)))
dat



# Function to calculate first-order Markov transition matrix.
# Each *row* corresponds to a single run of the Markov chain
trans.matrix <- function(X, prob=T)
{
  tt <- table( c(X[,-ncol(X)]), c(X[,-1]) )
  if(prob) tt <- tt / rowSums(tt)
  tt
}


# Counts instead of probabilities:
transitions_counts <- trans.matrix(as.matrix(dat), prob = F)
transitions_counts


# Transition probabilities from one state to another:
transitions <- trans.matrix(as.matrix(dat))
transitions






emissions <- ""












#https://www.rdocumentation.org/packages/HMM/versions/1.0/topics/viterbi

# Initialise HMM
#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)

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

hmm = initHMM(States = c("A","B"), # Hidden States
              Symbols = c("L","R"), # Symbols, or observations
              transProbs = matrix(c(.6,.4,.4,.6),2),
              emissionProbs = matrix(c(.6,.4,.4,.6),2),
              startProbs = matrix(c(0.1,0.9)))

print(hmm)


# Sequence of observations
observations = c("L","L","R","R")


# Calculate Viterbi path
viterbi = viterbi(hmm, observations)
print(viterbi)



# Simulate from the HMM
simHMM(hmm, 100)




#https://www.rdocumentation.org/packages/HMM/versions/1.0/topics/baumWelch
# Baum-Welch training to estimate HMM parameters given initial HMM parameters and observations.
# Think of this as updating the model given more observations. Viterbi training is also possible here
# and converges faster but is less theoretically justified.
# Caution: may get stuck in local optimum.
bw = baumWelch(hmm,observations,10)
print(bw$hmm)



# biofam example ----------------------------------------------------------


# Build model
sc_initmod <- build_hmm(observations = wsj, initial_probs = initial_probs,
                        transition_probs = transitions_matrix, emission_probs = emissions_matrix)

# Fit model
sc_fit <- fit_model(sc_initmod)






#https://arxiv.org/pdf/1704.00543.pdf

data("biofam", package = "TraMineR")
biofam_seq <- seqdef(biofam[, 10:25], start = 15, 
                     labels = c("parent","left", "married", "left+marr",
                                "child", "left+child", "left+marr+ch",
                                "divorced"))

data("biofam3c")
marr_seq <- seqdef(biofam3c$married, start = 15, 
                   alphabet = c("single","married", "divorced"))
child_seq <- seqdef(biofam3c$children, start = 15,
                    alphabet = c("childless", "children"))
left_seq <- seqdef(biofam3c$left, start = 15, 
                   alphabet = c("with parents","left home"))


sc_init <- c(0.9, 0.06, 0.02, 0.01, 0.01)
sc_trans <- matrix(c(0.80, 0.10, 0.05, 0.03, 0.02, 
                     0.02, 0.80, 0.10, 0.05, 0.03,
                     0.02, 0.03, 0.80, 0.10, 0.05, 
                     0.02, 0.03, 0.05, 0.80, 0.10,
                     0.02, 0.03, 0.05, 0.05, 0.85), 
                   nrow = 5, ncol = 5, byrow = TRUE)

sc_emiss <- matrix(NA, nrow = 5, ncol = 8)
sc_emiss[1,] <- seqstatf(biofam_seq[, 1:4])[, 2] + 0.1
sc_emiss[2,] <- seqstatf(biofam_seq[, 5:7])[, 2] + 0.1
sc_emiss[3,] <- seqstatf(biofam_seq[, 8:10])[, 2] + 0.1
sc_emiss[4,] <- seqstatf(biofam_seq[, 11:13])[, 2] + 0.1
sc_emiss[5,] <- seqstatf(biofam_seq[, 14:16])[, 2] + 0.1
sc_emiss <- sc_emiss / rowSums(sc_emiss)
rownames(sc_trans) <- colnames(sc_trans) <- rownames(sc_emiss) <- paste("State", 1:5)
colnames(sc_emiss) <- attr(biofam_seq, "labels")

sc_trans
round(sc_emiss, 3)


sc_initmod <- build_hmm(observations = biofam_seq, initial_probs = sc_init,
                        transition_probs = sc_trans, emission_probs = sc_emiss)

# Estimate parameters
sc_fit <- fit_model(sc_initmod)
sc_fit$logLik
sc_fit$model




mc_init <- c(0.9, 0.05, 0.02, 0.02, 0.01)
mc_trans <- matrix(c(0.80, 0.10, 0.05, 0.03, 0.02, 
                     0, 0.90, 0.05, 0.03, 0.02, 
                     0, 0, 0.90, 0.07, 0.03,
                     0, 0, 0, 0.90, 0.10,
                     0, 0, 0, 0, 1),
                   nrow = 5, ncol = 5, byrow = TRUE)

mc_emiss_marr <- matrix(c(0.90, 0.05, 0.05, 0.90, 0.05,
                          0.05, 0.05, 0.90, 0.05, 0.05,
                          0.90, 0.05, 0.30, 0.30, 0.40),
                        nrow = 5, ncol = 3,byrow = TRUE)
mc_emiss_child <- matrix(c(0.9, 0.1, 0.9, 0.1, 0.1,
                           0.9, 0.1, 0.9, 0.5,0.5),
                         nrow = 5, ncol = 2, byrow = TRUE)
mc_emiss_left <- matrix(c(0.9, 0.1, 0.1, 0.9, 0.1,
                          0.9, 0.1, 0.9, 0.5,0.5),
                        nrow = 5, ncol = 2, byrow = TRUE)
mc_obs <- list(marr_seq, child_seq, left_seq)
mc_emiss <- list(mc_emiss_marr, mc_emiss_child, mc_emiss_left)

mc_initmod <- build_hmm(observations = mc_obs, initial_probs = mc_init,
                        transition_probs = mc_trans, emission_probs = mc_emiss,
                        channel_names = c("Marriage", "Parenthood", "Residence"))

mc_initmod



mc_fit <- fit_model(mc_initmod, em_step = FALSE, local_step = TRUE,
                    threads = 4)

hmm_biofam <- mc_fit$model
BIC(hmm_biofam)


plot(hmm_biofam)

plot(hmm_biofam, vertex.size = 50, vertex.label.dist = 1.5,
     edge.curved = c(0, 0.6, -0.8, 0.6, 0, 0.6, 0), legend.prop = 0.3,
     combined.slice.label = "States with prob. < 0.05")


vertex_layout <- matrix(c(1, 2, 2, 3, 1, 0, 0.5, -0.5, 0, -1), ncol = 2)
plot(hmm_biofam, layout = vertex_layout, xlim = c(0.5, 3.5),
     ylim = c(-1.5, 1), rescale = FALSE, vertex.size = 50, 
     vertex.label.pos = c("left", "top", "bottom", "right", "left"),
     edge.curved = FALSE, edge.width = 1, edge.arrow.size = 1,
     with.legend = "left", legend.prop = 0.4, label.signif = 1,
     combine.slices = 0, cpal = colorpalette[[30]][c(14:5)])


ssplot(hmm_biofam, plots = "both", type = "I", sortv = "mds.hidden",
       title = "Observed and hidden state sequences", xtlab = 15:30,
       xlab = "Age")




