#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown tokens by copying them to the output with 
# probability 1. This is a sensible strategy when translating between 
# languages in Latin script since unknown tokens are often names or numbers.
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

########################################################################
# To alter the underlying dynamic program of the decoder, you need only
# modify the state data structure and the three functions below:
# * initial_state()
# * assign_stack(h)
# * extend_state(h,f)
########################################################################

# this data structure is the fundamental object of a dynamic program for
# monotone phrase-based decoding (also known as a semi-Markov model).
# Field i stores the number of words that have been translated (which are
# always the words from 1 to i). Field lm_state stores the language model
# conditioning context that should be used to compute the probability of
# the next English word in the translation.
state_h = namedtuple("state_h", "i, lm_state")
state_s = namedtuple("state_s" ,"k, j, i,lm_state")  #def state_s(i,j,k,lm_state):
#state_s(i=0,j=0,k=0,lm_state=lm.begin())

# generate an initial hypothesis
def initial_state():
  return state_h(i=0,lm_state=lm.begin())

# determine what stack a hypothesis should be placed in
def assign_stack(s):
  if type(s)==state_h:
    return s.i
  else:
    return s.k+s.i-s.j 
  
     # return s.i,s.j,s.k,s.lm_state
     #state_s stand for skip state_h stand for no skip



# Given an input consisting of partial translation state s and
# associated source sentence f, this function should return a list of
# all possible extensions to it. Each extension must be a tuple
# of the form (new_s, logprob, phrase), in which new_s is a new state
# object, and the edge from s to new_s should be labeled by phrase 
# with weight logprob.
def extend_state(s, f):
  
  if type(s)==state_h:  #state h 
    for j in xrange(s.i+1,len(f)+1):
      if f[s.i:j] in tm: 
        for phrase in tm[f[s.i:j]]:  
          # edge weight includes p_TM
          logprob = phrase.logprob

          # add p_LM probabilities for every word in phrase.english
          lm_state = s.lm_state
          for word in phrase.english.split(): 
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
          # special case: end of sentence
          if j == len(f):
            logprob += lm.end(lm_state)
          new_s = state_h(j, lm_state)
          yield (new_s, logprob, phrase)
        if j != len(f):
          for k in xrange(j+1,len(f)+1):
            if f[j:k] in tm: #translation before i 
              for phrase in tm[f[j:k]]:
                logprob = phrase.logprob
                lm_state=s.lm_state
                for word in phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob 
                new_s=state_s(s.i,j,k,lm_state)
                yield (new_s, logprob, phrase)
  else: #s state
    for phrase in tm[f[s.k:s.j]]:##fill skipped one after only one phrase translation
      logprob=phrase.logprob
      lm_state=s.lm_state
      for word in phrase.english.split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        logprob += word_logprob
      new_s=state_h(s.i,lm_state)
      yield (new_s, logprob, phrase)

    for m in xrange(s.i+1,len(f)+1):#fill skipped one after more than one phrase translation
      if f[s.i:m] in tm: 
        for phrase in tm[f[s.i:m]]:  
          logprob = phrase.logprob
          lm_state = s.lm_state
          for word in phrase.english.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
          new_s=state_s(s.k,s.j,m,lm_state)
          yield (new_s, logprob, phrase)
          # finally, return the new hypothesis

########################################################################
# End of functions requiring modification
########################################################################

# The following code implements a generic stack decoding algorithm
# that is agnostic to the form of a partial translation state.
# It does however assume that all states in stacks[i] represent 
# translations of exactly i source words (though they can be any words).
# It shouldn't be necessary to modify this code if you are only 
# changing the dynamic program, but you should understand how it works.
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # a hypothesis is a node in the decoding search graph. It is parameterized
  # by a state object, defined above.
  hypothesis = namedtuple("hypothesis", "logprob, predecessor, phrase, state")

  # create stacks and add initial state
  stacks = [{} for _ in f] + [{}] # add stack for case of no words are covered
  stacks[0][initial_state()] = hypothesis(0.0, None, None, initial_state())
  for stack in stacks[:-1]:
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for (new_state, logprob, phrase) in extend_state(h.state, f):
        new_h = hypothesis(logprob=h.logprob + logprob, 
                           predecessor=h, 
                           phrase=phrase, 
                           state=new_state)
        j = assign_stack(new_state)
        if new_state not in stacks[j] or stacks[j][new_state].logprob < new_h.logprob: # second case is recombination
          stacks[j][new_state] = new_h
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  # optionally report (Viterbi) log probability of best hypothesis
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
