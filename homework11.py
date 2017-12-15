############################################################
# CIS 521: Final Project / Homework 11
############################################################

student_name = "Joseph Franz"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import re
import string
import pickle
import math
from copy import deepcopy


############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    f = open(path, 'r')
    x = [corpus_helper(word) for line in f for word in line.split() if corpus_helper(word) is not ""]
    # print x
    return " ".join(x)
    # for line in f.readline():
    #     line = line.lower()
    #     for c in line:
    #         if c in string.punctation:

def corpus_helper(word):
    word = word.lower()
    word.replace(" ", "")
    # print word
    for c in word:
        if c in string.punctuation or c in string.whitespace:
            word = word.replace(c, "")
    # for c in string.punctuation:
    #     if c in word:        
    #         word = word.replace(c, "")
    # print word
    return word



def load_parameters(path):
    f = open(path)
    unpickler = pickle.Unpickler(f)
    pic = unpickler.load()

    for i in range(len(pic)):
        if i == 0:
            for state in pic[i]:
                pic[i][state] = math.log(pic[i][state])
        elif i == 1 or i == 2:
            for state in pic[i]:
                for state2 in pic[i][state]:
                    pic[i][state][state2] = math.log(pic[i][state][state2])
    return pic



class HMM(object):
    initial_probs = {}
    transition_probs = {}
    emission_probs = {}
    
    def __init__(self, probabilities):
        self.initial_probs = probabilities[0]
        self.transition_probs = probabilities[1]
        self.emission_probs = probabilities[2]    

    def get_parameters(self):
        return (self.convert_parameters(self.initial_probs, 0), self.convert_parameters(self.transition_probs, 1), self.emission_probs, 2)
    
    def convert_parameters(self, prob, i):
        if i == 0:
            initial = deepcopy(prob)
            for state in initial:
                initial[state] = math.exp(initial[state])
            return initial
        elif i == 1 or i == 2:
            vector = deepcopy(prob)
            for state in vector:
                for state2 in vector[state]:
                    vector[state][state2] = math.exp(vector[state][state2])
        return vector


    def forward(self, sequence):
        probs = [{} for i in range(len(sequence))]

        ##INITIALIZATION##
        probs[0] = {state: self.initial_probs[state] + self.emission_probs[state][sequence[0]] for state in self.initial_probs}
        
        ##INDUCTION##
        for i in range(1, len(sequence)):
            
            for state in self.initial_probs:
                prob_sum = 0
                # top = max([probs[i-1][prev_state] + self.transition_probs[prev_state][state] for prev_state in self.initial_probs])
                top = max([probs[i-1][prev_state] + self.transition_probs[prev_state][state] for prev_state in self.initial_probs])
                for previous_state in self.initial_probs:
                    # top = max([math.exp(probs[i-1][prev_state] + self.transition_probs[prev_state][state]) for p])    
                    val = math.exp(probs[i-1][previous_state] + self.transition_probs[previous_state][state] - top)
                    prob_sum = prob_sum + val

                p = top + math.log(prob_sum) + self.emission_probs[state][sequence[i]]

                probs[i][state] = p
        return probs 
         

    def forward_probability(self, alpha):
        termination = alpha[len(alpha)-1]
        high = max([termination[state] for state in termination])
        total = 0
        for state in termination:
            total = total + math.exp(termination[state] - high)
        result = high + math.log(total)
        return result

    def backward(self, sequence):
        probs = [{} for i in range(len(sequence))]
        probs[len(sequence)-1] = {state: math.log(1.0) for state in self.initial_probs}
        start = len(sequence)-2
        ##INDUCTION##
        for i in range(start, -1, -1):
            ## Sum tansition(i,j) * emission(j) (observation t+1) * prob (t+1) at j
            ## i = current state
            ## j = range of states
            ## current time + 1
            for state in self.initial_probs:
                probs_sum = 0
                high = max([self.transition_probs[state][next_state] + self.emission_probs[next_state][sequence[i+1]] + probs[i+1][next_state] for next_state in self.initial_probs])
                for next_state in self.initial_probs:
                    log_prob = self.transition_probs[state][next_state] + self.emission_probs[next_state][sequence[i+1]] + probs[i+1][next_state] - high
                    exp = math.exp(log_prob)
                    probs_sum = probs_sum + exp
                result = math.log(probs_sum) + high
                probs[i][state] = result
        return probs

    def backward_probability(self, beta, sequence):
        ##sum i = 1 to N sum pi[i] * b[i][o1] * probs[1][state]
        initial_trellis = beta[0]
        high = max([self.initial_probs[state] + self.emission_probs[state][sequence[0]] + initial_trellis[state] for state in self.initial_probs])
        # print high
        total = 0
        for state in initial_trellis:
            val = math.exp(self.initial_probs[state]+self.emission_probs[state][sequence[0]] + initial_trellis[state] - high)
            total = total + val 
        result = math.log(total) + high
        return result

    def forward_backward(self, sequence):
        pass    

    def xi_matrix(self, t, sequence, alpha, beta):
        greek_letter = {state: {state_2: 0.0 for state_2 in self.initial_probs} for state in self.initial_probs}
        for i in self.initial_probs:
            for j in self.initial_probs:
                num = alpha[t][i] + self.transition_probs[i][j] + self.emission_probs[j][sequence[t+1]] + beta[t+1][j]
                denom = 1
                greek_letter[i][j] = val

    def update(self, sequence, cutoff_value):
        pass

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
