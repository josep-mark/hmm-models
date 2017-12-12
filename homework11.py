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
        probs[0] = {state: self.initial_probs[state] + self.emission_probs[state][sequence[0]] for state in self.initial_probs}
        for i in range(1, len(sequence)):
            
            for state in self.initial_probs:
                prob_sum = 0
                # top = -float('inf')
                top = max([math.exp(probs[i-1][prev_state] + self.transition_probs[prev_state][state]) for prev_state in self.initial_probs])
                print top
                if top == 0:
                    print i, state
                    print [math.exp(probs[i-1][prev_state] + self.transition_probs[prev_state][state]) for prev_state in self.initial_probs]
                    print [probs[i-1][prev_state] + self.transition_probs[prev_state][state] for prev_state in self.initial_probs]
                    print [probs[i-1][prev_state] for prev_state in self.initial_probs]
                    print [self.transition_probs[prev_state][state] for prev_state in self.initial_probs]
                for previous_state in self.initial_probs:
                    # top = max([math.exp(probs[i-1][prev_state] + self.transition_probs[prev_state][state]) for p])    
                    val = math.exp(probs[i-1][previous_state] + self.transition_probs[previous_state][state] - top)
                    prob_sum = prob_sum + val
                    if top ==0:
                        print probs[i-1][previous_state], self.transition_probs[previous_state][state], val, prob_sum
                
                prob_sum = top + math.log(prob_sum) + self.emission_probs[state][sequence[0]]

                probs[i][state] = prob_sum
        return probs 
         

    def forward_probability(self, alpha):
        pass

    def backward(self, sequence):
        pass

    def backward_probability(self, beta, sequence):
        pass

    def forward_backward(self, sequence):
        pass    

    def xi_matrix(self, t, sequence, alpha, beta):
        pass

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
