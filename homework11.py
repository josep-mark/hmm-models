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
        alpha = self.forward(sequence)
        beta = self.backward(sequence)
      
        xi_matrices = {i: self.xi_matrix(i, sequence, alpha, beta) for i in range(len(sequence)-2)}
        initial_prime = deepcopy(self.initial_probs)
        transition_prime = deepcopy(self.transition_probs)
        emission_prime = deepcopy(self.emission_probs)
        ###INITIAL####
        for i in self.initial_probs:
            initial_prime[i] = self.calculate_yi(xi_matrices[0], i)
        ###TRANSITION####
        for i in self.initial_probs:
            for j in self.initial_probs:
                t_num = 0
                t_denom = 0
                t_num_max = max([xi_matrices[t][i][j] for t in range(len(sequence)-2)])
                denom_max = max([self.calculate_yi(m, i) for t, m in xi_matrices.iteritems()])
                for t in range(len(sequence)-2):
                    t_num = t_num + math.exp(xi_matrices[t][i][j] - t_num_max)
                    # e_num = e_num + math.exp(self.calculate_yi(xi_matrices[t], i))
                    t_denom = t_denom + math.exp(self.calculate_yi(xi_matrices[t], i) - denom_max)
                    # t_denom = math.exp(xi_matrices[t][i])
                t_num = math.log(t_num) + t_num_max
                # e_x = math.log(e_num) + e_num_max
                t_denom = math.log(t_denom) + denom_max
                # print transition_prime[i]
                # print transition_prime[i][j]
                transition_prime[i][j] = t_num - t_denom
                # emission_prime[i]
        ###EMISSION####
        ###0 -> T-1
        for i in self.initial_probs:
            for symbol in self.emission_probs[i]:
                num = 0
                denom = 0
                num_max = max([self.calculate_yi(xi_matrices[t], i) for t in range(len(sequence)-2) if sequence[t] == symbol ])
                denom_max = max([self.calculate_yi(m, i) for t, m in xi_matrices.iteritems()])
                
                final_num = alpha[len(sequence)-1][i] + beta[len(sequence)-1][i]
                final_max = max([(alpha[len(sequence)-1][i] + beta[len(sequence)-1][i]) for i in self.initial_probs])
                final_denom = sum([math.exp(alpha[len(sequence)-1][j] + beta[len(sequence)-1][j] - final_max) for j in self.initial_probs])
                final_denom = math.log(final_denom) + final_max
                final_yi = final_num - final_denom

                if final_yi > num_max:
                    num_max = final_yi
                if final_yi > denom_max:
                    denom_max = final_yi

                for t in range(len(sequence)-2):
                    if sequence[t] == symbol:
                        num = num + math.exp(self.calculate_yi(xi_matrices[t], i) - num_max)
                    denom = denom + math.exp(self.calculate_yi(xi_matrices[t], i) - denom_max)
                
                if sequence[len(sequence)-1] == symbol:
                    num = num + math.exp(final_yi - num_max)
                denom = denom + math.exp(final_yi - denom_max) 
                x = num_max + math.log(num)
                y = denom_max + math.log(denom)
                emission_prime[i][symbol] = x - y
        return (initial_prime, transition_prime, emission_prime)



    def calculate_yi(self, xi_matrix, i):
        top = max([xi_matrix[i][j] for j in self.initial_probs])
        total = sum([math.exp(xi_matrix[i][j]-top) for j in self.initial_probs])
        result = math.log(total) + top
        return result
    # def calculate_emission_num(self, t, sequence, xi_matrix, i):
        # observed = sequence[t]
        # for state in self.emission_probs:

        # top = max([self.emission_probs[state] *1 for t in range(len(sequence))])
        # for state in self.emission_probs[0]:


    def xi_matrix(self, t, sequence, alpha, beta):
        greek_letter = {state: {state_2: 0.0 for state_2 in self.initial_probs} for state in self.initial_probs}
        # denom = self.forward_probability(alpha)
        denom = self.xi_helper(t, alpha, beta, sequence)
        for i in self.initial_probs:
            for j in self.initial_probs:
                num = alpha[t][i] + self.transition_probs[i][j] + self.emission_probs[j][sequence[t+1]] + beta[t+1][j]
                greek_letter[i][j] = num - denom
        return greek_letter

    def xi_helper(self, t, alpha, beta, sequence):
        # tops = -float('inf')
        # for i in self.initial_probs: 
        #     for j in self.initial_probs:
        #         x = alpha[t][i] + self.transition_probs[i][j] + self.emission_probs[j][sequence[t+1]] + beta[t+1][j]
        #         if x > tops:
        #             tops = x
        tops = max([alpha[t][i] + self.transition_probs[i][j] + self.emission_probs[j][sequence[t+1]] + beta[t+1][j] 
            for i in self.initial_probs for j in self.initial_probs])
        total = 0
        for i in self.initial_probs:
            for j in self.initial_probs:
                total = total + math.exp(alpha[t][i] + self.transition_probs[i][j] + self.emission_probs[j][sequence[t+1]] + beta[t+1][j] - tops)
        result = math.log(total) + tops
        return result

    def update(self, sequence, cutoff_value):
        while True:
            alpha = self.forward(sequence)
            prev = self.forward_probability(alpha)
            probs = self.forward_backward(sequence)
            self.initial_probs = probs[0]
            self.transition_probs = probs[1]
            self.emission_probs = probs[2]
            alpha2 = self.forward(sequence)
            next = self.forward_probability(alpha2)
            if prev - next < 1:
                break

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
~10 hrs
"""

feedback_question_2 = """
most challenging was the forward-backward step and debugging.
"""

feedback_question_3 = """
The forward-backward algorithm was interesting but a pain to implement
"""
