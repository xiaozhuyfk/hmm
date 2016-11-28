"""
File: hmm.py
Author: Hongyu Li (hongyul)
"""

import numpy as np

class HMM(object):

    def __init__(self, transition, emission, priors):
        self.transition = transition
        self.emission = emission
        self.priors = priors
        self.M, self.N = emission.shape

    def forward(self, sequence):
        M = self.M
        T = len(sequence)
        if T == 0:
            return 0

        # initialize first column of alpha
        alpha = np.zeros((M, T))
        for i in xrange(M):
            alpha[i][0] = np.log(self.priors[i]) + np.log(self.emission[i][sequence[0]])

        for t in xrange(1,T):
            for i in xrange(M):
                # tmp = [alpha(t-1, 1) + log S(1, i), ..., alpha(t-1, M) + log S(M, i)]
                tmp = [alpha[k][t-1] + np.log(self.transition[k][i]) for k in range(M)]
                alpha[i][t] = np.log(self.emission[i][sequence[t]]) + reduce(np.logaddexp, tmp)

        # sum of alpha(T, i) in log space
        return reduce(np.logaddexp, alpha[:,-1])

    def backward(self, sequence):
        M = self.M
        T = len(sequence)
        if T == 0:
            return 0

        # initialize beta with 0, because log(1.0) = 0
        beta = np.zeros((M, T))
        for t in range(1,T)[::-1]:
            for i in xrange(M):
                tmp = [beta[k][t] + # beta(t, k)
                       np.log(self.transition[i][k]) + # S(i, k)
                       np.log(self.emission[k][sequence[t]]) # E(k, t)
                       for k in range(M)]
                # beta(i, t-1) = sum of (beta(t,k) * S(i,k) * E(k,t)) in log space
                beta[i][t-1] = reduce(np.logaddexp, tmp)

        tmp = [np.log(self.priors[i]) + # PI_i, prior of ith state
               np.log(self.emission[i][sequence[0]]) + # E(i, 1)
               beta[i][0] # beta(1, i)
               for i in range(M)]
        # sum of product of PI * E * beta in log space
        return reduce(np.logaddexp, tmp)

    def viterbi(self, sequence):
        M = self.M
        T = len(sequence)
        if T == 0:
            return []

        # initialize V matrix of shape (T, M)
        # each entry is an array of indices of POS tags
        V = []
        for t in xrange(T):
            v = []
            for i in xrange(M):
                v.append([i])
            V.append(v)

        # initialize B matrix of shape (T, M)
        B = np.zeros((T, M))
        for i in xrange(M):
            B[0][i] = self.priors[i] * self.emission[i][sequence[0]]

        for t in xrange(1, T):
            for i in xrange(M):
                tmp = [B[t-1][k] * self.transition[k][i] * self.emission[i][sequence[t]] for k in range(M)]
                k = np.argmax(tmp)
                B[t][i] = tmp[k]
                V[t][i] = V[t-1][k] + [i]
        k = np.argmax(B[-1])
        return V[-1][k]







