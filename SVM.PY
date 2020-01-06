import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import math
import cvxopt.solvers
import numpy.linalg as la
import logging

from sklearn.utils import shuffle

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVM(object):
    def __init__(self, C, gamma):

        self.gamma = gamma
        self._c = C

    def train(self, X, y):

        lagrange_multipliers = self._qp_solver(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):

        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._rbf_kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):

        support_vector_indices = \
            lagrange_multipliers > 1e-5

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                bias=0.0,
                gamma = self.gamma,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            bias=bias,
            gamma = self.gamma,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _qp_solver(self, X, y):

        n_samples, n_features = X.shape

        K = self._gram_matrix(X)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        A = cvxopt.matrix(np.double(y), (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def _rbf_kernel(self, x, y):
        return np.exp(-self.gamma * la.norm(np.subtract(x, y)))


class SVMPredictor(object):

    def __init__(self,
                 bias,
                 gamma,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._bias = bias
        self._gamma = gamma
        self._weights = weights  # shape = (sv_num, )
        self._support_vectors = support_vectors  # shape = (sv_num, feature, )
        self._support_vector_labels = support_vector_labels  # shape = (sv_num, )
        self._s = np.array(
            [alpha * y for alpha, y in zip(self._weights, self._support_vector_labels)])  # shape = (sv_num, )

    def predict(self, x):
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._rbf_kernel(x_i, x)
        return np.sign(result).item()

    def predict_vec(self, X):
        """
        X.shape = (batch, feature)
        """

        k = np.array(
            [[self._rbf_kernel(x_i, X_j) for x_i in self._support_vectors] for X_j in X])  # shape = (batch, sv_num, )
        score = np.dot(k, self._s.reshape(-1, 1))  # shape = (batch, )
        return np.sign(score.reshape(-1, 1))

    def _rbf_kernel(self, x, y):
        return np.exp(-self._gamma * la.norm(np.subtract(x, y)))
