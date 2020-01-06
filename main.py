import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import math
import cvxopt.solvers
import numpy.linalg as la
import logging
from GA import GA
from SVM import SVM

from sklearn.utils import shuffle

if __name__ == '__main__':
    boundary = np.array([[0, 0], [20, 2]])
    ga = GA(10, 2, boundary, 50, 7, 5)
    ga.start()