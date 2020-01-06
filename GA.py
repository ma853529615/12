import numpy
import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import math
import cvxopt.solvers
import numpy.linalg as la
import logging

from SVM import SVM
from tqdm import tqdm
from sklearn.utils import shuffle

class Chromosome:

    def __init__(self, vardim, bound):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        len = self.vardim
        rnd1 = np.random.random(size=len) - 0.5
        rnd2 = np.random.random(size=len) - 0.5
        self.chrom = np.zeros(len)
        self.cig = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                            (self.bound[1, i] - self.bound[0, i]) * rnd1[i]
            self.cig[i] = self.bound[0, i] + \
                          (self.bound[1, i] - self.bound[0, i]) * rnd2[i]

    def calculateFitness(self, dataset):
        self.fitness = SVMResult(
            self.vardim, self.chrom, self.bound, dataset)

    def print_(self):
        print("chrome".format(self.chrom))
        print("cigma".format(self.cig))

class GA:

    def __init__(self, sizepop, vardim, bound, MAXGEN, k, patience):

        self.q = 0
        self.patience = patience  # 引入早停
        self.remain = sizepop
        self.realsize = self.remain
        self.sizepop = self.remain * k
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.k = k
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.dataset = self.creat_dataset(pd.read_csv("diabetes.csv"))
        self.lr = 0.1
        self.flag = 0
        self.resize = []

    def creat_dataset(self, df):

        df = shuffle(df)
        size = len(df)
        df['split'] = 0
        df.iloc[0:math.ceil(0.7 * size), -1] = 'train'
        # df.iloc[math.ceil(0.7*size):math.ceil(0.85*size), -1] = 'test'
        df.iloc[math.ceil(0.15 * size):size, -1] = 'val'  # 调高验证集比例

        return df

    def initialize(self):

        for i in range(0, self.remain):
            chrom = Chromosome(self.vardim, self.bound)
            chrom.generate()
            self.population = np.append(self.population, chrom)
        self.population = np.array(self.population)

    def evaluate(self):

        fitness = []
        for i in tqdm(range(0, self.realsize)):
            self.population[i].calculateFitness(self.dataset)
            # print("###/n")
            fitness.append(self.population[i].fitness)
        self.fitness = np.array(fitness)
        best_idx = np.argmax(self.fitness)
        self.best_score = self.fitness[best_idx]
        self.best = self.population[best_idx]

    def start(self):

        self.t = 0
        self.initialize()
        self.evaluate()
        p = 0
        self.best_fitness = 0
        self.last_best = self.best_fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.ever_best = self.best
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = self.best_fitness
        self.trace[self.t, 1] = self.avefitness
        while (self.t < self.MAXGEN - 1):
            if self.best.fitness > self.best_fitness:
                self.p = 0
                self.ever_best = self.best
                self.best_fitness = self.ever_best.fitness
            elif self.best.fitness == self.ever_best.fitness \
                    or self.best.fitness == self.last_best:
                print(p)
                p += 1
            if p == self.patience:
                self.q += 5
                self.lr = 0.1
                self.k += 1
                self.remain += 5
                self.realsize = self.remain
                self.sizepop = self.remain * self.k
                p = 0
                self.flag = 1
                print("学习率重置")
                self.resize.append(self.t)
            print("第{}代".format(self.t))
            print("本代最好的染色体: {}".format(self.population[bestIndex].chrom))
            print("本代最高分: {}".format(self.population[bestIndex].fitness))
            print("历史最高分: {}".format(self.best_fitness))
            self.t += 1
            self.crossover()
            self.mutation()
            self.evaluate()
            self.selection()
            self.last_best = self.best.fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = self.best_fitness
            self.trace[self.t, 1] = self.avefitness

        print(self.best.chrom)
        self.show()

    def selection(self):

        sort_idx = np.argsort(self.fitness, axis=0).reshape(1, -1)[0]
        top_idx = sort_idx[len(self.fitness) - self.remain - 1:-1]
        new_fitness = copy.deepcopy(self.fitness[top_idx])
        self.fitness = new_fitness
        new = copy.deepcopy(self.population[top_idx])
        self.population = new
        self.realsize = len(self.population)

    """
     def selection(self):

        #轮盘赌
        #print("max: {}".format(np.argmax(self.fitness)))
        #print(self.fitness)
        sort_idx = np.argsort(self.fitness, axis=0).reshape(1,-1)[0]
        #print(sort_idx[0:self.remain])
        new = copy.deepcopy(self.population[sort_idx[0:self.remain]])
        self.population  = new
        self.realsize = len(self.population)
        #print(self.realsize)
    """

    def crossover(self):

        newpop = []
        for i in range(self.sizepop - self.remain - self.q + 5):
            if self.flag == 1:
                idx1 = random.randint(0, self.remain - 5 - 1)
                idx2 = random.randint(0, self.remain - 5 - 1)
                flag = 0
            else:
                idx1 = random.randint(0, self.remain - 1)
                idx2 = random.randint(0, self.remain - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.remain - 1)
            new = copy.deepcopy(self.population[idx1])
            new.chrom += self.population[idx2].chrom
            new.chrom /= 2
            new.cig += self.population[idx2].cig
            new.cig /= 2
            self.population = np.append(self.population, new)

        for i in range(self.remain + self.q - 1):
            new = Chromosome(self.vardim, self.bound)
            new.generate()
            self.population = np.append(self.population, new)
        self.realsize = len(self.population)

    def mutation(self):

        newpop = []
        self.lr *= 0.9
        for i in range(0, self.sizepop - 1):
            p = random.random() - 0.5
            newpop.append(copy.deepcopy(self.population[i]))
            rand = random.random() - 0.5
            for j in range(0, self.vardim):

                randn = random.random() - 0.5
                flag = 1
                while newpop[i].cig[j] == 0 or flag == 1:
                    flag = 0
                    newpop[i].cig[j] = newpop[i].cig[j] * np.exp(rand + randn)
                newpop[i].chrom[j] = newpop[i]. \
                                         cig[j] + newpop[i].cig[j] * randn * self.lr

        newpop.append(self.best)

        self.population = np.array(newpop)
        self.realsize = len(self.population)

    def show(self):

        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='best')
        plt.plot(x, y2, 'g', label='ave')
        plt.xlabel("iter")
        plt.ylabel("acc")
        plt.ylim((0.6, 0.85))
        plt.title("Optim process")
        plt.legend()
        plt.show()
def score(y_bar, val_y):
    Error = 0
    for i in range(len(y_bar)):
        miss = abs(y_bar[i - 1] - val_y[i - 1])
        Error += miss

    return 1 - (Error / len(y_bar))

def SVMResult(vardim, x, bound, dataset):
    X = dataset.loc[dataset['split'] == 'train'].iloc[:, 0:-2].values
    y = dataset.loc[dataset['split'] == 'train'].iloc[:, -2].values
    val_X = dataset.loc[dataset['split'] == 'val'].iloc[:, 0:-2].values
    val_y = dataset.loc[dataset['split'] == 'val'].iloc[:, -2].values
    c = abs(x[0])
    g = abs(x[1])
    # f = x[2]#四参数
    svm = SVM(C=c, gamma=g)
    predictor = svm.train(X, y)
    y_bar = predictor.predict_vec(val_X)

    return score(y_bar, val_y)