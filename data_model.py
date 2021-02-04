import numpy as np
import torch

from utils import *
from functools import reduce
from operator import mul


class DataModel:
    def __init__(self,
                 p: int,
                 rho: float,
                 theta: float,
                 lamb: float,
                 model: str = 'ar'
                 ):
        """
        :param p: number of features in each example.
        :param rho: tunable parameter.
        :param theta: true value of theta.
        :param lamb: tunable parameter.
        :param model: data model to use.
                options: 'ar';auto-regression (default), 'mm';majority-minority, other; multivariate Gaussian.
        """
        self.p = p
        self.rho = rho
        self.theta = theta
        self.lamb = lamb
        if model == 'ar':
            self.mu, self.sigma = self.covariance_AR1(p, self.rho)
        elif model == 'mm':
            self.mu, self.sigma = self.majority_minority(p, self.rho)
        else:
            self.mu, self.sigma = np.zeros(p,), np.identity(p)

    def g0(self, X: torch.Tensor):
        """
        :param X: a tensor of size (num_samples,num_features).
        :return: the actual value of g0(X).
        """
        out = - X[:, 0] ** 2 + 2 * torch.log(0.1 + X[:, 1] ** 2)
        return out

    def m0(self, X: torch.Tensor):
        """
        :param X: a tensor of size (num_samples,num_features).
        :return: the actual value of m0(X).
        """
        out = self.lamb * torch.relu(X[:,1])
        return out

    def generate(self, n):
        """
        Return n samples from the data model.
        :param n: number of samples to generate.
        :return: a tuple of tensors of the double machine learning model.
        """
        with torch.no_grad():
            mvn = np.random.multivariate_normal(self.mu, self.sigma, n)
            X = np_to_torch(mvn)
            D = self.m0(X) + 0.1 * np_to_torch(np.random.randn(n,))
            Y = D * self.theta + self.g0(X) + 0.1 * np_to_torch(np.random.randn(n,))
        return Y, D, X

    def covariance_AR1(self, p: int, rho: float):
        """
        Return mean and covariance of an AR1(rho) process.
        :param p: number of features in each example.
        :param rho: correlation between neighbour variables.
        :return: a tuple with mean vector and covariance matrix.
        """
        mu = np.zeros(p,)
        rho = np.ones(p,) * rho
        sigma = np.zeros(shape=(p,p))
        for i in range(p):
            for j in range(i,p):
                sigma[i][j] = reduce(mul, [rho[l] for l in range(i,j)], 1)
        sigma = np.triu(sigma)+np.triu(sigma).T-np.diag(np.diag(sigma))
        return mu, sigma

    def majority_minority(self, p: int, rho: float=0.9, mmu: float = 3.):
        """
        Return mean and covariance of a majority-minority process.
        :param p: number of features in each example.
        :param rho: the majority ratio.
        :param mmu: the minority mean.
        :return: a tuple with mean vector and covariance matrix.
        """
        p1 = round(rho * p)
        p2 = p - p1
        mu = np.concatenate((np.zeros((p1,)), mmu * np.ones(p2,)), )
        sigma = np.identity(p)
        return mu, sigma