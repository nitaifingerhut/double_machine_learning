import numpy as np

from functools import reduce
from operator import mul
from typing import Tuple


class CovarianceAR1(object):
    """
    Data model of AR1(rho) process.
    """

    def __init__(self, rho: float):
        self.rho = rho

    def __call__(self, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return mean and covariance of data model.
        :param p: dimension of features.
        :return: a tuple with mean vector and covariance matrix.
        """
        mu = np.zeros(p,)
        rho = np.ones(p,) * self.rho
        sigma = np.zeros(shape=(p, p))
        for i in range(p):
            for j in range(i, p):
                sigma[i][j] = reduce(mul, [rho[k] for k in range(i, j)], 1)
        sigma = np.triu(sigma) + np.triu(sigma).T - np.diag(np.diag(sigma))
        return mu, sigma


class MajorityMinority(object):
    """
    Data model of majority-minority groups.
    """

    def __init__(self, rho: float, mmu: float = 3.0):
        self.rho = rho
        self.mmu = mmu

    def __call__(self, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return mean and covariance of data model.
        :param p: dimension of features.
        :return: a tuple with mean vector and covariance matrix.
        """
        p1 = round(self.rho * p)
        p2 = p - p1
        mu = np.concatenate((np.zeros((p1,)), self.mmu * np.ones(p2,)),)
        sigma = np.identity(p)
        return mu, sigma


class MultivariateGaussian(object):
    """
    Data model of multivariate Gaussian.
    """

    def __init__(self, rho: float):
        self.rho = rho

    def __call__(self, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return mean and covariance of data model.
        :param p: dimension of features.
        :return: a tuple with mean vector and covariance matrix.
        """
        return np.zeros(p,), np.identity(p)
